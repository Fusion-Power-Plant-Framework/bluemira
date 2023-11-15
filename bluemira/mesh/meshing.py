# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Core functionality for the bluemira mesh module.
"""

from __future__ import annotations

import inspect
import pprint
from dataclasses import asdict, dataclass
from enum import Enum, IntEnum, auto
from typing import TYPE_CHECKING, Callable, Dict, Iterable, List, Optional, Union

import gmsh

from bluemira.base.look_and_feel import bluemira_print
from bluemira.mesh.error import MeshOptionsError

if TYPE_CHECKING:
    from bluemira.base.components import Component


# Mesh options for the moment are limited to definition of mesh size for each point (
# quantity called lcar to be consistent with gmsh) and the definition of physical
# groups.


@dataclass
class DefaultMeshOptions:
    """Default mesh options"""

    lcar: Optional[float] = None
    physical_group: Optional[float] = None


class MeshTags(IntEnum):
    """Mesh tags and dimensions"""

    POINTS = 0
    CNTRPOINTS = 0
    CURVE = 1
    SURFACE = 2
    CURVELOOP = -1  # TODO what is the num


class MeshTagsNC(IntEnum):
    """Mesh tags and dimensions

    CURVELOOP is not in this class.
    All entries are equal to the equivalent entry in MeshTags
    """

    POINTS = MeshTags.POINTS
    CNTRPOINTS = MeshTags.CNTRPOINTS
    CURVE = MeshTags.CURVE
    SURFACE = MeshTags.SURFACE


class GEOS(IntEnum):
    """Supported geometry types and thier dimesions"""

    BluemiraWire = 1
    BluemiraFace = 2
    BluemiraShell = 2
    BluemiraCompound = 2


SUPPORTED_GEOS = tuple(GEOS.__members__.keys())


def get_default_options() -> DefaultMeshOptions:
    """
    Returns the default display options.
    """
    return DefaultMeshOptions()


class MeshOptions:
    """
    The options that are available for meshing objects.
    """

    def __init__(self, **kwargs):
        self._options = get_default_options()
        self.modify(**kwargs)

    @property
    def lcar(self) -> Union[float, None]:
        """
        Mesh size of points.
        """
        return self._options.lcar

    @lcar.setter
    def lcar(self, val: float):
        self._options.lcar = val

    @property
    def physical_group(self) -> Union[float, None]:
        """
        Definition of physical groups.
        """
        return self._options.physical_group

    @physical_group.setter
    def physical_group(self, val: float):
        self._options.physical_group = val

    def as_dict(self) -> dict[str, Union[float, None]]:
        """
        Returns the instance as a dictionary.
        """
        return asdict(self._options)

    def modify(self, **kwargs):
        """
        Function to override meshing options.
        """
        for k in kwargs:
            if hasattr(self._options, k):
                setattr(self._options, k, kwargs[k])

    def __repr__(self) -> str:
        """
        Representation string of the DisplayOptions.
        """
        return f"{type(self).__name__}({pprint.pformat(self._options)}" + "\n)"


class Meshable:
    """Mixin class to make a class meshable"""

    def __init__(self):
        super().__init__()
        self._mesh_options = MeshOptions()

    @property
    def mesh_options(self) -> MeshOptions:
        """
        The options that will be used to mesh the object.
        """
        return self._mesh_options

    @mesh_options.setter
    def mesh_options(self, value: Union[MeshOptions, Dict]):
        if isinstance(value, MeshOptions):
            self._mesh_options = value
        elif isinstance(value, Dict):
            if self.mesh_options is None:
                self.mesh_options = MeshOptions()
            self.mesh_options.modify(**value)
        else:
            raise MeshOptionsError("Mesh options must be set to a MeshOptions instance.")


class _GmshEnum(Enum):
    SHELL = "BluemiraShell"
    COMPOUND = "BluemiraCompound"


class GmshFileType(Enum):
    """Gmsh file output types"""

    DEFAULT = auto()
    GMSH = auto()


class Mesh:
    """
    A class for supporting the creation of meshes and writing out those meshes to files.
    """

    def __init__(
        self,
        modelname: str = "Mesh",
        terminal: int = 0,
        meshfile: Optional[Union[str, List[str]]] = None,
        logfile: str = "gmsh.log",
    ):
        self.modelname = modelname
        self.terminal = terminal
        self.meshfile = (
            ["Mesh.geo_unrolled", "Mesh.msh"] if meshfile is None else meshfile
        )
        self.logfile = logfile

    @staticmethod
    def _check_meshfile(meshfile: Union[str, list]) -> List[str]:
        """
        Check the mesh file input.
        """
        # todo: should be implemented also a check on the file extension. Only a
        # limited type of file extensions is allowed by gmsh.
        if isinstance(meshfile, str):
            meshfile = [meshfile]
        elif isinstance(meshfile, list):
            if len(meshfile) < 1:
                raise ValueError("meshfile is an empty list")
        else:
            raise TypeError("meshfile must be a string or a list of strings")
        return meshfile

    @property
    def meshfile(self) -> List[str]:
        """
        The path(s) to the file(s) containing the meshes.
        """
        return self._meshfile

    @meshfile.setter
    def meshfile(self, meshfile: Union[str, List[str]]):
        self._meshfile = self._check_meshfile(meshfile)

    def __call__(self, obj: Union[Component, Meshable], dim: int = 2):
        """
        Generate the mesh and save it to file.
        """
        bluemira_print("Starting mesh process...")

        if "Component" in [c.__name__ for c in inspect.getmro(type(obj))]:
            from bluemira.base.tools import (  # noqa: PLC0415
                create_compound_from_component,
            )

            obj = create_compound_from_component(obj)

        if isinstance(obj, Meshable):
            # gmsh is initialised
            _FreeCADGmsh._initialise_mesh(self.terminal, self.modelname)
            # Mesh the object. A dictionary with the geometrical and internal
            # information that are used by gmsh is returned. In particular,
            # a gmsh key is added to any meshed entity.
            buffer = self.__mesh_obj(obj, dim=dim)
            # Check for possible intersection (only allowed at the boundary to adjust
            # the gmsh_dictionary
            Mesh.__iterate_gmsh_dict(buffer, Mesh._check_intersections)

            # Create the physical groups
            self._apply_physical_group(buffer)

            # apply the mesh size
            self._apply_mesh_size(buffer)

            # generate the mesh
            _FreeCADGmsh._generate_mesh()

            # save the mesh file
            for file in self.meshfile:
                _FreeCADGmsh._save_mesh(file)

            # close gmsh
            _FreeCADGmsh._finalize_mesh(self.logfile)
        else:
            raise TypeError("Only Meshable objects can be meshed")

        bluemira_print("Mesh process completed.")

        return buffer

    def __mesh_obj(self, obj, dim: int):
        """
        Function to mesh the object.
        """
        from bluemira.geometry.tools import serialize_shape  # noqa: PLC0415

        if not hasattr(obj, "ismeshed") or not obj.ismeshed:
            if type(obj).__name__ not in SUPPORTED_GEOS:
                raise ValueError(
                    f"Mesh procedure not implemented for {obj.__class__.__name__} type."
                )

            # object is serialised into a dictionary
            buffer = serialize_shape(obj)

            # Each object is recreated into gmsh. Here there is a trick: in order to
            # allow the correct mesh in case of intersection, the procedure
            # is made meshing the objects with increasing dimension.
            for d in range(1, dim + 1, 1):
                self.__convert_item_to_gmsh(buffer, d)
            obj.ismeshed = True
        else:
            bluemira_print("Object already meshed")
        return buffer

    def __convert_item_to_gmsh(self, buffer: dict, dim: int):
        for k in buffer:
            if k == "BluemiraWire":
                self.__convert_wire_to_gmsh(buffer, dim)
            if k == "BluemiraFace":
                self.__convert_face_to_gmsh(buffer, dim)
            if k in {"BluemiraShell", "BluemiraCompound"}:
                self.__convert_compound_shell_to_gmsh(
                    buffer, dim, converter=_GmshEnum(k)
                )

    def _apply_physical_group(self, buffer: dict):
        """
        Function to apply physical groups
        """
        for k, v in buffer.items():
            if k in SUPPORTED_GEOS:
                if "physical_group" in v:
                    _FreeCADGmsh.add_physical_group(
                        GEOS[k].value,
                        self.get_gmsh_dict(buffer, GmshFileType.DEFAULT)[
                            MeshTags(GEOS[k].value)
                        ],
                        v["physical_group"],
                    )
                for o in v["boundary"]:
                    self._apply_physical_group(o)

    def _apply_mesh_size(self, buffer: dict):
        """
        Function to apply mesh size.
        """
        # mesh size is applied not only to the vertexes of the defined geometry,
        # but also to the intersection points (new vertexes). For this reason,
        # it is important to do this operation after the completion of the mesh
        # procedure.
        points_lcar2 = self.__create_dict_for_mesh_size(buffer)
        if len(points_lcar2) > 0:
            for p in points_lcar2:
                _FreeCADGmsh._set_mesh_size([(0, p[0])], p[1])

    def __create_dict_for_mesh_size(self, buffer: dict):
        """
        Function to create the correct dictionary format for the
        application of the mesh size.
        """
        points_lcar = []
        for k, v in buffer.items():
            if k in SUPPORTED_GEOS:
                if "lcar" in v and v["lcar"] is not None:
                    points_tags = self.get_gmsh_dict(buffer, GmshFileType.GMSH)[
                        MeshTags(GEOS[k].value)
                    ]
                    if len(points_tags) > 0:
                        points_lcar += [(p[1], v["lcar"]) for p in points_tags]
                for o in v["boundary"]:
                    points_lcar += self.__create_dict_for_mesh_size(o)
        points_lcar = sorted(points_lcar, key=lambda element: (element[0], element[1]))
        points_lcar.reverse()
        points_lcar = dict(points_lcar)
        return list(points_lcar.items())

    @staticmethod
    def __apply_fragment(
        buffer: dict,
        dim: Iterable[int] = (2, 1, 0),
        all_ent=None,
        tools: Optional[list] = None,
        remove_object: bool = True,
        remove_tool: bool = True,
    ):
        """
        Apply the boolean fragment operation.
        """
        all_ent, _oo, oov = _FreeCADGmsh._fragment(
            dim, all_ent, [] if tools is None else tools, remove_object, remove_tool
        )
        Mesh.__iterate_gmsh_dict(buffer, _FreeCADGmsh._map_mesh_dict, all_ent, oov)

    @staticmethod
    def _check_intersections(gmsh_dict: dict):
        """
        Check intersection and add the necessary vertexes to the gmsh dict.
        """
        if len(gmsh_dict[MeshTags.CURVE]) > 0:
            gmsh_curve_tag = [(1, tag) for tag in gmsh_dict[MeshTags.CURVE]]
            gmsh_dict[MeshTags.POINTS] = list(
                {tag[1] for tag in _FreeCADGmsh._get_boundary(gmsh_curve_tag)}
            )

    @staticmethod
    def __iterate_gmsh_dict(buffer: dict, function: Callable, *args):
        """
        Supporting function to iterate over a gmsh dict.
        """
        if "BluemiraWire" in buffer:
            boundary = buffer["BluemiraWire"]["boundary"]
            if "gmsh" in buffer["BluemiraWire"]:
                function(buffer["BluemiraWire"]["gmsh"], *args)
            for item in boundary:
                for k in item:
                    if k == "BluemiraWire":
                        Mesh.__iterate_gmsh_dict(item, function, *args)

        for buffer_type in ("BluemiraFace", "BluemiraShell", "BluemiraCompound"):
            if buffer_type in buffer:
                boundary = buffer[buffer_type]["boundary"]
                if "gmsh" in buffer[buffer_type]:
                    function(buffer[buffer_type]["gmsh"], *args)
                for item in boundary:
                    Mesh.__iterate_gmsh_dict(item, function, *args)

    def __convert_wire_to_gmsh(self, buffer: dict, dim: int):
        """
        Converts a wire to gmsh. If dim is not equal to 1, wire is not meshed.
        """
        for type_, value in buffer.items():
            if type_ == "BluemiraWire":
                boundary = value["boundary"]
                if dim == 1:
                    value["gmsh"] = {
                        MeshTags.POINTS: [],
                        MeshTags.CNTRPOINTS: [],
                        MeshTags.CURVE: [],
                        MeshTags.CURVELOOP: [],
                        MeshTags.SURFACE: [],
                    }
                    for item in boundary:
                        for btype_, bvalue in item.items():
                            if btype_ == "BluemiraWire":
                                self.__convert_wire_to_gmsh(item, dim)
                            else:
                                for curve in bvalue:
                                    curve_gmsh_dict = _FreeCADGmsh.create_gmsh_curve(
                                        curve
                                    )
                                    for key in (
                                        MeshTags.POINTS,
                                        MeshTags.CNTRPOINTS,
                                        MeshTags.CURVE,
                                    ):
                                        value["gmsh"][key] += curve_gmsh_dict[key]

                    # get the dictionary of the BluemiraWire defined in buffer
                    # as default and gmsh format
                    dict_gmsh = self.get_gmsh_dict(buffer, GmshFileType.GMSH)

                    # fragment points_tag and curves
                    self.__apply_fragment(
                        buffer,
                        dict_gmsh[MeshTags.POINTS] + dict_gmsh[MeshTags.CURVE],
                        [],
                        False,
                        False,
                    )
            else:
                raise NotImplementedError(f"Serialisation non implemented for {type_}")

    def __convert_face_to_gmsh(self, buffer: dict, dim: int):
        """
        Converts a face to gmsh.
        """
        for type_, value in buffer.items():
            if type_ == "BluemiraFace":
                boundary = value["boundary"]
                if dim == 1:
                    value["gmsh"] = {}
                    for item in boundary:
                        for btype_ in item:
                            if btype_ == "BluemiraWire":
                                self.__convert_wire_to_gmsh(item, dim)

                    # get the dictionary of the BluemiraWire defined in buffer
                    # as default and gmsh format
                    dict_gmsh = self.get_gmsh_dict(buffer, GmshFileType.GMSH)

                    # fragment points_tag and curves
                    self.__apply_fragment(
                        buffer,
                        all_ent=dict_gmsh[MeshTags.POINTS] + dict_gmsh[MeshTags.CURVE],
                    )
                elif dim == 2:  # noqa: PLR2004
                    value["gmsh"][MeshTags.CURVELOOP] = [
                        gmsh.model.occ.addCurveLoop(
                            self.get_gmsh_dict(item)[MeshTags.CURVE]
                        )
                        for item in boundary
                    ]
                    gmsh.model.occ.synchronize()
                    value["gmsh"][MeshTags.SURFACE] = [
                        gmsh.model.occ.addPlaneSurface(value["gmsh"][MeshTags.CURVELOOP])
                    ]
                    gmsh.model.occ.synchronize()

    def __convert_compound_shell_to_gmsh(
        self, buffer: dict, dim: int, converter: _GmshEnum
    ):
        """
        Converts a shell to gmsh.
        """
        if converter == _GmshEnum.SHELL:
            convert_f = self.__convert_face_to_gmsh
        elif converter == _GmshEnum.COMPOUND:
            convert_f = self.__convert_item_to_gmsh

        for type_, value in buffer.items():
            if type_ == converter.value:
                boundary = value["boundary"]
                if dim == 1:
                    value["gmsh"] = {}
                    for item in boundary:
                        convert_f(item, dim)
                        # dictionary of the BluemiraShell or Component defined in buffer
                        dict_gmsh = self.get_gmsh_dict(buffer, GmshFileType.GMSH)

                        # fragment points_tag and curves
                        self.__apply_fragment(
                            buffer,
                            all_ent=dict_gmsh[MeshTags.POINTS]
                            + dict_gmsh[MeshTags.CURVE],
                        )
                elif dim == 2:  # noqa: PLR2004
                    for item in boundary:
                        convert_f(item, dim)

    def get_gmsh_dict(
        self, buffer: dict, file_format: Union[str, GmshFileType] = GmshFileType.DEFAULT
    ) -> dict[MeshTags, list]:
        """
        Returns the gmsh dict in a default (only tags) or gmsh (tuple(dim,
        tag)) format.
        """
        if isinstance(file_format, str):
            file_format = GmshFileType[file_format.upper()]

        gmsh_dict = {d: [] for d in MeshTagsNC}

        def _extract_mesh_from_buffer(buffer, obj_name):
            if obj_name not in buffer:
                raise ValueError(f"No {obj_name} to mesh.")

            boundary = buffer[obj_name]["boundary"]
            if "gmsh" in buffer[obj_name]:
                for d in MeshTagsNC:
                    if d in buffer[obj_name]["gmsh"]:
                        gmsh_dict[d] += buffer[obj_name]["gmsh"][d]

            for item in boundary:
                if obj_name == "BluemiraWire":
                    for k in item:
                        if k == obj_name:
                            temp_dict = self.get_gmsh_dict(item)
                            for d in MeshTagsNC:
                                gmsh_dict[d] += temp_dict[d]
                else:
                    temp_dict = self.get_gmsh_dict(item)
                    for d in MeshTagsNC:
                        gmsh_dict[d] += temp_dict[d]

        for geo_name in SUPPORTED_GEOS:
            if geo_name in buffer:
                _extract_mesh_from_buffer(buffer, geo_name)

        gmsh_dict = {d: list(dict.fromkeys(gmsh_dict[d])) for d in MeshTagsNC}

        if file_format == GmshFileType.DEFAULT:
            return gmsh_dict
        return {d: [(d.value, tag) for tag in gmsh_dict[d]] for d in MeshTagsNC}


class _FreeCADGmsh:
    @staticmethod
    def _initialise_mesh(terminal: int = 1, modelname: str = "Mesh"):
        # GMSH file generation
        # Before using any functions in the Python API,
        # Gmsh must be initialised:
        gmsh.initialize()

        # By default Gmsh will not print out any messages:
        # in order to output messages
        # on the terminal, just set the "General.Terminal" option to 1:
        gmsh.option.setNumber("General.Terminal", terminal)

        gmsh.logger.start()

        # gmsh.option.setNumber("Mesh.MshFileVersion", 2.0)

        # Next we add a new model named "t1" (if gmsh.model.add() is
        # not called a new
        # unnamed model will be created on the fly, if necessary):
        gmsh.model.add(modelname)

    @staticmethod
    def _save_mesh(meshfile: str = "Mesh.geo_unrolled"):
        # ... and save it to disk
        gmsh.write(meshfile)

    @staticmethod
    def _finalize_mesh(logfile: str = "gmsh.log"):
        with open(logfile, "w") as file_handler:
            file_handler.write("\n".join(str(item) for item in gmsh.logger.get()))

        # This should be called when you are done using the Gmsh Python API:
        # gmsh.logger.stop()
        # gmsh.finalize()

    @staticmethod
    def _generate_mesh(mesh_dim: int = 3):
        # Before it can be meshed, the internal CAD representation must
        # be synchronized with the Gmsh model, which will create the
        # relevant Gmsh data structures. This is achieved by the
        # gmsh.model.occ.synchronize() API call for the built-in
        # geometry kernel. Synchronizations can be called at any time,
        # but they involve a non trivial amount of processing;
        # so while you could synchronize the internal CAD data after
        # every CAD command, it is usually better to minimize
        # the number of synchronization points.
        gmsh.model.occ.synchronize()

        # We can then generate a mesh...
        gmsh.model.mesh.generate(mesh_dim)

    @staticmethod
    def create_gmsh_curve(buffer: dict):
        """
        Function to create gmsh curve from a dictionary (buffer).
        """
        gmsh_dict = {}

        points_tag = []
        cntrpoints_tag = []
        curve_tag = []
        for type_ in buffer:
            if type_ == "LineSegment":
                points_tag.extend(
                    _add_points(buffer[type_]["StartPoint"], buffer[type_]["EndPoint"])
                )
                curve_tag.append(gmsh.model.occ.addLine(points_tag[0], points_tag[1]))
            elif type_ == "BezierCurve":
                cntrpoints_tag.extend(_add_points(*buffer[type_]["Poles"]))
                curve_tag.append(gmsh.model.occ.addBezier(cntrpoints_tag))
                points_tag.extend((cntrpoints_tag[0], cntrpoints_tag[-1]))
            elif type_ == "BSplineCurve":
                cntrpoints_tag.extend(_add_points(*buffer[type_]["Poles"]))
                curve_tag.append(gmsh.model.occ.addBSpline(cntrpoints_tag))
                points_tag.extend((cntrpoints_tag[0], cntrpoints_tag[-1]))
            elif type_ == "ArcOfCircle":
                start_tag, end_tag, centre_tag = _add_points(
                    buffer[type_]["StartPoint"],
                    buffer[type_]["EndPoint"],
                    buffer[type_]["Center"],
                )
                points_tag.extend((start_tag, end_tag))
                curve_tag.append(
                    gmsh.model.occ.addCircleArc(start_tag, centre_tag, end_tag)
                )
                cntrpoints_tag.append(centre_tag)
            elif type_ == "ArcOfEllipse":
                start_tag, end_tag, focus_tag, centre_tag = _add_points(
                    buffer[type_]["StartPoint"],
                    buffer[type_]["EndPoint"],
                    buffer[type_]["Focus1"],
                    buffer[type_]["Center"],
                )
                points_tag.extend((start_tag, end_tag))
                curve_tag.append(
                    gmsh.model.occ.addEllipseArc(
                        start_tag, centre_tag, focus_tag, end_tag
                    )
                )
                cntrpoints_tag.extend((centre_tag, focus_tag))
            else:
                raise NotImplementedError(
                    f"Gmsh curve creation non implemented for {type_}"
                )

        gmsh_dict[MeshTags.POINTS] = points_tag
        gmsh_dict[MeshTags.CNTRPOINTS] = cntrpoints_tag
        gmsh_dict[MeshTags.CURVE] = curve_tag
        gmsh.model.occ.synchronize()
        return gmsh_dict

    @staticmethod
    def _fragment(
        dim: Union[int, Iterable[int]] = (2, 1, 0),
        all_ent: Optional[List[int]] = None,
        tools: Optional[list] = None,
        remove_object: bool = True,
        remove_tool: bool = True,
    ):
        if isinstance(dim, int):
            dim = [dim]
        if all_ent is None:
            all_ent = []
            for d in dim:
                all_ent += gmsh.model.getEntities(d)
        oo = []
        oov = []
        if len(all_ent) > 1:
            oo, oov = gmsh.model.occ.fragment(
                objectDimTags=all_ent,
                toolDimTags=[] if tools is None else tools,
                removeObject=remove_object,
                removeTool=remove_tool,
            )
            gmsh.model.occ.synchronize()

        return all_ent, oo, oov

    @staticmethod
    def _map_mesh_dict(mesh_dict: dict, all_ent, oov: Optional[list] = None):
        if oov is None:
            oov = []

        new_gmsh_dict = {key: [] for key in MeshTagsNC}

        for tagtype, values in mesh_dict.items():
            if tagtype != MeshTags.CURVELOOP:
                dim = tagtype.value
                for v in values:
                    if (dim, v) in all_ent:
                        if len(oov) > 0:
                            new_gmsh_dict[tagtype].extend([
                                o[1] for o in oov[all_ent.index((dim, v))]
                            ])
                    else:
                        new_gmsh_dict[tagtype].append(v)

        for key in MeshTagsNC:
            mesh_dict[key] = list(dict.fromkeys(new_gmsh_dict[key]))

        return new_gmsh_dict

    @staticmethod
    def set_mesh_size(dim_tags, size):
        gmsh.model.occ.mesh.setSize(dim_tags, size)
        gmsh.model.occ.synchronize()

    @staticmethod
    def add_physical_group(dim, tags, name: Optional[str] = None):
        tag = gmsh.model.addPhysicalGroup(dim, tags)
        if name is not None:
            gmsh.model.setPhysicalName(dim, tag, name)

    @staticmethod
    def _set_mesh_size(dim_tags, size):
        gmsh.model.mesh.setSize(dim_tags, size)

    @staticmethod
    def _get_boundary(dimtags, combined=False, recursive=False):
        return gmsh.model.getBoundary(dimtags, combined, recursive)


def _add_points(*point: Iterable) -> List:
    """
    Add gmsh model points
    """
    return [gmsh.model.occ.addPoint(p[0], p[1], p[2]) for p in point]
