# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

# import mesher lib (gmsh)
import gmsh

# import mirapy modules
import bluemira.geometry as geo

from .error import MeshOptionsError

import copy
import pprint

from typing import Dict, Union

DEFAULT_MESH_OPTIONS = {
    "lcar": 0.5,
    "physical_group": None,
}


def get_default_options():
    """
    Returns the default display options.
    """
    return copy.deepcopy(DEFAULT_MESH_OPTIONS)


class MeshOptions:
    """
    The options that are available for meshing objects
    """

    def __init__(self, **kwargs):
        self._options = get_default_options()
        if kwargs:
            for k in kwargs:
                if k in self._options:
                    self._options[k] = kwargs[k]
        # TODO: in this way class attributes are not seen till runtime. Not sure if
        #  this should be changed manually declaring all the attributes.
        for k in self._options:
            setattr(self, k, self._options[k])

    def as_dict(self):
        """
        Returns the instance as a dictionary.
        """
        return copy.deepcopy(self._options)

    def modify(self, **kwargs):
        """
        Function to override plotting options.
        """
        if kwargs:
            for k in kwargs:
                if k in self._options:
                    self._options[k] = kwargs[k]

    def __repr__(self):
        """
        Representation string of the DisplayOptions.
        """
        return f"{self.__class__.__name__}({pprint.pformat(self._options)}" + "\n)"


class Meshable:
    """Mixin class to make a class meshable"""

    def __init__(self):
        super().__init__()
        self._mesh_options = None

    @property
    def mesh_options(self) -> MeshOptions:
        """
        The options that will be used to display the object.
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

    def remove_mesh_options(self):
        self._mesh_options = None


class Mesh:
    def __init__(
        self,
        modelname="Mesh",
        terminal=1,
        meshfile="Mesh.geo_unrolled",
    ):

        self.modelname = modelname
        self.terminal = terminal
        self.meshfile = meshfile

    def _check_meshfile(self, meshfile):
        if isinstance(meshfile, str):
            meshfile = [meshfile]
        elif isinstance(meshfile, list):
            if len(meshfile) < 1:
                raise ValueError("meshfile is an empty list")
        else:
            raise ValueError("meshfile must be a string or a list of strings")
        return meshfile

    @property
    def meshfile(self):
        return self._meshfile

    @meshfile.setter
    def meshfile(self, meshfile):
        self._meshfile = self._check_meshfile(meshfile)

    def __call__(self, obj):
        if isinstance(obj, Meshable):
            _freecadGmsh._initialize_mesh(self.terminal, self.modelname)
            buffer = self.__mesh_obj(obj)

            _freecadGmsh._generate_mesh()

            #### Mesh size test - to be deleted
            temp_dict = self.get_gmsh_dict(buffer)
            print(f"temp_dict: {temp_dict}")
            dimTags = [(0, tag) for tag in temp_dict["points_tag"]]
            _freecadGmsh.set_mesh_size(dimTags, 0.1)
            #####

            for file in self.meshfile:
                _freecadGmsh._save_mesh(file)
            _freecadGmsh._finalize_mesh()
        else:
            raise ValueError(f"Only Meshable objects can be meshed")
        return buffer

    def __mesh_obj(self, obj, dim = 2):
        if not hasattr(obj, "ismeshed") or not obj.ismeshed:
            buffer = geo.tools.serialize_shape(obj)
            for d in range(1, dim+1, 1):
                print(f"dim = {d}")
                for k, v in buffer.items():
                    if k == "BluemiraWire":
                        print(f"meshing BluemiraWire: {buffer}")
                        self.__convert_wire_to_gmsh(buffer, d)
                    if k == "BluemiraFace":
                        print(f"meshing BluemiraFace: {buffer}")
                        self.__convert_face_to_gmsh(buffer, d)
                    if k == "BluemiraShell":
                        print(f"meshing BluemiraShell: {buffer}")
                        self.__convert_shell_to_gmsh(buffer,d)
            obj.ismeshed = True
        else:
            print("Obj already meshed")
        return buffer

    def __apply_fragment(self, buffer, dim=[2, 1, 0], all_ent=None, tools=[],
                         remove_object=True, remove_tool=True):
        all_ent, oo, oov = _freecadGmsh._fragment(dim, all_ent, tools, remove_object,
                                                  remove_tool)
        Mesh.__iterate_gmsh_dict(buffer, _freecadGmsh._map_mesh_dict, all_ent, oov)

    @staticmethod
    def __iterate_gmsh_dict(buffer, function, *args):
        if "BluemiraWire" in buffer:
            boundary = buffer["BluemiraWire"]["boundary"]
            if "gmsh" in buffer["BluemiraWire"]:
                function(buffer["BluemiraWire"]["gmsh"], *args)
            for item in boundary:
                for k, v1 in item.items():
                    if k == "BluemiraWire":
                        Mesh.__iterate_gmsh_dict(item, function, *args)

        if "BluemiraFace" in buffer:
            boundary = buffer["BluemiraFace"]["boundary"]
            if "gmsh" in buffer["BluemiraFace"]:
                function(buffer["BluemiraFace"]["gmsh"], *args)
            for item in boundary:
                for k, v1 in item.items():
                    if k == "BluemiraWire":
                        Mesh.__iterate_gmsh_dict(item, function, *args)

        if "BluemiraShell" in buffer:
            boundary = buffer["BluemiraShell"]["boundary"]
            if "gmsh" in buffer["BluemiraShell"]:
                function(buffer["BluemiraShell"]["gmsh"], *args)
            for item in boundary:
                for k, v1 in item.items():
                    if k == "BluemiraFace":
                        Mesh.__iterate_gmsh_dict(item, function, *args)

    def __convert_wire_to_gmsh(self, buffer, dim=1):
        # print(f"__convert_wire_to_gmsh: buffer = {buffer}")
        for type_, value in buffer.items():
            if type_ == "BluemiraWire":
                label = value["label"]
                boundary = value["boundary"]
                if dim == 1:
                    value["gmsh"] = {'points_tag': [], 'cntrpoints_tag': [], 'curve_tag':
                        [], 'curveloop_tag': [], 'surface_tag': []}
                    for item in boundary:
                        for btype_, bvalue in item.items():
                            if btype_ == "BluemiraWire":
                                self.__convert_wire_to_gmsh(item)
                            else:
                                for curve in bvalue:
                                    curve_gmsh_dict = _freecadGmsh.create_gmsh_curve(curve)
                                    value["gmsh"]["points_tag"] += curve_gmsh_dict["points_tag"]
                                    value["gmsh"]["cntrpoints_tag"] += curve_gmsh_dict[
                                        "cntrpoints_tag"]
                                    value["gmsh"]["curve_tag"] += curve_gmsh_dict["curve_tag"]

                    # get the dictionary of the BluemiraWire defined in buffer
                    # as default and gmsh format
                    dict_gmsh = self.get_gmsh_dict(buffer, "gmsh")

                    # fragment points_tag and curves
                    all_ent = dict_gmsh["points_tag"] + dict_gmsh["curve_tag"]
                    self.__apply_fragment(buffer, all_ent, [], False, False)
                else:
                    pass
            else:
                raise NotImplementedError(f"Serialization non implemented for {type_}")

    def __convert_face_to_gmsh(self, buffer, dim):
        print(f"__convert_face_to_gmsh: buffer = {buffer}")
        for type_, value in buffer.items():
            if type_ == "BluemiraFace":
                label = value["label"]
                boundary = value["boundary"]
                if dim == 1:
                    value["gmsh"] = {}
                    for item in boundary:
                        for btype_, bvalue in item.items():
                            if btype_ == "BluemiraWire":
                                self.__convert_wire_to_gmsh(item)

                    # get the dictionary of the BluemiraWire defined in buffer
                    # as default and gmsh format
                    dict_gmsh = self.get_gmsh_dict(buffer, "gmsh")

                    # fragment points_tag and curves
                    all_ent = dict_gmsh["points_tag"] + dict_gmsh["curve_tag"]
                    self.__apply_fragment(buffer, all_ent=all_ent)
                elif dim == 2:
                    value["gmsh"]["curveloop_tag"] = []
                    # print(f"buffer = {buffer}")
                    for item in boundary:
                        dict_curve = self.get_gmsh_dict(item)
                        value["gmsh"]["curveloop_tag"].append(
                            gmsh.model.occ.addCurveLoop(dict_curve['curve_tag'])
                        )
                    gmsh.model.occ.synchronize()
                    value["gmsh"]["surface_tag"] = [
                        gmsh.model.occ.addPlaneSurface(value["gmsh"]["curveloop_tag"])]
                    gmsh.model.occ.synchronize()
                else:
                    pass


    def __convert_shell_to_gmsh(self, buffer, dim):
        # print(f"__convert_shell_to_gmsh: buffer = {buffer}")
        for type_, value in buffer.items():
            if type_ == "BluemiraShell":
                label = value["label"]
                boundary = value["boundary"]
                if dim == 1:
                    value["gmsh"] = {}
                    for item in boundary:
                        # print(f"item = {item}")
                        self.__convert_face_to_gmsh(item, dim)
                        # get the dictionary of the BluemiraShell defined in buffer
                        # as default and gmsh format
                        # print(f"buffer: {buffer}")
                        dict_gmsh = self.get_gmsh_dict(buffer, "gmsh")

                        # fragment points_tag and curves
                        all_ent = dict_gmsh["points_tag"] + dict_gmsh["curve_tag"]
                        self.__apply_fragment(buffer, all_ent=all_ent)
                elif dim == 2:
                    for item in boundary:
                        # print(f"item = {item}")
                        self.__convert_face_to_gmsh(item, dim)
                else:
                    pass

    def get_gmsh_dict(self, buffer, format="default"):
        gmsh_dict = {}
        output = None

        data = {"points_tag": 0, "cntrpoints_tag": 0, "curve_tag": 1, "surface_tag": 2}
        for d in data:
            gmsh_dict[d] = []

        if "BluemiraWire" in buffer:
            boundary = buffer["BluemiraWire"]["boundary"]
            if "gmsh" in buffer["BluemiraWire"]:
                for d in data:
                    if d in buffer["BluemiraWire"]["gmsh"]:
                        gmsh_dict[d] += buffer["BluemiraWire"]["gmsh"][d]
            for item in boundary:
                for k, v1 in item.items():
                    if k == "BluemiraWire":
                        temp_dict = self.get_gmsh_dict(item)
                        for d in data:
                            gmsh_dict[d] += temp_dict[d]

        if "BluemiraFace" in buffer:
            boundary = buffer["BluemiraFace"]["boundary"]
            if "gmsh" in buffer["BluemiraFace"]:
                for d in data:
                    if d in buffer["BluemiraFace"]["gmsh"]:
                        gmsh_dict[d] += buffer["BluemiraFace"]["gmsh"][d]
            for item in boundary:
                temp_dict = self.get_gmsh_dict(item)
                for d in data:
                    gmsh_dict[d] += temp_dict[d]

        if "BluemiraShell" in buffer:
            boundary = buffer["BluemiraShell"]["boundary"]
            if "gmsh" in buffer["BluemiraShell"]:
                for d in data:
                    if d in buffer["BluemiraShell"]["gmsh"]:
                        gmsh_dict[d] += buffer["BluemiraShell"]["gmsh"][d]
            for item in boundary:
                temp_dict = self.get_gmsh_dict(item)
                for d in data:
                    gmsh_dict[d] += temp_dict[d]

        for d in data:
            gmsh_dict[d] = list(dict.fromkeys(gmsh_dict[d]))

        if format == "default":
            output = gmsh_dict
        if format == "gmsh":
            output = {}
            for d in data:
                output[d] = [(data[d], tag) for tag in gmsh_dict[d]]
        return output


class _freecadGmsh:
    @staticmethod
    def _initialize_mesh(terminal=1, modelname="Mesh"):

        # GMSH file generation #######################
        # Before using any functions in the Python API,
        # Gmsh must be initialized:
        gmsh.initialize()

        # By default Gmsh will not print out any messages:
        # in order to output messages
        # on the terminal, just set the "General.Terminal" option to 1:
        gmsh.option.setNumber("General.Terminal", terminal)

        # Next we add a new model named "t1" (if gmsh.model.add() is
        # not called a new
        # unnamed model will be created on the fly, if necessary):
        gmsh.model.add(modelname)

    @staticmethod
    def _save_mesh(meshfile="Mesh.geo_unrolled"):
        # ... and save it to disk
        gmsh.write(meshfile)

    @staticmethod
    def _finalize_mesh():
        # This should be called when you are done using the Gmsh Python API:
        gmsh.finalize()

    @staticmethod
    def _generate_mesh(mesh_dim=3):
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
    def create_gmsh_curve(buffer):
        gmsh_dict = {}

        points_tag = []
        cntrpoints_tag = []
        curve_tag = []
        for type_ in buffer:
            if type_ == "LineSegment":
                start_point = buffer[type_]["StartPoint"]
                points_tag.append(
                    gmsh.model.occ.addPoint(
                        start_point[0], start_point[1], start_point[2]
                    )
                )
                end_point = buffer[type_]["EndPoint"]
                points_tag.append(
                    gmsh.model.occ.addPoint(end_point[0], end_point[1], end_point[2])
                )
                curve_tag.append(gmsh.model.occ.addLine(points_tag[0], points_tag[1]))
            elif type_ == "BezierCurve":
                poles = buffer[type_]["Poles"]
                for p in poles:
                    cntrpoints_tag.append(gmsh.model.occ.addPoint(p[0], p[1], p[2]))
                curve_tag.append(gmsh.model.occ.addBezier(cntrpoints_tag))
                points_tag.append(cntrpoints_tag[0])
                points_tag.append(cntrpoints_tag[-1])
            elif type_ == "BSplineCurve":
                poles = buffer[type_]["Poles"]
                for p in poles:
                    cntrpoints_tag.append(gmsh.model.occ.addPoint(p[0], p[1], p[2]))
                curve_tag.append(gmsh.model.occ.addBSpline(cntrpoints_tag))
                points_tag.append(cntrpoints_tag[0])
                points_tag.append(cntrpoints_tag[-1])
            else:
                raise NotImplementedError(
                    f"Gmsh curve creation non implemented for {type_}"
                )

        gmsh_dict["points_tag"] = points_tag
        gmsh_dict["cntrpoints_tag"] = cntrpoints_tag
        gmsh_dict["curve_tag"] = curve_tag
        gmsh.model.occ.synchronize()
        return gmsh_dict

    @staticmethod
    def _fragment(dim=[2, 1, 0], all_ent=None, tools=[], remove_object=True,
                  remove_tool=True):
        print("_fragment")
        if not hasattr(dim, "__len__"):
            dim = [dim]
        if all_ent is None:
            all_ent = []
            for d in dim:
                all_ent += gmsh.model.getEntities(d)
        oo = []
        oov = []
        if len(all_ent) > 1:
            oo, oov = gmsh.model.occ.fragment(objectDimTags=all_ent, toolDimTags=tools,
                                              removeObject=remove_object,
                                              removeTool=remove_tool)
            print(all_ent)
            print(oo)
            print(oov)
            gmsh.model.occ.synchronize()

        return all_ent, oo, oov

    @staticmethod
    def _map_mesh_dict(mesh_dict, all_ent, oov):
        dim_dict = {"points_tag": 0, "cntrpoints_tag": 0, "curve_tag": 1, "surface_tag": 2}
        new_gmsh_dict = {}

        for key in dim_dict:
            new_gmsh_dict[key] = []

        for type_, values in mesh_dict.items():
            if type_ != 'curveloop_tag':
                for v in values:
                    dim = dim_dict[type_]
                    if (dim, v) in all_ent:
                        if len(oov) > 0:
                            for o in oov[all_ent.index((dim, v))]:
                                new_gmsh_dict[type_].append(o[1])
                    else:
                        new_gmsh_dict[type_].append(v)

        for key in dim_dict:
            mesh_dict[key] = list(dict.fromkeys(new_gmsh_dict[key]))

        return new_gmsh_dict

    @staticmethod
    def set_mesh_size(dimTags, size):
        gmsh.model.occ.mesh.setSize(dimTags, size)
        gmsh.model.occ.synchronize()

    @staticmethod
    def add_physical_group(dim, tags):
        gmsh.model.addPhysicalGroup(dim, tags)
