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
from bluemira.geometry.wire import BluemiraWire
from bluemira.geometry.face import BluemiraFace


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

    def __call__(self, obj, clean=True):
        objlist = (BluemiraWire, BluemiraFace)
        if isinstance(obj, objlist):
            _freecadGmsh._initialize_mesh(self.terminal, self.modelname)
            buffer = self.__mesh_obj(obj)

            #### Mesh size test - to be deleted
            temp_dict = self.get_gmsh_dict(buffer)
            dimTags = [(0, tag) for tag in temp_dict["points_tag"]]
            _freecadGmsh.set_mesh_size(dimTags, 0.5)
            #####

            _freecadGmsh._generate_mesh()
            for file in self.meshfile:
                _freecadGmsh._save_mesh(file)
            _freecadGmsh._finalize_mesh()
        else:
            raise ValueError(f"Only {objlist} can be meshed")
        return buffer

    def __mesh_obj(self, obj):
        if not hasattr(obj, "ismeshed") or not obj.ismeshed:
            buffer = geo.tools.serialize_shape(obj)
            for k, v in buffer.items():
                if k == "BluemiraWire":
                    self.__convert_wire_to_gmsh(buffer)
                if k == "BluemiraFace":
                    self.__convert_face_to_gmsh(buffer)
            obj.ismeshed = True
        else:
            print("Obj already meshed")
        return buffer

    def __apply_fragment(self, buffer, dim=[0, 1, 2], all_ent=None, tools=[]):
        print(f"prev_buffer: {buffer}")
        all_ent, oo, oov = _freecadGmsh._fragment(dim, all_ent, tools)
        Mesh.__iterate_gmsh_dict(buffer, _freecadGmsh._map_mesh_dict, all_ent, oov)
        print(f"post_buffer: {buffer}")

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

    def __convert_wire_to_gmsh(self, buffer):
        print(f"__convert_wire_to_gmsh: buffer = {buffer}")
        for type_, value in buffer.items():
            if type_ == "BluemiraWire":
                label = value["label"]
                boundary = value["boundary"]
                value["gmsh"] = {}
                for item in boundary:
                    for btype_, bvalue in item.items():
                        if btype_ == "BluemiraWire":
                            self.__convert_wire_to_gmsh(item)
                        else:
                            for curve in bvalue:
                                value["gmsh"] = {
                                    **value["gmsh"],
                                    **_freecadGmsh.create_gmsh_curve(curve),
                                }
                # get the dictionary of the BluemiraWire defined in buffer
                # as default and gmsh format
                dict_default = self.get_gmsh_dict(buffer)
                dict_gmsh = self.get_gmsh_dict(buffer, "gmsh")

                # In order to avoid fragmentation of curves given by construction points
                # the fragment function is applied in two steps:
                # 1) fragment only points
                all_ent = dict_gmsh["points_tag"] + dict_gmsh["cntrpoints_tag"]
                self.__apply_fragment(buffer, all_ent=all_ent)

                # 2) fragment points_tag and curves
                all_ent = dict_gmsh["points_tag"] + dict_gmsh["curve_tag"]
                self.__apply_fragment(buffer, all_ent=all_ent)

            else:
                raise NotImplementedError(f"Serialization non implemented for {type_}")

    def __convert_face_to_gmsh(self, buffer):
        print(f"__convert_face_to_gmsh: buffer = {buffer}")
        for type_, value in buffer.items():
            if type_ == "BluemiraFace":
                label = value["label"]
                boundary = value["boundary"]
                value["gmsh"] = {}
                for item in boundary:
                    for btype_, bvalue in item.items():
                        if btype_ == "BluemiraWire":
                            self.__convert_wire_to_gmsh(item)

                # get the dictionary of the BluemiraWire defined in buffer
                # as default and gmsh format
                dict_default = self.get_gmsh_dict(buffer)
                dict_gmsh = self.get_gmsh_dict(buffer, "gmsh")

                # In order to avoid fragmentation of curves given by construction points
                # the fragment function is applied in two steps:
                # 1) fragment only points
                all_ent = dict_gmsh["points_tag"] + dict_gmsh["cntrpoints_tag"]
                self.__apply_fragment(buffer, all_ent=all_ent)

                # 2) fragment points_tag and curves
                all_ent = dict_gmsh["points_tag"] + dict_gmsh["curve_tag"]
                self.__apply_fragment(buffer, all_ent=all_ent)

                value["gmsh"]["curveloop_tag"] = []
                for item in boundary:
                    dict_curve = self.get_gmsh_dict(item)
                    value["gmsh"]["curveloop_tag"].append(
                        gmsh.model.occ.addCurveLoop(dict_curve['curve_tag'])
                    )
                value["gmsh"]["surface_tag"] = [gmsh.model.occ.addPlaneSurface(value["gmsh"]["curveloop_tag"])]

    def get_gmsh_dict(self, buffer, format="default"):
        gmsh_dict = {}
        output = None

        data = {"points_tag": 0, "cntrpoints_tag": 0, "curve_tag": 1}
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
    def _fragment(dim=[0, 1, 2], all_ent=None, tools=[]):
        if not hasattr(dim, "__len__"):
            dim = [dim]
        if all_ent is None:
            all_ent = []
            for d in dim:
                all_ent += gmsh.model.getEntities(d)
        oo = []
        oov = []
        if len(all_ent) > 1:
            oo, oov = gmsh.model.occ.fragment(all_ent, tools)
            gmsh.model.occ.synchronize()

        return all_ent, oo, oov

    @staticmethod
    def _map_mesh_dict(mesh_dict, all_ent, oov):
        dim_dict = {"points_tag": 0, "cntrpoints_tag": 0, "curve_tag": 1}
        new_gmsh_dict = {}
        for key in dim_dict:
            new_gmsh_dict[key] = []

        for type_, values in mesh_dict.items():
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
