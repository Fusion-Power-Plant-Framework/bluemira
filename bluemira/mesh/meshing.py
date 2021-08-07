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
        objlist = (BluemiraWire)
        if isinstance(obj, objlist):
            _freecadGmsh._initialize_mesh(self.terminal, self.modelname)
            mesh_dict = self.__mesh_obj(obj)
            _freecadGmsh._generate_mesh()
            for file in self.meshfile:
                _freecadGmsh._save_mesh(file)
            _freecadGmsh._finalize_mesh()
        else:
            raise ValueError(f"Only {objlist} can be meshed")
        return mesh_dict

    def __mesh_obj(self, obj):
        if not hasattr(obj, "ismeshed") or not obj.ismeshed:
            buffer = geo.tools.serialize_shape(obj)
            for k, v in buffer.items():
                if k == "BluemiraWire":
                    self.__convert_to_gmsh(buffer)
            obj.ismeshed = True
        else:
            print("Obj already meshed")
        return buffer

    def get_mesh_dict(self, obj, dim=None):
        mesh_dict = []
        if hasattr(obj, "mesh_dict"):
            mesh_dict += obj.mesh_dict
        if isinstance(obj, BluemiraWire):
            for o in obj.boundary:
                mesh_dict += self.get_mesh_dict(o)
        if dim is not None:
            mesh_dict = [v for k, v in mesh_dict if k == dim]
        return mesh_dict

    def __convert_to_gmsh(self, buffer):
        for type_, v in buffer.items():
            if type_ == "BluemiraWire":
                label = v['label']
                boundary = v['boundary']
                temp_list = []
                for item in boundary:
                    for k, v1 in item.items():
                        if k == "BluemiraWire":
                            wire = self.__convert_to_gmsh(item)
                        else:
                            wire = v1
                            for c in wire:
                                _freecadGmsh.create_gmsh_curve(c)
                        temp_list = temp_list + wire
                return temp_list
        raise NotImplementedError(f"Serialization non implemented for {type_}")


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

    def create_gmsh_curve(buffer):
        points_tag = []
        cntrpoints_tag = []
        curve_tag = []

        for type_, v in buffer.items():
            if type_ == "BezierCurve":
                buffer[type_]['gmsh'] = {}
                num = 0
                poles = v['Poles']
                for p in poles:
                    cntrpoints_tag.append(gmsh.model.occ.addPoint(p[0], p[1], p[2]))
                curve_tag.append(gmsh.model.occ.addBezier([cptag for cptag in cntrpoints_tag]))
                points_tag.append(cntrpoints_tag[0])
                points_tag.append(cntrpoints_tag[-1])
                buffer[type_]['gmsh']['points_tag'] = points_tag
                buffer[type_]['gmsh']['cntrpoints_tag'] = cntrpoints_tag
                buffer[type_]['gmsh']['curve_tag'] = curve_tag
            else:
                raise NotImplementedError(f"Gmsh curve creation non implemented for {type_}")
