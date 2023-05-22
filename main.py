import inspect

import dolfinx.io.gmshio as gmshio
import gmsh
from mpi4py import MPI

import bluemira.mesh.meshing as msh
from bluemira.base.look_and_feel import bluemira_print
from bluemira.geometry import tools
from bluemira.geometry.face import BluemiraFace


def test_override_lcar_surf_and_mesh_poly(lcar, nodes_num, half):
    poly = tools.make_polygon(
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], closed=True, label="poly"
    )

    poly.mesh_options = {"lcar": lcar, "physical_group": "poly"}

    surf = BluemiraFace(poly, label="surf")
    if half:
        surf.mesh_options = {"lcar": lcar / 2, "physical_group": "coil"}
    else:
        surf.mesh_options = {"physical_group": "coil"}

    m = msh.Mesh()
    m(surf)

    mesh, *meshtags = gmshio.model_to_mesh(
        gmsh.model, comm=MPI.COMM_WORLD, rank=0, gdim=2
    )

    msh._FreeCADGmsh._finalize_mesh(m.logfile)
    return mesh, meshtags


def mesh_call(self, obj, dim=2):
    """
    Generate the mesh and save it to file.
    """
    bluemira_print("Starting mesh process...")

    if "Component" in [c.__name__ for c in inspect.getmro(type(obj))]:
        from bluemira.base.tools import create_compound_from_component

        obj = create_compound_from_component(obj)

    if isinstance(obj, msh.Meshable):
        # gmsh is initialized
        msh._FreeCADGmsh._initialize_mesh(self.terminal, self.modelname)
        # Mesh the object. A dictionary with the geometrical and internal
        # information that are used by gmsh is returned. In particular,
        # a gmsh key is added to any meshed entity.
        buffer = self._Mesh__mesh_obj(obj, dim=dim)
        # Check for possible intersection (only allowed at the boundary to adjust
        # the gmsh_dictionary
        msh.Mesh._Mesh__iterate_gmsh_dict(buffer, msh.Mesh._check_intersections)

        # Create the physical groups
        self._apply_physical_group(buffer)

        # apply the mesh size
        self._apply_mesh_size(buffer)

        # generate the mesh
        msh._FreeCADGmsh._generate_mesh()
        return
        # save the mesh file
        # for file in self.meshfile:
        #     _FreeCADGmsh._save_mesh(file)

        # # close gmsh
        # _FreeCADGmsh._finalize_mesh(self.logfile)
    else:
        raise ValueError("Only Meshable objects can be meshed")

    bluemira_print("Mesh process completed.")

    return buffer


# hack out the gmsh closing for now
msh.Mesh.__call__ = mesh_call

lcars = [0.1, 0.25, 0.5, 0.1, 0.25, 0.5]
#  lcars = [0.1, 0.25, 0.5, 0.1 * 2, 0.25 * 2, 0.5 * 2]  # This works...
half = [True, True, True, False, False, False]
nodes_num = [80, 32, 16, 40, 16, 8]

meshes, meshtags = [], []
for lcar, nn in zip(lcars, nodes_num):
    mesh, _meshtags = test_override_lcar_surf_and_mesh_poly(lcar, nn, half=half)
    meshes += [mesh]
    meshtags += [_meshtags]
    assert _meshtags[-1].indices.size == nn
    assert all(_meshtags[0].values == 1)
    assert all(_meshtags[1].values == 2)
