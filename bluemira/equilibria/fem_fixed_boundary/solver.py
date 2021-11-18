# copyright

import tools
from plasma import Plasma
from bluemira.codes.plasmod.plasmodapi import (
    PlasmodSolver
)
from dolfinSolver import GradShafranovLagrange
from bluemira.mesh import meshing
# creation of a plasma
## create a geometry
## create a face

plasma = Plasma(name = "plasma", shape=face)
m = meshing.Mesh()
m(plasma.shape)

plasmod_parameters = {}
mhd_solver = PlasmodSolver(...)
gs_solver = GradShafranovLagrange(...)

plasma.set_mhd_solver = mhd_solver
plasma.set_gs_solver = gs_solver

# Remember that this
#     g = plasma.J_to_dolfinFunction(solver.V)
# will become this
# g = tools.func_to_dolfinFunction(plasma.curr_density, gs_solver.V)

# implement the dolfinUpdate that adjust the current density
# check the method in core.py for PlasmaFreeGS



#
# plasma = self.getPlasma()
#
# if plasma is None:
#     raise ValueError("No plasma has been found")
#
# if solver is None:
#     if (not hasattr(plasma.shape,
#                     'physicalGroups')) or plasma.shape.physicalGroups is None:
#         plasma.shape.physicalGroups = {1: "external", 2: "plasma"}
#     else:
#         if not 1 in plasma.shape.physicalGroups:
#             plasma.shape.physicalGroups[1] = "external"
#
#         if not 2 in plasma.shape.physicalGroups:
#             plasma.shape.physicalGroups[2] = "plasma"
#
#     mesh_dim = 2
#
#     if plasma.J is None:
#         raise ValueError('Plamsa Jp must to be defined')
#
#     fullmeshfile = os.path.join(meshdir, meshfile)
#
#     print(fullmeshfile)
#
#     if createmesh:
#         #### Mesh Generation ####
#         mesh = mirapy.core.Mesh("plasma")
#         mesh.meshfile = fullmeshfile
#         if not Pax is None:
#             # P0lcar = plasma.shape.lcar/2.
#             mesh.embed = [(Pax, Pax_lcar)]
#         mesh(plasma)
#
#     # Run the conversion
#     mirapy.msh2xdmf.msh2xdmf(meshfile, dim=mesh_dim)
#
#     # Run the import
#     prefix, _ = os.path.splitext(fullmeshfile)
#
#     mesh, boundaries, subdomains, labels = mirapy.msh2xdmf.import_mesh_from_xdmf(
#         prefix=prefix,
#         dim=mesh_dim,
#         directory=meshdir,
#         subdomains=True,
#     )
#
#     solver = mirapy.dolfinSolver.GradShafranovLagrange(mesh, p=p)
#
# # Calculate plasma geometrical parameters
# plasma.calculatePlasmaParameters(solver.mesh)
#
# eps = 1.0  # error measure ||u-u_k||
# i = 0  # iteration counter
# while eps > tol and i < maxiter:
#     prev = solver.psi.compute_vertex_values()
#     i += 1
#     plasma.psi = solver.psi
#     g = plasma.J_to_dolfinFunction(solver.V)
#     solver.solve(g)
#     diff = solver.psi.compute_vertex_values() - prev
#     eps = numpy.linalg.norm(diff, ord=numpy.Inf)
#     print('iter = {} eps = {}'.format(i, eps))
#     plasma.dolfinUpdate(solver.V)
#
# self.__solvers['fixed_boundary'] = solver
# plasma.updateFilaments(solver.V)