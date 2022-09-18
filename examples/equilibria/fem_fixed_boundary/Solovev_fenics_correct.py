"""
Example to test Solovev solution
"""

import time
import numpy
import matplotlib

# matplotlib.use('Agg')
import matplotlib.pyplot as pyplot
from bluemira.equilibria.fem_fixed_boundary.utilities import Solovev

from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import FemMagnetostatic2d
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf

from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import interpolate_bspline, make_polygon
from bluemira.geometry.wire import BluemiraWire

import dolfin

# ------------------------------------------------------------------------------
# Input parameters

p = 3 # the order of the finite elements

R0 = 8.9
A = 3.1
delta = 0.5
kappa = 1.7

a = R0 / A

A1 = -6.84256806e-02
A2 = -6.52918977e-02

solovev = Solovev(R0, a, kappa, delta, A1, A2)
levels = 20
r_min = 5
z_min = -6
dr = 8
dz = 12
r_max = r_min + dr
z_max = z_min + dz

nx = 20
nz = 20

l_car = min([dr/nx, dz/nz])
l_car_fine = l_car/5

axis, cntr, cntrf, points, psi_exact = solovev.plot_psi(
    r_min, z_min, dr, dz, nx, nz, levels=levels
)
pyplot.show()
print(max(psi_exact))
print(min(psi_exact))

# ------------------------------------------------------------------------------

print('\nSetting up parameters for Grad-Shafranov solver...')
print('  p = %2d,  N_r = %4d,  N_z = %4d' % (p, nx, nz))

# ------------------------------------------------------------------------------
# boundary conditions definition
c = solovev.coeff
A1 = solovev.A1
A2 = solovev.A2

dirichletBCFunction = dolfin.Expression(
    'c1 + c2*pow(x[0],2) + c3*(pow(x[0],4) - 4*pow(x[0],2)*pow(x[1],2)) + c4*(pow(x['
    '0],2)*std::log(x[0]) -pow(x[1],2)) + A1*pow(x[0],4)/8 -A2*pow(x[1],2)/2',
    c1 = c[0], c2 = c[1], c3 = c[2], c4 = c[3], A1 = A1, A2 = A2, degree=p) # the Dirichlet boundary condition (in this case it is the exact
# solution)
neumannBCFunction = dolfin.Expression(
    '0.0', degree=p)  # the Neumann boundary conditions (in this case they are set to 0
# since there are no Neumann conditions)

# Initialize mesh, boundary conditions and Grad-Shafranov solver
dolfin_mesh = False
# mesh
p0 = dolfin.Point(r_min, z_min)
p1 = dolfin.Point(r_max, z_max)

if dolfin_mesh:
    mesh = dolfin.RectangleMesh(p0, p1, nx, nz)  # initialize mesh
    # identify the Dirichlet BC's nodes
    class dirichlet_boundary(dolfin.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary  # all entities on the boundary have Dirichlet B.C.


    myDirichletBoundary = dirichlet_boundary()  # define the function for marking entities where
    # Dirichlet B.C. are defined

    boundaries = dolfin.MeshFunction("size_t", mesh,
                                     mesh.topology().dim() - 1)  # initialize the MeshFunction
    myDirichletBoundary.mark(boundaries, 2)  # mark the boundary entities with 2
else:
    # create a corresponding geometrical domain
    d_points = Coordinates({'x':[r_min, r_max, r_max, r_min],
                            'y':[0, 0, 0, 0],
                            'z':[z_min, z_min, z_max, z_max]})

    rect = make_polygon(d_points, "boundary", True)
    rect.mesh_options = {'lcar': l_car, 'physical_group': 'boundary'}
    face = BluemiraFace(rect)
    face.mesh_options = {'lcar': l_car, 'physical_group': 'domain'}

    m = meshing.Mesh()
    buffer = m(face)

    msh_to_xdmf("Mesh.msh", dimensions=(0, 2), directory=".", verbose=True)

    mesh, boundaries, subdomains, labels = import_mesh(
        "Mesh",
        directory=".",
        subdomains=True,
    )
    dolfin.plot(mesh)
    pyplot.show()

# initialize the Grad-Shafranov solver
gs_solver = FemMagnetostatic2d(mesh, boundaries, p)

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Set the right hand side of the Grad-Shafranov equation

from bluemira.base.constants import MU_0
import numpy as np

f = dolfin.Expression(
    "1/mu0*(-x[0]*A1 + A2/x[0])", pi=np.pi, A1=A1, A2=A2, mu0=MU_0, degree=p
)

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# Solve the equation

print('\nSolving...')

# solve the Grad-Shafranov equation
solve_start = time.time()  # compute the time it takes to solve
psi = gs_solver.solve(f, dirichletBCFunction, 2, neumannBCFunction)
solve_end = time.time()

# define a fine mesh for plotting
if dolfin_mesh:
    mesh_plot = dolfin.RectangleMesh(p0, p1, 250, 250)
else:
    rect.mesh_options = {'lcar': l_car_fine, 'physical_group': 'boundary'}
    face = BluemiraFace(rect)
    face.mesh_options = {'lcar': l_car_fine, 'physical_group': 'domain'}

    m = meshing.Mesh()
    buffer = m(face)
    msh_to_xdmf("Mesh.msh", dimensions=(0, 2), directory=".", verbose=True)
    mesh_plot, _, _, _ = import_mesh(
        "Mesh",
        directory=".",
        subdomains=True,
    )

# generate the refined function space
V_plot = dolfin.FunctionSpace(mesh_plot, 'CG', 1)

# interpolate the analytical solution into the mesh
psi_exact = dolfin.interpolate(dirichletBCFunction, V_plot)
psi = dolfin.interpolate(dirichletBCFunction, gs_solver.V)

# interpolate the solution into a finer plotting grid only if higher order elements are used
if p > 1:
    # refine the solution
    psi_plot = dolfin.interpolate(psi, V_plot)

# print the error
L2_error = dolfin.errornorm(dirichletBCFunction, psi)
print('\nL2 error = %.16f    t = %.4fs\n' % (L2_error, solve_end - solve_start))

# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# make publication quality plots

# first extract mesh data
if p > 1:
    x = mesh_plot.coordinates()[:, 0]  # coordinates of the points
    y = mesh_plot.coordinates()[:, 1]

    cells = mesh_plot.cells()  # the cells definition, that is the connectivity

else:
    x = mesh.coordinates()[:, 0]  # coordinates of the points
    y = mesh.coordinates()[:, 1]

    cells = mesh.cells()  # the cells definition, that is the connectivity

# now extract the data to plot, reordering to be compatible with the coordinates ordering
if p > 1:
    # the numerical solution
    psi_data = psi_plot.vector()[:][dolfin.vertex_to_dof_map(V_plot)]
else:
    # the numerical solution
    psi_data = psi.vector()[:][dolfin.vertex_to_dof_map(gs_solver.V)]

# the exact solution
x_exact = mesh_plot.coordinates()[:, 0]  # coordinates of the points
y_exact = mesh_plot.coordinates()[:, 1]
cells_exact = mesh_plot.cells()  # the cells definition, that is the connectivity
psi_exact_data = psi_exact.vector()[:][dolfin.vertex_to_dof_map(V_plot)]

# plot the data using pylab triplot
# pyplot.tripcolor(x,y,cells,psi_data,shading='gouraud',edgecolors='none')
pyplot.tricontourf(x, y, cells, psi_data, 20)
pyplot.colorbar()

# make a contour plot
contour_plot_numerical = pyplot.tricontour(x, y, cells, psi_data,
                                           levels=20,
                                           colors='g')
contour_plot_exact = pyplot.tricontour(x_exact, y_exact, cells_exact, psi_exact_data,
                                       levels=20, colors='k')

pyplot.clabel(contour_plot_numerical, fontsize=9, inline=1)  # add the labels to the contours
pyplot.clabel(contour_plot_exact, fontsize=9, inline=1)  # add the labels to the contours

pyplot.xlabel('R')  # add labels to the axis
pyplot.ylabel('z')

pyplot.xlim(r_min, r_max)  # set ranges of the axis
pyplot.ylim(z_min, z_max)

pyplot.savefig(
    'comparison_psi_exact_vs_psi_p_%d_Nr_%d_Nz_%d.png' % (p, nx, nz),
    bbox_inches='tight', dpi=300)  # save the figure

pyplot.show()
# ------------------------------------------------------------------------------