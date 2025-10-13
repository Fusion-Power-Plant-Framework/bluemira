import numpy as np
from dolfinx import default_scalar_type, fem, io, mesh
from dolfinx.fem.petsc import LinearProblem
from mpi4py import MPI
from ufl import (
    Identity,
    Measure,
    TestFunction,
    TrialFunction,
    dot,
    ds,
    dx,
    grad,
    inner,
    nabla_div,
    sqrt,
    sym,
    tr,
)

L = 1
W = 0.2
delta = W / L
gamma = 0.4 * delta**2
beta = 1.25
g = gamma

domain = mesh.create_box(
    MPI.COMM_WORLD,
    [np.array([0, 0, 0]), np.array([L, W, W])],
    [20, 6, 6],
    cell_type=mesh.CellType.hexahedron,
)


E = fem.Constant(domain, 200000000000.0)
nu = fem.Constant(domain, 0.3)

lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)

rho = 8000.0
g = 9.81


# Define Function Space
V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))


def clamped_boundary(x):
    return np.isclose(x[0], 0)


fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, clamped_boundary)

u_D = np.array([0, 0, 0], dtype=default_scalar_type)
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets), V)
bcs = [bc]

dx = Measure("dx", domain=domain)
# dS = Measure("dS", domain=domain, subdomain_data=facet_markers)
ds = Measure("ds", domain=domain)


T = fem.Constant(domain, default_scalar_type((0, 0, 0)))


def epsilon(u):
    return sym(grad(u))  # Equivalent to 0.5*(ufl.nabla_grad(u) + ufl.nabla_grad(u).T)


def sigma(u):
    return lmbda * nabla_div(u) * Identity(len(u)) + 2 * mu * epsilon(u)


u = TrialFunction(V)
v = TestFunction(V)
f = fem.Constant(domain, default_scalar_type((0, 0, -rho * g)))
a = inner(sigma(u), epsilon(v)) * dx
L = dot(f, v) * dx + dot(T, v) * ds


problem = LinearProblem(
    a, L, bcs, petsc_options={"ksp_type": "preonly", "pc_type": "lu"}
)
uh = problem.solve()


with io.XDMFFile(domain.comm, "../../trry.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    uh.name = "Deformation"
    xdmf.write_function(uh)


s = sigma(uh) - 1.0 / 3 * tr(sigma(uh)) * Identity(len(uh))
von_Mises = sqrt(3.0 / 2 * inner(s, s))

V_von_mises = fem.functionspace(domain, ("DG", 0))
stress_expr = fem.Expression(von_Mises, V_von_mises.element.interpolation_points())
stresses = fem.Function(V_von_mises)
stresses.interpolate(stress_expr)


with io.XDMFFile(domain.comm, "../../fry3.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    uh.name = "Deformation"
    xdmf.write_function(uh)
    stresses.name = "Von_Mises"
    xdmf.write_function(stresses)
