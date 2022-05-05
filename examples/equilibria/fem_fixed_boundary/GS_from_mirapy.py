#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __doc__ = """

# Grad-Shafranov solver
# =====================

# The main class for solving the Grad-Shafranov equation

# Description
# -----------
# This class implements a general solver for the Grad-Shafranov equation where any
# right hand side, as function of (P(\psi),f(\psi)), can be prescribed.
# """

import dolfin
import numpy as np
from bluemira.base.constants import MU_0
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.base.components import PhysicalComponent
from bluemira.geometry.face import BluemiraFace
from bluemira.equilibria.fem_fixed_boundary.utilities import plot_scalar_field

class GradShafranovLagrange:
    def __init__(self, mesh, boundaries=None, p=3):
        # """
        # Grad-Shafranov solver implementation as a class.

        # DESCRIPTION
        # -----------
        # Solves the Grad-Shafranov equation:

        # Lagrange interpolants of order p are used for the unknown quantity.

        # INPUTS
        # ------
        # mesh : dolfin.mesh or string
        #        the filename of the xml file with the mesh definition
        #        or a dolfin mesh

        # boundaries : dolfin.MeshFunction or string
        #              the filename of the xml file with the boundaries definition
        #              or a MeshFunction that defines the boundaries

        # p : int
        #     the order of the approximating polynomial basis functions

        # """

        """
        Grad-Shafranov solver implementation as a class.

        :param constrList: list of constraints of the type
                [(P1, angle1),(P2, angle2), ...].
                Points must to have a "list" rappresentation, i.e. the x and y
                coordinates must be accessible using P[0] and P[1].
                Note: FreeCAD.Base.Vector fits the requirements
        :type constrList: list(tuple)

        :param degree: option to consider angle as degree, defaults to False
        :type degree: boolean

        :return: center coordinates and major and minor axis length
        :rtype: tuple

        Example:
            constrainList = [(P1,angle1), (P2,), (P3,)]
        """

        # ======================================================================
        # define the geometry
        if isinstance(
            mesh, str
        ):  # check wether mesh is a filename or a mesh, then load it or use it
            self.mesh = dolfin.Mesh(mesh)  # define the mesh
        else:
            self.mesh = mesh  # use the mesh

        # ======================================================================
        # define boundaries
        if boundaries is None:  # Dirichlet B.C. are defined
            self.boundaries = dolfin.MeshFunction(
                "size_t", mesh, mesh.topology().dim() - 1
            )  # initialize the MeshFunction
        elif isinstance(
            boundaries, str
        ):  # check wether boundaries is a filename or a MeshFunction, then load it or use it
            self.boundaries = dolfin.MeshFunction(
                "size_t", self.mesh, boundaries
            )  # define the boundaries
        else:
            self.boundaries = boundaries

        # ======================================================================
        # define the function space and bilinear forms

        self.V = dolfin.FunctionSpace(self.mesh, "CG", p)  # the solution function space

        # define trial and test functions
        self.u = dolfin.TrialFunction(self.V)
        self.v = dolfin.TestFunction(self.V)

        # Define r
        r = dolfin.Expression("x[0]", degree=p)

        self.a = (
            1
            / (2.0 * dolfin.pi * 4 * dolfin.pi * 1e-7)
            * (1 / r * dolfin.dot(dolfin.grad(self.u), dolfin.grad(self.v)))
            * dolfin.dx
        )

        # initialize solution
        self.psi = dolfin.Function(self.V)
        self.psi.set_allow_extrapolation(True)

    def solve(
        self, g, dirichletBCFunction=None, dirichlet_marker=None, neumannBCFunction=None
    ):
        """
        Solve the Grad-Shafranov equation given a right hand side g, Dirichlet and
        Neumann boundary conditions and convergence tolerance error.

        INPUTS
        ------
        g : dolfin.Expression or dolfin.Function
            the right hand side function of the Poisson problem

        dirichletBCFunction : dolfin.Expression o dolfin.Function
                              the Dirichlet boundary condition function

        neumannBCFunction : dolfin.Expression or dolfin.Function
                            the Neumann boundary condition function

        tol : float64
              the error goal to stop the iteration process

        dirichlet_marker : int
                           the identification number for the dirichlet boundary

        OUTPUTS
        -------
        psi : the solution of the Grad-Shafranov problem

        """
        # print(g.compute_vertex_values()[:4])
        if neumannBCFunction is None:
            neumannBCFunction = dolfin.Expression("0.0", degree=2)

        # define the right hand side
        self.L = g * self.v * dolfin.dx - neumannBCFunction * self.v * dolfin.ds

        # define the Dirichlet boundary conditions
        if dirichletBCFunction is None:
            dirichletBCFunction = dolfin.Expression("0.0", degree=2)
            dirichletBC = dolfin.DirichletBC(self.V, dirichletBCFunction, "on_boundary")
        else:
            dirichletBC = dolfin.DirichletBC(
                self.V, dirichletBCFunction, self.boundaries, dirichlet_marker
            )  # dirichlet_marker is the identification of Dirichlet BC in the mesh
        bcs = [dirichletBC]

        # solve the system taking into account the boundary conditions
        dolfin.solve(self.a == self.L, self.psi, bcs)

        self.__calculateB()

        # return the solution
        return self.psi

    def __calculateB(self):
        # POSTPROCESSING
        W = dolfin.VectorFunctionSpace(
            self.mesh, "P", 1
        )  # new function space for mapping B as vector

        r = dolfin.Expression("x[0]", degree=1)

        # calculate derivatives
        Bx = -self.psi.dx(1) / (2 * dolfin.pi * r)
        Bz = self.psi.dx(0) / (2 * dolfin.pi * r)

        self.B = dolfin.project(
            dolfin.as_vector((Bx, Bz)), W
        )  # project B as vector to new function space


def create_plasma(lcar=0.15):
    p = JohnerLCFS()
    p.adjust_variable('r_0', 8.983)
    p.adjust_variable('a', 8.983/3.1)
    p.adjust_variable('kappa_u', 1.75)
    p.adjust_variable('delta_u', 0.5)
    p.adjust_variable('kappa_l', 1.85)
    p.adjust_variable('delta_l', 0.5)
    p.adjust_variable('phi_u_neg', 180)
    p.adjust_variable('phi_u_pos', 0, -10, 10)
    p.adjust_variable('phi_l_neg', -120)
    p.adjust_variable('phi_l_pos', 30)

    lcfs = p.create_shape(label="LCFS")
    lcfs.mesh_options = {"lcar": lcar, "physical_group": "LCFS"}
    face = BluemiraFace(lcfs, label="plasma_surface")
    face.mesh_options = {"lcar": lcar, "physical_group": "surface"}
    c_plasma = PhysicalComponent(name="plasma", shape=face)
    return c_plasma


class PlasmaCurrent():
    def __init__(self, pprime, ffprime, target_curr, plasma_area):
        self.pprime = pprime
        self.ffprime = ffprime
        self.target_curr = target_curr
        self.plasma_area = plasma_area
        self.k = 1

    def curr_dens(self, points, x2D):
        j = np.zeros(len(points))
        if x2D is None:
            j[:] = self.target_curr / self.plasma_area
        else:
            a = np.array([points[i, 0] * self.pprime(x2D[i]) for i in range(len(
                x2D))])
            b = np.array([1/ MU_0 / points[i, 0] * self.ffprime(x2D[i]) for i in
                 range(len(x2D))])
            j = self.k * (a + b)
        return j

    def convert_to_dolfin(self, V, psi, psi_ax, psi_b):
        f = dolfin.Function(V)
        p = V.ufl_element().degree()
        mesh = V.mesh()
        points = mesh.coordinates()
        if psi_ax == 0:
            x2D = None
        else:
            x2D = calculate_x2D(psi, psi_ax, psi_b)
        data = self.curr_dens(points, x2D)

        if p > 1:
            # generate a 1-degree function space
            V1 = dolfin.FunctionSpace(mesh, 'CG', 1)
            f1 = dolfin.Function(V1)
            d2v = dolfin.dof_to_vertex_map(V1)
            new_data = [data[d2v[i]] for i in range(mesh.num_vertices())]
            f1.vector().set_local(new_data)
            f = dolfin.interpolate(f1, V)
        else:
            d2v = dolfin.dof_to_vertex_map(V)
            new_data = [data[d2v[i]] for i in range(mesh.num_vertices())]
            f.vector().set_local(new_data)
        return f


def convert_to_dolfin(V, data):
    f = dolfin.Function(V)
    p = V.ufl_element().degree()
    mesh = V.mesh()
    if p > 1:
        # generate a 1-degree function space
        V1 = dolfin.FunctionSpace(mesh, 'CG', 1)
        f1 = dolfin.Function(V1)
        d2v = dolfin.dof_to_vertex_map(V1)
        new_data = [data[d2v[i]] for i in range(mesh.num_vertices())]
        f1.vector().set_local(new_data)
        f = dolfin.interpolate(f1, V)
    else:
        d2v = dolfin.dof_to_vertex_map(V)
        new_data = [data[d2v[i]] for i in range(mesh.num_vertices())]
        f.vector().set_local(new_data)
    return f


def calculate_x2D(psi_data, psi_ax, psi_b):
    return np.sqrt(abs((psi_data - psi_ax) / (psi_b - psi_ax)))


def find_psi_ax(points, psi):
    ind = np.argmax(psi)
    return points[ind, :], psi[ind]

def find_psi_ax_2(V, psi):
    points = V.tabulate_dof_coordinates()
    psi_data = psi.vector()[:]
    ind = np.argmax(psi_data)
    return points[ind, :], psi_data[ind]

def find_psi_ax_3(V, psi):
    point_ax_0, psi_ax = find_psi_ax_2(V, psi)
    grad = dolfin.project(dolfin.grad(psi))
    import scipy.optimize as optimize
    sol = optimize.root(grad, point_ax_0)
    return sol.x, psi(sol.x)


def plasma_fixed_boundary(
    plasma,
    createmesh=True,
    meshfile="Mesh.msh",
    meshdir=".",
    Pax=None,
    Pax_lcar=0.1,
    maxiter=100,
    tol=1e-6,
    p=5,
    transport_solver=None,
):

    import os

    fullmeshfile = os.path.join(meshdir, meshfile)

    print(fullmeshfile)

    if createmesh:
        #### Mesh Generation ####
        import bluemira.mesh.meshing as meshing
        from bluemira.mesh.tools import msh_to_xdmf, import_mesh
        import matplotlib.pyplot as plt

        m = meshing.Mesh()
        buffer = m(plasma)

    prefix, _ = os.path.splitext(fullmeshfile)

    msh_to_xdmf(meshfile, dimensions=(0, 2), directory=meshdir, verbose=True)
    mesh, boundaries, subdomains, labels = import_mesh(
        prefix,
        directory=meshdir,
        subdomains=True,
    )
    dolfin.plot(mesh)
    plt.show()

    solver = GradShafranovLagrange(mesh, p=p)

    dx = dolfin.Measure("dx", domain=mesh)
    points = mesh.coordinates()

    print(transport_solver.I_p)

    j_plasma = PlasmaCurrent(transport_solver.pprime, transport_solver.ffprime,
                             transport_solver.I_p, plasma.shape.area)
    psi_data = solver.psi.compute_vertex_values()
    # point_ax, psi_ax = find_psi_ax_3(solver.V, solver.psi)
    psi_b = np.min(psi_data)
    g = j_plasma.convert_to_dolfin(solver.V, psi_data, 0, 0)
    # print('point_ax = {} psi_ax = {}'.format(point_ax, psi_ax))

    solver.solve(g)

    psi_data = solver.psi.compute_vertex_values()
    point_ax, psi_ax = find_psi_ax_3(solver.V, solver.psi)
    psi_b = np.min(psi_data)
    j_plasma.k = 1
    g = j_plasma.convert_to_dolfin(solver.V, solver.psi.compute_vertex_values(),
                                   psi_ax, psi_b)
    j_plasma.k = j_plasma.target_curr / dolfin.assemble(g * dx)
    g = j_plasma.convert_to_dolfin(solver.V, solver.psi.compute_vertex_values(),
                                   psi_ax, psi_b)
#    print('point_ax = {} psi_ax = {}'.format(point_ax, psi_ax))

    solver.psi_ax = psi_ax
    solver.point_ax = point_ax
    solver.psi_b = psi_b

    eps = 1.0  # error measure ||u-u_k||
    i = 0  # iteration counter
    while eps > tol and i < maxiter:
        i += 1

        J_data = g.compute_vertex_values()

        psi_data = solver.psi.compute_vertex_values()
        x2D = calculate_x2D(psi_data, psi_ax, psi_b)
        # print('PREV: psi_b = {} psi_ax = {}'.format(psi_b, psi_ax))

        dolfin.plot(g, mesh)
        plt.title(f'J at iteration {i-1}')
        plt.show()
        # print(dolfin.assemble(g * dx))

        solver = GradShafranovLagrange(mesh, p=p)
        solver.solve(g)

        new_psi_data = solver.psi.compute_vertex_values()
        point_ax, psi_ax = find_psi_ax_3(solver.V, solver.psi)
        print('point_ax = {} psi_ax = {}'.format(point_ax, psi_ax))
        psi_b = np.min(new_psi_data)

        new_x2D = calculate_x2D(new_psi_data, psi_ax, psi_b)

        # diff = new_psi_data - psi_data

        diff = (new_x2D - x2D)**2
        eps = np.linalg.norm(diff, ord=np.Inf)
        print('iter = {} eps = {}'.format(i, eps))

        diff = convert_to_dolfin(solver.V, diff)
        eps = dolfin.assemble(diff * dx) #/dolfin.assemble(convert_to_dolfin(
        # solver.V, x2D) * dx)
        print('iter = {} eps = {}'.format(i, eps))

        eps = dolfin.errornorm(convert_to_dolfin(solver.V,new_x2D), convert_to_dolfin(
            solver.V,x2D))
        print('iter = {} eps = {}'.format(i, eps))

        j_plasma.k = 1
        g = j_plasma.convert_to_dolfin(solver.V, solver.psi.compute_vertex_values(),
                                       psi_ax, psi_b)
        j_plasma.k = j_plasma.target_curr / dolfin.assemble(g * dx)
        g = j_plasma.convert_to_dolfin(solver.V, solver.psi.compute_vertex_values(),
                                       psi_ax, psi_b)

        solver.psi_ax = psi_ax
        solver.point_ax = point_ax
        solver.psi_b = psi_b

        new_J_data = g.compute_vertex_values()
        J_theta = 0.9
        g = convert_to_dolfin(solver.V, J_theta * new_J_data + (1 - J_theta) * J_data )

    return solver

if __name__ == "__main__":

    new_params = {
        "A": 3.1,
        "R_0": 8.9830e00,
        "I_p": 19,
        "B_0": 5.31,
        "V_p": -2500,
        "v_burn": -1.0e6,
        "kappa_95": 1.652,
        "delta_95": 0.333,
        "delta": 0.575,
        "kappa": 1.7,
        "q_95": 3.25,
        "f_ni": 0,
    }

    # plasmod options
    PLASMOD_PATH = "/home/ivan/Desktop/bluemira_project/plasmod/bin/"
    binary = f"{PLASMOD_PATH}plasmod"

    from bluemira.base.config import Configuration
    plasmod_params = Configuration(new_params)

    problem_settings = {
        "amin": new_params['R_0'] / new_params['A'],
        "pfus_req": 2000.0,
        "pheat_max": 100.0,
        "q_control": 50.0,
        "i_impmodel": "PED_FIXED",
        "i_modeltype": "GYROBOHM_2",
        "i_equiltype": "q95_sawtooth",
        "i_pedestal": "SAARELMA",
    }

    plasmod_build_config = {
        "problem_settings": problem_settings,
        "mode": "read",
        "binary": binary,
    }

    plasmod_options = {"params": plasmod_params, "build_config": plasmod_build_config}

    from bluemira.equilibria.fem_fixed_boundary.transport_solver import PlasmodTransportSolver

    plasmod_solver = PlasmodTransportSolver(
        params=plasmod_options["params"],
        build_config=plasmod_options["build_config"],
        read_dir="/home/ivan/Desktop/bluemira_project/bluemira/examples/equilibria"
                 "/fem_fixed_boundary/mira_matlab_data/PLASMOD"
    )

    from bluemira.equilibria.fem_fixed_boundary.utilities import plot_profile

    plot_profile(
        plasmod_solver.x, plasmod_solver.pprime(plasmod_solver.x), "pprime", "-"
    )
    plot_profile(
        plasmod_solver.x, plasmod_solver.ffprime(plasmod_solver.x), "ffprime", "-"
    )

    plasma = create_plasma(0.1)

    gs_solver = plasma_fixed_boundary(
        plasma=plasma,
        createmesh=True,
        meshfile="Mesh.msh",
        meshdir=".",
        Pax=None,
        Pax_lcar=0.1,
        maxiter=30,
        tol=1e-5,
        p=2,
        transport_solver=plasmod_solver,
    )

    lcar_fine = 0.05
    # create the finer mesh to calculate the isofluxes
    plasma.shape.boundary[0].mesh_options = {
        "lcar": lcar_fine,
        "physical_group": "lcfs",
    }
    plasma.shape.mesh_options = {"lcar": lcar_fine, "physical_group": "plasma_face"}

    import bluemira.mesh.meshing as meshing
    from bluemira.mesh.tools import msh_to_xdmf, import_mesh
    from bluemira.equilibria.fem_fixed_boundary.utilities import calculate_plasma_shape_params

    m = meshing.Mesh()
    buffer = m(plasma)

    msh_to_xdmf("Mesh.msh", dimensions=(0, 2), directory=".", verbose=True)

    mesh, boundaries, subdomains, labels = import_mesh(
        "Mesh",
        directory=".",
        subdomains=True,
    )

    points = mesh.coordinates()
    psi_data = np.array([gs_solver.psi(x) for x in points])

    # calculate kappa_95 and delta_95
    R_geo, kappa_95, delta_95 = calculate_plasma_shape_params(
        points, psi_data, [gs_solver.psi_ax * 0.05]
    )
    R_geo, kappa_95, delta_95 = R_geo[0], kappa_95[0], delta_95[0]

    delta95_t = 0.333
    kappa95_t = 1.652

    # calculate the iteration error
    err_delta = abs(delta_95 - delta95_t) / delta95_t
    err_kappa = abs(kappa_95 - kappa95_t) / kappa95_t
    iter_err = max(err_delta, err_kappa)

    print(" ")
    print(f"bluemira delta95 = {delta_95}")
    print(f"target delta95 = {delta95_t}")
    print(f"|Target - bluemira|/Target = {err_delta}")
    print(" ")
    print(f"bluemira kappa95 = {kappa_95}")
    print(f"target kappa95 = {kappa95_t}")
    print(f"|Target - bluemira|/Target = {err_delta}")

    point_ax, psi_ax = find_psi_ax(gs_solver.V.mesh().coordinates(),
                                   gs_solver.psi.compute_vertex_values())
    point_ax2, psi_ax2 = find_psi_ax_2(gs_solver.V, gs_solver.psi)
    point_ax3, psi_ax3 = find_psi_ax_3(gs_solver.V, gs_solver.psi)