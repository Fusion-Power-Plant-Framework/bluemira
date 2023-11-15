import gmsh
import matplotlib.pyplot as plt
import numpy as np
import scipy
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from bluemira.base.constants import MU_0
from bluemira.equilibria.fem_fixed_boundary.utilities import (
    calculate_plasma_shape_params,
    get_flux_surfaces_from_mesh,
    get_mesh_boundary,
    plot_scalar_field,
)
from bluemira.magnetostatics.fem_utils import (
    closest_point_in_mesh,
    model_to_mesh,
    read_from_msh,
)
from bluemira.magnetostatics.finite_element_2d import FemMagnetostatic2d


class Solovev:
    """
    Solov'ev analytical solution to a fixed boundary equilibrium problem with a symmetric
    plasma boundary sa described in [Zheng1996]. Used for verification purposes.

    .. [Zheng1996] S. B. Zheng, A. J. Wootton, and Emilia R. Solano , "Analytical
        tokamak equilibrium for shaped plasmas", Physics of Plasmas 3, 1176-1178 (
        1996) https://doi.org/10.1063/1.871772
    """

    def __init__(self, R_0, a, kappa, delta, A1, A2, psi_b: float = 0):  # noqa: N803
        self.R_0 = R_0
        self.a = a
        self.kappa = kappa
        self.delta = delta
        self.A1 = A1
        self.A2 = A2
        self._find_params()
        self._psi_ax = None
        self.psi_b = psi_b
        self._rz_ax = None

    def _find_params(self):
        ri = self.R_0 - self.a
        ro = self.R_0 + self.a
        rt = self.R_0 - self.delta * self.a
        zt = self.kappa * self.a
        ri_2, ri_4 = ri**2, ri**4
        ro_2, ro_4 = ro**2, ro**4
        rt_lg, rt_2 = np.log(rt), rt**2
        zt_2 = zt**2

        m = np.array([
            [1.0, ri_2, ri_4, ri_2 * np.log(ri)],
            [1.0, ro_2, ro_4, ro_2 * np.log(ro)],
            [1.0, rt_2, rt_2 * (rt_2 - 4 * zt_2), rt_2 * rt_lg - zt_2],
            [0.0, 2.0, 4 * (rt_2 - 2 * zt_2), 2 * rt_lg + 1.0],
        ])

        b = np.sum(
            np.array([
                [-ri_4 * 0.125, 0],
                [-ro_4 * 0.125, 0],
                [-(rt**4) * 0.125, zt_2 * 0.5],
                [-rt_2 * 0.5, 0],
            ])
            * np.array([self.A1, self.A2]),
            axis=1,
        )

        self.coeff = scipy.linalg.solve(m, b)

    def psi(self, point):
        """
        Calculate psi analytically at a point.
        """

        def psi_func(x):
            x_0_2 = x[0] ** 2
            x_1_2 = x[1] ** 2
            return np.array([
                1.0,
                x_0_2,
                x_0_2 * (x_0_2 - 4 * x_1_2),
                x_0_2 * np.log(x[0]) - x_1_2,
                (x[0] ** 4) / 8.0,
                -(x_1_2) / 2.0,
            ])

        m = np.concatenate((self.coeff, np.array([self.A1, self.A2])))
        return 2 * np.pi * np.sum(psi_func(point) * m)

    def plot_psi(self, ri, zi, dr, dz, nr, nz, levels=20, axis=None, tofill=True):
        """
        Plot psi
        """
        r = np.linspace(ri, ri + dr, nr)
        z = np.linspace(zi, zi + dz, nz)
        rv, zv = np.meshgrid(r, z)
        points = np.vstack([rv.ravel(), zv.ravel()]).T
        psi = np.array([self.psi(point) for point in points])
        ax, cntr, cntrf = plot_scalar_field(
            points[:, 0], points[:, 1], psi, levels=levels, ax=axis, tofill=tofill
        )
        output = {"ax": ax, "cntr": cntr, "cntrf": cntrf}
        output["points"] = points
        output["values"] = psi
        return output

    def psi_gradient(self, point):
        return scipy.optimize.approx_fprime(point, self.psi)

    @property
    def psi_ax(self):
        """Poloidal flux on the magnetic axis"""
        if self._psi_ax is None:
            result = scipy.optimize.minimize(lambda x: -self.psi(x), (self.R_0, 0))
            self._psi_ax = self.psi(result.x)
            self._rz_ax = result.x
        return self._psi_ax

    @property
    def psi_b(self):
        """Poloidal flux on the boundary"""
        return self._psi_b

    @psi_b.setter
    def psi_b(self, value: float):
        self._psi_b = value

    @property
    def psi_norm_2d(self):
        """Normalized flux function in 2-D"""

        def myfunc(x):
            return np.sqrt(
                np.abs((self.psi(x) - self.psi_ax) / (self.psi_b - self.psi_ax))
            )

        return myfunc

    @property
    def pprime(self):
        return lambda x: -self.A1 / MU_0

    @property
    def ffprime(self):
        return lambda x: self.A2

    @property
    def jp(self):
        return lambda x: x[0] * self.pprime(x) + self.ffprime(x) / (MU_0 * x[0])


if __name__ == "__main__":
    # set problem parameters
    R_0 = 9.07
    A = 3.1
    delta = 0.5
    kappa = 1.7
    a = R_0 / A

    # Solovev parameters for pprime and ffprime
    A1 = -6.84256806e-02
    A2 = -6.52918977e-02

    # create the Solovev instance to get the exact psi
    solovev = Solovev(R_0, a, kappa, delta, A1, A2)

    levels = np.linspace(solovev.psi_b, solovev.psi_ax, 20)
    plot_info = solovev.plot_psi(5.0, -6, 8.0, 12.0, 100, 100, levels=levels)
    plt.show()

    # Find the LCFS.
    # Note: the points returned by matplotlib can have a small "interpolation" error,
    # thus psi on the LCFS could not be exaclty 0.
    LCFS = plot_info["cntr"].collections[0].get_paths()[0].vertices

    # create the mesh
    lcar = 1

    gmsh.initialize()
    # points
    point_tags = [gmsh.model.occ.addPoint(v[0], 0, v[1], lcar) for v in LCFS[:-1]]
    line_tags = [
        gmsh.model.occ.addLine(point_tags[i + 1], point_tags[i])
        for i in range(len(point_tags) - 1)
    ]
    line_tags.append(gmsh.model.occ.addLine(point_tags[0], point_tags[-1]))
    gmsh.model.occ.synchronize()
    curve_loop = gmsh.model.occ.addCurveLoop(line_tags)
    surf = gmsh.model.occ.addPlaneSurface([curve_loop])
    gmsh.model.occ.synchronize()

    # embed psi_ax point with a finer mesh
    psi_ax = solovev.psi_ax
    rz_ax = solovev._rz_ax
    psi_ax_tag = gmsh.model.occ.addPoint(rz_ax[0], 0, rz_ax[1], lcar / 50)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.embed(0, [psi_ax_tag], 2, surf)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(1, line_tags, 0)
    gmsh.model.addPhysicalGroup(2, [surf], 1)
    gmsh.model.occ.synchronize()

    # Generate mesh
    gmsh.option.setNumber("Mesh.Algorithm", 2)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lcar)
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.optimize("Netgen")

    (mesh, ct, ft), labels = model_to_mesh(gmsh.model, gdim=[0, 2])

    gmsh.write("Mesh.geo_unrolled")
    gmsh.write("Mesh.msh")
    gmsh.finalize()

    (mesh1, ct1, ft1), labels = read_from_msh("Mesh.msh", gdim=2)

    # Inizialize the em solever
    gs_solver = FemMagnetostatic2d(2)
    gs_solver.set_mesh(mesh, ct)

    # create the plasma density current function
    g = fem.Function(gs_solver.V)
    # select the dofs coordinates in the xz plane
    dof_points = gs_solver.V.tabulate_dof_coordinates()[:, 0:2]
    g.x.array[:] = np.array([solovev.jp(x) for x in dof_points])

    """
    Solve the linear GSE for the Solovev' plasma using zero boundary conditions on the LCFS
    """
    # interpolate the exact solution on the solver function space
    psi_exact_fun = fem.Function(gs_solver.V)
    # extract the dof coordinates (only on the xz plane)
    dof_points = gs_solver.V.tabulate_dof_coordinates()[:, 0:2]
    psi_exact_fun.x.array[:] = [solovev.psi(x) for x in dof_points]

    mean_err = []
    Itot = []
    # boundary conditions
    dirichlet_bcs = None

    for bc_tag in range(4):
        if bc_tag == 0:
            dirichlet_bcs = None
        elif bc_tag == 1:
            dofs = fem.locate_dofs_topological(
                gs_solver.V, mesh.topology.dim - 1, ft.find(0)
            )
            psi_exact_boundary = fem.Function(gs_solver.V)
            psi_exact_boundary.x.array[dofs] = psi_exact_fun.x.array[dofs] * 0
            dirichlet_bcs = [fem.dirichletbc(psi_exact_boundary, dofs)]
        elif bc_tag == 2:
            tdim = mesh.topology.dim
            facets = dmesh.locate_entities_boundary(
                mesh, tdim - 1, lambda x: np.full(x.shape[1], True)
            )
            dofs = fem.locate_dofs_topological(gs_solver.V, tdim - 1, facets)
            dirichlet_bcs = [fem.dirichletbc(psi_exact_fun, dofs)]
        elif bc_tag == 3:
            dofs = fem.locate_dofs_topological(
                gs_solver.V, mesh.topology.dim - 1, ft.find(0)
            )
            psi_exact_boundary = fem.Function(gs_solver.V)
            psi_exact_boundary.x.array[dofs] = psi_exact_fun.x.array[dofs]
            dirichlet_bcs = [fem.dirichletbc(psi_exact_boundary, dofs)]

        # solve the Grad-Shafranov equation
        gs_solver.define_g(g)
        gs_solver.setup_problem(dirichlet_bcs)
        gs_solver.solve()

        dx = ufl.Measure("dx", subdomain_data=ct, domain=mesh)
        Itot.append(fem.assemble_scalar(fem.form(g * dx)))

        err = fem.form((gs_solver.psi - psi_exact_fun) ** 2 * dx)
        comm = gs_solver.psi.function_space.mesh.comm
        mean_err.append(np.sqrt(comm.allreduce(fem.assemble_scalar(err), MPI.SUM)))

    np.testing.assert_allclose(Itot, Itot[0])
    assert mean_err[0] == mean_err[1]
    assert mean_err[2] == mean_err[3]
    assert mean_err[0] < 2e-1
    assert mean_err[2] < 6e-3  # TODO(je-cook) convergence is not very tight old:1e-5

    points_x, points_y = get_mesh_boundary(mesh)
    plt.plot(points_x, points_y, "r-")
    plt.title("Check mesh boundary function")
    plt.show()

    dofs_points = gs_solver.psi.function_space.tabulate_dof_coordinates()[:, 0:2]
    data = gs_solver.psi(dofs_points)
    plot_scalar_field(dofs_points[:, 0], dofs_points[:, 1], data)
    plt.title("Plot psi from recalculated dof_points")
    plt.show()

    dofs_points = gs_solver.psi.function_space.tabulate_dof_coordinates()[:, 0:2]
    plot_scalar_field(dofs_points[:, 0], dofs_points[:, 1], gs_solver.psi.x.array[:])
    plt.title("Plot psi from dof_points")
    plt.show()

    from bluemira.equilibria.fem_fixed_boundary.utilities import find_magnetic_axis

    o_point = find_magnetic_axis(gs_solver.psi, mesh)

    gs_solver.psi_ax = max(gs_solver.psi.x.array)
    gs_solver.psi_b = 0

    def psi_norm_func(x):
        return np.sqrt(
            np.abs(
                (gs_solver.psi(x) - gs_solver.psi_ax)
                / (gs_solver.psi_b - gs_solver.psi_ax)
            )
        )

    print(calculate_plasma_shape_params(psi_norm_func, mesh, 0.95, True))
    print(get_flux_surfaces_from_mesh(mesh, psi_norm_func, None, 40))

    points = np.array([[-1, 0.3, 0], [9, 0.3, 0]])
    print(closest_point_in_mesh(mesh, points))

    point = [9, 0.3, 0]
    print(closest_point_in_mesh(mesh, point))
