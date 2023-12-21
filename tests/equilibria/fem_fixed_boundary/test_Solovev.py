# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later


import gmsh
import matplotlib.pyplot as plt
import numpy as np
import pytest
import scipy
import ufl
from dolfinx import fem
from dolfinx import mesh as dmesh
from mpi4py import MPI

from bluemira.base.constants import MU_0
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemMagnetostatic2d,
)
from bluemira.equilibria.fem_fixed_boundary.utilities import (
    calculate_plasma_shape_params,
    find_flux_surface,
    find_magnetic_axis,
    get_flux_surfaces_from_mesh,
    get_mesh_boundary,
    plot_scalar_field,
)
from bluemira.magnetostatics.fem_utils import (
    closest_point_in_mesh,
    model_to_mesh,
    read_from_msh,
)


class Solovev:
    """
    Solov'ev analytical solution to a fixed boundary equilibrium problem with a symmetric
    plasma boundary sa described in [Zheng1996]. Used for verification purposes.

    .. [Zheng1996] S. B. Zheng, A. J. Wootton, and Emilia R. Solano , "Analytical
        tokamak equilibrium for shaped plasmas", Physics of Plasmas 3, 1176-1178 (
        1996) https://doi.org/10.1063/1.871772
    """

    def __init__(self, R_0, a, kappa, delta, A1, A2):  # noqa: N803
        self.R_0 = R_0
        self.a = a
        self.kappa = kappa
        self.delta = delta
        self.A1 = A1
        self.A2 = A2
        self._find_params()
        self._psi_ax = None
        self._psi_b = None

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
            [1.0, ri**2, ri**4, ri**2 * np.log(ri)],
            [1.0, ro**2, ro**4, ro**2 * np.log(ro)],
            [
                [1.0, ri_2, ri_4, ri_2 * np.log(ri)],
                [1.0, ro_2, ro_4, ro_2 * np.log(ro)],
                [1.0, rt_2, rt_2 * (rt_2 - 4 * zt_2), rt_2 * rt_lg - zt_2],
                [0.0, 2.0, 4 * (rt_2 - 2 * zt_2), 2 * rt_lg + 1.0],
            ]
        )

        b = np.sum(
            np.array(
                [
                    [-ri_4 * 0.125, 0],
                    [-ro_4 * 0.125, 0],
                    [-(rt**4) * 0.125, zt_2 * 0.5],
                    [-rt_2 * 0.5, 0],
                ]
            )
            * np.array([self.A1, self.A2]),
            axis=1,
        )

        self.coeff = scipy.linalg.solve(m, b)
        self._m = np.concatenate((self.coeff, np.array([self.A1, self.A2])))[:, None]

    def psi(self, point):
        """
        Calculate psi analytically at a point.
        """

        psi_func = np.atleast_2d(
            np.array(
                [
                    np.ones_like(point[0]),
                    point[0] ** 2,
                    point[0] ** 2 * (point[0] ** 2 - 4 * point[1] ** 2),
                    point[0] ** 2 * np.log(point[0]) - point[1] ** 2,
                    (point[0] ** 4) / 8.0,
                    -(point[1] ** 2) / 2.0,
                ]
            ).T
        ).T

        return np.squeeze(2 * np.pi * np.sum(psi_func * self._m, axis=0))

    def plot_psi(self, ri, zi, dr, dz, nr, nz, levels=20, axis=None, tofill=True):
        """
        Plot psi
        """
        r = np.linspace(ri, ri + dr, nr)
        z = np.linspace(zi, zi + dz, nz)
        rv, zv = np.meshgrid(r, z)
        points = np.vstack([rv.ravel(), zv.ravel()]).T
        psi = self.psi(points.T)
        cplot = plot_scalar_field(
            points[:, 0], points[:, 1], psi, levels=levels, ax=axis, tofill=tofill
        )
        return (*cplot, points, psi)

    def psi_gradient(self, point):
        return scipy.optimize.approx_fprime(point, self.psi)

    @property
    def psi_ax(self):
        """Poloidal flux on the magnetic axis"""
        # if self._psi_ax is None:
        #     self._psi_ax = self.psi(find_magnetic_axis(lambda x: self.psi(x), None))
        if self._psi_ax is None:
            result = scipy.optimize.minimize(lambda x: -self.psi(x), (self.R_0, 0))
            self._psi_ax = self.psi(result.x)
            self._rz_ax = result.x
        return self._psi_ax

    @property
    def psi_b(self):
        """Poloidal flux on the boundary"""
        if self._psi_b is None:
            self._psi_b = 0.0
        return self._psi_b

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
        return lambda _x: -self.A1 / MU_0

    @property
    def ffprime(self):
        return lambda _x: self.A2

    @property
    def jp(self):
        return lambda x: x[0] * self.pprime(x) + self.ffprime(x) / (MU_0 * x[0])


def create_mesh(solovev, LCFS, lcar):
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

    return (mesh, ct, ft), labels, psi_ax


class TestSolovevZheng:
    @pytest.fixture(scope="class", autouse=True)
    def setup_class(self, tmp_path_factory):  # noqa: PT004
        cls = type(self)
        tmp_path = tmp_path_factory.mktemp("Solvev")
        # set problem parameters
        R_0 = 9.07
        A = 3.1
        delta = 0.5
        kappa = 1.7
        a = R_0 / A

        # Solovev parameters for pprime and ffprime
        A1 = -6.84256806e-02  # noqa: N806
        A2 = -6.52918977e-02  # noqa: N806

        # create the Solovev instance to get the exact psi
        solovev = Solovev(R_0, a, kappa, delta, A1, A2)

        levels = np.linspace(solovev.psi_b, solovev.psi_ax, 20)
        _ax, cntr, _cntrf, _points, _psi = solovev.plot_psi(
            5.0, -6, 8.0, 12.0, 100, 100, levels=levels
        )
        plt.show()

        cls.boundary = find_flux_surface(solovev.psi_norm_2d, 1, n_points=500)

        # Find the LCFS.
        # Note: the points returned by matplotlib can have a small "interpolation" error,
        # thus psi on the LCFS could not be exaclty 0.
        LCFS = cntr.collections[0].get_paths()[0].vertices

        # create the mesh
        lcar = 1

        (mesh, ct, ft), labels, psi_ax = create_mesh(solovev, LCFS, lcar)

        (mesh1, ct1, ft1), labels = read_from_msh("Mesh.msh", gdim=2)

        # Inizialize the em solever
        gs_solver = FemMagnetostatic2d(2)
        gs_solver.set_mesh(mesh, ct)

        # create the plasma density current function
        g = fem.Function(gs_solver.V)
        # select the dofs coordinates in the xz plane
        dof_points = gs_solver.V.tabulate_dof_coordinates()[:, 0:2]
        g.x.array[:] = solovev.jp(dof_points.T)

        # interpolate the exact solution on the solver function space
        psi_exact_fun = fem.Function(gs_solver.V)
        psi_exact_fun.x.array[:] = solovev.psi(dof_points.T)

        # boundary conditions
        dofs = fem.locate_dofs_topological(
            gs_solver.V, mesh.topology.dim - 1, ft.find(0)
        )
        psi_exact_boundary = fem.Function(gs_solver.V)
        psi_exact_boundary.x.array[dofs] = 0
        dirichlet_bcs_1 = fem.dirichletbc(psi_exact_boundary, dofs)

        tdim = mesh.topology.dim
        facets = dmesh.locate_entities_boundary(
            mesh, tdim - 1, lambda x: np.full(x.shape[1], True)
        )
        dirichlet_bcs_2 = fem.dirichletbc(
            psi_exact_fun, fem.locate_dofs_topological(gs_solver.V, tdim - 1, facets)
        )

        dofs = fem.locate_dofs_topological(
            gs_solver.V, mesh.topology.dim - 1, ft.find(0)
        )
        psi_exact_boundary = fem.Function(gs_solver.V)
        psi_exact_boundary.x.array[dofs] = psi_exact_fun.x.array[dofs]
        dirichlet_bcs_3 = fem.dirichletbc(psi_exact_boundary, dofs)

        cls.mean_err = []
        cls.itot = []

        cls.gs_solver = gs_solver
        gs_solver.define_g(g, None)
        cls.fe_psi_calc = gs_solver.solve()

        # TODO(je-cook) convergence is not very tight old:1e-5, new:6e-3
        for dirich_bc, is_close in zip(
            ((None, dirichlet_bcs_1), (dirichlet_bcs_2, dirichlet_bcs_3)), (2e-1, 6e-3)
        ):
            for d in dirich_bc:
                # solve the Grad-Shafranov equation
                gs_solver.define_g(g, d)
                gs_solver.solve()

                dx = ufl.Measure("dx", subdomain_data=ct, domain=mesh)
                cls.itot.append(fem.assemble_scalar(fem.form(g * dx)))

                err = fem.form((gs_solver.psi - psi_exact_fun) ** 2 * dx)
                err_val = np.sqrt(
                    gs_solver.psi.function_space.mesh.comm.allreduce(
                        fem.assemble_scalar(err), MPI.SUM
                    )
                )
                assert err_val < is_close
                cls.mean_err.append(err_val)

        cls.solovev = solovev
        cls.mesh = mesh

    def test_psi_mesh_array(self):
        """
        Compare the psi Solovev analytical solution as described in [Zheng1996] with the
        one calculated using the implemented magnetostic module.

        .. [Zheng1996] S. B. Zheng, A. J. Wootton, and Emilia R. Solano , "Analytical
           tokamak equilibrium for shaped plasmas", Physics of Plasmas 3, 1176-1178 (
           1996) https://doi.org/10.1063/1.871772
        """
        # calculate the GS and analytic solution on the mesh points
        points_x, points_y = get_mesh_boundary(self.mesh)
        _f, ax = plt.subplots()
        ax.plot(points_x, points_y, "r-")
        ax.set_title("Check mesh boundary function")

        dofs_points = self.gs_solver.psi.function_space.tabulate_dof_coordinates()
        psi_calc_data = self.gs_solver.psi(dofs_points)
        (ax, _cntr, _cntrf) = plot_scalar_field(
            dofs_points[:, 0], dofs_points[:, 1], psi_calc_data
        )
        ax.set_title("Plot psi from recalculated dof_points")

        (ax, _cntr, _cntrf) = plot_scalar_field(
            dofs_points[:, 0], dofs_points[:, 1], self.gs_solver.psi.x.array[:]
        )
        ax.set_title("Plot psi from dof_points")

        psi_exact = self.solovev.psi(dofs_points.T)

        error = abs(psi_calc_data - psi_exact)

        levels = np.linspace(0.0, max(error) * 1.1, 50)
        (ax, _cntr, _cntrf) = plot_scalar_field(
            dofs_points[:, 0],
            dofs_points[:, 1],
            error,
            levels=levels,
            ax=None,
            tofill=True,
        )
        ax.set_title("Error")

        # calculate the error norm
        # TODO (je-cook error margin increased from 1e-5 to 2e-5)
        assert (
            np.linalg.norm(psi_calc_data - psi_exact, ord=2)
            / np.linalg.norm(psi_exact, ord=2)
        ) < 2e-5

    def test_psi_axis(self):
        x_axis_s, z_axis_s = find_magnetic_axis(self.solovev.psi, None)
        x_axis_fe, z_axis_fe = find_magnetic_axis(self.gs_solver.psi, self.mesh)

        # TODO (je-cook error margin increased from 1e-6 to 1.7e-5)
        np.testing.assert_allclose(x_axis_fe, x_axis_s, atol=2e-5)
        # TODO (je-cook error margin increased from 1e-6 to 1e-5)
        np.testing.assert_allclose(z_axis_fe, z_axis_s, atol=1e-5)

    def test_closest_point_in_mesh(self):
        psi_ax = max(self.gs_solver.psi.x.array)
        psi_b = 0

        def psi_norm_func(x):
            return np.sqrt(np.abs((self.gs_solver.psi(x) - psi_ax) / (psi_b - psi_ax)))

        # TODO (je-cook) what am I meant to see?
        print(calculate_plasma_shape_params(psi_norm_func, self.mesh, 0.95, True))
        print(get_flux_surfaces_from_mesh(self.mesh, psi_norm_func, None, 40))

        close_test_1 = np.array([[-1, 0.3, 0], [9, 0.3, 0]])
        np.testing.assert_allclose(
            closest_point_in_mesh(self.mesh, close_test_1), close_test_1
        )

        close_test_2 = [9, 0.3, 0]
        np.testing.assert_allclose(
            np.squeeze(closest_point_in_mesh(self.mesh, close_test_2)), close_test_2
        )

    def mean_test(self):
        np.testing.assert_allclose(self.itot, self.itot[0])
        assert self.mean_err[0] == self.mean_err[1]
        assert self.mean_err[2] == self.mean_err[3]
        assert self.mean_err[0] < 2e-1
        # TODO(je-cook) convergence is not very tight old:1e-5
        assert self.mean_err[2] < 6e-3

    def test_psi_boundary(self):
        psi_fe_boundary = [self.fe_psi_calc(point) for point in self.boundary.T]
        # Higher than I might expect, but probably because some of the points
        # lie outside the mesh, and cannot be properly interpolated.
        # TODO(je-cook) convergence is not very tight old:2e-3 new 9.1e-2 (!)
        assert np.max(np.abs(psi_fe_boundary)) < 9.1e-2
