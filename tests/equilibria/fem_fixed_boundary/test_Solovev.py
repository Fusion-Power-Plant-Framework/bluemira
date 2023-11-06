# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path

import gmsh
import numpy as np
import pytest
import scipy

from bluemira.base.constants import MU_0
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemMagnetostatic2d,
)
from bluemira.equilibria.fem_fixed_boundary.utilities import (
    find_magnetic_axis,
    plot_scalar_field,
)
from bluemira.magnetostatics.fem_utils import model_to_mesh, read_from_msh


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

        m = np.array([
            [1.0, ri**2, ri**4, ri**2 * np.log(ri)],
            [1.0, ro**2, ro**4, ro**2 * np.log(ro)],
            [
                1.0,
                rt**2,
                rt**2 * (rt**2 - 4 * zt**2),
                rt**2 * np.log(rt) - zt**2,
            ],
            [0.0, 2.0, 4 * (rt**2 - 2 * zt**2), 2 * np.log(rt) + 1.0],
        ])

        b = np.array([
            [-(ri**4) / 8.0, 0],
            [-(ro**4) / 8.0, 0.0],
            [-(rt**4) / 8.0, +(zt**2) / 2.0],
            [-(rt**2) / 2.0, 0.0],
        ])
        b = b * np.array([self.A1, self.A2])
        b = np.sum(b, axis=1)

        self.coeff = scipy.linalg.solve(m, b)

    def psi(self, point):
        """
        Calculate psi analytically at a point.
        """

        def psi_func(x):
            return np.array([
                1.0,
                x[0] ** 2,
                x[0] ** 2 * (x[0] ** 2 - 4 * x[1] ** 2),
                x[0] ** 2 * np.log(x[0]) - x[1] ** 2,
                (x[0] ** 4) / 8.0,
                -(x[1] ** 2) / 2.0,
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
        cplot = plot_scalar_field(
            points[:, 0], points[:, 1], psi, levels=levels, ax=axis, tofill=tofill
        )
        return (*cplot, points, psi)

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

    def psi_gradient(self, point):
        return scipy.optimize.approx_fprime(point, self.psi)

    @property
    def pprime(self):
        return lambda x: -self.A1 / MU_0

    @property
    def ffprime(self):
        return lambda x: self.A2

    @property
    def jp(self):
        return lambda x: x[0] * self.pprime(x) + self.ffprime(x) / (MU_0 * x[0])


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
        cls.solovev = Solovev(R_0, a, kappa, delta, A1, A2)

        levels = 50
        plot_info = cls.solovev.plot_psi(5.0, -6, 8.0, 12.0, 100, 100, levels=levels)

        # n_points = 500
        # boundary = find_flux_surface(cls.solovev.psi_norm_2d, 1, n_points=n_points)
        # cls.boundary = boundary
        # boundary = np.array([boundary[0, :], np.zeros(n_points), boundary[1, :]])

        # curve1 = interpolate_bspline(boundary[:, : n_points // 2 + 1], "curve1")
        # curve2 = interpolate_bspline(boundary[:, n_points // 2 :], "curve2")
        # lcfs = BluemiraWire([curve1, curve2], "LCFS")

        # # Tweaked discretisation and mesh size to get error below 1e-5 but still be fast.
        # # Keep as is until we move to Fenics-X where we will need to see how it performs.
        # lcfs.mesh_options = {"lcar": 0.024, "physical_group": "lcfs"}

        # plasma_face = BluemiraFace(lcfs, "plasma_face")
        # plasma_face.mesh_options = {"lcar": 0.5, "physical_group": "plasma_face"}

        # plasma = PhysicalComponent("Plasma", shape=plasma_face)

        # # mesh the plasma
        # meshing.Mesh()(plasma)

        # msh_to_xdmf("Mesh.msh", dimensions=(0, 2), directory=".")

        # cls.mesh, boundaries, _, _ = import_mesh(
        #     "Mesh",
        #     directory=".",
        #     subdomains=True,
        # )

        # Find the LCFS.
        # Note: the points returned by matplotlib can have a small "interpolation" error,
        # thus psi on the LCFS could not be exaclty 0.
        LCFS = plot_info[1].collections[0].get_paths()[0].vertices

        # create the mesh
        model_rank = 0
        mesh_comm = MPI.COMM_WORLD
        lcar = 1

        gmsh.initialize()
        # points
        point_tags = [gmsh.model.occ.addPoint(v[0], 0, v[1], lcar) for v in LCFS[:-1]]
        # point_tags = [gmsh.model.occ.addPoint(v[0], 0, v[1], lcar) for v in LCFS[:-1]]
        line_tags = []
        for i in range(len(point_tags) - 1):
            line_tags.append(gmsh.model.occ.addLine(point_tags[i + 1], point_tags[i]))
        line_tags.append(gmsh.model.occ.addLine(point_tags[0], point_tags[-1]))
        gmsh.model.occ.synchronize()
        curve_loop = gmsh.model.occ.addCurveLoop(line_tags)
        surf = gmsh.model.occ.addPlaneSurface([curve_loop])
        gmsh.model.occ.synchronize()

        # embed psi_ax point with a finer mesh
        psi_ax = cls.solovev.psi_ax
        rz_ax = cls.solovev._rz_ax
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

        (mesh, ct, ft), labels = model_to_mesh(
            gmsh.model, mesh_comm, model_rank, gdim=[0, 2]
        )

        gmsh.write("Mesh.geo_unrolled")
        gmsh.write("Mesh.msh")
        gmsh.finalize()

        (mesh1, ct1, ft1), labels = read_from_msh(
            "Mesh.msh", mesh_comm, model_rank, gdim=2
        )

        # initialize the Grad-Shafranov solver
        p = 2
        gs_solver = FemMagnetostatic2d(p_order=p)
        gs_solver.set_mesh(mesh, ct)

        # create the plasma density current function
        g = fem.Function(gs_solver.V)
        # select the dofs coordinates in the xz plane
        dof_points = gs_solver.V.tabulate_dof_coordinates()[:, 0:2]
        g.x.array[:] = np.array([cls.solovev.jp(x) for x in dof_points])

        # solve the Grad-Shafranov equation
        cls.fe_psi_calc = gs_solver.solve()

    def test_psi_mesh_array(self):
        """
        Compare the psi Solovev analytical solution as described in [Zheng1996] with the
        one calculated using the implemented magnetostic module.

        .. [Zheng1996] S. B. Zheng, A. J. Wootton, and Emilia R. Solano , "Analytical
           tokamak equilibrium for shaped plasmas", Physics of Plasmas 3, 1176-1178 (
           1996) https://doi.org/10.1063/1.871772
        """
        # calculate the GS and analytic solution on the mesh points
        mesh_points = self.mesh.coordinates()
        psi_calc_data = np.array([self.fe_psi_calc(x) for x in mesh_points])
        psi_exact = [self.solovev.psi(point) for point in mesh_points]

        plot_scalar_field(
            mesh_points[:, 0],
            mesh_points[:, 1],
            psi_exact,
            levels=20,
            ax=None,
            tofill=True,
        )

        error = abs(psi_calc_data - psi_exact)

        levels = np.linspace(0.0, max(error) * 1.1, 50)
        plot_scalar_field(
            mesh_points[:, 0],
            mesh_points[:, 1],
            error,
            levels=levels,
            ax=None,
            tofill=True,
        )

        # calculate the error norm
        diff = psi_calc_data - psi_exact
        eps = np.linalg.norm(diff, ord=2) / np.linalg.norm(psi_exact, ord=2)
        assert eps < 1e-5

    def test_psi_axis(self):
        x_axis_s, z_axis_s = find_magnetic_axis(self.solovev.psi, None)
        x_axis_fe, z_axis_fe = find_magnetic_axis(self.fe_psi_calc, self.mesh)
        np.testing.assert_allclose(x_axis_fe, x_axis_s, atol=1e-6)
        np.testing.assert_allclose(z_axis_fe, z_axis_s, atol=1e-6)

    def test_psi_boundary(self):
        psi_fe_boundary = [self.fe_psi_calc(point) for point in self.boundary.T]
        # Higher than I might expect, but probably because some of the points lie outside
        # the mesh, and cannot be properly interpolated.
        assert np.max(np.abs(psi_fe_boundary)) < 2e-3
