# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Solovev example
"""

import gmsh
import matplotlib.pyplot as plt
import numpy as np
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
    find_flux_surface,
    find_magnetic_axis,
    get_mesh_boundary,
    plot_scalar_field,
)
from bluemira.magnetostatics.fem_utils import BluemiraFemFunction, model_to_mesh


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

        m = np.array(
            [
                [1.0, ri_2, ri_4, ri_2 * np.log(ri)],
                [1.0, ro_2, ro_4, ro_2 * np.log(ro)],
                [1.0, rt_2, rt_2 * (rt_2 - 4 * zt_2), rt_2 * rt_lg - zt_2],
                [0.0, 2.0, 4 * (rt_2 - 2 * zt_2), 2 * rt_lg + 1.0],
            ],
        )

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
        self._m = np.concatenate((self.coeff, np.array([self.A1, self.A2])))[:, None]

    def psi(self, point):
        """
        Calculate psi analytically at a point.
        """
        psi_func = np.atleast_2d(
            np.array([
                np.ones_like(point[0]),
                point[0] ** 2,
                point[0] ** 2 * (point[0] ** 2 - 4 * point[1] ** 2),
                point[0] ** 2 * np.log(point[0]) - point[1] ** 2,
                (point[0] ** 4) / 8.0,
                -(point[1] ** 2) / 2.0,
            ]).T
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
        """Calculate the gradient of psi in the specified point"""
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
        """Pprime"""
        return lambda _x: -self.A1 / MU_0

    @property
    def ffprime(self):
        """FFprime"""
        return lambda _x: self.A2

    @property
    def jp(self):
        """Current density"""
        return lambda x: x[0] * self.pprime(x) + self.ffprime(x) / (MU_0 * x[0])


def create_mesh(solovev, LCFS, lcar):
    """Create the mesh"""
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
    _ax, cntr, _cntrf, _points, _psi = solovev.plot_psi(
        5.0, -6, 8.0, 12.0, 100, 100, levels=levels
    )
    plt.show()

    n_points = 500
    boundary = find_flux_surface(solovev.psi_norm_2d, 1, n_points=n_points)

    # Find the LCFS.
    # Note: the points returned by matplotlib can have a small "interpolation" error,
    # thus psi on the LCFS could not be exaclty 0.
    LCFS = cntr.collections[0].get_paths()[0].vertices

    # create the mesh
    lcar = 0.5

    (mesh, ct, ft), labels, psi_ax = create_mesh(solovev, LCFS, lcar)
    labels["lcfs"] = (1, 0)

    # Inizialize the em solever
    p_order = 2
    gs_solver = FemMagnetostatic2d(p_order=p_order)
    gs_solver.set_mesh(mesh, ct)

    # create the plasma density current function
    g = BluemiraFemFunction(gs_solver.V)
    # select the dofs coordinates in the xz plane
    dof_points = gs_solver.V.tabulate_dof_coordinates()[:, 0:2]
    g.x.array[:] = solovev.jp(dof_points.T)

    # interpolate the exact solution on the solver function space
    psi_exact_fun = fem.Function(gs_solver.V)
    psi_exact_fun.x.array[:] = solovev.psi(dof_points.T)

    # boundary conditions
    dofs_boundary = fem.locate_dofs_topological(
        gs_solver.V, mesh.topology.dim - 1, ft.find(0)
    )

    # boundary 1 (set to 0)
    psi_boundary1 = fem.Function(gs_solver.V)
    psi_boundary1.x.array[:] = 0
    dirichlet_bcs_1 = fem.dirichletbc(psi_boundary1, dofs_boundary)

    # boundary 2 (set using psi_exact func)
    tdim = mesh.topology.dim
    facets = dmesh.locate_entities_boundary(
        mesh, tdim - 1, lambda x: np.full(x.shape[1], True)
    )
    dirichlet_bcs_2 = fem.dirichletbc(
        psi_exact_fun, fem.locate_dofs_topological(gs_solver.V, tdim - 1, facets)
    )

    # boundary 3 (set using psi_exact values at boundary)
    psi_boundary3 = fem.Function(gs_solver.V)
    psi_boundary3.x.array[dofs_boundary] = psi_exact_fun.x.array[dofs_boundary]
    dirichlet_bcs_3 = fem.dirichletbc(psi_boundary3, dofs_boundary)

    points_x, points_y = get_mesh_boundary(mesh)
    plt.plot(points_x, points_y, "r.")
    plt.scatter(dof_points[dofs_boundary, 0], dof_points[dofs_boundary, 1], 1)
    plt.scatter(boundary.T[:, 0], boundary.T[:, 1], 1)
    plt.title("Check mesh boundary function")
    plt.show()

    mean_err = []
    itot = []

    i = 0
    # TODO(je-cook) convergence is not very tight old:1e-5, new:6e-3
    for dirich_bc, is_close in zip(
        ((None, dirichlet_bcs_1), (dirichlet_bcs_2, dirichlet_bcs_3)), (2e-1, 6e-3)
    ):
        for d in dirich_bc:
            i = i + 1
            print(f"i = {i}")
            print(f"is_close: {is_close}")
            print(f"dirich_bc: {dirich_bc}")

            # solve the Grad-Shafranov equation
            gs_solver.define_g(g, d)
            fe_psi_calc = gs_solver.solve()

            dx = ufl.Measure("dx", subdomain_data=ct, domain=mesh)
            itot.append(fem.assemble_scalar(fem.form(g * dx)))

            err = fem.form((gs_solver.psi - psi_exact_fun) ** 2 * dx)
            err_val = np.sqrt(
                gs_solver.psi.function_space.mesh.comm.allreduce(
                    fem.assemble_scalar(err), MPI.SUM
                )
            )
            # assert err_val < is_close
            mean_err.append(err_val)

            data = gs_solver.psi(dof_points)
            plot_scalar_field(dof_points[:, 0], dof_points[:, 1], data)
            plt.title(f"Plot psi from recalculated dof_points {i}")
            plt.show()

            plot_scalar_field(
                dof_points[:, 0], dof_points[:, 1], gs_solver.psi.x.array[:]
            )
            plt.title(f"Plot psi from dof_points {i}")
            plt.show()

            # calculate the error norm
            diff = gs_solver.psi.x.array - psi_exact_fun.x.array
            eps = np.linalg.norm(diff, ord=2) / np.linalg.norm(
                psi_exact_fun.x.array, ord=2
            )

            plot_scalar_field(
                dof_points[:, 0],
                dof_points[:, 1],
                diff,
                levels=20,
                ax=None,
                tofill=True,
            )
            plt.title(f"Diff between psi exact and fem solution with bcs {i}")
            plt.show()

            mesh_points = mesh.geometry.x
            psi_calc_data = np.array([fe_psi_calc(x) for x in mesh_points])
            psi_exact = [solovev.psi(point) for point in mesh_points]

            plot_scalar_field(
                mesh_points[:, 0],
                mesh_points[:, 1],
                psi_exact,
                levels=20,
                ax=None,
                tofill=True,
            )
            plt.title(f"Psi exact on mesh points {i}")
            plt.show()

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
            plt.title(f"Absolute error on mesh points {i}")
            plt.show()

            # calculate the error norm
            diff = psi_calc_data - psi_exact
            eps = np.linalg.norm(diff, ord=2) / np.linalg.norm(psi_exact, ord=2)
            print(f"eps: {eps}")
            print(f"mean_err: {mean_err}")

            x_axis_s, z_axis_s = find_magnetic_axis(solovev.psi, None)
            x_axis_fe, z_axis_fe = find_magnetic_axis(gs_solver.psi, mesh)

            print(f"x_axis_fe - x_axis_s: {x_axis_fe - x_axis_s}")
            print(f"z_axis_fe - z_axis_s: {z_axis_fe - z_axis_s}")

            psi_fe_boundary = np.array([
                fe_psi_calc(point) for point in dof_points[dofs_boundary]
            ])
            psi_exact_boundary1 = np.array([
                solovev.psi(point) for point in dof_points[dofs_boundary]
            ])

            # Higher than I might expect, but probably because some of the points
            # lie outside the mesh, and cannot be properly interpolated.
            # TODO(je-cook) convergence is not very tight old:2e-3 new 9.1e-2 (!)
            print(
                f"np.max(np.abs(psi_fe_boundary)):"
                f"{np.max(np.abs(psi_fe_boundary - psi_exact_boundary1))}"
            )

            diff = np.abs(psi_fe_boundary - psi_exact_boundary1)
            plt.scatter(
                dof_points[dofs_boundary, 0], dof_points[dofs_boundary, 1], 10, diff
            )
            plt.show()
