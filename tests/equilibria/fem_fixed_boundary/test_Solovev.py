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

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.components import PhysicalComponent
from bluemira.base.constants import MU_0
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemMagnetostatic2d,
)
from bluemira.equilibria.fem_fixed_boundary.utilities import (
    Solovev,
    find_flux_surface_no_mesh,
    plot_scalar_field,
)
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import interpolate_bspline
from bluemira.geometry.wire import BluemiraWire
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf

import dolfin  # isort:skip


class TestSolovev:
    def test_Solovev_Zheng(self):  # noqa: N803
        """
        Compare the psi Solovev analytical solution as described in [Zheng1996] with the
        one calculated using the implemented magnetostic module.

        .. [Zheng1996] S. B. Zheng, A. J. Wootton, and Emilia R. Solano , "Analytical
           tokamak equilibrium for shaped plasmas", Physics of Plasmas 3, 1176-1178 (
           1996) https://doi.org/10.1063/1.871772
        """

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

        levels = 50
        axis, _, _, _, psi_exact = solovev.plot_psi(
            5.0, -6, 8.0, 12.0, 100, 100, levels=levels
        )
        plt.show()

        n_points = 500
        boundary = find_flux_surface_no_mesh(solovev.psi_norm_2d, 1, n_points=n_points)

        boundary = np.array([boundary[0, :], np.zeros(n_points), boundary[1, :]])

        curve1 = interpolate_bspline(boundary[:, : n_points // 2 + 1], "curve1")
        curve2 = interpolate_bspline(boundary[:, n_points // 2 :], "curve2")
        lcfs = BluemiraWire([curve1, curve2], "LCFS")

        # Tweaked discretisation and mesh size to get error below 1e-5 but still be fast.
        # Keep as is until we move to Fenics-X where we will need to see how it performs.
        lcfs.mesh_options = {"lcar": 0.024, "physical_group": "lcfs"}

        plasma_face = BluemiraFace(lcfs, "plasma_face")
        plasma_face.mesh_options = {"lcar": 0.5, "physical_group": "plasma_face"}

        plasma = PhysicalComponent("Plasma", shape=plasma_face)

        plasma.plot_options.view = "xz"
        plasma.plot_2d()
        plt.show()

        # mesh the plasma
        meshing.Mesh()(plasma)

        msh_to_xdmf("Mesh.msh", dimensions=(0, 2), directory=".")

        mesh, boundaries, _, _ = import_mesh(
            "Mesh",
            directory=".",
            subdomains=True,
        )

        # initialize the Grad-Shafranov solver
        p = 2
        gs_solver = FemMagnetostatic2d(p_order=p)
        gs_solver.set_mesh(mesh, boundaries)

        # Set the right hand side of the Grad-Shafranov equation, as a function of psi
        g = dolfin.Expression(
            "1/mu0*(-x[0]*A1 + A2/x[0])",
            A1=solovev.A1,
            A2=solovev.A2,
            mu0=MU_0,
            degree=p,
        )
        gs_solver.define_g(g)

        # solve the Grad-Shafranov equation
        psi_calc = gs_solver.solve()

        # calculate the GS and analytic solution on the mesh points
        mesh_points = mesh.coordinates()
        psi_calc_data = np.array([psi_calc(x) for x in mesh_points])
        psi_exact = [solovev.psi(point) for point in mesh_points]

        levels = np.linspace(min(psi_exact), max(psi_exact), 25)
        axis, cntr, _ = plot_scalar_field(
            mesh_points[:, 0],
            mesh_points[:, 1],
            psi_exact,
            levels=levels,
            axis=None,
            tofill=False,
        )

        plt.show()

        axis = None
        axis, cntr, _ = plot_scalar_field(
            mesh_points[:, 0],
            mesh_points[:, 1],
            psi_exact,
            levels=20,
            axis=axis,
            tofill=True,
        )
        plt.show()

        error = abs(psi_calc_data - psi_exact)

        levels = np.linspace(0.0, max(error) * 1.1, 50)
        axis, cntr, _ = plot_scalar_field(
            mesh_points[:, 0],
            mesh_points[:, 1],
            error,
            levels=levels,
            axis=None,
            tofill=True,
        )
        plt.show()

        # calculate the error norm
        diff = psi_calc_data - psi_exact
        eps = np.linalg.norm(diff, ord=2) / np.linalg.norm(psi_exact, ord=2)
        raise ValueError(eps)
        assert eps < 1e-5
