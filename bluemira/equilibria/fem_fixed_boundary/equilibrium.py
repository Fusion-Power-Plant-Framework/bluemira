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

"""Fixed boundary equilibrium class"""
import numpy as np

from bluemira.base.look_and_feel import bluemira_debug, bluemira_print
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemGradShafranovFixedBoundary,
)
from bluemira.equilibria.fem_fixed_boundary.transport_solver import (
    PlasmodTransportSolver,
)
from bluemira.equilibria.fem_fixed_boundary.utilities import (
    calculate_plasma_shape_params,
    plot_profile,
)
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf


def solve_plasmod_fixed_boundary(
    builder_plasma,
    plasmod_options,
    gs_options,
    delta95_t,
    kappa95_t,
    lcar_coarse=0.15,
    lcar_fine=0.05,
    niter_max=30,
    iter_err_max=1e-5,
    theta=0.8,
    gs_i_theta=0.5,
    plot=False,
    verbose=False,
):
    """
    Solve the plasma fixed boundary problem using delta95 and kappa95 as target
    values and iterating on plasmod to have consistency with pprime and ffprime.

    Parameters
    ----------
    builder_plasma: bluemira.base.builder
        plasma poloidal cross section builder object
    plasmod_options: dict
        set of options used to set up and run plasmod
    gs_options: dict
        set of options used to set up and run the FemGradShafranovFixedBoundary
    delta95_t: float
        target value for delta at 95%
    kappa95_t: float
        target value for kappa at 95%
    lcar_coarse: float
        value of the characteristic length used to generate the mesh to solve the
        Grad-Shafranov problem
    lcar_fine: float
        value of the characteristic length used to extrapolate the isoflux for the
        calculation of kappa and delta at 95%
    niter_max: int
        maximum number of iteration between Grad-Shafranov and Plasmod
    iter_err_max: float
        convergence maximum error to stop the iteration
    theta: float
        iteration relaxing factor
    gs_i_theta: float
        FemGradShafranovFixedBoundary iteration relaxing factor
    plot: bool
        Whether or not to plot
    verbose: bool
        Whether or not to print

    Notes
    -----
    This function directly modifies the parameters of builder_plasma
    """
    niter = 0
    delta_95 = delta95_t
    kappa_95 = kappa95_t

    while niter < niter_max:
        # source string to be used in changed parameters
        source = f"from equilibrium iteration {niter}"

        # build the plasma
        # - xz - plasma toroidal cross section
        # - xyz - to get the volume
        plasma = builder_plasma.build_xz().get_component("xz").get_component("LCFS")
        plasma_volume = (
            builder_plasma.build_xyz()
            .get_component("xyz")
            .get_component("LCFS")
            .shape.volume
        )

        if plot:
            plasma.plot_options.show_faces = False
            plasma.plot_2d(show=True)

        kappa_u = builder_plasma.params.get_param("kappa_u")
        kappa_l = builder_plasma.params.get_param("kappa_l")
        delta_u = builder_plasma.params.get_param("delta_u")
        delta_l = builder_plasma.params.get_param("delta_l")

        kappa = (kappa_u + kappa_l) / 2
        delta = (delta_u + delta_l) / 2

        if verbose:
            print(f"kappa_u: {kappa_u}, delta_u: {delta_u}")
            print(f"kappa_l: {kappa_l}, delta_l: {delta_l}")
            print(f"kappa: {kappa}, delta: {delta}")
            print(f"volume: {plasma_volume}")

        # initialize plasmod solver
        # - V_p is set equal to plasma volume
        plasmod_options["params"].set_parameter("V_p", plasma_volume, "m^3", source)

        plasmod_options["params"].set_parameter("kappa", kappa, "dimensionless", source)
        plasmod_options["params"].set_parameter("delta", delta, "dimensionless", source)
        plasmod_options["params"].set_parameter(
            "kappa_95", kappa_95, "dimensionless", source
        )
        plasmod_options["params"].set_parameter(
            "delta_95", delta_95, "dimensionless", source
        )

        plasmod_solver = PlasmodTransportSolver(
            params=plasmod_options["params"],
            build_config=plasmod_options["build_config"],
        )
        plasmod_solver.execute()

        if plot:
            plot_profile(
                plasmod_solver.x, plasmod_solver.pprime(plasmod_solver.x), "pprime", "-"
            )
            plot_profile(
                plasmod_solver.x, plasmod_solver.ffprime(plasmod_solver.x), "ffrime", "-"
            )

        # generate mesh for the Grad-Shafranov solver
        plasma.shape.boundary[0].mesh_options = {
            "lcar": lcar_coarse,
            "physical_group": "lcfs",
        }
        plasma.shape.mesh_options = {
            "lcar": lcar_coarse,
            "physical_group": "plasma_face",
        }

        m = meshing.Mesh()
        m(plasma)

        msh_to_xdmf("Mesh.msh", dimensions=(0, 2), directory=".", verbose=True)

        mesh, boundaries, subdomains, labels = import_mesh(
            "Mesh",
            directory=".",
            subdomains=True,
        )

        # initialize the Grad-Shafranov solver
        p_order = gs_options["p_order"]
        gs_solver = FemGradShafranovFixedBoundary(mesh, p_order=p_order)

        bluemira_print("Solving fixed boundary Grad-Shafranov...")
        # solve the Grad-Shafranov equation
        gs_solver.solve(
            plasmod_solver.pprime,
            plasmod_solver.ffprime,
            plasmod_solver.I_p,
            tol=gs_options["tol"],
            max_iter=gs_options["max_iter"],
            i_theta=gs_i_theta,
            verbose_plot=gs_options["verbose_plot"],
        )

        # create the finer mesh to calculate the isofluxes
        plasma.shape.boundary[0].mesh_options = {
            "lcar": lcar_fine,
            "physical_group": "lcfs",
        }
        plasma.shape.mesh_options = {"lcar": lcar_fine, "physical_group": "plasma_face"}

        m = meshing.Mesh()
        m(plasma)

        msh_to_xdmf("Mesh.msh", dimensions=(0, 2), directory=".", verbose=verbose)

        mesh, boundaries, subdomains, labels = import_mesh(
            "Mesh",
            directory=".",
            subdomains=True,
        )

        points = mesh.coordinates()
        x2d_data = np.array([gs_solver.psi_norm_2d(x) for x in points])

        # calculate kappa_95 and delta_95
        r_geo, kappa_95, delta_95 = calculate_plasma_shape_params(
            points, x2d_data, [np.sqrt(0.95)]
        )
        r_geo, kappa_95, delta_95 = r_geo[0], kappa_95[0], delta_95[0]

        # calculate the iteration error
        err_delta = abs(delta_95 - delta95_t) / delta95_t
        err_kappa = abs(kappa_95 - kappa95_t) / kappa95_t
        iter_err = max(err_delta, err_kappa)

        # calculate the new kappa_u and delta_u
        kappa_u_0 = builder_plasma.params.get_param("kappa_u")
        delta_u_0 = builder_plasma.params.get_param("delta_u")

        kappa_u = theta * kappa_u_0 * (kappa95_t / kappa_95) + (1 - theta) * kappa_u_0
        delta_u = theta * delta_u_0 * (delta95_t / delta_95) + (1 - theta) * delta_u_0

        if verbose:
            print("previous shape parameters")
            print(f"{kappa_u_0}, {delta_u_0}")

            print("recalculated shape parameters")
            print(f"{kappa_u}, {delta_u}")

            print(" ")
            print(f"bluemira delta95 = {delta_95}")
            print(f"target delta95 = {delta95_t}")

            print(f"|Target - bluemira|/Target = {err_delta}")

            print(" ")
            print(f"bluemira kappa95 = {kappa_95}")
            print(f"target kappa95 = {kappa95_t}")

            print(f"|Target - bluemira|/Target = {err_kappa}")

        print("\n")
        bluemira_print(f"iter_err: {iter_err:.3E}, iter_err_max: {iter_err_max:.3E}")
        print("\n")

        if iter_err <= iter_err_max:
            break

        # increase iteration number
        niter += 1

        # update builder_plasma parameters
        builder_plasma.params.kappa_u = kappa_u
        builder_plasma.params.delta_u = delta_u
        bluemira_debug(f"{builder_plasma.params}")
