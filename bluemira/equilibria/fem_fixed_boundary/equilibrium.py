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

"""Fixed boundary equilibrium solve"""
import os

import numpy as np

from bluemira.base.file import get_bluemira_path
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print, bluemira_warn
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
    params,
    build_config,
    gs_options,
    delta95_t,
    kappa95_t,
    lcar_mesh=0.15,
    max_iter=30,
    iter_err_max=1e-5,
    relaxation=0.2,
    plot=False,
):
    """
    Solve the plasma fixed boundary problem using delta95 and kappa95 as target
    values and iterating on PLASMOD to have consistency with pprime and ffprime.

    Parameters
    ----------
    builder_plasma: Builder
        Plasma poloidal cross section builder object
    params: Configuration
        Parameters to use in the PLASMOD solve
    build_config: dict
        Build configuration to use in the PLASMOD solve
    gs_options: dict
        Set of options used to set up and run the FemGradShafranovFixedBoundary
    delta95_t: float
        Target value for delta at 95%
    kappa95_t: float
        Target value for kappa at 95%
    lcar_mesh: float
        Value of the characteristic length used to generate the mesh to solve the
        Grad-Shafranov problem
    max_iter: int
        Maximum number of iteration between Grad-Shafranov and PLASMOD
    iter_err_max: float
        Convergence maximum error to stop the iteration
    relaxation: float
        Iteration relaxing factor
    plot: bool
        Whether or not to plot

    Notes
    -----
    This function directly modifies the parameters of builder_plasma
    """
    delta_95 = delta95_t
    kappa_95 = kappa95_t

    directory = get_bluemira_path("", subfolder="generated_data")
    mesh_name = "FixedBoundaryEquilibriumMesh"
    mesh_name_msh = mesh_name + ".msh"

    for n_iter in range(max_iter):
        # source string to be used in changed parameters
        source = f"from equilibrium iteration {n_iter}"

        # build the plasma x-z cross-section and get its volume
        plasma = builder_plasma.build_xz().get_component("xz").get_component("LCFS")
        lcfs = plasma.shape
        plasma_volume = 2 * np.pi * lcfs.center_of_mass[0] * lcfs.area

        if plot:
            plasma.plot_options.show_faces = False
            plasma.plot_2d(show=True)

        kappa_u = builder_plasma.params.get_param("kappa_u")
        kappa_l = builder_plasma.params.get_param("kappa_l")
        delta_u = builder_plasma.params.get_param("delta_u")
        delta_l = builder_plasma.params.get_param("delta_l")

        kappa = (kappa_u + kappa_l) / 2
        delta = (delta_u + delta_l) / 2

        bluemira_debug(
            f"{kappa_u=}, {delta_u=}\n"
            f"{kappa_l=}, {delta_l=}\n"
            f"{kappa=}, {delta=}\n"
            f"{plasma_volume=}"
        )

        # initialize plasmod solver
        # - V_p is set equal to plasma volume
        params.set_parameter("V_p", plasma_volume, "m^3", source)

        params.set_parameter("kappa", kappa, "dimensionless", source)
        params.set_parameter("delta", delta, "dimensionless", source)
        params.set_parameter("kappa_95", kappa_95, "dimensionless", source)
        params.set_parameter("delta_95", delta_95, "dimensionless", source)

        plasmod_solver = PlasmodTransportSolver(
            params=params,
            build_config=build_config,
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
            "lcar": lcar_mesh,
            "physical_group": "lcfs",
        }
        plasma.shape.mesh_options = {
            "lcar": lcar_mesh,
            "physical_group": "plasma_face",
        }

        meshing.Mesh(meshfile=os.path.join(directory, mesh_name_msh))(plasma)

        msh_to_xdmf(mesh_name_msh, dimensions=(0, 2), directory=directory)

        mesh = import_mesh(
            mesh_name,
            directory=directory,
            subdomains=True,
        )[0]

        # initialize the Grad-Shafranov solver
        gs_solver = FemGradShafranovFixedBoundary(mesh, p_order=gs_options["p_order"])

        bluemira_print("Solving fixed boundary Grad-Shafranov...")

        gs_solver.solve(
            plasmod_solver.pprime,
            plasmod_solver.ffprime,
            plasmod_solver.I_p,
            iter_err_max=gs_options["iter_err_max"],
            max_iter=gs_options["max_iter"],
            relaxation=gs_options["relaxation"],
            plot=gs_options["plot"],
        )

        _, kappa_95, delta_95 = calculate_plasma_shape_params(
            gs_solver.psi_norm_2d,
            mesh,
            np.sqrt(0.95),
        )

        # calculate the iteration error
        err_delta = abs(delta_95 - delta95_t) / delta95_t
        err_kappa = abs(kappa_95 - kappa95_t) / kappa95_t
        iter_err = max(err_delta, err_kappa)

        # calculate the new kappa_u and delta_u
        kappa_u_0 = builder_plasma.params.get_param("kappa_u")
        delta_u_0 = builder_plasma.params.get_param("delta_u")

        kappa_u = (1 - relaxation) * kappa_u_0 * (
            kappa95_t / kappa_95
        ) + relaxation * kappa_u_0
        delta_u = (1 - relaxation) * delta_u_0 * (
            delta95_t / delta_95
        ) + relaxation * delta_u_0

        bluemira_debug(
            "Previous shape parameters:\n"
            f"\t {kappa_u_0=:.3f}, {delta_u_0=:.3f}\n"
            "Recalculated shape parameters:\n"
            f"\t {kappa_u=:.3f}, {delta_u=:.3f}\n"
            "\n"
            f"|Target - Actual|/Target = {err_delta:.3f}\n"
            f"|Target - bluemira|/Target = {err_kappa:.3f}\n"
        )

        bluemira_print(f"PLASMOD <-> Fixed boundary G-S iter {n_iter} : {iter_err:.3E}")

        if iter_err <= iter_err_max:
            break

        # update builder_plasma parameters
        builder_plasma.params.kappa_u = kappa_u
        builder_plasma.params.delta_u = delta_u
        builder_plasma.reinitialise(builder_plasma.params)
        bluemira_debug(f"{builder_plasma.params}")

    else:
        bluemira_warn(
            f"PLASMOD <-> Fixed boundary G-S did not converge within {max_iter} iterations:\n"
            f"\t Target kappa_95: {kappa95_t:.3f}\n"
            f"\t Actual kappa_95: {kappa_95:.3f}\n"
            f"\t Target delta_95: {delta95_t:.3f}\n"
            f"\t Actual delta_95: {delta_95:.3f}\n"
            f"\t Error: {iter_err:.3E} > {iter_err_max:.3E}\n"
        )
        return

    bluemira_print(
        f"PLASMOD <-> Fixed boundary G-S successfully converged within {n_iter} iterations:\n"
        f"\t Target kappa_95: {kappa95_t:.3f}\n"
        f"\t Actual kappa_95: {kappa_95:.3f}\n"
        f"\t Target delta_95: {delta95_t:.3f}\n"
        f"\t Actual delta_95: {delta_95:.3f}\n"
        f"\t Error: {iter_err:.3E} > {iter_err_max:.3E}\n"
    )
