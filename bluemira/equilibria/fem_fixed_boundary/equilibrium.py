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
from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.interpolate import interp1d

from bluemira.base.file import get_bluemira_path
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print, bluemira_warn
from bluemira.base.parameter_frame import NewParameterFrame as ParameterFrame
from bluemira.codes import transport_code_solver
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemGradShafranovFixedBoundary,
)
from bluemira.equilibria.fem_fixed_boundary.utilities import (
    calculate_plasma_shape_params,
    plot_profile,
)
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf


@dataclass
class PlasmaFixedBoundary:
    r_0: float
    a: float
    kappa_u: float
    kappa_l: float
    delta_u: float
    delta_l: float

    def to_dict(self):
        shape_params = {}
        for param_name in self.__dataclass_fields__:
            param_data = getattr(self, param_name)
            shape_params[param_name] = {"value": param_data}
        return shape_params


@dataclass
class FemGradShafranovOptions:
    p_order: int
    iter_err_max: float
    max_iter: int
    relaxation: float


def _interpolate_profile(x, profile_data):
    return interp1d(x, profile_data, kind="linear", fill_value="extrapolate")


def solve_transport_fixed_boundary(
    plasma_parameterisation: GeometryParameterisation,
    params: PlasmaFixedBoundary,
    transport_params: ParameterFrame,
    build_config: Dict,
    gs_options: FemGradShafranovOptions,
    delta95_t: float,
    kappa95_t: float,
    lcar_mesh: float = 0.15,
    max_iter: int = 30,
    iter_err_max: float = 1e-5,
    relaxation: float = 0.2,
    plot: bool = False,
    transport_code_module: str = "PLASMOD",
) -> PlasmaFixedBoundary:
    """
    Solve the plasma fixed boundary problem using delta95 and kappa95 as target
    values and iterating on a transport solver to have consistency with pprime
    and ffprime.

    Parameters
    ----------
    plasma_parameterisation: GeometryParameterisation
        Geometry parameterisation of the plasma
    params: Configuration
        Parameters to use in the PLASMOD solve
    build_config: dict
        Build configuration to use in the PLASMOD solve
    gs_options: FemGradShafranovOptions
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

    """
    delta_95 = delta95_t
    kappa_95 = kappa95_t

    directory = get_bluemira_path("", subfolder="generated_data")
    mesh_name = "FixedBoundaryEquilibriumMesh"
    mesh_name_msh = mesh_name + ".msh"

    plasma = plasma_parameterisation(params.to_dict())

    lcfs_boundary_options = {"lcar": lcar_mesh, "physical_group": "lcfs"}
    lcfs_options = {"lcar": lcar_mesh, "physical_group": "plasma_face"}

    for n_iter in range(max_iter):
        # build the plasma x-z cross-section and get its volume
        lcfs = plasma.create_shape()
        plasma_volume = 2 * np.pi * lcfs.center_of_mass[0] * lcfs.area

        if plot:
            lcfs.plot_options.show_faces = False
            lcfs.plot_2d(show=True)

        transport_params.kappa = (params.kappa_u + params.kappa_l) / 2
        transport_params.delta = (params.delta_u + params.delta_l) / 2

        transport_params.V_p = plasma_volume
        transport_params.kappa95 = kappa_95
        transport_params.delta95 = delta_95

        bluemira_debug(
            f"{params.kappa_u=}, {params.delta_u=}\n"
            f"{params.kappa_l=}, {params.delta_l=}\n"
            f"{transport_params.kappa=}, {transport_params.delta=}\n"
            f"{transport_params.V_p=}"
        )

        # initialize transport solver
        transport_solver = transport_code_solver(
            params=transport_params,
            build_config=build_config,
            module=transport_code_module,
        )
        transp_out_params = transport_solver.execute()

        x = transport_solver.get_profile("x")
        pprime = transport_solver.get_profile("pprime")
        ffprime = transport_solver.get_profile("ffprime")

        if plot:
            plot_profile(x, _interpolate_profile(x, pprime)(x), "pprime", "-")
            plot_profile(x, _interpolate_profile(x, ffprime)(x), "ffrime", "-")

        # generate mesh for the Grad-Shafranov solver
        lcfs.boundary[0].mesh_options = lcfs_boundary_options
        lcfs.mesh_options = lcfs_options

        meshing.Mesh(meshfile=os.path.join(directory, mesh_name_msh))(lcfs)

        msh_to_xdmf(mesh_name_msh, dimensions=(0, 2), directory=directory)

        mesh = import_mesh(mesh_name, directory=directory, subdomains=True)[0]

        # initialize the Grad-Shafranov solver
        gs_solver = FemGradShafranovFixedBoundary(mesh, p_order=gs_options.p_order)

        bluemira_print("Solving fixed boundary Grad-Shafranov...")

        gs_solver.solve(
            _interpolate_profile(x, pprime),
            _interpolate_profile(x, ffprime),
            transp_out_params.I_p,
            iter_err_max=gs_options.iter_err_max,
            max_iter=gs_options.max_iter,
            relaxation=gs_options.relaxation,
            plot=plot,
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
        kappa_u_0 = params.kappa_u
        delta_u_0 = params.delta_u

        kappa_u = (1 - relaxation) * kappa_u_0 * (
            kappa95_t / kappa_95
        ) + relaxation * kappa_u_0
        delta_u = (1 - relaxation) * delta_u_0 * (
            delta95_t / delta_95
        ) + relaxation * delta_u_0

        bluemira_debug(
            "Previous shape parameters:\n\t"
            f"{kappa_u_0=:.3f}, {delta_u_0=:.3f}\n"
            "Recalculated shape parameters:\n\t"
            f"{kappa_u=:.3f}, {delta_u=:.3f}\n\n"
            f"|Target - Actual|/Target = {err_delta:.3f}\n"
            f"|Target - bluemira|/Target = {err_kappa:.3f}\n"
        )

        bluemira_print(f"PLASMOD <-> Fixed boundary G-S iter {n_iter} : {iter_err:.3E}")

        if iter_err <= iter_err_max:
            break

        # update parameters
        params.kappa_u = kappa_u
        params.delta_u = delta_u
        plasma.adjust_variables(params.to_dict())
        bluemira_debug(f"{params}")

    if n_iter == max_iter - 1:
        message = bluemira_warn
        line_1 = f"did not converge within {max_iter}"

    else:
        message = bluemira_print
        line_1 = f"successfully converged within {n_iter}"

    message(
        f"PLASMOD <-> Fixed boundary G-S {line_1} iterations:\n\t"
        f"Target kappa_95: {kappa95_t:.3f}\n\t"
        f"Actual kappa_95: {kappa_95:.3f}\n\t"
        f"Target delta_95: {delta95_t:.3f}\n\t"
        f"Actual delta_95: {delta_95:.3f}\n\t"
        f"Error: {iter_err:.3E} > {iter_err_max:.3E}\n"
    )

    return params
