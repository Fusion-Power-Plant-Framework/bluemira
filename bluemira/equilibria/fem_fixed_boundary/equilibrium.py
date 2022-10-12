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
from dataclasses import asdict, dataclass
from typing import Callable, Dict, Tuple, Type, Union

import numpy as np
from scipy.interpolate import interp1d
from tabulate import tabulate

from bluemira.base.components import PhysicalComponent
from bluemira.base.file import get_bluemira_path
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print, bluemira_warn
from bluemira.base.parameter_frame import ParameterFrame
from bluemira.base.solver import RunMode
from bluemira.codes import transport_code_solver
from bluemira.codes.interface import CodesSolver
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemGradShafranovFixedBoundary,
)
from bluemira.equilibria.fem_fixed_boundary.utilities import (
    calculate_plasma_shape_params,
    plot_profile,
)
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.wire import BluemiraWire
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf


@dataclass
class PlasmaFixedBoundaryParams:
    """Plasma Transport Fixed Boundary parameters"""

    r_0: float
    a: float
    kappa_u: float
    kappa_l: float
    delta_u: float
    delta_l: float

    def tabulate(self) -> str:
        """
        Tabulate dataclass
        """
        return tabulate(
            list(asdict(self).items()),
            headers=["name", "value"],
            tablefmt="simple",
            numalign="right",
        )


@dataclass
class FemGradShafranovOptions:
    """Fem Grad-Shafranov solver options"""

    p_order: int
    iter_err_max: float
    max_iter: int
    relaxation: float


def _interpolate_profile(
    x: np.ndarray, profile_data: np.ndarray
) -> Callable[np.ndarray, np.ndarray]:
    """Interpolate profile data"""
    return interp1d(x, profile_data, kind="linear", fill_value="extrapolate")


def _create_plasma_xz_cross_section(
    parameterisation: GeometryParameterisation,
    transport_params: ParameterFrame,
    params: PlasmaFixedBoundaryParams,
    delta_95: float,
    kappa_95: float,
    lcfs_options: Dict[str, Dict],
):
    """
    Build the plasma x-z cross-section, get its volume and update transport solver
    parameters
    """
    plasma = PhysicalComponent(
        "Plasma", shape=BluemiraFace(parameterisation.create_shape())
    )
    lcfs = plasma.shape
    plasma_volume = 2 * np.pi * lcfs.center_of_mass[0] * lcfs.area

    # Update transport parameter values
    transport_params.kappa.value = (params.kappa_u + params.kappa_l) / 2
    transport_params.delta.value = (params.delta_u + params.delta_l) / 2

    transport_params.V_p.value = plasma_volume
    transport_params.kappa_95.value = kappa_95
    transport_params.delta_95.value = delta_95

    # Set mesh options
    lcfs.boundary[0].mesh_options = lcfs_options["lcfs"]
    lcfs.mesh_options = lcfs_options["face"]

    bluemira_debug(
        f"FB Params\n\n"
        f"{params.tabulate()}\n\n"
        f"Transport Params\n\n"
        f"{transport_params.tabulate(keys=['name', 'value', 'unit'], tablefmt='simple')}"
    )
    return plasma, lcfs


def _run_transport_solver(
    transport_params: ParameterFrame,
    transport_solver: CodesSolver,
    transport_run_mode: Union[str, RunMode],
) -> Tuple[ParameterFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Run transport solver"""
    transport_solver.params.update_from_frame(transport_params)
    transp_out_params = transport_solver.execute(transport_run_mode)

    return (
        transp_out_params,
        transport_solver.get_profile("x"),
        transport_solver.get_profile("pprime"),
        transport_solver.get_profile("ffprime"),
    )


def _solve_GS_problem(
    plasma: PhysicalComponent,
    lcfs: BluemiraWire,
    pprime: Callable[np.ndarray, np.ndarray],
    ffprime: Callable[np.ndarray, np.ndarray],
    I_p: float,
    directory: str,
    mesh_filename: str,
    mesh_name_msh: str,
    gs_options: FemGradShafranovOptions,
    plot: bool = False,
) -> Tuple[float, float]:
    """
    Solve Grad-Shafranov problem
    """
    # generate mesh for the Grad-Shafranov solver
    meshing.Mesh(meshfile=os.path.join(directory, mesh_name_msh))(plasma)

    msh_to_xdmf(mesh_name_msh, dimensions=(0, 2), directory=directory)

    mesh = import_mesh(mesh_filename, directory=directory, subdomains=True)[0]

    # initialize the Grad-Shafranov solver
    gs_solver = FemGradShafranovFixedBoundary(mesh, p_order=gs_options.p_order)

    bluemira_print("Solving fixed boundary Grad-Shafranov...")

    gs_solver.solve(
        pprime,
        ffprime,
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

    return delta_95, kappa_95


def _calculate_iteration_err(
    params: PlasmaFixedBoundaryParams,
    delta_95: float,
    delta95_t: float,
    kappa_95: float,
    kappa95_t: float,
    relaxation: float,
) -> Tuple[float, float, float]:
    """
    Calculate the iteration error
    """
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
        f"{kappa_u_0=:.3e}, {delta_u_0=:.3e}\n"
        "Recalculated shape parameters:\n\t"
        f"{kappa_u=:.3e}, {delta_u=:.3e}\n\n"
        f"|Target - Actual|/Target = {err_delta:.3e}\n"
        f"|Target - bluemira|/Target = {err_kappa:.3e}\n"
    )
    return iter_err, delta_u, kappa_u


def solve_transport_fixed_boundary(
    plasma_parameterisation: Type[GeometryParameterisation],
    params: PlasmaFixedBoundaryParams,
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
    transport_run_mode: str = "run",
    mesh_filename: str = "FixedBoundaryEquilibriumMesh",
) -> PlasmaFixedBoundaryParams:
    """
    Solve the plasma fixed boundary problem using delta95 and kappa95 as target
    values and iterating on a transport solver to have consistency with pprime
    and ffprime.

    Parameters
    ----------
    plasma_parameterisation: Type[GeometryParameterisation]
        Geometry parameterisation class for the plasma
    params: PlasmaFixedBoundaryParams
        dataclass of variables to use in the PLASMOD solve
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
    transport_code_module: str
        transport code to run
    transport_run_mode: str
        transport code run mode
    mesh_filename: str
        filename for mesh output file

    """
    delta_95 = delta95_t
    kappa_95 = kappa95_t

    directory = get_bluemira_path("", subfolder="generated_data")
    mesh_name_msh = mesh_filename + ".msh"

    parameterisation = plasma_parameterisation(
        {k: {"value": v} for k, v in asdict(params).items()}
    )

    lcfs_options = {
        "face": {"lcar": lcar_mesh, "physical_group": "plasma_face"},
        "lcfs": {"lcar": lcar_mesh, "physical_group": "lcfs"},
    }

    transport_solver = transport_code_solver(
        params=transport_params,
        build_config=build_config,
        module=transport_code_module,
    )

    for n_iter in range(max_iter):

        plasma, lcfs = _create_plasma_xz_cross_section(
            parameterisation,
            transport_params,
            params,
            delta_95,
            kappa_95,
            lcfs_options,
        )

        if plot:
            lcfs.plot_options.show_faces = False
            lcfs.plot_2d(show=True)

        transp_out_params, x, pprime, ffprime = _run_transport_solver(
            transport_params, transport_solver, transport_run_mode
        )

        if plot:
            plot_profile(x, _interpolate_profile(x, pprime)(x), "pprime", "-")
            plot_profile(x, _interpolate_profile(x, ffprime)(x), "ffrime", "-")

        delta_95, kappa_95 = _solve_GS_problem(
            plasma,
            lcfs,
            _interpolate_profile(x, pprime),
            _interpolate_profile(x, ffprime),
            transp_out_params.I_p.value,
            directory,
            mesh_filename,
            mesh_name_msh,
            gs_options,
            plot,
        )

        iter_err, delta_u, kappa_u = _calculate_iteration_err(
            params,
            delta_95,
            delta95_t,
            kappa_95,
            kappa95_t,
            relaxation,
        )

        bluemira_print(f"PLASMOD <-> Fixed boundary G-S iter {n_iter} : {iter_err:.3E}")

        # update parameters
        params.delta_u = delta_u
        params.kappa_u = kappa_u

        if iter_err <= iter_err_max:
            message = bluemira_print
            line_1 = f"successfully converged within {n_iter}"
            ltgt = "<"
            break

        for name, value in asdict(params).items():
            parameterisation.adjust_variable(name, value)

    else:
        # If we don't break we didn't converge
        message = bluemira_warn
        line_1 = f"did not converge within {max_iter}"
        ltgt = ">"

    message(
        f"PLASMOD <-> Fixed boundary G-S {line_1} iterations:\n\t"
        f"Target kappa_95: {kappa95_t:.3f}\n\t"
        f"Actual kappa_95: {kappa_95:.3f}\n\t"
        f"Target delta_95: {delta95_t:.3f}\n\t"
        f"Actual delta_95: {delta_95:.3f}\n\t"
        f"Error: {iter_err:.3E} {ltgt} {iter_err_max:.3E}\n"
    )

    return params
