# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
from dolfin import Mesh
from scipy.interpolate import interp1d
from tabulate import tabulate

from bluemira.base.components import PhysicalComponent
from bluemira.base.file import get_bluemira_path
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print, bluemira_warn
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.codes.interface import CodesSolver, RunMode
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemGradShafranovFixedBoundary,
)
from bluemira.equilibria.fem_fixed_boundary.utilities import (
    calculate_plasma_shape_params,
    plot_profile,
)
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf

__all__ = ["solve_transport_fixed_boundary"]


@dataclass
class PlasmaFixedBoundaryParams:
    """Plasma Transport Fixed Boundary parameters"""

    r_0: float
    a: float
    kappa_u: float
    kappa_l: float
    delta_u: float
    delta_l: float

    _fields = None

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

    @classmethod
    def fields(cls) -> List:
        """List of fields in the dataclass"""
        if cls._fields is None:
            cls._fields = [k.name for k in fields(cls)]
        return cls._fields


@dataclass
class TransportSolverParams(ParameterFrame):
    """Transport Solver ParameterFrame"""

    A: Parameter[float]
    R_0: Parameter[float]
    I_p: Parameter[float]
    B_0: Parameter[float]
    V_p: Parameter[float]
    v_burn: Parameter[float]
    kappa_95: Parameter[float]
    delta_95: Parameter[float]
    delta: Parameter[float]
    kappa: Parameter[float]
    q_95: Parameter[float]
    f_ni: Parameter[float]


def _interpolate_profile(
    x: np.ndarray, profile_data: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Interpolate profile data"""
    return interp1d(x, profile_data, kind="linear", fill_value="extrapolate")


def create_plasma_xz_cross_section(
    parameterisation: GeometryParameterisation,
    transport_params: ParameterFrame,
    params: PlasmaFixedBoundaryParams,
    kappa_95: float,
    delta_95: float,
    lcfs_options: Dict[str, Dict],
    source: str,
    plot: bool,
) -> PhysicalComponent:
    """
    Build the plasma x-z cross-section, get its volume and update transport solver
    parameters
    """
    plasma = PhysicalComponent(
        "Plasma", shape=BluemiraFace(parameterisation.create_shape())
    )
    lcfs = plasma.shape
    plasma_volume = 2 * np.pi * lcfs.center_of_mass[0] * lcfs.area

    # Update transport parameter valupes
    transport_params.kappa.set_value((params.kappa_u + params.kappa_l) / 2, source)
    transport_params.delta.set_value((params.delta_u + params.delta_l) / 2, source)

    transport_params.V_p.set_value(plasma_volume, source)
    transport_params.kappa_95.set_value(kappa_95, source)
    transport_params.delta_95.set_value(delta_95, source)

    # Set mesh options
    lcfs.boundary[0].mesh_options = lcfs_options["lcfs"]
    lcfs.mesh_options = lcfs_options["face"]

    if plot:
        plasma.plot_options.show_faces = False
        plasma.plot_2d(show=True)

    bluemira_debug(
        f"FB Params\n\n"
        f"{params.tabulate()}\n\n"
        f"Transport Params\n\n"
        f"{transport_params.tabulate(keys=['name', 'value', 'unit'], tablefmt='simple')}"
    )
    return plasma


def _run_transport_solver(
    transport_solver: CodesSolver,
    transport_params: ParameterFrame,
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


def create_mesh(
    plasma: PhysicalComponent,
    directory: str,
    mesh_filename: str,
    mesh_name_msh: str,
) -> Mesh:
    """
    Create mesh
    """
    meshing.Mesh(meshfile=os.path.join(directory, mesh_name_msh))(plasma)
    msh_to_xdmf(mesh_name_msh, dimensions=(0, 2), directory=directory)
    return import_mesh(mesh_filename, directory=directory, subdomains=True)[0]


def _update_delta_kappa(
    params: PlasmaFixedBoundaryParams,
    kappa_95: float,
    kappa95_t: float,
    delta_95: float,
    delta95_t: float,
    relaxation: float,
) -> float:
    """
    Recalculate Delta and Kappa and calculate the iteration error
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
    params.kappa_u = kappa_u
    params.delta_u = delta_u

    return iter_err


def solve_transport_fixed_boundary(
    parameterisation: GeometryParameterisation,
    transport_solver: CodesSolver,
    gs_solver: FemGradShafranovFixedBoundary,
    kappa95_t: float,
    delta95_t: float,
    lcar_mesh: float = 0.15,
    max_iter: int = 30,
    iter_err_max: float = 1e-5,
    relaxation: float = 0.2,
    transport_run_mode: Union[str, RunMode] = "run",
    mesh_filename: str = "FixedBoundaryEquilibriumMesh",
    plot: bool = False,
    debug: bool = False,
    gif: bool = False,
):
    """
    Solve the plasma fixed boundary problem using delta95 and kappa95 as target
    values and iterating on a transport solver to have consistency with pprime
    and ffprime.

    Parameters
    ----------
    parameterisation: Type[GeometryParameterisation]
        Geometry parameterisation class for the plasma
    transport_solver: CodesSolver
        Transport Solver to call
    gs_solver: FemGradShafranovFixedBoundary
        Grad-Shafranov Solver instance
    kappa95_t: float
        Target value for kappa at 95%
    delta95_t: float
        Target value for delta at 95%
    lcar_mesh: float
        Value of the characteristic length used to generate the mesh to solve the
        Grad-Shafranov problem
    max_iter: int
        Maximum number of iteration between Grad-Shafranov and the transport solver
    iter_err_max: float
        Convergence maximum error to stop the iteration
    relaxation: float
        Iteration relaxing factor
    transport_run_mode: str
        Run mode for transport solver
    mesh_filename: str
        filename for mesh output file
    plot: bool
        Whether or not to plot

    """
    kappa_95 = kappa95_t
    delta_95 = delta95_t

    directory = get_bluemira_path("", subfolder="generated_data")
    mesh_name_msh = mesh_filename + ".msh"

    paramet_params = PlasmaFixedBoundaryParams(
        **{
            k: v
            for k, v in zip(
                parameterisation.variables.names, parameterisation.variables.values
            )
            if k in PlasmaFixedBoundaryParams.fields()
        }
    )

    transport_params = TransportSolverParams.from_frame(
        deepcopy(transport_solver.params)
    )

    lcfs_options = {
        "face": {"lcar": lcar_mesh, "physical_group": "plasma_face"},
        "lcfs": {"lcar": lcar_mesh, "physical_group": "lcfs"},
    }

    for n_iter in range(max_iter):

        plasma = create_plasma_xz_cross_section(
            parameterisation,
            transport_params,
            paramet_params,
            kappa_95,
            delta_95,
            lcfs_options,
            f"from equilibrium iteration {n_iter}",
            plot,
        )

        transp_out_params, x, pprime, ffprime = _run_transport_solver(
            transport_solver, transport_params, transport_run_mode
        )

        if plot:
            plot_profile(x, _interpolate_profile(x, pprime)(x), "pprime", "-")
            plot_profile(x, _interpolate_profile(x, ffprime)(x), "ffrime", "-")

        mesh = create_mesh(
            plasma,
            directory,
            mesh_filename,
            mesh_name_msh,
        )

        gs_solver.set_mesh(mesh)
        gs_solver.define_g(
            _interpolate_profile(x, pprime),
            _interpolate_profile(x, ffprime),
            transp_out_params.I_p.value,
        )

        bluemira_print("Solving fixed boundary Grad-Shafranov...")

        gs_solver.solve(plot=plot, debug=debug, gif=gif)

        _, kappa_95, delta_95 = calculate_plasma_shape_params(
            gs_solver.psi_norm_2d,
            mesh,
            np.sqrt(0.95),
        )

        iter_err = _update_delta_kappa(
            paramet_params,
            kappa_95,
            kappa95_t,
            delta_95,
            delta95_t,
            relaxation,
        )

        bluemira_print(f"PLASMOD <-> Fixed boundary G-S iter {n_iter} : {iter_err:.3E}")

        if iter_err <= iter_err_max:
            message = bluemira_print
            line_1 = f"successfully converged in {n_iter} iterations"
            ltgt = "<"
            break

        # update parameters
        for name, value in asdict(paramet_params).items():
            parameterisation.adjust_variable(name, value)

    else:
        # If we don't break we didn't converge
        message = bluemira_warn
        line_1 = f"did not converge within {max_iter} iterations"
        ltgt = ">"

    message(
        f"PLASMOD <-> Fixed boundary G-S {line_1}:\n\t"
        f"Target kappa_95: {kappa95_t:.3f}\n\t"
        f"Actual kappa_95: {kappa_95:.3f}\n\t"
        f"Target delta_95: {delta95_t:.3f}\n\t"
        f"Actual delta_95: {delta_95:.3f}\n\t"
        f"Error: {iter_err:.3E} {ltgt} {iter_err_max:.3E}\n"
    )
