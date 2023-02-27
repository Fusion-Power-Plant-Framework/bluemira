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
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
from dolfin import Mesh
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from tabulate import tabulate

from bluemira.base.components import PhysicalComponent
from bluemira.base.constants import MU_0
from bluemira.base.file import get_bluemira_path, try_get_bluemira_path
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print, bluemira_warn
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.codes.interface import CodesSolver, RunMode
from bluemira.codes.plasmod import plot_default_profiles
from bluemira.equilibria.constants import DPI_GIF, PLT_PAUSE
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemGradShafranovFixedBoundary,
)
from bluemira.equilibria.fem_fixed_boundary.utilities import (
    calculate_plasma_shape_params,
)
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.tools import BluemiraFace
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf
from bluemira.utilities.plot_tools import make_gif, save_figure

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


def create_plasma_xz_cross_section(
    parameterisation: GeometryParameterisation,
    transport_params: ParameterFrame,
    params: PlasmaFixedBoundaryParams,
    kappa_95: float,
    delta_95: float,
    lcfs_options: Dict[str, Dict],
    source: str,
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

    Returns
    -------
    equilibrium: FixedBoundaryEquilibrium
        Final fixed boundary equilibrium result from the transport <-> fixed boundary
        equilibrium solve
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

    plot = any((plot, debug, gif))
    folder = try_get_bluemira_path("", subfolder="generated_data", allow_missing=False)
    figname = "Transport iteration "
    f, ax = None, None

    for n_iter in range(max_iter):
        plasma = create_plasma_xz_cross_section(
            parameterisation,
            transport_params,
            paramet_params,
            kappa_95,
            delta_95,
            lcfs_options,
            f"from equilibrium iteration {n_iter}",
        )

        transp_out_params, x, pprime, ffprime = _run_transport_solver(
            transport_solver, transport_params, transport_run_mode
        )

        if plot:
            if ax is not None:
                for axis in ax.flat:
                    axis.clear()
            f, ax = plot_default_profiles(transport_solver, show=False, f=f, ax=ax)
            f.suptitle(figname + str(n_iter))
            plt.pause(PLT_PAUSE)
            if debug or gif:
                save_figure(
                    f,
                    figname + str(n_iter),
                    save=True,
                    folder=folder,
                    dpi=DPI_GIF,
                )

        mesh = create_mesh(
            plasma,
            directory,
            mesh_filename,
            mesh_name_msh,
        )

        gs_solver.set_mesh(mesh)
        gs_solver.set_profiles(
            pprime,
            ffprime,
            transp_out_params.I_p.value,
            transp_out_params.B_0.value,
            transp_out_params.R_0.value,
        )

        bluemira_print("Solving fixed boundary Grad-Shafranov...")

        equilibrium = gs_solver.solve(
            plot=plot,
            debug=debug,
            gif=gif,
            figname=f"{n_iter} Fixed boundary equilibrium iteration ",
        )

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

        bluemira_print(
            f"{transport_solver.name} <-> Fixed boundary G-S iter {n_iter} : {iter_err:.3E}"
        )

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
        f"{transport_solver.name} <-> Fixed boundary G-S {line_1}:\n\t"
        f"Target kappa_95: {kappa95_t:.3f}\n\t"
        f"Actual kappa_95: {kappa_95:.3f}\n\t"
        f"Target delta_95: {delta95_t:.3f}\n\t"
        f"Actual delta_95: {delta_95:.3f}\n\t"
        f"Error: {iter_err:.3E} {ltgt} {iter_err_max:.3E}\n"
    )

    if gif:
        make_gif(folder, figname, clean=not debug)
    return equilibrium


def calc_metric_coefficients(
    flux_surfaces,
    psi_2D_func: callable,
    psi_norm_2D_func: callable,
    psi_norm_1D: np.ndarray,
):
    """
    Calculate metric coefficients of a set of flux surfaces.

    Parameters
    ----------
    flux_surfaces: List[ClosedFluxSurface]
        List of closed flux surfaces on which to calculate the coefficients
    psi_2D_func:
        Callable which calculates psi of the form f(p: Iterable[2]) = float
    psi_norm_2D_func:
        Callable which calculates psi norm of the form f(p: Iterable[2]) = float
    psi_norm_1D:
        Array of 1-D normalised psi values

    Returns
    -------
    psi_norm_1D: np.ndarray
        1-D vector of normalised psi values at which the coefficients were calculated
    volume: np.ndarray
        1-D volume vector
    g1: np.ndarray
        1-D g1 vector
    g2: np.ndarray
        1-D g2 vector
    g3: np.ndarray
        1-D g3 vector
    """
    if psi_norm_1D[0] != 0:
        # Initialise with 0 at axis
        psi_norm_1D = np.insert(psi_norm_1D, 0, 0)
    nx = psi_norm_1D.size

    g1 = np.zeros(nx)
    g2 = np.zeros(nx)
    g3 = np.zeros(nx)
    volume = np.zeros(nx)
    volume[1:] = [fs.volume for fs in flux_surfaces]

    volume_func = interp1d(psi_norm_1D, volume, fill_value="extrapolate")
    grad_vol_1D = nd.Gradient(volume_func)
    grad_psinorm_2D = nd.Gradient(psi_norm_2D_func)
    grad_psi_2D = nd.Gradient(psi_2D_func)

    def grad_psi_norm_norm(x):
        return np.hypot(*grad_psinorm_2D(x))

    def grad_psi_norm(x):
        return np.hypot(*grad_psi_2D(x))

    def grad_vol_norm(x):
        """GradV norm"""
        return grad_psi_norm_norm(x) * grad_vol_1D(psi_norm_2D_func(x))

    for i, fs in enumerate(flux_surfaces):
        points = fs.coords.xz.T
        dx = np.diff(fs.coords.x)
        dz = np.diff(fs.coords.z)
        dl = np.hypot(dx, dz)
        x_data = np.concatenate([np.array([0.0]), np.cumsum(dl)])
        # Poloidal field
        bp = np.array([grad_psi_norm(p) for p in points]) / (2 * np.pi * fs.coords.x)

        grad_V_norm_2 = np.array([grad_vol_norm(p) ** 2 for p in points])
        y0_data = 1 / bp
        y1_data = grad_V_norm_2 * y0_data
        y3_data = 1 / (fs.coords.x**2 * bp)
        y2_data = grad_V_norm_2 * y3_data

        denom = np.trapz(y0_data, x_data)
        g1[i + 1] = np.trapz(y1_data, x_data) / denom
        g2[i + 1] = np.trapz(y2_data, x_data) / denom
        g3[i + 1] = np.trapz(y3_data, x_data) / denom
        # NOTE: To future self, g1 is not used (right now), and the calculation could be removed
        # to speed things up.

    g2_temp = interp1d(psi_norm_1D[1:-1], g2[1:-1], fill_value="extrapolate")
    g2[-1] = g2_temp(psi_norm_1D[-1])

    g3_temp = interp1d(psi_norm_1D[1:-1], g3[1:-1], fill_value="extrapolate")
    g3[0] = g3_temp(psi_norm_1D[0])
    g3[-1] = g3_temp(psi_norm_1D[-1])

    return psi_norm_1D, volume, g1, g2, g3


def calc_curr_dens_profiles(
    psi_norm_1D: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    g2: np.ndarray,
    g3: np.ndarray,
    volume: np.ndarray,
    I_p: float,
    B_0: float,
    R_0: float,
    psi_ax: float,
    psi_b: float,
):
    """
    Calculate pprime and ffprime from metric coefficients, emulating behaviour
    in PLASMOD.

    Parameters
    ----------

    Returns
    -------
    """
    psi_1D = psi_ax - psi_norm_1D**2 * (psi_ax - psi_b)

    Psi1D_0 = psi_1D
    for i in range(100):
        # calculate pprime profile from p
        p_fun_psi1D = interp1d(psi_1D, p, fill_value="extrapolate")
        pprime_psi1D = nd.Derivative(p_fun_psi1D)
        pprime_psi1D_data = pprime_psi1D(psi_1D)

        q3 = q / g3
        AA = g2 / q3**2 + (16 * np.pi**4) * g3
        C = -4 * np.pi**2 * MU_0 * np.gradient(p, psi_norm_1D) / AA
        dum3 = g2 / q3
        dum2 = np.gradient(dum3, psi_norm_1D)
        B = -dum2 / q3 / AA
        Fb = -R_0 * B_0 / (2 * np.pi)
        yb = 0.5 * Fb**2
        dum2 = cumulative_trapezoid(B, psi_norm_1D, initial=0)
        dum1 = np.exp(2.0 * dum2)
        dum1 = dum1 / dum1[-1]
        dum3 = cumulative_trapezoid(C / dum1, psi_norm_1D, initial=0)
        dum3 = dum3 - dum3[-1]
        C1 = yb
        y = dum1 * (dum3 + C1)
        dum2 = g2 / q3
        dum3 = np.gradient(dum2, psi_1D)
        betahat = dum3 / q3 / AA
        chat = -4 * np.pi**2 * MU_0 * pprime_psi1D_data / AA
        FF = np.sqrt(2.0 * y)
        FFprime = 4 * np.pi**2 * (chat - betahat * FF**2)

        Phi1D = -cumulative_trapezoid(q, psi_1D, initial=0)
        Phib = Phi1D[-1]

        F = 2 * np.pi * FF
        dPsidV = -F / q * g3 / (2.0 * np.pi)
        psi_1D = np.flip(
            cumulative_trapezoid(np.flip(dPsidV), np.flip(volume), initial=0)
        )

        rms_error = np.sqrt(np.square(np.subtract(psi_1D, Psi1D_0)).mean())
        Psi1D_0 = psi_1D

        if rms_error <= 1e-5:
            break
    else:
        bluemira_warn(
            "Jackpot, you've somehow found a set of inputs for which this calculation does not converge immediately."
        )

    if I_p == 0:
        I_p = -g2[-1] * dPsidV[-1] / (4 * np.pi**2 * MU_0)

    return I_p, Phi1D, psi_1D, pprime_psi1D_data, F, FFprime
