# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Designer for TF Coil XY cross section."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matproplib import OperationalConditions
from matproplib.material import MaterialFraction
from scipy.optimize import minimize_scalar

from bluemira.base.constants import MU_0, MU_0_2PI, MU_0_4PI
from bluemira.base.designer import Designer
from bluemira.base.look_and_feel import (
    bluemira_debug,
    bluemira_error,
    bluemira_print,
    bluemira_warn,
)
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.base.parameter_frame.typed import ParameterFrameLike
from bluemira.magnets.cable import RectangularCable
from bluemira.magnets.case_tf import CaseTF
from bluemira.magnets.conductor import SymmetricConductor
from bluemira.magnets.utils import delayed_exp_func
from bluemira.utilities.tools import get_class_from_module


@dataclass
class TFCoilXYDesignerParams(ParameterFrame):
    """
    Parameters needed for all aspects of the TF coil design
    """

    # base params
    R0: Parameter[float]
    """Major radius [m]"""
    B0: Parameter[float]
    """Magnetic field at R0 [T]"""
    A: Parameter[float]
    """Aspect ratio"""
    n_TF: Parameter[float]
    """Number of TF coils"""
    ripple: Parameter[float]
    """Maximum plasma ripple"""
    d: Parameter[float]
    """Additional distance to calculate max external radius of inner TF leg"""
    S_VV: Parameter[float]
    """Vacuum vessel steel limit"""
    safety_factor: Parameter[float]
    """Allowable stress values"""
    B_ref: Parameter[float]
    """Reference value for B field (LTS limit) [T]"""

    # # strand params
    # d_strand_sc: Parameter[float]
    # """Diameter of superconducting strand"""
    # d_strand: Parameter[float]
    # """Diameter of stabilising strand"""
    # operating_temperature: Parameter[float]
    # """Operating temperature for the strands [K]"""

    # # cable params
    # n_sc_strand: Parameter[int]
    # """Number of superconducting strands."""
    # n_stab_strand: Parameter[int]
    # """Number of stabilising strands."""
    # d_cooling_channel: Parameter[float]
    # """Diameter of the cooling channel [m]."""
    # void_fraction: Parameter[float]
    # """Ratio of material volume to total volume [unitless]."""
    # cos_theta: Parameter[float]
    # """Correction factor for twist in the cable layout."""
    # dx: Parameter[float]
    # """Cable half-width in the x-direction [m]."""

    # # conductor params
    # dx_jacket: Parameter[float]
    # """x-thickness of the jacket [m]."""
    # dy_jacket: Parameter[float]
    # """y-tickness of the jacket [m]."""
    # dx_ins: Parameter[float]
    # """x-thickness of the insulator [m]."""
    # dy_ins: Parameter[float]
    # """y-thickness of the insulator [m]."""

    # # winding pack params
    # nx: Parameter[int]
    # """Number of conductors along the x-axis."""
    # ny: Parameter[int]
    # """Number of conductors along the y-axis."""

    # # case params
    # Ri: Parameter[float]
    # """External radius of the TF coil case [m]."""
    # Rk: Parameter[float]
    # """Internal radius of the TF coil case [m]."""
    # theta_TF: Parameter[float]
    # """Toroidal angular span of the TF coil [degrees]."""
    # dy_ps: Parameter[float]
    # """Radial thickness of the poloidal support region [m]."""
    # dy_vault: Parameter[float]
    # """Radial thickness of the vault support region [m]."""

    Iop: Parameter[float]
    """Operational current in conductor"""
    T_sc: Parameter[float]
    """Operational temperature of superconducting cable"""
    T_margin: Parameter[float]
    """Temperature margin"""
    t_delay: Parameter[float]
    """Time delay for exponential functions"""
    strain: Parameter[float]
    """Strain on system"""

    # # optimisation params
    # t0: Parameter[float]
    # """Initial time"""
    # Tau_discharge: Parameter[float]
    # """Characteristic time constant"""
    # hotspot_target_temperature: Parameter[float]
    # """Target temperature for hotspot for cable optimisiation"""
    # layout: Parameter[str]
    # """Cable layout strategy"""
    # wp_reduction_factor: Parameter[float]
    # """Fractional reduction of available toroidal space for WPs"""
    # n_layers_reduction: Parameter[int]
    # """Number of layers to remove after each WP"""
    # bounds_cond_jacket: Parameter[np.ndarray]
    # """Min/max bounds for conductor jacket area optimisation [m²]"""
    # bounds_dy_vault: Parameter[np.ndarray]
    # """Min/max bounds for the case vault thickness optimisation [m]"""
    # max_niter: Parameter[int]
    # """Maximum number of optimisation iterations"""
    # eps: Parameter[float]
    # """Convergence threshold for the combined optimisation loop."""


@dataclass
class DerivedTFCoilXYDesignerParams:
    a: float
    Ri: float
    Re: float
    B_TF_i: float
    pm: float
    t_z: float
    T_op: float
    s_y: float
    n_cond: float
    min_gap_x: float
    I_fun: Callable[[float], float]
    B_fun: Callable[[float], float]
    strain: float


@dataclass
class TFCoilXY:
    case: CaseTF
    convergence: npt.NDArray
    derived_params: DerivedTFCoilXYDesignerParams
    op_config: dict[str, float]

    def plot_I_B(self, ax, n_steps=300):
        time_steps = np.linspace(
            self.op_config["t0"], self.op_config["Tau_discharge"], n_steps
        )
        I_values = [self.derived_params.I_fun(t) for t in time_steps]  # noqa: N806
        B_values = [self.derived_params.B_fun(t) for t in time_steps]

        ax.plot(time_steps, I_values, "g", label="Current [A]")
        ax.set_ylabel("Current [A]", color="g", fontsize=10)
        ax.tick_params(axis="y", labelcolor="g", labelsize=9)
        ax.grid(visible=True)

        ax_right = ax.twinx()
        ax_right.plot(time_steps, B_values, "m--", label="Magnetic field [T]")
        ax_right.set_ylabel("Magnetic field [T]", color="m", fontsize=10)
        ax_right.tick_params(axis="y", labelcolor="m", labelsize=9)

        # Labels
        ax.set_xlabel("Time [s]", fontsize=10)
        ax.tick_params(axis="x", labelsize=9)

        # Combined legend for both sides
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax_right.get_legend_handles_labels()
        ax.legend(lines + lines2, labels + labels2, loc="best", fontsize=9)

        ax.figure.tight_layout()

    def plot_cable_temperature_evolution(self, ax, n_steps=100):
        solution = self.case.solution

        ax.plot(solution.t, solution.y[0], "r*", label="Simulation points")
        time_steps = np.linspace(
            self.op_config["t0"], self.op_config["Tau_discharge"], n_steps
        )
        ax.plot(time_steps, solution.sol(time_steps)[0], "b", label="Interpolated curve")
        ax.grid(visible=True)
        ax.set_ylabel("Temperature [K]", fontsize=10)
        ax.set_title("Quench temperature evolution", fontsize=11)
        ax.legend(fontsize=9)

        ax.tick_params(axis="y", labelcolor="k", labelsize=9)

        props = {"boxstyle": "round", "facecolor": "white", "alpha": 0.8}
        ax.text(
            0.65,
            0.5,
            self.case.info_text,
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="top",
            bbox=props,
        )
        ax.figure.tight_layout()

    def plot_summary(self, n_steps, show=False):
        f, (ax_temp, ax_ib) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        self.plot_cable_temperature_evolution(ax_temp, n_steps)
        self.plot_I_B(ax_ib, n_steps * 3)
        return f

    def plot(
        self,
        ax: plt.Axes | None = None,
        *,
        show: bool = False,
        homogenised: bool = False,
    ) -> plt.Axes:
        return self.case.plot(ax=ax, show=show, homogenised=homogenised)

    def plot_convergence(self):
        """
        Plot the evolution of thicknesses and error values over optimisation iterations.

        Raises
        ------
        RuntimeError
            If no convergence data available
        """
        iterations = self.convergence[:, 0]
        dy_jacket = self.convergence[:, 1]
        dy_vault = self.convergence[:, 2]
        err_dy_jacket = self.convergence[:, 3]
        err_dy_vault = self.convergence[:, 4]
        dy_wp_tot = self.convergence[:, 5]
        Ri_minus_Rk = self.convergence[:, 6]  # noqa: N806

        _, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # Top subplot: Thicknesses
        axs[0].plot(iterations, dy_jacket, marker="o", label="dy_jacket [m]")
        axs[0].plot(iterations, dy_vault, marker="s", label="dy_vault [m]")
        axs[0].plot(iterations, dy_wp_tot, marker="^", label="dy_wp_tot [m]")
        axs[0].plot(iterations, Ri_minus_Rk, marker="v", label="Ri - Rk [m]")
        axs[0].set_ylabel("Thickness [m]")
        axs[0].set_title("Evolution of Jacket, Vault, and WP Thicknesses")
        axs[0].legend()
        axs[0].grid(visible=True)

        # Bottom subplot: Errors
        axs[1].plot(iterations, err_dy_jacket, marker="o", label="err_dy_jacket")
        axs[1].plot(iterations, err_dy_vault, marker="s", label="err_dy_vault")
        axs[1].set_ylabel("Relative Error")
        axs[1].set_xlabel("Iteration")
        axs[1].set_title("Evolution of Errors during Optimisation")
        axs[1].set_yscale("log")  # Log scale for better visibility if needed
        axs[1].legend()
        axs[1].grid(visible=True)

        plt.tight_layout()
        plt.show()


class TFCoilXYDesigner(Designer[TFCoilXY]):
    """
    Handles initialisation of TF Coil XY cross section from the individual parts:
        - Strands
        - Cable
        - Conductor
        - Winding Pack
        - Casing

    Will output a CaseTF object that allows for the access of all constituent parts
    and their properties.
    """

    param_cls: type[TFCoilXYDesignerParams] = TFCoilXYDesignerParams

    def __init__(
        self,
        params: ParameterFrameLike,
        build_config: dict,
    ):
        super().__init__(params=params, build_config=build_config)

    def _derived_values(self, op_config):
        # Needed params that are calculated using the base params
        R0 = self.params.R0.value
        n_TF = self.params.n_TF.value
        B0 = self.params.B0.value
        a = R0 / self.params.A.value
        Ri = R0 - a - self.params.d.value
        Re = (R0 + a) * (1 / self.params.ripple.value) ** (1 / n_TF)
        B_TF_i = 1.08 * (MU_0_2PI * n_TF * (B0 * R0 / MU_0_2PI / n_TF) / Ri)
        t_z = 0.5 * np.log(Re / Ri) * MU_0_4PI * n_TF * (B0 * R0 / MU_0_2PI / n_TF) ** 2
        return DerivedTFCoilXYDesignerParams(
            a=a,
            Ri=Ri,
            Re=Re,
            B_TF_i=B_TF_i,
            pm=B_TF_i**2 / (2 * MU_0),
            t_z=t_z,
            T_op=self.params.T_sc.value + self.params.T_margin.value,
            s_y=1e9 / self.params.safety_factor.value,
            n_cond=int(
                self.params.B0.value * R0 / MU_0_2PI / n_TF // self.params.Iop.value
            ),
            # 2 * thickness of the plate before the WP
            min_gap_x=2 * (R0 * 2 / 3 * 1e-2),
            I_fun=delayed_exp_func(
                self.params.Iop.value,
                op_config["Tau_discharge"],
                self.params.t_delay.value,
            ),
            B_fun=delayed_exp_func(
                B_TF_i, op_config["Tau_discharge"], self.params.t_delay.value
            ),
            strain=self.params.strain.value,
        )

    def B_TF_r(self, tf_current, r):
        """
        Compute the magnetic field generated by the TF coils,
        including ripple correction.

        Parameters
        ----------
        tf_current : float
            Toroidal field coil current [A].
        n_TF : int
            Number of toroidal field coils.
        r : float
            Radial position from the tokamak center [m].

        Returns
        -------
        float
            Magnetic field intensity [T].
        """
        return 1.08 * (MU_0_2PI * self.params.n_TF.value * tf_current / r)

    def run(self):
        """
        Run the TF coil XY design problem.

        Returns
        -------
        case:
            TF case object all parts that make it up.
        """
        wp_config = self.build_config.get("winding_pack")
        n_WPs = int(wp_config.get("sets"))

        optimisation_params = self.build_config.get("optimisation_params")
        derived_params = self._derived_values(optimisation_params)

        # param frame optimisation stuff?
        cable = self.optimise_cable_n_stab_ths(
            self._make_cable(n_WPs, WP_i=0),
            t0=optimisation_params["t0"],
            tf=optimisation_params["Tau_discharge"],
            initial_temperature=derived_params.T_op,
            target_temperature=optimisation_params["hotspot_target_temperature"],
            B_fun=derived_params.B_fun,
            I_fun=derived_params.I_fun,
            bounds=[1, 10000],
        )
        conductor = self._make_conductor(cable.cable, n_WPs, WP_i=0)
        wp_params = self._check_arrays_match(n_WPs, wp_config.pop("params"))
        winding_pack = [
            self._make_winding_pack(conductor, i_WP, wp_config, wp_params)
            for i_WP in range(n_WPs)
        ]

        # param frame optimisation stuff?
        case, convergence_array = self.optimise_jacket_and_vault(
            self._make_case(winding_pack, derived_params, optimisation_params),
            pm=derived_params.pm,
            fz=derived_params.t_z,
            op_cond=OperationalConditions(
                temperature=derived_params.T_op,
                magnetic_field=derived_params.B_TF_i,
                strain=derived_params.strain,
            ),
            allowable_sigma=derived_params.s_y,
            bounds_cond_jacket=optimisation_params["bounds_cond_jacket"],
            bounds_dy_vault=optimisation_params["bounds_dy_vault"],
            layout=optimisation_params["layout"],
            wp_reduction_factor=optimisation_params["wp_reduction_factor"],
            min_gap_x=derived_params.min_gap_x,
            n_layers_reduction=optimisation_params["n_layers_reduction"],
            max_niter=optimisation_params["max_niter"],
            eps=optimisation_params["eps"],
            n_conds=derived_params.n_cond,
        )
        return TFCoilXY(case, convergence_array, derived_params, optimisation_params)

    def optimise_cable_n_stab_ths(
        self,
        cable,
        t0: float,
        tf: float,
        initial_temperature: float,
        target_temperature: float,
        B_fun: Callable[[float], float],
        I_fun: Callable[[float], float],  # noqa: N803
        bounds: np.ndarray | None = None,
    ):
        """
        Optimise the number of stabiliser strand in the superconducting cable using a
        0-D hot spot criteria.

        Parameters
        ----------
        t0:
            Initial time [s].
        tf:
            Final time [s].
        initial_temperature:
            Temperature [K] at initial time.
        target_temperature:
            Target temperature [K] at final time.
        B_fun :
            Magnetic field [T] as a time-dependent function.
        I_fun :
            Current [A] as a time-dependent function.
        bounds:
            Lower and upper limits for the number of stabiliser strands.

        Returns
        -------
        :
            The result of the optimisation process.

        Raises
        ------
        ValueError
            If the optimisiation process does not converge.

        Notes
        -----
        - The number of stabiliser strands in the cable is modified directly.
        - Cooling material contribution is neglected when applying the hot spot criteria.
        """
        result = minimize_scalar(
            fun=cable.final_temperature_difference,
            args=(t0, tf, initial_temperature, target_temperature, B_fun, I_fun),
            bounds=bounds,
            method=None if bounds is None else "bounded",
        )

        if not result.success:
            raise ValueError(
                "n_stab optimisation did not converge. Check your input parameters "
                "or initial bracket."
            )

        # Here we re-ensure the n_stab_strand to be an integer
        cable.n_stab_strand = int(np.ceil(cable.n_stab_strand))

        solution = cable._temperature_evolution(
            t0, tf, initial_temperature, B_fun, I_fun
        )
        final_temperature = solution.y[0][-1]

        if final_temperature > target_temperature:
            bluemira_error(
                f"Final temperature ({final_temperature:.2f} K) exceeds target "
                f"temperature "
                f"({target_temperature} K) even with maximum n_stab = "
                f"{cable.n_stab_strand}."
            )
            raise ValueError(
                "Optimisation failed to keep final temperature ≤ target. "
                "Try increasing the upper bound of n_stab or adjusting cable parameters."
            )
        bluemira_print(f"Optimal n_stab: {cable.n_stab_strand}")
        bluemira_print(
            f"Final temperature with optimal n_stab: {final_temperature:.2f} Kelvin"
        )

        @dataclass
        class StabilisingStrandRes:
            cable: Any
            solution: Any
            info_text: str

        return StabilisingStrandRes(
            cable,
            solution,
            (
                f"Target T: {target_temperature:.2f} K\n"
                f"Initial T: {initial_temperature:.2f} K\n"
                f"SC Strand: {cable.sc_strand.name}\n"
                f"n. sc. strand = {cable.n_sc_strand}\n"
                f"Stab. strand = {cable.stab_strand.name}\n"
                f"n. stab. strand = {cable.n_stab_strand}\n"
            ),
        )

    def optimise_jacket_and_vault(
        self,
        case: CaseTF,
        pm: float,
        fz: float,
        op_cond: OperationalConditions,
        allowable_sigma: float,
        bounds_cond_jacket: np.ndarray | None = None,
        bounds_dy_vault: np.ndarray | None = None,
        layout: str = "auto",
        wp_reduction_factor: float = 0.8,
        min_gap_x: float = 0.05,
        n_layers_reduction: int = 4,
        max_niter: int = 10,
        eps: float = 1e-8,
        n_conds: int | None = None,
    ):
        """
        Jointly optimise the conductor jacket and case vault thickness
        under electromagnetic loading constraints.

        This method performs an iterative optimisation of:
        - The cross-sectional area of the conductor jacket.
        - The vault radial thickness of the TF coil casing.

        The optimisation loop continues until the relative change in
        jacket area and vault thickness drops below the specified
        convergence threshold `eps`, or `max_niter` is reached.

        Parameters
        ----------
        pm:
            Radial magnetic pressure on the conductor [Pa].
        fz:
            Axial electromagnetic force on the winding pack [N].
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material properties.
        allowable_sigma:
            Maximum allowable stress for structural material [Pa].
        bounds_cond_jacket:
            Min/max bounds for conductor jacket area optimisation [m²].
        bounds_dy_vault:
            Min/max bounds for the case vault thickness optimisation [m].
        layout:
            Cable layout strategy; "auto" or predefined layout name.
        wp_reduction_factor:
            Reduction factor applied to WP footprint during conductor rearrangement.
        min_gap_x:
            Minimum spacing between adjacent conductors [m].
        n_layers_reduction:
            Number of conductor layers to remove when reducing WP height.
        max_niter:
            Maximum number of optimisation iterations.
        eps:
            Convergence threshold for the combined optimisation loop.
        n_conds:
            Target total number of conductors in the winding pack. If None, the self
            number of conductors is used.

        Notes
        -----
        The function modifies the internal state of `conductor` and `self.dy_vault`.
        """
        debug_msg = ["Method optimise_jacket_and_vault"]

        # Initialise convergence array
        convergence_array = []

        if n_conds is None:
            n_conds = case.n_conductors

        conductor = case.WPs[0].conductor

        case._check_WPs(case.WPs)

        err_conductor_area_jacket = 10000 * eps
        err_dy_vault = 10000 * eps
        tot_err = err_dy_vault + err_conductor_area_jacket

        convergence_array.append([
            0,
            conductor.dy_jacket,
            case.dy_vault,
            err_conductor_area_jacket,
            err_dy_vault,
            case.dy_wp_tot,
            case.geometry.variables.Ri.value - case.geometry.variables.Rk.value,
        ])

        damping_factor = 0.3

        for i in range(1, max_niter):
            if tot_err <= eps:
                bluemira_print(
                    f"Optimisation of jacket and vault reached after "
                    f"{i - 1} iterations. Total error: {tot_err} < {eps}."
                )

                ax = case.plot(show=False, homogenised=False)
                ax.set_title("Case design after optimisation")
                plt.show()
                break
            debug_msg.append(f"Internal optimazion - iteration {i}")

            # Store current values
            cond_dx_jacket0 = conductor.dx_jacket
            case_dy_vault0 = case.dy_vault

            debug_msg.append(
                f"before optimisation: conductor jacket area = {conductor.area_jacket}"
            )
            cond_area_jacket0 = conductor.area_jacket
            t_z_cable_jacket = (
                fz
                * case.area_wps_jacket
                / (case.area_case_jacket + case.area_wps_jacket)
                / case.n_conductors
            )
            self.optimise_jacket_conductor(
                conductor,
                pm,
                t_z_cable_jacket,
                op_cond,
                allowable_sigma,
                bounds_cond_jacket,
            )
            debug_msg.extend([
                f"t_z_cable_jacket: {t_z_cable_jacket}",
                f"after optimisation: conductor jacket area = {conductor.area_jacket}",
            ])

            conductor.dx_jacket = (
                1 - damping_factor
            ) * cond_dx_jacket0 + damping_factor * conductor.dx_jacket

            err_conductor_area_jacket = (
                abs(conductor.area_jacket - cond_area_jacket0) / cond_area_jacket0
            )

            case.rearrange_conductors_in_wp(
                n_conds,
                wp_reduction_factor,
                min_gap_x,
                n_layers_reduction,
                layout=layout,
            )

            debug_msg.append(f"before optimisation: case dy_vault = {case.dy_vault}")
            result = self.optimise_vault_radial_thickness(
                case,
                pm=pm,
                fz=fz,
                op_cond=op_cond,
                allowable_sigma=allowable_sigma,
                bounds=bounds_dy_vault,
            )

            # case.dy_vault = result.x
            # print(f"Optimal dy_vault: {case.dy_vault}")
            # print(f"Tresca sigma: {case._tresca_stress(pm, fz, T=T, B=B) / 1e6} MPa")

            case.dy_vault = (
                1 - damping_factor
            ) * case_dy_vault0 + damping_factor * result.x

            delta_case_dy_vault = abs(case.dy_vault - case_dy_vault0)
            err_dy_vault = delta_case_dy_vault / case.dy_vault
            tot_err = err_dy_vault + err_conductor_area_jacket

            debug_msg.append(
                f"after optimisation: case dy_vault = {case.dy_vault}\n"
                f"err_dy_jacket = {err_conductor_area_jacket}\n "
                f"err_dy_vault = {err_dy_vault}\n "
                f"tot_err = {tot_err}"
            )

            # Store iteration results in convergence array
            convergence_array.append([
                i,
                conductor.dy_jacket,
                case.dy_vault,
                err_conductor_area_jacket,
                err_dy_vault,
                case.dy_wp_tot,
                case.geometry.variables.Ri.value - case.geometry.variables.Rk.value,
            ])

        else:
            bluemira_warn(
                f"Maximum number of optimisation iterations {max_niter} "
                f"reached. A total of {tot_err} > {eps} has been obtained."
            )

        return case, np.array(convergence_array)

    def optimise_jacket_conductor(
        self,
        conductor,
        pressure: float,
        f_z: float,
        op_cond: OperationalConditions,
        allowable_sigma: float,
        bounds: np.ndarray | None = None,
        direction: str = "x",
    ):
        """
        Optimise the jacket dimension of a conductor based on allowable stress using
        the Tresca criterion.

        Parameters
        ----------
        pressure:
            The pressure applied along the specified direction (Pa).
        f_z:
            The force applied in the z direction, perpendicular to the conductor
            cross-section (N).
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material properties.
        allowable_sigma:
            The allowable stress (Pa) for the jacket material.
        bounds:
            Optional bounds for the jacket thickness optimisation (default is None).
        direction:
            The direction along which the pressure is applied ('x' or 'y'). Default is
            'x'.

        Returns
        -------
        :
            The result of the optimisation process containing information about the
            optimal jacket thickness.

        Raises
        ------
        ValueError
            If the optimisation process did not converge.

        Notes
        -----
        This function uses the Tresca yield criterion to optimise the thickness of the
        jacket surrounding the conductor.
        This function directly update the conductor's jacket thickness along the x
        direction to the optimal value.
        """
        debug_msg = ["Method optimise_jacket_conductor:"]

        if direction == "x":
            debug_msg.append(f"Previous dx_jacket: {conductor.dx_jacket}")
        else:
            debug_msg.append(f"Previous dy_jacket: {conductor.dy_jacket}")

        method = "bounded" if bounds is not None else None

        if method == "bounded":
            debug_msg.append(f"bounds: {bounds}")

        result = minimize_scalar(
            fun=conductor.sigma_difference,
            args=(pressure, f_z, op_cond, allowable_sigma),
            bounds=bounds,
            method=method,
            options={"xatol": 1e-4},
        )

        if not result.success:
            raise ValueError("Optimisation of the jacket conductor did not converge.")
        if direction == "x":
            conductor.dx_jacket = result.x
            debug_msg.append(f"Optimal dx_jacket: {conductor.dx_jacket}")
        else:
            conductor.dy_jacket = result.x
            debug_msg.append(f"Optimal dy_jacket: {conductor.dy_jacket}")
        debug_msg.append(
            f"Averaged sigma in the {direction}-direction: "
            f"{conductor._tresca_sigma_jacket(pressure, f_z, op_cond) / 1e6} MPa\n"
            f"Allowable stress in the {direction}-direction: {allowable_sigma / 1e6} "
            f"MPa."
        )
        bluemira_debug("\n".join(debug_msg))

        return result

    def optimise_vault_radial_thickness(
        self,
        case,
        pm: float,
        fz: float,
        op_cond: OperationalConditions,
        allowable_sigma: float,
        bounds: np.array = None,
    ):
        """
        Optimise the vault radial thickness of the case

        Parameters
        ----------
        pm:
            The magnetic pressure applied along the radial direction (Pa).
        f_z:
            The force applied in the z direction, perpendicular to the case
            cross-section (N).
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material properties.
        allowable_sigma:
            The allowable stress (Pa) for the jacket material.
        bounds:
            Optional bounds for the jacket thickness optimisation (default is None).

        Returns
        -------
        :
            The result of the optimisation process containing information about the
            optimal vault thickness.

        Raises
        ------
        ValueError
            If the optimisation process did not converge.
        """
        method = None
        if bounds is not None:
            method = "bounded"

        result = minimize_scalar(
            fun=case._sigma_difference,
            args=(pm, fz, op_cond, allowable_sigma),
            bounds=bounds,
            method=method,
            options={"xatol": 1e-4},
        )

        if not result.success:
            raise ValueError("dy_vault optimisation did not converge.")

        return result

    def _check_arrays_match(self, n_WPs, param_list):
        if n_WPs > 1:
            for param in param_list:
                if np.size(param_list[param]) != n_WPs:
                    param_list[param] = [param_list[param] for _ in range(n_WPs)]
            return param_list
        if n_WPs == 1:
            return param_list
        raise ValueError(
            f"Invalid value {n_WPs} for winding pack 'sets' in config."
            "Value should be an integer >= 1."
        )

    def _make_strand(self, i_WP, config, params):
        cls_name = config["class"]
        stab_strand_cls = get_class_from_module(
            cls_name, default_module="bluemira.magnets.strand"
        )
        material_mix = []
        for m in config.get("materials"):
            material_data = m["material"]
            if isinstance(material_data, str):
                raise TypeError(
                    "Material data must be a Material instance, not a string - "
                    "TEMPORARY."
                )
            material_obj = material_data

            material_mix.append(
                MaterialFraction(material=material_obj, fraction=m["fraction"])
            )
        return stab_strand_cls(
            materials=material_mix,
            d_strand=params["d_strand"][i_WP],
            operating_temperature=params["operating_temperature"][i_WP],
            name="stab_strand",
        )

    def _make_cable_cls(self, stab_strand, sc_strand, i_WP, config, params):
        cls_name = config["class"]
        cable_cls = get_class_from_module(
            cls_name, default_module="bluemira.magnets.cable"
        )
        return cable_cls(
            sc_strand=sc_strand,
            stab_strand=stab_strand,
            n_sc_strand=params["n_sc_strand"][i_WP],
            n_stab_strand=params["n_stab_strand"][i_WP],
            d_cooling_channel=params["d_cooling_channel"][i_WP],
            void_fraction=params["void_fraction"][i_WP],
            cos_theta=params["cos_theta"][i_WP],
            name=config.get("name", cls_name.rsplit("::", 1)[-1]),
            **(
                {"dx": params["dx"][i_WP], "E": params["E"][i_WP]}
                if issubclass(cable_cls, RectangularCable)
                else {"E": params["E"][i_WP]}
            ),
        )

    def _make_cable(self, n_WPs, WP_i):
        stab_strand_config = self.build_config.get("stabilising_strand")
        sc_strand_config = self.build_config.get("superconducting_strand")
        cable_config = self.build_config.get("cable")

        stab_strand_params = self._check_arrays_match(
            n_WPs, stab_strand_config.get("params")
        )
        sc_strand_params = self._check_arrays_match(
            n_WPs, sc_strand_config.get("params")
        )

        cable_params = self._check_arrays_match(n_WPs, cable_config.get("params"))

        stab_strand = self._make_strand(WP_i, stab_strand_config, stab_strand_params)
        sc_strand = self._make_strand(WP_i, sc_strand_config, sc_strand_params)
        return self._make_cable_cls(
            stab_strand, sc_strand, WP_i, cable_config, cable_params
        )

    def _make_conductor_cls(self, cable, i_WP, config, params):
        cls_name = config["class"]
        conductor_cls = get_class_from_module(
            cls_name, default_module="bluemira.magnets.conductor"
        )
        return conductor_cls(
            cable=cable,
            mat_jacket=config["jacket_material"],
            mat_ins=config["ins_material"],
            dx_jacket=params["dx_jacket"][i_WP],
            dx_ins=params["dx_ins"][i_WP],
            name=config.get("name", cls_name.rsplit("::", 1)[-1]),
            **(
                {}
                if issubclass(conductor_cls, SymmetricConductor)
                else {
                    "dy_jacket": params["dy_jacket"][i_WP],
                    "dy_ins": params["dy_ins"][i_WP],
                }
            ),
        )

    def _make_conductor(self, cable, n_WPs, WP_i=0):
        # current functionality requires conductors are the same for both WPs
        # in future allow for different conductor objects so can vary cable and strands
        # between the sets of the winding pack?
        conductor_config = self.build_config.get("conductor")
        conductor_params = self._check_arrays_match(
            n_WPs, conductor_config.get("params")
        )

        return self._make_conductor_cls(cable, WP_i, conductor_config, conductor_params)

    def _make_winding_pack(self, conductor, i_WP, config, params):
        cls_name = config["class"]
        winding_pack_cls = get_class_from_module(
            cls_name, default_module="bluemira.magnets.winding_pack"
        )
        return winding_pack_cls(
            conductor=conductor,
            nx=int(params["nx"][i_WP]),
            ny=int(params["ny"][i_WP]),
            name="winding_pack",
        )

    def _make_case(self, WPs, derived_params, optimisation_params):  # noqa: N803
        config = self.build_config.get("case")
        params = config.get("params")

        cls_name = config["class"]
        case_cls = get_class_from_module(
            cls_name, default_module="bluemira.magnets.case_tf"
        )

        case = case_cls(
            Ri=params["Ri"],
            theta_TF=params["theta_TF"],
            dy_ps=params["dy_ps"],
            dy_vault=params["dy_vault"],
            mat_case=config["material"],
            WPs=WPs,
            name=config.get("name", cls_name.rsplit("::", 1)[-1]),
        )

        # param frame optimisation stuff?
        case.rearrange_conductors_in_wp(
            n_conductors=derived_params.n_cond,
            wp_reduction_factor=optimisation_params["wp_reduction_factor"],
            min_gap_x=derived_params.min_gap_x,
            n_layers_reduction=optimisation_params["n_layers_reduction"],
            layout=optimisation_params["layout"],
        )
        return case
