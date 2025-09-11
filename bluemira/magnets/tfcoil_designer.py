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
    bluemira_print,
    bluemira_warn,
    bluemira_debug,
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
    # """Number of stabilizing strands."""
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

    def _make_conductor(self, optimisation_params, derived_params, n_WPs, WP_i=0):
        # current functionality requires conductors are the same for both WPs
        # in future allow for different conductor objects so can vary cable and strands
        # between the sets of the winding pack?
        stab_strand_config = self.build_config.get("stabilising_strand")
        sc_strand_config = self.build_config.get("superconducting_strand")
        cable_config = self.build_config.get("cable")
        conductor_config = self.build_config.get("conductor")

        stab_strand_params = self._check_arrays_match(
            n_WPs, stab_strand_config.get("params")
        )
        sc_strand_params = self._check_arrays_match(
            n_WPs, sc_strand_config.get("params")
        )
        conductor_params = self._check_arrays_match(
            n_WPs, conductor_config.get("params")
        )

        cable_params = self._check_arrays_match(n_WPs, cable_config.get("params"))

        stab_strand = self._make_strand(WP_i, stab_strand_config, stab_strand_params)
        sc_strand = self._make_strand(WP_i, sc_strand_config, sc_strand_params)
        cable = self._make_cable(
            stab_strand, sc_strand, WP_i, cable_config, cable_params
        )
        # param frame optimisation stuff?
        result = cable.optimise_n_stab_ths(
            t0=optimisation_params["t0"],
            tf=optimisation_params["Tau_discharge"],
            initial_temperature=derived_params.T_op,
            target_temperature=optimisation_params["hotspot_target_temperature"],
            B_fun=derived_params.B_fun,
            I_fun=derived_params.I_fun,
            bounds=[1, 10000],
        )

        return self._make_conductor_cls(cable, WP_i, conductor_config, conductor_params)

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

        conductor, conductor_result = self._make_conductor(
            optimisation_params, derived_params, n_WPs, WP_i=0
        )
        wp_params = self._check_arrays_match(n_WPs, wp_config.pop("params"))
        winding_pack = [
            self._make_winding_pack(conductor, i_WP, wp_config, wp_params)
            for i_WP in range(n_WPs)
        ]

        case = self._make_case(winding_pack, derived_params, optimisation_params)

        # param frame optimisation stuff?
        case.optimise_jacket_and_vault(
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
        return TFCoilXY(case, derived_params, optimisation_params)

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

    def _make_cable(self, WP_i, n_WPs):
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
