# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Designer for TF Coil XY cross section."""

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matproplib import OperationalConditions

from bluemira.base.constants import MU_0, MU_0_2PI, MU_0_4PI
from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.magnets.cable import RectangularCable, RoundCable, SquareCable
from bluemira.magnets.case_tf import TrapezoidalCaseTF
from bluemira.magnets.conductor import Conductor, SymmetricConductor
from bluemira.magnets.strand import Strand, SuperconductingStrand
from bluemira.magnets.utils import delayed_exp_func
from bluemira.magnets.winding_pack import WindingPack


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

    # strand params
    d_strand_sc: Parameter[float]
    """Diameter of superconducting strand"""
    d_strand: Parameter[float]
    """Diameter of stabilising strand"""
    operating_temperature: Parameter[float]
    """Operating temperature for the strands [K]"""

    # cable params
    n_sc_strand: Parameter[int]
    """Number of superconducting strands."""
    n_stab_strand: Parameter[int]
    """Number of stabilizing strands."""
    d_cooling_channel: Parameter[float]
    """Diameter of the cooling channel [m]."""
    void_fraction: Parameter[float]
    """Ratio of material volume to total volume [unitless]."""
    cos_theta: Parameter[float]
    """Correction factor for twist in the cable layout."""
    dx: Parameter[float]
    """Cable half-width in the x-direction [m]."""

    # conductor params
    dx_jacket: Parameter[float]
    """x-thickness of the jacket [m]."""
    dy_jacket: Parameter[float]
    """y-tickness of the jacket [m]."""
    dx_ins: Parameter[float]
    """x-thickness of the insulator [m]."""
    dy_ins: Parameter[float]
    """y-thickness of the insulator [m]."""

    # winding pack params
    nx: Parameter[int]
    """Number of conductors along the x-axis."""
    ny: Parameter[int]
    """Number of conductors along the y-axis."""

    # case params
    Ri: Parameter[float]
    """External radius of the TF coil case [m]."""
    Rk: Parameter[float]
    """Internal radius of the TF coil case [m]."""
    theta_TF: Parameter[float]
    """Toroidal angular span of the TF coil [degrees]."""
    dy_ps: Parameter[float]
    """Radial thickness of the poloidal support region [m]."""
    dy_vault: Parameter[float]
    """Radial thickness of the vault support region [m]."""

    Iop: Parameter[float]
    """Operational current in conductor"""
    T_sc: Parameter[float]
    """Operational temperature of superconducting cable"""
    T_margin: Parameter[float]
    """Temperature margin"""
    t_delay: Parameter[float]
    """Time delay for exponential functions"""

    # optimisation params
    t0: Parameter[float]
    """Initial time"""
    Tau_discharge: Parameter[float]
    """Characteristic time constant"""
    hotspot_target_temperature: Parameter[float]
    """Target temperature for hotspot for cable optimisiation"""
    layout: Parameter[str]
    """Cable layout strategy"""
    wp_reduction_factor: Parameter[float]
    """Fractional reduction of available toroidal space for WPs"""
    n_layers_reduction: Parameter[int]
    """Number of layers to remove after each WP"""
    bounds_cond_jacket: Parameter[np.ndarray]
    """Min/max bounds for conductor jacket area optimization [m²]"""
    bounds_dy_vault: Parameter[np.ndarray]
    """Min/max bounds for the case vault thickness optimization [m]"""
    max_niter: Parameter[int]
    """Maximum number of optimization iterations"""
    eps: Parameter[float]
    """Convergence threshold for the combined optimization loop."""


class TFCoilXYDesigner(Designer):
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

    def __init__(
        self,
        params: dict | ParameterFrame,
        build_config: dict,
    ):
        super().__init__(params, build_config)

    def _derived_values(self):
        # Needed params that are calculated using the base params
        a = self.params.R0.value / self.params.A.value
        Ri = self.params.R0.value - a - self.params.d.value  # noqa: N806
        Re = (self.params.R0.value + a) * (1 / self.params.ripple.value) ** (  # noqa: N806
            1 / self.params.n_TF.value
        )
        B_TF_i = 1.08 * (
            MU_0_2PI
            * self.params.n_TF.value
            * (
                self.params.B0.value
                * self.params.R0.value
                / MU_0_2PI
                / self.params.n_TF.value
            )
            / Ri
        )
        pm = B_TF_i**2 / (2 * MU_0)
        t_z = (
            0.5
            * np.log(Re / Ri)
            * MU_0_4PI
            * self.params.n_TF.value
            * (
                self.params.B0.value
                * self.params.R0.value
                / MU_0_2PI
                / self.params.n_TF.value
            )
            ** 2
        )
        T_op = self.params.T_sc.value + self.params.T_margin.value  # noqa: N806
        self.params.operating_temperature.value = (
            T_op  # this necessary? Or just remove T_sc and T_margin
        )
        s_y = 1e9 / self.params.safety_factor.value
        n_cond = int(
            np.floor(
                (
                    self.params.B0.value
                    * self.params.R0.value
                    / MU_0_2PI
                    / self.params.n_TF.value
                )
                / self.params.Iop.value
            )
        )
        min_gap_x = int(
            np.floor(
                (
                    self.params.B0.value
                    * self.params.R0.value
                    / MU_0_2PI
                    / self.params.n_TF.value
                )
                / self.params.Iop.value
            )
        )
        I_fun = delayed_exp_func(  # noqa: N806
            self.params.Iop.value,
            self.params.Tau_discharge.value,
            self.params.t_delay.value,
        )
        B_fun = delayed_exp_func(
            B_TF_i, self.params.Tau_discharge.value, self.params.t_delay.value
        )
        return {
            "a": a,
            "Ri": Ri,
            "Re": Re,
            "B_TF_I": B_TF_i,
            "pm": pm,
            "t_z": t_z,
            "T_op": T_op,
            "s_y": s_y,
            "n_cond": n_cond,
            "min_gap_x": min_gap_x,
            "I_fun": I_fun,
            "B_fun": B_fun,
        }

    def run(self):
        """
        Run the TF coil XY design problem.

        Returns
        -------
        case:
            TF case object all parts that make it up.
        """
        # params that are function of another param
        derived_params = self._derived_values()

        n_WPs = len(self.params.nx.value)
        if n_WPs > 1:
            self._check_arrays_match()
        winding_pack = []
        for i_WP in n_WPs:
            stab_strand = self._make_stab_strand(i_WP)
            sc_strand = self._make_sc_strand(i_WP)
            initial_cable = self._make_cable(stab_strand, sc_strand, i_WP)
            # param frame optimisation stuff?
            optimised_cable = initial_cable.optimise_n_stab_ths(
                t0=self.params.t0.value,
                tf=self.params.Tau_discharge.value,
                T_for_hts=derived_params["T_op"],
                hotspot_target_temperature=self.params.hotspot_target_temperature.value,
                B_fun=derived_params["B_fun"],
                I_fun=derived_params["I_fun"],
                bounds=[1, 10000],
            )
            conductor = self._make_conductor(optimised_cable, i_WP)
            winding_pack += [self._make_winding_pack(conductor, i_WP)]
        case = self._make_case(winding_pack)
        # param frame optimisation stuff?
        case.rearrange_conductors_in_wp(
            n_conductors=derived_params["n_cond"],
            wp_reduction_factor=self.params.wp_reduction_factor.value,
            min_gap_x=derived_params["min_gap_x"],
            n_layers_reduction=self.params.n_layers_reduction.value,
            layout=self.params.layout.value,
        )
        # param frame optimisation stuff?
        case.optimize_jacket_and_vault(
            pm=derived_params["pm"],
            fz=derived_params["t_z"],
            op_cond=OperationalConditions(
                temperature=derived_params["T_op"],
                magnetic_field=derived_params["B_TF_i"],
            ),
            allowable_sigma=derived_params["s_y"],
            bounds_cond_jacket=self.params.bounds_cond_jacket.value,
            bounds_dy_vault=self.params.bounds_dy_vault.value,
            layout=self.params.layout.value,
            wp_reduction_factor=self.params.wp_reduction_factor.value,
            min_gap_x=derived_params["min_gap_x"],
            n_layers_reduction=self.params.n_layers_reduction.value,
            max_niter=self.params.max_niter.value,
            eps=self.params.eps.value,
            n_conds=derived_params["n_cond"],
        )
        return case

    def _check_arrays_match(self):
        n = len(self.params.nx.value)
        param_list = [
            "d_strand_sc",
            "d_strand",
            "operating_temperature",
            "n_sc_strand",
            "n_stab_strand",
            "d_cooling_channel",
            "void_fraction",
            "cos_theta",
            "dx",
            "dx_jacket",
            "dy_jacket",
            "dx_ins",
            "dy_ins",
            "ny",
        ]
        for param in param_list:
            if len(self.params.get(param).value) != n:
                self.params.get(param).value = [
                    self.params.get(param).value for _ in range(n)
                ]

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

    def _make_stab_strand(self, i_WP):
        stab_strand_config = self.build_config.get("stabilising_strand")
        return Strand(
            materials=stab_strand_config.get("materials"),
            d_strand=self.params.d_strand.value[i_WP],
            operating_temperature=self.params.operating_temperature.value[i_WP],
            name="stab_strand",
        )

    def _make_sc_strand(self, i_WP):
        sc_strand_config = self.build_config.get("superconducting_strand")
        return SuperconductingStrand(
            materials=sc_strand_config.get("materials"),
            d_strand=self.params.d_strand_sc.value[i_WP],
            operating_temperature=self.params.operating_temperature.value[i_WP],
            name="sc_strand",
        )

    def _make_cable(self, stab_strand, sc_strand, i_WP):
        cable_config = self.build_config.get("cable")
        if cable_config.get("type") == "Rectangular":
            cable = RectangularCable(
                sc_strand=sc_strand,
                stab_strand=stab_strand,
                n_sc_strand=self.params.n_sc_strand.value[i_WP],
                n_stab_strand=self.params.n_stab_strand.value[i_WP],
                d_cooling_channel=self.params.d_cooling_channel.value[i_WP],
                void_fraction=self.params.void_fraction.value[i_WP],
                cos_theta=self.params.cos_theta.value[i_WP],
                dx=self.params.dx.value[i_WP],
                name="RectangularCable",
            )
        elif cable_config.get("type") == "Square":
            cable = SquareCable(
                sc_strand=sc_strand,
                stab_strand=stab_strand,
                n_sc_strand=self.params.n_sc_strand.value[i_WP],
                n_stab_strand=self.params.n_stab_strand.value[i_WP],
                d_cooling_channel=self.params.d_cooling_channel.value[i_WP],
                void_fraction=self.params.void_fraction.value[i_WP],
                cos_theta=self.params.cos_theta.value[i_WP],
                name="SquareCable",
            )
        elif cable_config.get("type") == "Round":
            cable = RoundCable(
                sc_strand=sc_strand,
                stab_strand=stab_strand,
                n_sc_strand=self.params.n_sc_strand.value[i_WP],
                n_stab_strand=self.params.n_stab_strand.value[i_WP],
                d_cooling_channel=self.params.d_cooling_channel.value[i_WP],
                void_fraction=self.params.void_fraction.value[i_WP],
                cos_theta=self.params.cos_theta.value[i_WP],
                name="RoundCable",
            )
        else:
            raise ValueError(
                f"Cable type {cable_config.get('type')} is not known."
                "Available options are 'Rectangular', 'Square' and 'Round'."
            )
        return cable

    def _make_conductor(self, cable, i_WP):
        conductor_config = self.build_config.get("conductor")
        if conductor_config.get("type") == "Conductor":
            conductor = Conductor(
                cable=cable,
                mat_jacket=conductor_config.get("jacket_material"),
                mat_ins=conductor_config.get("ins_material"),
                dx_jacket=self.params.dx_jacket.value[i_WP],
                dy_jacket=self.params.dy_jacket.value[i_WP],
                dx_ins=self.params.dx_ins.value[i_WP],
                dy_ins=self.params.dy_ins.value[i_WP],
                name="Conductor",
            )
        elif conductor_config.get("type") == "SymmetricConductor":
            conductor = SymmetricConductor(
                cable=cable,
                mat_jacket=conductor_config.get("jacket_material"),
                mat_ins=conductor_config.get("ins_material"),
                dx_jacket=self.params.dx_jacket.value[i_WP],
                dx_ins=self.params.dx_ins.value[i_WP],
                name="SymmetricConductor",
            )
        else:
            raise ValueError(
                f"Conductor type {conductor_config.get('type')} is not known."
                "Available options are 'Conductor' and 'SymmetricConductor'."
            )
        return conductor

    def _make_winding_pack(self, conductor, i_WP):
        return WindingPack(
            conductor=conductor,
            nx=self.params.nx.value[i_WP],
            ny=self.params.ny.value[i_WP],
            name="winding_pack",
        )

    def _make_case(self, WPs):  # noqa: N803
        case_config = self.build_config.get("case")
        if case_config.get("type") == "Trapezoidal":
            case = TrapezoidalCaseTF(
                Ri=self.params.Ri.value,
                theta_TF=self.params.theta_TF.value,
                dy_ps=self.params.dy_ps.value,
                dy_vault=self.params.dy_vault.value,
                mat_case=case_config.get("material"),
                WPs=WPs,
                name="TrapezoidalCase",
            )
        else:
            raise ValueError(
                f"Case type {case_config.get('type')} is not known."
                "Available options are 'Trapezoidal'."
            )
        return case


def plot_cable_temperature_evolution(result, t0, tf, ax, n_steps=100):
    solution = result.solution

    ax.plot(solution.t, solution.y[0], "r*", label="Simulation points")
    time_steps = np.linspace(t0, tf, n_steps)
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
        result.info_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=props,
    )
    ax.figure.tight_layout()


def plot_I_B(I_fun, B_fun, t0, tf, ax, n_steps=300):
    time_steps = np.linspace(t0, tf, n_steps)
    I_values = [I_fun(t) for t in time_steps]  # noqa: N806
    B_values = [B_fun(t) for t in time_steps]

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


def plot_summary(result, t0, tf, I_fun, B_fun, n_steps, show=False):
    f, (ax_temp, ax_ib) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    plot_cable_temperature_evolution(result, t0, tf, ax_temp, n_steps)
    plot_I_B(I_fun, B_fun, t0, tf, ax_ib, n_steps * 3)
    return f
