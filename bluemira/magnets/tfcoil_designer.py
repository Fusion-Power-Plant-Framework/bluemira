# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Designer for TF Coil XY cross section."""

from dataclasses import dataclass

import numpy as np

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
    Iop: Parameter[float]
    """Operational current in conductor"""
    T_sc: Parameter[float]
    """Operational temperature of superconducting cable"""
    T_margin: Parameter[float]
    """Temperature margin"""
    t_delay: Parameter[float]
    """Time delay for exponential functions"""
    t0: Parameter[float]
    """Initial time"""
    hotspot_target_temperature: Parameter[float]
    """Target temperature for hotspot for cable optimisiation"""
    S_VV: Parameter[float]
    """Vacuum vessel steel limit"""
    d_strand_sc: Parameter[float]
    """Diameter of superconducting strand"""
    d_strand_stab: Parameter[float]
    """Diameter of stabilising strand"""
    safety_factor: Parameter[float]
    """Allowable stress values"""
    dx: Parameter[float]
    """Cable length"""
    B_ref: Parameter[float]
    """Reference value for B field (LTS limit) [T]"""
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
    Tau_discharge: Parameter[float]
    """Characteristic time constant"""


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
        # sort configs
        stab_strand_config = self.build_config.get("stabilising_strand")
        sc_strand_config = self.build_config.get("superconducting_strand")
        conductor_config = self.build_config.get("conductor")
        case_config = self.build_config.get("case")
        # sort params (break down into smaller parts rather than pass in all params?)
        stab_strand_params = self.params.get("stab_strand")
        sc_strand_params = self.params.get("sc_strand")
        cable_params = self.params.get("cable")
        conductor_params = self.params.get("conductor")
        winding_pack_params = self.params.get("winding_pack")
        case_params = self.params.get("case")
        optimisation_params = self.params.get("optimisation")
        derived_params = self._derived_values()

        stab_strand = self._make_stab_strand(stab_strand_config, stab_strand_params)
        sc_strand = self._make_sc_strand(sc_strand_config, sc_strand_params)
        initial_cable = self._make_cable(cable_params, stab_strand, sc_strand)
        # param frame optimisation stuff?
        optimised_cable = initial_cable.optimise_n_stab_ths(
            t0=optimisation_params.t0.value,
            tf=optimisation_params.Tau_discharge.value,
            T_for_hts=derived_params["T_op"],
            hotspot_target_temperature=optimisation_params.hotspot_target_temperature.value,
            B_fun=derived_params["B_fun"],
            I_fun=derived_params["I_fun"],
            bounds=[1, 10000],
        )
        conductor = self._make_conductor(
            conductor_config, conductor_params, optimised_cable
        )
        winding_pack = self._make_winding_pack(winding_pack_params, conductor)
        case = self._make_case(case_config, case_params, [winding_pack])
        # param frame optimisation stuff?
        case.rearrange_conductors_in_wp(
            n_conductors=derived_params["n_cond"],
            wp_reduction_factor=optimisation_params.wp_reduction_factor.value,
            min_gap_x=derived_params["min_gap_x"],
            n_layers_reduction=optimisation_params.n_layers_reduction.value,
            layout=optimisation_params.layout.value,
        )
        # param frame optimisation stuff?
        case.optimize_jacket_and_vault(
            pm=derived_params["pm"],
            fz=derived_params["t_z"],
            temperature=derived_params["T_op"],
            B=derived_params["B_TF_i"],
            allowable_sigma=derived_params["s_y"],
            bounds_cond_jacket=optimisation_params.bounds_cond_jacket.value,
            bounds_dy_vault=optimisation_params.bounds_dy_vault.value,
            layout=optimisation_params.layout.value,
            wp_reduction_factor=optimisation_params.wp_reduction_factor.value,
            min_gap_x=derived_params["min_gap_x"],
            n_layers_reduction=optimisation_params.n_layers_reduction.value,
            max_niter=optimisation_params.max_niter.value,
            eps=optimisation_params.eps.value,
            n_conds=derived_params["n_cond"],
        )
        return case

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

    def _make_stab_strand(self):
        stab_strand_config = self.build_config.get("stabilising_strand")
        stab_strand_params = self.params.get("stab_strand")
        return Strand(
            materials=stab_strand_config.get("material"),
            params=stab_strand_params,
            name="stab_strand",
        )

    def _make_sc_strand(self):
        sc_strand_config = self.build_config.get("superconducting_strand")
        sc_strand_params = self.params.get("sc_strand")
        return SuperconductingStrand(
            materials=sc_strand_config.get("material"),
            params=sc_strand_params,
            name="sc_strand",
        )

    def _make_cable(self, stab_strand, sc_strand):
        cable_params = self.params.get("cable")
        if cable_params.cable_type == "Rectangular":
            cable = RectangularCable(
                sc_strand=sc_strand,
                stab_strand=stab_strand,
                params=cable_params,
                name="RectangularCable",
            )
        elif cable_params.cable_type == "Square":
            cable = SquareCable(
                sc_strand=sc_strand,
                stab_strand=stab_strand,
                params=cable_params,
                name="SquareCable",
            )
        elif cable_params.cable_type == "Round":
            cable = RoundCable(
                sc_strand=sc_strand,
                stab_strand=stab_strand,
                params=cable_params,
                name="RoundCable",
            )
        else:
            raise ValueError(
                f"Cable type {cable_params.cable_type} is not known."
                "Available options are 'Rectangular', 'Square' and 'Round'."
            )
        return cable

    def _make_conductor(self, cable):
        conductor_config = self.build_config.get("conductor")
        conductor_params = self.params.get("conductor")
        if conductor_params.conductor_type == "Conductor":
            conductor = Conductor(
                cable=cable,
                mat_jacket=conductor_config.get("mat_jacket"),
                mat_ins=conductor_config.get("mat_ins"),
                params=conductor_params,
                name="Conductor",
            )
        elif conductor_params.conductor_type == "SymmetricConductor":
            conductor = SymmetricConductor(
                cable=cable,
                mat_jacket=conductor_config.get("mat_jacket"),
                mat_ins=conductor_config.get("mat_ins"),
                params=conductor_params,
                name="SymmetricConductor",
            )
        else:
            raise ValueError(
                f"Conductor type {conductor_params.conductor_type} is not known."
                "Available options are 'Conductor' and 'SymmetricConductor'."
            )
        return conductor

    def _make_winding_pack(self, conductor):
        winding_pack_params = self.params.get("winding_pack")
        return WindingPack(
            conductor=conductor, params=winding_pack_params, name="winding_pack"
        )

    def _make_case(self, WPs):  # noqa: N803
        case_config = self.build_config.get("case")
        case_params = self.params.get("case")
        if case_params.case_type == "Trapezoidal":
            case = TrapezoidalCaseTF(
                params=case_params,
                mat_case=case_config.get("material"),
                WPs=WPs,
                name="TrapezoidalCase",
            )
        else:
            raise ValueError(
                f"Case type {case_params.case_type} is not known."
                "Available options are 'Trapezoidal'."
            )
        return case
