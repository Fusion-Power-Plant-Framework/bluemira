# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
PROCESS's parameter definitions.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

from bluemira.base.parameter_frame import Parameter  # noqa: TC001
from bluemira.codes.params import MappedParameterFrame, ParameterMapping
from bluemira.codes.process._inputs import ProcessInputs
from bluemira.codes.process.constants import NAME
from bluemira.codes.process.mapping import mappings

if TYPE_CHECKING:
    from bluemira.codes.process.api import _INVariable


@dataclass
class ProcessSolverParams(MappedParameterFrame):
    """Parameters required in :class:`bluemira.codes.process._solver.Solver`."""

    # In parameters
    C_Ejima: Parameter[float]
    """Ejima constant [dimensionless]."""

    e_mult: Parameter[float]
    """
    Energy multiplication factor [dimensionless]. Instantaneous energy
    multiplication due to neutron multiplication and the like.
    """

    e_nbi: Parameter[float]
    """Neutral beam energy [kiloelectron_volt]."""

    eta_nb: Parameter[float]
    """NB electrical efficiency [dimensionless]. Check units!."""

    n_TF: Parameter[int]
    """Number of TF coils [dimensionless]."""

    P_el_net: Parameter[float]
    """Net electrical power output [megawatt]."""

    tau_flattop: Parameter[float]
    """Flat-top duration [second]."""

    P_hcd_ss: Parameter[float]
    """Steady-state HCD power [megawatt]."""

    TF_ripple_limit: Parameter[float]
    """TF coil ripple limit [percent]."""

    tk_cr_vv: Parameter[float]
    """Cryostat VV thickness [meter]."""

    tk_sh_bot: Parameter[float]
    """Lower shield thickness [meter]. DO NOT USE - PROCESS has VV = VV + shield."""

    tk_sh_out: Parameter[float]
    """Outboard shield thickness [meter]. DO NOT USE - PROCESS has VV = VV + shield."""

    tk_sh_top: Parameter[float]
    """Upper shield thickness [meter]. DO NOT USE - PROCESS has VV = VV + shield."""

    tk_tf_front_ib: Parameter[float]
    """TF coil inboard steel front plasma-facing [meter]."""

    tk_tf_side: Parameter[float]
    """TF coil inboard case minimum side wall thickness [meter]."""

    tk_vv_bot: Parameter[float]
    """Lower vacuum vessel thickness [meter]."""

    tk_vv_out: Parameter[float]
    """Outboard vacuum vessel thickness [meter]."""

    tk_vv_top: Parameter[float]
    """Upper vacuum vessel thickness [meter]."""

    PsepB_qAR_max: Parameter[float]
    """Maximum PsepB/q95AR vale [MW.T/m]"""

    q_0: Parameter[float]
    """Plasma safety factor on axis [dimensionless]"""

    q_95: Parameter[float]
    """Plasma safety factor at the 95th percentile flux surface [dimensionless]"""

    m_s_limit: Parameter[float]
    """Margin to vertical stability [dimensionless]"""

    delta: Parameter[float]
    """Triangularity [dimensionless]"""

    sigma_tf_case_max: Parameter[float]
    """Maximum von Mises stress in the TF coil case nose [pascal]."""

    sigma_tf_wp_max: Parameter[float]
    """Maximum von Mises stress in the TF coil winding pack [pascal]."""

    sigma_cs_wp_max: Parameter[float]
    """Maximum von Mises stress in the CS coil winding pack [pascal]."""

    H_star: Parameter[float]
    """H factor (radiation corrected) [dimensionless]."""

    bb_pump_eta_el: Parameter[float]
    """Breeding blanket pumping electrical efficiency [dimensionless]"""

    bb_pump_eta_isen: Parameter[float]
    """Breeding blanket pumping isentropic efficiency [dimensionless]"""

    bb_t_inlet: Parameter[float]
    """Breeding blanket inlet temperature [K]"""

    bb_t_outlet: Parameter[float]
    """Breeding blanket outlet temperature [K]"""

    eta_ecrh: Parameter[float]
    """Electron cyclotron resonce heating wallplug efficiency [dimensionless]"""

    eta_cd_norm_ecrh: Parameter[float]
    """Electron cyclotron resonce heating current drive efficiency [TODO: UNITS!]"""

    # Out parameters
    B_0: Parameter[float]
    """Toroidal field at R_0 [tesla]."""

    B_cs_peak_flat_top_end: Parameter[float]
    """Peak poloidal field in the CS at the end of flat-top [T]"""

    B_cs_peak_pulse_start: Parameter[float]
    """Peak poloidal field in the CS at the start of a pulse [T]"""

    beta_p: Parameter[float]
    """Ratio of plasma pressure to poloidal magnetic pressure [dimensionless]."""

    beta: Parameter[float]
    """Total ratio of plasma pressure to magnetic pressure [dimensionless]."""

    condrad_cryo_heat: Parameter[float]
    """Conduction and radiation heat loads on cryogenic components [megawatt]."""

    delta_95: Parameter[float]
    """95th percentile plasma triangularity [dimensionless]."""

    """Last closed surface plasma triangularity [dimensionless]."""

    f_bs: Parameter[float]
    """Bootstrap fraction [dimensionless]."""

    g_vv_ts: Parameter[float]
    """Gap between VV and TS [meter]."""

    I_p: Parameter[float]
    """Plasma current [megaampere]."""

    j_cs_critical: Parameter[float]
    """Maximum allowable current density in the central solenoid [A/m**2]"""

    kappa_95: Parameter[float]
    """95th percentile plasma elongation [dimensionless]."""

    kappa: Parameter[float]
    """Last closed surface plasma elongation [dimensionless]."""

    P_bd_in: Parameter[float]
    """total auxiliary injected power [megawatt]."""

    P_brehms: Parameter[float]
    """Bremsstrahlung [megawatt]."""

    P_fus_DD: Parameter[float]
    """D-D fusion power [megawatt]."""

    P_fus_DT: Parameter[float]
    """D-T fusion power [megawatt]."""

    P_fus: Parameter[float]
    """Total fusion power [megawatt]."""

    P_line: Parameter[float]
    """Line radiation [megawatt]."""

    P_rad_core: Parameter[float]
    """Core radiation power [megawatt]."""

    P_rad_edge: Parameter[float]
    """Edge radiation power [megawatt]."""

    P_rad: Parameter[float]
    """Radiation power [megawatt]."""

    P_sep: Parameter[float]
    """Separatrix power [megawatt]."""

    P_sync: Parameter[float]
    """Synchrotron radiation [megawatt]."""

    R_0: Parameter[float]
    """Major radius [meter]."""

    r_cp_top: Parameter[float]
    """Radial Position of Top of TF coil taper [meter]."""

    r_cs_in: Parameter[float]
    """Central Solenoid inner radius [meter]."""

    r_fw_ib_in: Parameter[float]
    """Inboard first wall inner radius [meter]."""

    r_fw_ob_in: Parameter[float]
    """Outboard first wall inner radius [meter]."""

    r_tf_in_centre: Parameter[float]
    """Inboard TF leg centre radius [meter]."""

    r_tf_in: Parameter[float]
    """Inboard radius of the TF coil inboard leg [meter]."""

    r_tf_out_centre: Parameter[float]
    """Outboard TF leg centre radius [meter]."""

    r_ts_ib_in: Parameter[float]
    """Inboard TS inner radius [meter]."""

    r_vv_ib_in: Parameter[float]
    """Inboard vessel inner radius [meter]."""

    r_vv_ob_in: Parameter[float]
    """Outboard vessel inner radius [meter]."""

    tau_e: Parameter[float]
    """Energy confinement time [second]."""

    TF_currpt_ob: Parameter[float]
    """TF coil current per turn [ampere]."""

    TF_E_stored: Parameter[float]
    """total stored energy in the toroidal field [gigajoule]."""

    TF_res_bus: Parameter[float]
    """TF Bus resistance [meter]."""

    TF_res_tot: Parameter[float]
    """Total resistance for TF coil set [ohm]."""

    TF_respc_ob: Parameter[float]
    """TF coil leg resistance [ohm]."""

    tf_wp_depth: Parameter[float]
    """TF coil winding pack depth (in y) [meter]. Including insulation."""

    tf_wp_width: Parameter[float]
    """TF coil winding pack radial width [meter]. Including insulation."""

    tk_cs: Parameter[float]
    """Central Solenoid radial thickness [meter]."""

    tk_fw_in: Parameter[float]
    """Inboard first wall thickness [meter]."""

    tk_fw_out: Parameter[float]
    """Outboard first wall thickness [meter]."""

    tk_tf_inboard: Parameter[float]
    """TF coil inboard thickness [meter]."""

    tk_tf_ins: Parameter[float]
    """TF coil ground insulation thickness [meter]."""

    tk_tf_insgap: Parameter[float]
    """
    TF coil WP insertion gap [meter]. Backfilled with epoxy resin (impregnation).
    This is an average value; can be less or more due to manufacturing tolerances.
    """

    tk_tf_nose: Parameter[float]
    """TF coil inboard nose thickness [meter]."""

    v_burn: Parameter[float]
    """Loop voltage during burn [volt]."""

    # In-out parameters
    A: Parameter[float]
    """Plasma aspect ratio [dimensionless]."""

    g_cs_tf: Parameter[float]
    """Gap between CS and TF [meter]."""

    g_ts_tf: Parameter[float]
    """Gap between TS and TF [meter]."""

    g_vv_bb: Parameter[float]
    """Gap between VV and BB [meter]."""

    tk_bb_ib: Parameter[float]
    """Inboard blanket thickness [meter]."""

    tk_bb_ob: Parameter[float]
    """Outboard blanket thickness [meter]."""

    tk_sh_in: Parameter[float]
    """Inboard shield thickness [meter]. DO NOT USE - PROCESS has VV = VV + shield."""

    tk_sol_ib: Parameter[float]
    """Inboard SOL thickness [meter]."""

    tk_sol_ob: Parameter[float]
    """Outboard SOL thickness [meter]."""

    tk_ts: Parameter[float]
    """TS thickness [meter]."""

    tk_vv_in: Parameter[float]
    """Inboard vacuum vessel thickness [meter]."""

    # Other parameters
    B_tf_peak: Parameter[float]
    """Peak field inside the TF coil winding pack [tesla]."""

    f_ni: Parameter[float]
    """Non-inductive current drive fraction [dimensionless]."""

    z_cp_top: Parameter[float]
    """Height of the TF coil inboard Tapered section end [meter]."""

    h_tf_max_in: Parameter[float]
    """Plasma side TF coil maximum height [meter]."""

    l_i: Parameter[float]
    """Normalised internal plasma inductance [dimensionless]."""

    r_tf_inboard_out: Parameter[float]
    """Outboard Radius of the TF coil inboard leg tapered region [meter]."""

    T_profile_alpha: Parameter[float]
    """Temperature profile alpha exponent [dimensionless]."""

    T_profile_beta: Parameter[float]
    """Temperature profile beta exponent [dimensionless]."""

    n_profile_alpha: Parameter[float]
    """Density profile alpha exponent [dimensionless]."""

    profile_rho_ped: Parameter[float]
    """Pedestal location in normalized radius [dimensionless]."""

    T_e: Parameter[float]
    """Volumed-averaged plasma electron temperature [kiloelectron_volt]."""

    T_e_core: Parameter[float]
    """Core electron temperature [kiloelectron_volt]."""

    T_e_ped: Parameter[float]
    """Pedestal electron temperature [kiloelectron_volt]."""

    T_e_sep: Parameter[float]
    """Electron temperature at the separatrix [kiloelectron_volt]."""

    n_e: Parameter[float]
    """Volumed-averaged plasma electron density [1/metre ** 3]."""

    n_e_core: Parameter[float]
    """Core electron density [1/metre ** 3]."""

    n_e_ped: Parameter[float]
    """Pedestal electron density [1/metre ** 3]."""

    n_e_sep: Parameter[float]
    """Electron density at the separatrix [1/metre ** 3]."""

    tk_tf_outboard: Parameter[float]
    """TF coil outboard thickness [meter]."""

    V_p: Parameter[float]
    """Plasma volume [meter ** 3]."""

    Z_eff: Parameter[float]
    """Effective particle radiation atomic mass [unified_atomic_mass_unit]."""

    _mappings = deepcopy(mappings)

    @property
    def _defaults(self):
        try:
            return self.__defaults
        except AttributeError:
            self.__defaults = ProcessInputs()
            return self.__defaults

    @_defaults.setter
    def _defaults(self, value: ProcessInputs):
        self.__defaults = value

    @property
    def mappings(self) -> dict[str, ParameterMapping]:
        """Define mappings between these parameters and PROCESS's."""
        return self._mappings

    @property
    def defaults(self) -> dict[str, float | list | dict]:
        """
        Default values for Process
        """
        return self._defaults.to_dict()

    @property
    def template_defaults(self) -> dict[str, _INVariable]:
        """
        Template defaults for process
        """
        return self._defaults.to_invariable()

    @classmethod
    def from_defaults(cls, template: ProcessInputs | None = None) -> ProcessSolverParams:
        """
        Initialise from defaults
        """  # noqa: DOC201
        if template is None:
            template = ProcessInputs()
            self = super().from_defaults(template.to_dict())
        else:
            self = super().from_defaults(
                template.to_dict(), source=f"{NAME} user input template"
            )
        self.__defaults = template
        return self
