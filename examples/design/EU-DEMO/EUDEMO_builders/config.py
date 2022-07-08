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

"""
Configuration classes
"""
from bluemira.base.config import Configuration
from bluemira.base.config_schema import ConfigurationSchema
from bluemira.base.parameter import Parameter, ParameterFrame


class EUDEMOConfigurationSchema(ConfigurationSchema):
    """
    EUDEMO configuration schema.
    """

    blanket_type: Parameter
    n_CS: Parameter
    n_PF: Parameter

    # Radiation and charged particles
    f_core_rad_fw: Parameter
    f_fw_aux: Parameter
    f_sol_ch_fw: Parameter
    f_sol_rad: Parameter
    f_sol_rad_fw: Parameter

    # Plasma
    shaf_shift: Parameter

    # Blanket
    bb_min_angle: Parameter
    bb_p_inlet: Parameter
    bb_p_outlet: Parameter
    bb_pump_eta_el: Parameter
    bb_pump_eta_isen: Parameter
    bb_t_inlet: Parameter
    bb_t_outlet: Parameter
    n_bb_inboard: Parameter
    n_bb_outboard: Parameter

    # Divertor
    div_pump_eta_el: Parameter
    div_pump_eta_isen: Parameter

    # Divertor profile
    div_L2D_ib: Parameter
    div_L2D_ob: Parameter
    div_Ltarg: Parameter  # noqa :N815 - mixed case to match PROCESS
    div_open: Parameter
    n_div_cassettes: Parameter

    # First wall profile
    fw_psi_n: Parameter

    # Central solenoid
    CS_bmax: Parameter
    CS_jmax: Parameter

    # PF magnets
    PF_bmax: Parameter
    PF_jmax: Parameter

    # Gaps and clearances
    c_rm: Parameter
    g_cs_mod: Parameter

    # Vacuum vessel
    vvpfrac: Parameter

    # TF coils
    r_tf_current_ib: Parameter
    tk_tf_wp: Parameter
    tk_tf_wp_y: Parameter

    # PF coils
    r_cs_corner: Parameter
    r_pf_corner: Parameter
    tk_cs_casing: Parameter
    tk_cs_insulation: Parameter
    tk_pf_casing: Parameter
    tk_pf_insulation: Parameter

    # Neutronics
    e_decay_mult: Parameter

    # Powercycle
    eta_ss: Parameter
    f_recirc: Parameter


class EUDEMOConfiguration(Configuration, EUDEMOConfigurationSchema):
    """
    EUDEMO Configuration
    """

    # fmt: off
    new_params = [

        # Reactor
        ["blanket_type", "Blanket type", "HCPB", "dimensionless", None, "Input"],
        ["n_CS", "Number of CS coil divisions", 5, "dimensionless", None, "Input"],
        ["n_PF", "Number of PF coils", 6, "dimensionless", None, "Input"],

        # Radiation and charged particles
        ["f_core_rad_fw", "Fraction of core radiation power that is distributed to the blanket FW", 0.9, "dimensionless", None, "Input (MC guess)"],
        ["f_fw_aux", "Fraction of first wall power that goes into auxiliary systems", 0.09, "dimensionless", None, "Input (F. Maviglia standard)"],
        ["f_sol_ch_fw", "Fraction of SOL charged particle power that is distributed to the blanket FW", 0.8, "dimensionless", None, "Input (F. Maviglia standard)"],
        ["f_sol_rad", "Fraction of SOL power radiated", 0.75, "dimensionless", "The rest is assumed to be in the form of charged particles", "Input (F. Maviglia standard)"],
        ["f_sol_rad_fw", "Fraction of radiated SOL power that is distributed to the blanket FW", 0.8, "dimensionless", None, "Input (MC guess)"],

        # Plasma
        ["shaf_shift", "Shafranov shift of plasma (geometric=>magnetic)", 0.5, "m", None, "equilibria"],

        # Blanket
        ["bb_min_angle", "Minimum module angle", 70, "°", "Sharpest cut of a module possible", "Input (Lorenzo Boccaccini said this in a meeting in 2015, Garching, Germany)"],
        ["bb_p_inlet", "Breeding blanket inlet pressure", 8e6, "Pa", None, "Input (HCPB classic)"],
        ["bb_p_outlet", "Breeding blanket outlet pressure", 7.5e6, "Pa", None, "Input (HCPB classic)"],
        ["bb_pump_eta_el", "Breeding blanket pumping electrical efficiency", 0.87, "dimensionless", None, "Input (D.J. Ward, W.E. Han. Results of system studies for DEMO. Report of DEMO study, Task TW6-TRP-002. July 2007)"],
        ["bb_pump_eta_isen", "Breeding blanket pumping isentropic efficiency", 0.9, "dimensionless", None, "Input (Fabio Cismondi 08/12/16)"],
        ["bb_t_inlet", "Breeding blanket inlet temperature", 300, "°C", None, "Input (HCPB classic)"],
        ["bb_t_outlet", "Breeding blanket outlet temperature", 500, "°C", None, "Input (HCPB classic)"],
        ["n_bb_inboard", "Number of inboard blanket segments", 2, "dimensionless", None, "Input"],
        ["n_bb_outboard", "Number of outboard blanket segments", 3, "dimensionless", None, "Input"],

        # Divertor
        ["div_pump_eta_el", "Divertor pumping electrical efficiency", 0.87, "dimensionless", None, "Input (F. Cismondi)"],
        ["div_pump_eta_isen", "Divertor pumping isentropic efficiency", 0.99, "dimensionless", None, "Input (F. Cismondi)"],

        # Divertor profile
        ["div_L2D_ib", "Inboard divertor leg length", 1.1, "m", None, "Input"],
        ["div_L2D_ob", "Outboard divertor leg length", 1.45, "m", None, "Input"],
        ["div_Ltarg", "Divertor target length", 0.5, "m", None, "Input"],
        ["div_open", "Divertor open/closed configuration", False, "dimensionless", None, "Input"],
        ["n_div_cassettes", "Number of divertor cassettes per sector", 3, "dimensionless", None, "Input"],

        # First wall profile
        ["fw_psi_n", "Normalised psi boundary to fit FW to", 1.07, "dimensionless", None, "Input"],

        # Central solenoid
        ["CS_bmax", "Maximum peak field to use in CS modules", 13, "T", None, "Input"],
        ["CS_jmax", "Maximum current density to use in CS modules", 16, "MA/m^2", None, "Input"],

        # PF magnets
        ["PF_bmax", "Maximum peak field to use in PF modules", 11, "T", None, "Input"],
        ["PF_jmax", "Maximum current density to use in PF modules", 12.5, "MA/m^2", None, "Input"],

        # Gaps and clearances
        ["c_rm", "Remote maintenance clearance", 0.02, "m", "Distance between IVCs", "Input"],
        ["g_cs_mod", "Gap between CS modules", 0.1, "m", None, "Input"],

        # Vacuum vessel
        ["vvpfrac", "Fraction of neutrons deposited in VV", 0.04, "dimensionless", "simpleneutrons needs a correction for VV n absorbtion", "Input (Bachmann, probably thanks to P. Pereslavtsev)"],

        # TF Coil
        ["r_tf_current_ib", "Radius of the TF coil current centroid on the inboard", 0, "m", None, "Input"],
        ["tk_tf_wp", "TF coil winding pack radial thickness", 0.5, "m", "Excluding insulation", "Input"],
        ["tk_tf_wp_y", "TF coil winding pack toroidal thickness", 0.5, "m", "Excluding insulation", "Input"],

        # PF coils
        ["r_cs_corner", "Corner radius of the CS coil winding pack", 0.05, "m", None, "Input"],
        ["r_pf_corner", "Corner radius of the PF coil winding pack", 0.05, "m", None, "Input"],
        ["tk_cs_casing", "Thickness of the CS coil casing", 0.07, "m", None, "Input"],
        ["tk_cs_insulation", "Thickness of the CS coil insulation", 0.05, "m", None, "Input"],
        ["tk_pf_casing", "Thickness of the PF coil casing", 0.07, "m", None, "Input"],
        ["tk_pf_insulation", "Thickness of the PF coil insulation", 0.05, "m", None, "Input"],

        # Neutronics
        ["e_decay_mult", "Decay heat multiplication factor", 1.0175, "dimensionless", "Quasi-instantaneous energy multiplication; still present when plasma is off", "Input (PPCS FWBL Helium Cooled Model P PPCS04 D5part1)"],

        # Powercycle
        ["eta_ss", "Steady-state power cycle efficiency", 0, "dimensionless", "Including energy multiplication in denominator", "Input"],
        ["f_recirc", "Recirculating power fraction", 0, "dimensionless", None, "Input"],

    ]
    # fmt: on
    ParameterFrame._clean()
    params = Configuration.params + new_params
    ParameterFrame.set_default_parameters(params)
