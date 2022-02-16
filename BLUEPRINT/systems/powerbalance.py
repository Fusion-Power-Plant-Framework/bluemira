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
Balance of plant system
"""
from typing import Type

import matplotlib.pyplot as plt
import numpy as np

from bluemira.balance_of_plant.plotting import SuperSankey
from bluemira.base.constants import (
    HE3_MOLAR_MASS,
    HE_MOLAR_MASS,
    NEUTRON_MOLAR_MASS,
    to_kelvin,
)
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter import ParameterFrame
from BLUEPRINT.base.palettes import B_PAL_MAP
from BLUEPRINT.systems.baseclass import ReactorSystem


def cryo_power(s_tf, m_cold, nucl_heating, e_pf_max, t_pulse, tf_current, n_TF):
    """
    Calculates cryogenic loads (taken from PROCESS) \n

    Parameters
    ----------
    s_tf: float
        TF coil total surface area [m^2]
    m_cold: float
        Total cold mass [kg]
    nucl_heating: float
        Total coil nuclear heating [MW]
    e_pf_max: float
        Maximum stored energy in the PF coils [MJ]
    t_pulse: float
        Pulse length [s]
    tf_current: float
        TF coil current per turn [A]
    n_TF: int
        Number of TF coils

    Returns
    -------
    Pcryo: float
        Total power required to cool cryogenic components

    Note
    ----
    Author: P J Knight, CCFE, Culham Science Centre
    D. Slack memo SCMDG 88-5-1-059, LLNL ITER-88-054, Aug. 1988
    """
    # TODO: Temperature!
    # Steady-state loads
    qss = 4.3e-4 * m_cold + 2 * s_tf
    # AC losses
    qac = 1e3 * e_pf_max / t_pulse
    # Current leads
    qcl = 13.6e-3 * n_TF * tf_current
    # Misc. loads (piping and reserves)
    fmisc = 0.45
    return (1 + fmisc) * (nucl_heating + qcl + qac + qss)


def He_pumping(  # noqa :N802
    pressure_in, d_pressure, t_in, t_out, blanket_power, eta_isen, eta_el
):
    """
    Isso aqui calcula o poder necessario pra os blankets
    A pressão esta em MPa e não em Bar!!

    Parameters
    ----------
    pressure_in: float
        Inlet pressure [MPa]
    d_pressure: float
        Pressure drop [MPa]
    t_in: float
        Inlet temperature [°C]
    t_out: float
        Outlet temperature [°C]
    blanket_power: float
        Total blanket power excluding pumping power [MW]
    eta_isen: float
        Isentropic efficiency of the He compressors
    eta_el: float
        Electrical efficiency of the He compressors

    Returns
    -------
    P_pump_is: float
        The isentropic pumping power (added to the working fluid)
    P_pump_el: float
        The eletrical pumping power (parasitic load)

    \t:math:`T_{in_{comp}} = \\dfrac{T_{in_{BB}}}{\\dfrac{P}{P-dP}^{\\dfrac{\\gamma-1}{\\gamma}}}`\n
    \t:math:`f_{p} = \\dfrac{T_{in_{comp}}}{\\eta_{is}dT}\\Bigg(\\dfrac{P}{P-dP}^{\\dfrac{\\gamma-1}{\\gamma}}-1\\Bigg)`

    Notes
    -----
    \t:math:`f_{p} = \\dfrac{T_{in_{BB}}}{\\eta_{is}dT}\\Bigg(1-\\dfrac{P-dP}{P}^{\\dfrac{\\gamma-1}{\\gamma}}\\Bigg)`
    **Outputs:**\n
    \t:math:`P_{pump} = \\dfrac{f_{p}P_{plasma}}{1-f_p}` [MW]\n
    \t:math:`P_{pump,el} = \\dfrac{P_{pump}}{\\eta_{el}}` [MW]\n
    **No longer in use:**
    \t:math:`f_{pump}=\\dfrac{dP}{dTc_P\\rho_{av}}`
    """  # noqa :W505
    d_temp = t_out - t_in
    t_bb_inlet = to_kelvin(t_in)
    # Modèle gaz idéal monoatomique - small compression ratios
    t_comp_inlet = t_bb_inlet / ((pressure_in / (pressure_in - d_pressure)) ** (2 / 5))
    # Ivo not sure why can't refind it - probably right but very little
    # difference ~ 1 K
    # T_comp_inlet = eta_isen*T_bb_inlet/((P/(P-dP))**(6/15)+eta_isen-1)
    f_pump = (t_comp_inlet / (eta_isen * d_temp)) * (
        (pressure_in / (pressure_in - d_pressure)) ** (2 / 5) - 1
    )  # kJ/kg
    p_pump_is = f_pump * blanket_power / (1 - f_pump)
    p_pump_el = p_pump_is / eta_el
    return p_pump_is, p_pump_el


def H2O_pumping(p_blanket, f_pump, eta_isen, eta_el):  # noqa :N802
    # TODO: Add proper pump model
    f_pump /= eta_isen

    p_pump_is = f_pump * p_blanket / (1 - f_pump)
    p_pump_el = p_pump_is / eta_el
    return p_pump_is, p_pump_el


def superheated_rankine(blanket_power, div_power, bb_outlet_temp):
    """
    PROCESS C. Harrington correlation. Accounts for low-grade heat penalty.
    Used for He-cooled blankets. Not applicable to H2O temperatures.
    """
    d_t_turb = 20  # Turbine inlet delta-T to BB_out [K]
    t_turb = to_kelvin(bb_outlet_temp - d_t_turb)
    if t_turb < 657 or t_turb > 915:
        bluemira_warn("BoP turbine inlet temperature outside range of validity.")
    f_lgh = div_power / (blanket_power + div_power)
    delta_eta = 0.339 * f_lgh
    return 0.1802 * np.log(t_turb) - 0.7823 - delta_eta


class BalanceOfPlant(ReactorSystem):
    """
    Balance of plant system for a fusion power reactor

    .. math::
        P_{el}={\\eta}_{BOP}\\Bigg[\\Bigg(\\frac{4}{5}P_{fus}f_{nrgm}-\\
        P_{n_{aux}}-P_{n_{DIV}}+f_{SOL_{rad}}f_{SOL_{ch}}\\Big(\\frac{P_{fus}}{5}+P_{HCD}\\Big)\\Bigg)\\
        \\Big(1+\\frac{f_{p_{BB}}}{1-f_{p_{BB}}}\\Big)
        +\\Bigg(P_{n_{DIV}}+f_{SOL_{rad}}f_{SOL_{ch}}f_{fw}\\Big(\\frac{P_{fus}}{5}+P_{HCD}\\Big)\\Bigg)\\
        \\Big(1+\\frac{f_{p_{DIV}}}{1-f_{p_{DIV}}}\\Big)\\Bigg]

    """  # noqa :W505

    config: Type[ParameterFrame]
    inputs: dict

    # fmt: off
    default_params = [
        ['P_fus_DT', 'D-T fusion power', 1995, 'MW', None, 'PLASMOD'],
        ['P_fus_DD', 'D-D fusion power', 5, 'MW', None, 'PLASMOD'],
        ['P_rad', 'Radiation power', 400, 'MW', None, 'PLASMOD'],
        ['op_mode', 'Mode of operation', 'Pulsed', 'dimensionless', None, 'Input'],
        ['pradfw', 'Fraction of core radiation deposited in the FW', 0.9, 'dimensionless', None, None],
        ['eta_el_He', 'He compressor electrical efficiency', 0.87, 'dimensionless', None, 'D.J. Ward, W.E. Han. Results of system studies for DEMO. Report of DEMO study, Task TW6-TRP-002. July 2007'],
        ['eta_isen_He', 'He compressor isentropic efficiency', 0.9, 'dimensionless', None, 'Fabio Cismondi08/12/16'],
        ['fsolrad', 'SOL radiation fraction', 0.75, 'dimensionless', None, 'F. Maviglia standard'],
        ['fsolradfw', 'Fraction of SOL radiation deposited in the FW', 0.8, 'dimensionless', None, 'MC guess'],
        ['fsepchblk', 'SOL power in the form of charged particles', 0.8, 'dimensionless', None, 'F. Maviglia standard'],
        ['f_fw_a_blk', 'Fraction of alpha and aux power deposited on the blanket FW', 0.91, 'dimensionless', None, 'Taken from some Bachmann crap'],
        ['eta_el_H2O', 'H2O pump electrical efficiency', 0.87, 'dimensionless', None, 'F. Cismondi'],
        ['eta_isen_H2O', 'H2O pump isentropic efficiency', 0.99, 'dimensionless', None, 'F. Cismondi'],
        ['f_pump_H2O_BB', 'BB pumping fraction for WC blanket', 0.004, 'dimensionless', None, 'F. Cismondi 08/12/16'],
        ['f_pump_H2O_DIV', 'DIV pumping fraction', 0.05, 'dimensionless', None, 'MC guess x 10-20 lit numbers for BB'],
        ['f_alpha_plasma', 'Fraction of charged particle power deposited in the plasma', 0.95, 'dimensionless', None, 'PROCESS reference value in 2019']
    ]
    # fmt: on

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs

        self._init_params(self.config)

        self._plotter = BalanceOfPlantPlotter()

        # Constructors
        self.flow_dict = {}

    def build(self):
        """
        Perform the balance of plant operations.
        """
        # Plasma
        p_fusion = self.params.P_fus_DT + self.params.P_fus_DD
        f_neutron_DT = HE_MOLAR_MASS / (HE_MOLAR_MASS + NEUTRON_MOLAR_MASS)
        f_neutron_DD = 0.5 * HE3_MOLAR_MASS / (NEUTRON_MOLAR_MASS + HE3_MOLAR_MASS)
        # Neutron power
        n_p_DT = f_neutron_DT * self.params.P_fus_DT  # noqa neutron power in D-T
        n_p_DD = f_neutron_DD * self.params.P_fus_DD  # noqa neutron power in D-D
        p_neutron = n_p_DT + n_p_DD
        # Charged particle power
        a_p = (1 - f_neutron_DT) * self.params.P_fus_DT  # alpha power from D-T
        # charged particle power from D-D
        other_p = (1 - f_neutron_DD) * self.params.P_fus_DD
        p_charged = a_p + other_p
        # NOTE: There is a difference between how PROCESS and BLUEPRINT handle
        # charged particle power. The "falpha" variable in PROCESS is not used
        # here, and instead we take "fsolradfw" and "fsolrad" to separate out
        # charged particle power and radiation, and divvy it up between the
        # plasma-facing components.
        # Rad / sep power
        p_hcd = self.inputs["P_hcd"]
        p_radiation = self.params.P_rad
        # NOTE: P_sep is calculated and will be slightly different to PROCESS
        p_separatrix = p_charged + p_hcd - p_radiation
        p_hcd_el = self.inputs["P_hcd_el"]
        blk_p_frac = self.inputs["blkpfrac"]
        nrgm = self.inputs["nrgm"]
        # neutron power deposited in blanket
        p_n_blk = blk_p_frac * nrgm * p_neutron
        p_n_div = (
            self.inputs["divpfrac"] * p_neutron
        )  # neutron power deposited in divertor
        p_n_aux = (
            self.inputs["auxpfrac"] * p_neutron
        )  # neutron power deposited in aux systems
        # neutron power deposited in VV
        p_n_vv = self.inputs["vvpfrac"] * p_neutron
        p_nrgm = (nrgm - 1) * p_neutron * blk_p_frac

        # Alphas
        # core radiation power deposited in blanket
        p_rad_fw = self.params.pradfw * p_radiation
        p_rad_div = p_radiation * (
            1 - self.params.pradfw
        )  # core radiation power deposited in divertor
        # power from the SOL deposited by radiation in the FW
        p_sep_fw = p_separatrix * self.params.fsolradfw * self.params.fsolrad
        # power from the SOL deposited by radiation in the divertor
        p_sep_div = p_separatrix * (1 - self.params.fsolradfw) * self.params.fsolrad
        # fraction of charged particle SOL power deposited in the blanket
        p_sep_charged = p_separatrix * (1 - self.params.fsolrad)
        # charged particle SOL power deposited in the blanket FW
        p_sep_ch_blk = p_sep_charged * self.params.fsepchblk
        # charged particle SOL power deposited in the divertor
        p_sep_ch_div = p_sep_charged * (1 - self.params.fsepchblk)
        # SOL charged particle and radiation power in the divertor
        p_div_a = p_rad_div + p_sep_div + p_sep_ch_div
        p_div = p_n_div + p_div_a
        p_fw_a = p_rad_fw + p_sep_fw + p_sep_ch_blk
        # Alpha and aux power fraction deposited on the aux surfaces of the FW
        p_fw_a_blk = p_fw_a * self.params.f_fw_a_blk
        p_fw_a_aux = p_fw_a * (1 - self.params.f_fw_a_blk)
        # Decay heat (in steady-state operation)
        p_decayheat = self.inputs["f_decayheat"] * p_fusion
        p_blk_decayheat = p_decayheat * blk_p_frac
        # Total blanket heat excluding pumping power
        p_blanket = p_n_blk + p_fw_a_blk + p_blk_decayheat

        if self.inputs["BBcoolant"] == "He":
            p_pump, p_pump_el = He_pumping(
                self.inputs["BB_P_in"],
                self.inputs["BB_dP"],
                self.inputs["BB_T_in"],
                self.inputs["BB_T_out"],
                p_blanket,
                self.params.eta_isen_He,
                self.params.eta_el_He,
            )
            n_bop = superheated_rankine(p_blanket, p_div, self.inputs["BB_T_out"])
        elif self.inputs["BBcoolant"] == "Water":
            p_pump, p_pump_el = H2O_pumping(
                p_blanket,
                self.params.f_pump_H2O_BB,
                self.params.eta_isen_H2O,
                self.params.eta_el_H2O,
            )
            n_bop = 0.33  # H2O cycle efficiency  Ciattaglia
        p_blanket += p_pump  # Add pump power to fluid

        # NOTE: P_pumpdiv is not added to P_div as in the blanket for
        # plotting purposes only!
        p_pump_div, p_pump_div_el = H2O_pumping(
            p_div,
            self.params.f_pump_H2O_DIV,
            self.params.eta_isen_H2O,
            self.params.eta_el_H2O,
        )
        # Normalised values for now (no better model)
        fnorm = p_fusion / 2037
        p_cryo = fnorm * 29  # [MW] Ciattaglia
        p_mag = fnorm * 44  # [MW] Ciattaglia
        p_t_plant = fnorm * 15.5  # [MW] Ciattaglia
        p_other = fnorm * 31  # [MW] Ciattaglia
        p_bop = (p_blanket + p_div + p_pump_div) * n_bop
        p_bop_loss = (1 - n_bop) * p_bop / n_bop
        p_el_net = (
            p_bop
            - p_hcd_el
            - p_pump_el
            - p_cryo
            - p_mag
            - p_other
            - p_t_plant
            - p_pump_div_el
        )

        etaplant = p_el_net / p_fusion

        # Set up power flows and plotting
        d = {
            "Plasma": [p_fusion, p_hcd, -p_neutron, -p_separatrix - p_radiation],
            "H&CD": [p_hcd_el, -p_hcd, -(p_hcd_el - p_hcd)],
            "Neutrons": [p_neutron, p_nrgm, -p_n_blk, -p_n_div, -p_n_vv - p_n_aux],
            "Radiation and \nseparatrix": [
                p_separatrix + p_radiation,
                -p_fw_a,
                -p_div_a,
            ],
            "Blanket": [p_n_blk, p_fw_a_blk, p_pump, p_blk_decayheat, -p_blanket],
            "Divertor": [p_n_div, p_div_a, -p_div],
            "First wall": [p_fw_a, -p_fw_a_aux, -p_fw_a_blk],
            "BoP": [p_blanket, p_div + p_pump_div, -p_bop_loss, -p_bop],
            "Electricity": [
                p_bop,
                -p_t_plant,
                -p_other,
                -p_cryo,
                -p_mag,
                -p_hcd_el,
                -p_pump_div_el,
                -p_pump_el,
                -p_el_net,
            ],
            "_H&CD loop": [p_hcd_el, -p_hcd_el],
            "_Divertor 2": [p_div_a, -(p_div - p_n_div)],
            "_H&CD loop 2": [p_hcd_el, -p_hcd_el],
            "_DIV to BOP": [p_div, p_pump_div, -p_div - p_pump_div],
            "_BB coolant loop turn": [p_pump_el, -p_pump_el + p_pump, -p_pump],
            "_BB coolant loop blanket": [p_pump, -p_pump],
            "_DIV coolant loop turn": [
                p_pump_div_el,
                -p_pump_div_el + p_pump_div,
                -p_pump_div,
            ],
            "_DIV coolant loop divertor": [p_pump_div, -p_pump_div],
        }
        self.flow_dict = d

        self.sanity()
        # Formalise output parameters
        # fmt: off

        p = [['P_charged', 'Charged particle power', p_charged, 'MW', None, 'BLUEPRINT'],
             ['P_neutron', 'Neutron power (D-T and D-D)', p_neutron, 'MW', None, 'BLUEPRINT'],
             ['P_nrgm', 'Neutron multiplication power', p_nrgm, 'MW', None, 'BLUEPRINT'],
             ['P_nBB', 'Blanket neutron power', p_n_blk, 'MW', None, 'BLUEPRINT'],
             ['P_nDIV', 'Divertor neutron power', p_n_div, 'MW', None, 'BLUEPRINT'],
             ['P_nHCD', 'H&CD neutron power', p_n_aux, 'MW', None, 'BLUEPRINT'],
             ['P_nVV', 'Vacuum vessel neutron power', p_n_vv, 'MW', None, 'BLUEPRINT'],
             ['P_DIV', 'Divertor power', p_div, 'MW', None, 'BLUEPRINT'],
             ['P_BB', 'Blanket power', p_blanket, 'MW', None, 'BLUEPRINT'],
             ['P_pumpBB', 'Blanket pumping power', p_pump, 'MW', None, 'BLUEPRINT'],
             ['P_pumpelBB', 'BB pumping electrical power', p_pump_el, 'MW', None, 'BLUEPRINT'],
             ['P_pumpDIV', 'Divertor pumping power', p_pump_div, 'MW', None, 'BLUEPRINT'],
             ['P_pumpelDIV', 'DIV pumping electrical power', p_pump_div_el, 'MW', None, 'BLUEPRINT'],
             ['P_BBdecay', 'BB decay heat', p_decayheat, 'MW', None, 'BLUEPRINT'],
             ['P_SOLBB', 'SOL power in BB', p_fw_a_blk, 'MW', None, 'BLUEPRINT'],
             ['P_SOLHCD', 'SOL power in HCD systems', p_fw_a_aux, 'MW', None, 'BLUEPRINT'],
             ['P_SOLDIV', 'SOL power in DIV', p_div_a, 'MW', None, 'BLUEPRINT'],
             ['P_el', 'Total electric power generated', p_bop, 'MW', None, 'BLUEPRINT'],
             ['P_el_net', 'Net electric power generated', p_el_net, 'MW', None, 'BLUEPRINT'],
             ['P_cryo', 'Cryoplant power', p_cryo, 'MW', None, 'BLUEPRINT'],
             ['P_T', 'Tritium plant power', p_t_plant, 'MW', None, 'BLUEPRINT'],
             ['P_MAG', 'Magnet power', p_mag, 'MW', None, 'BLUEPRINT'],
             ['P_oth', 'Miscellaneous power', p_other, 'MW', None, 'BLUEPRINT'],
             ['eta_plant', 'Global plant efficiency', etaplant * 100, '%', None, 'BLUEPRINT']]
        # fmt: on

        self.add_parameters(p)
        return p_el_net, etaplant

    def sanity(self):
        """
        Performs a series of sanity checks
        """
        # Per Sankey block check
        for label, flow in self.flow_dict.items():
            delta = sum(flow)
            block = label
            if round(delta) != 0:
                bluemira_warn(
                    f"O bloque {block} agora ta fodido.. {delta:.2f} MW perdidos"
                )

        # Global check
        delta_truth = sum(np.sum(list(self.flow_dict.values())))
        if round(delta_truth) != 0:
            bluemira_warn(
                f"Você não sabe o que está fazendo, cara. O seu modelo "
                f"BOP tem {delta_truth:.2f} MW perdidos."
            )

    def plot(self):
        """
        Plots the BalanceOfPlant system in the form of a Sankey diagram.
        """
        self._plotter.plot(self.inputs, self.params.op_mode, self.flow_dict)


class BalanceOfPlantPlotter:
    """
    The plotting object for the BalanceOfPlant system. Builds a relatively
    complicated Sankey diagram, connecting the various flows of energy in the
    reactor.
    """

    def __init__(self):
        # Sankey diagram scaling and sizing defaults
        self.scale = 0.001
        self.gap = 0.25
        self.trunk_length = 0.0007 / self.scale
        self.l_standard = 0.0006 / self.scale  # standard arrow length
        self.l_medium = 0.001 / self.scale  # medium arrow length

        self.inputs = None
        self.flow_dict = None
        self.fig = None
        self.sankey = None

    def plot(self, inputs, op_mode, flow_dict):
        """
        Plots the BalanceOfPlant system, based on the inputs and flows.

        Parameters
        ----------
        inputs: dict
            The inputs to BalanceOfPlant (used here to format the title)
        op_mode: str
            The operation mode of the reactor
        flow_dict: dict
            The dictionary of flows for each of the Sankey diagrams.
        """
        self.inputs = inputs
        self.flow_dict = flow_dict

        # Text processing
        phcd = self.inputs["P_hcd"]
        if phcd != self.inputs["P_hcd_nb"]:
            f_nbi = self.inputs["P_hcd_nb"] / phcd * 100
            hcd = "{0:.0f}% NBI & {1:.0f}% ECD)".format(f_nbi, 100 - f_nbi)
        elif phcd == self.inputs["P_hcd_nb"]:
            hcd = "NBI"
        elif phcd == self.inputs["P_hcd_ec"]:
            hcd = "ECD"
        coolant = self.inputs["BBcoolant"]
        mult = self.inputs["multiplier"]

        # Build the base figure object
        self.fig = plt.figure(figsize=(14, 10), facecolor="k")
        ax = self.fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
        plt.axis("off")

        self.fig.suptitle(
            f"Reactor Power Flows - {op_mode}, {coolant}, {mult}, {hcd}",
            color="white",
            fontsize=24,
            weight="bold",
        )
        self.fig.subplots_adjust(top=0.9)
        self.sankey = SuperSankey(
            ax=ax,
            scale=self.scale,
            format="%.0f",
            unit="MW",
            gap=self.gap,
            radius=0,
            shoulder=0,
            head_angle=150,
        )
        self._build_diagram()
        self._polish()

    def _build_diagram(self):
        """
        Builds the Sankey diagram. This is much more verbose than looping over
        some structs, but that's how it used to be and it was hard to modify.
        This is easier to read and modify.
        """
        trunk_length = self.trunk_length
        l_s = self.l_standard
        l_m = self.l_medium

        # 0: Plasma
        self.sankey.add(
            patchlabel="Plasma",
            labels=["Fusion Power", None, "Neutrons", "Alphas + Aux"],
            flows=self.flow_dict["Plasma"],
            orientations=[0, -1, 0, -1],
            prior=None,
            connect=None,
            trunklength=trunk_length,
            pathlengths=[l_m, l_s / 1.5, l_s, l_s],
            facecolor=B_PAL_MAP["blue"],
        )
        # 1: H&CD (first block)
        self.sankey.add(
            patchlabel="H&CD",
            labels=["", "H&CD power", "Losses"],
            flows=self.flow_dict["H&CD"],
            orientations=[-1, 1, -1],
            prior=0,
            connect=(1, 1),
            trunklength=trunk_length,
            pathlengths=[l_s, l_s / 1.5, l_s],
            facecolor=B_PAL_MAP["pink"],
        )
        # 2: Neutrons
        self.sankey.add(
            patchlabel="Neutrons",
            labels=[None, "Energy Multiplication", "Blanket n", "Divertor n", "Aux n"],
            flows=self.flow_dict["Neutrons"],
            orientations=[0, 1, 0, -1, -1],
            prior=0,
            connect=(2, 0),
            trunklength=trunk_length,
            pathlengths=[l_s, l_s, l_s, 3 * l_m, l_m],
            facecolor=B_PAL_MAP["orange"],
        )
        # 3: Radiation and separatrix
        self.sankey.add(
            patchlabel="Radiation and\nseparatrix",
            labels=[None, "", "Divertor rad and\n charged p"],
            flows=self.flow_dict["Radiation and \nseparatrix"],
            orientations=[1, 0, -1],
            prior=0,
            connect=(3, 0),
            trunklength=trunk_length,
            pathlengths=[l_s, l_s, l_s],
            facecolor=B_PAL_MAP["red"],
        )
        # 4: Blanket
        self.sankey.add(
            patchlabel="Blanket",
            labels=[None, "", "", "Decay heat", ""],
            flows=self.flow_dict["Blanket"],
            orientations=[0, -1, -1, 1, 0],
            prior=2,
            connect=(2, 0),
            trunklength=trunk_length,
            pathlengths=[l_s, l_s, l_s, l_s, l_s],
            facecolor=B_PAL_MAP["yellow"],
        )
        # 5: Divertor
        self.sankey.add(
            patchlabel="Divertor",
            labels=[None, None, ""],
            flows=self.flow_dict["Divertor"],
            orientations=[1, 0, 0],
            prior=2,
            connect=(3, 0),
            trunklength=trunk_length,
            pathlengths=[l_m, l_s, l_s],
            facecolor=B_PAL_MAP["cyan"],
        )
        # 6: First wall
        self.sankey.add(
            patchlabel="First wall",
            labels=[None, "Auxiliary \n FW", None],
            flows=self.flow_dict["First wall"],
            orientations=[0, -1, 1],
            prior=3,
            future=4,
            connect=[(1, 0), (1, 2)],
            trunklength=trunk_length,
            pathlengths=[0, l_s, 0],
            facecolor=B_PAL_MAP["grey"],
        )
        # 7: BoP
        self.sankey.add(
            patchlabel="BoP",
            labels=[None, None, "Losses", None],
            flows=self.flow_dict["BoP"],
            orientations=[0, -1, -1, 0],
            prior=4,
            connect=(4, 0),
            trunklength=trunk_length,
            pathlengths=[l_s, l_m, l_m, 0],
            facecolor=B_PAL_MAP["purple"],
        )
        # 8: Electricity
        # Check if we have net electric power
        labels = [
            "$P_{el}$",
            "T plant",
            "P_oth...",
            "Cryoplant",
            "Magnets",
            None,
            None,
            "BB Pumping \n electrical \n power",
            "",
        ]
        orientations = [0, -1, -1, -1, -1, -1, -1, -1, 0]

        if self.flow_dict["Electricity"][-1] > 0:
            # Conversely, this means "net electric loss"
            labels[-1] = "Grid"
            orientations[-1] = 1

        self.sankey.add(
            patchlabel="Electricity",
            labels=labels,
            flows=self.flow_dict["Electricity"],
            orientations=orientations,
            prior=7,
            connect=(3, 0),
            trunklength=trunk_length,
            pathlengths=[
                l_m,
                2 * l_m,
                3 * l_m,
                4 * l_m,
                5 * l_m,
                7 * l_m,
                5 * l_m,
                3 * l_m,
                l_s,
            ],
            facecolor=B_PAL_MAP["green"],
        )
        # 9: H&CD return leg
        self.sankey.add(
            patchlabel="",
            labels=[None, "H&CD Power"],
            flows=self.flow_dict["_H&CD loop"],
            orientations=[-1, 0],
            prior=8,
            connect=(5, 0),
            trunklength=trunk_length,
            pathlengths=[l_s / 2, 7 * l_m],
            facecolor=B_PAL_MAP["green"],
        )
        # 10: Divertor (second block)
        self.sankey.add(
            patchlabel="",
            labels=[None, None],
            flows=self.flow_dict["_Divertor 2"],
            orientations=[1, 0],
            prior=3,
            future=5,
            connect=[(2, 0), (1, 1)],
            trunklength=trunk_length,
            pathlengths=[0, 0],
            facecolor=B_PAL_MAP["cyan"],
        )
        # 11: H&CD return leg (second half)
        self.sankey.add(
            patchlabel="",
            labels=[None, None],
            flows=self.flow_dict["_H&CD loop 2"],
            orientations=[-1, 0],
            prior=9,
            future=1,
            connect=[(1, 0), (0, 1)],
            trunklength=trunk_length,
            pathlengths=[0, 0],
            facecolor=B_PAL_MAP["green"],
        )
        # 12: Divertor back into BoP
        self.sankey.add(
            patchlabel="",
            labels=[None, "", ""],
            flows=self.flow_dict["_DIV to BOP"],
            orientations=[0, -1, 1],
            prior=5,
            future=7,
            connect=[(2, 0), (1, 2)],
            trunklength=trunk_length,
            pathlengths=[0, l_s / 2, 0],
            facecolor=B_PAL_MAP["cyan"],
        )
        # 13: BB electrical pumping loss turn leg
        self.sankey.add(
            patchlabel="",
            labels=[None, "Losses", "BB coolant \n pumping"],
            flows=self.flow_dict["_BB coolant loop turn"],
            orientations=[0, 0, -1],
            prior=8,
            connect=(7, 0),
            trunklength=trunk_length,
            pathlengths=[l_s, l_s, l_m * 3],
            facecolor=B_PAL_MAP["green"],
        )
        # 14: BB electrical pumping return leg into blanket
        self.sankey.add(
            patchlabel="",
            labels=[None, None],
            flows=self.flow_dict["_BB coolant loop blanket"],
            orientations=[0, -1],
            prior=13,
            future=4,
            connect=[(2, 0), (2, 1)],
            trunklength=trunk_length,
            pathlengths=[0, 0],
            facecolor=B_PAL_MAP["green"],
        )
        # 15: Divertor electrical pumping loss turn leg
        self.sankey.add(
            patchlabel="",
            labels=[None, "Losses", "Div coolant \n pumping"],
            flows=self.flow_dict["_DIV coolant loop turn"],
            orientations=[0, 0, -1],
            prior=8,
            connect=(6, 0),
            trunklength=trunk_length,
            pathlengths=[l_s, l_s / 2, l_m],
            facecolor=B_PAL_MAP["green"],
        )
        # 16: Divertor electrical pumping return into divertor
        self.sankey.add(
            patchlabel="",
            labels=[None, None],
            flows=self.flow_dict["_DIV coolant loop divertor"],
            orientations=[0, -1],
            prior=15,
            future=12,
            connect=[(2, 0), (1, 1)],
            trunklength=trunk_length,
            pathlengths=[0, 0],
            facecolor=B_PAL_MAP["green"],
        )

    def _polish(self):
        """
        Finish up and polish figure, and format text
        """
        diagrams = self.sankey.finish()
        for diagram in diagrams:
            diagram.text.set_fontweight("bold")
            diagram.text.set_fontsize("14")
            diagram.text.set_color("white")
            for text in diagram.texts:
                text.set_fontsize("11")
                text.set_color("white")

        self.fig.tight_layout()
