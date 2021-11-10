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

import abc
from typing import Type, List
import numpy as np

from bluemira.base.parameter import ParameterFrame
from bluemira.base.constants import HE3_MOLAR_MASS, HE_MOLAR_MASS, NEUTRON_MOLAR_MASS
from bluemira.balance_of_plant.error import BalanceOfPlantError
from bluemira.balance_of_plant.calculations import (
    He_pumping,
    H2O_pumping,
    superheated_rankine,
)
from bluemira.balance_of_plant.plotting import BalanceOfPlantPlotter


class CoolantPumping(abc.ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def pump(self, power) -> List[float]:
        pass


class HePumping(CoolantPumping):
    def __init__(
        self, pressure_in, pressure_out, temp_in, temp_out, eta_isentropic, eta_electric
    ):
        self.p_in = pressure_in
        self.p_out = pressure_out
        self.t_in = temp_in
        self.t_out = temp_out
        self.eta_isen = eta_isentropic
        self.eta_el = eta_electric

    def pump(self, power):
        p_pump, p_electric = He_pumping(
            power,
            self.pressure_in,
            self.pressure_out,
            self.t_in,
            self.t_out,
            self.eta_isen,
            self.eta_el,
        )
        return p_pump, p_electric


class H2OPumping(CoolantPumping):
    def __init__(self, f_pump, eta_isentropic, eta_electric):
        self.f_pump = f_pump
        self.eta_isen = eta_isentropic
        self.eta_el = eta_electric

    def pump(self, power):
        p_pump, p_electric = H2O_pumping(power, self.f_pump, self.eta_isen, self.eta_el)
        return p_pump, p_electric


class PowerCycleEfficiencyCalc(abc.ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def calculate(self, *args) -> float:
        pass


class PredeterminedEfficiency(PowerCycleEfficiencyCalc):
    def __init__(self, efficiency):
        self.efficiency

    def calculate(self, p_blanket, p_divertor) -> float:
        return self.efficiency


class SuperheatedRankine(PowerCycleEfficiencyCalc):
    def __init__(self, bb_t_out):
        self.bb_t_out = bb_t_out

    def calculate(self, p_blanket, p_divertor) -> float:
        return superheated_rankine(p_blanket, p_divertor, self.bb_t_out)


class FractionSplitStrategy(abc.ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def split(self, *args):
        pass

    def check_fractions(self, fractions):
        frac_sum = sum(fractions)
        if not np.isclose(frac_sum, 1.0, atol=1e-6, rtol=0):
            raise BalanceOfPlantError(
                f"{self.__class__.__name__} fractions sum to {frac_sum:.7f}"
            )


class FusionPowerStrategy(FractionSplitStrategy):
    def __init__(self):
        self.f_neutron_DT = HE_MOLAR_MASS / (HE_MOLAR_MASS + NEUTRON_MOLAR_MASS)
        self.f_neutron_DD = 0.5 * HE3_MOLAR_MASS / (NEUTRON_MOLAR_MASS + HE3_MOLAR_MASS)

    def split(self, p_fusion_DT, p_fusion_DD):
        p_neutron = self.f_neutron_DT * p_fusion_DT + self.f_neutron_DD * p_fusion_DD
        p_charged = p_fusion_DT + p_fusion_DD - p_neutron
        return p_neutron, p_charged


class NeutronPowerStrategy(FractionSplitStrategy):
    def __init__(self, f_blanket, f_divertor, f_vessel, f_other, energy_multiplication):
        self.check_fractions([f_blanket, f_divertor, f_vessel, f_other])
        self.f_blanket = f_blanket
        self.f_divertor = f_divertor
        self.f_vessel = f_vessel
        self.f_other = f_other
        self.nrg_mult = energy_multiplication

    def split(self, neutron_power):
        blk_power = self.f_blanket * self.nrg_mult * neutron_power
        div_power = self.f_divertor * neutron_power
        vv_power = self.f_vessel * neutron_power
        aux_power = self.f_other * neutron_power
        mult_power = self.f_blanket * neutron_power - blk_power
        return blk_power, div_power, vv_power, aux_power, mult_power


class RadChargedPowerStrategy(FractionSplitStrategy):
    def __init__(self, f_core_rad_fw, f_sol_rad, f_sol_rad_fw, f_sol_ch_fw, f_fw_blk):
        self.f_core_rad_fw = f_core_rad_fw
        self.f_sol_rad = f_sol_rad
        self.f_sol_rad_fw = f_sol_rad_fw
        self.f_sol_ch_fw = f_sol_ch_fw
        self.f_fw_blk = f_fw_blk

    def split(self, p_radiation, p_separatrix):
        # Core radiation
        p_core_rad_fw = p_radiation * self.f_core_rad_fw
        p_core_rad_div = p_radiation - p_core_rad_fw

        # Scrape-off layer radiation
        p_sol_rad = p_separatrix * self.f_sol_rad
        p_sol_rad_fw = p_sol_rad * self.f_sol_rad_fw
        p_sol_rad_div = p_sol_rad - p_sol_rad_fw

        # Scrape-off layer charged particles
        p_sol_charged = p_separatrix - p_sol_rad
        p_sol_charged_fw = p_sol_charged * self.f_sol_ch_fw
        p_sol_charged_div = p_sol_charged - p_sol_charged_fw

        # Split first wall into blanket and auxiliary
        p_rad_sep_fw = p_core_rad_fw + p_sol_rad_fw + p_sol_charged_fw
        p_rad_sep_blk = p_rad_sep_fw * self.f_fw_blk
        p_rad_sep_aux = p_rad_sep_fw - p_rad_sep_blk
        p_rad_sep_div = p_core_rad_div + p_sol_rad_div + p_sol_charged_div
        return p_rad_sep_blk, p_rad_sep_div, p_rad_sep_aux


class ParasiticLoadStrategy:
    """
    Ciattaglia reference point from the mid 2010's...
    """

    def __init__(self):
        self.p_fusion_ref = 2037
        self.p_cryo = 44
        self.p_mag = 44
        self.p_t_plant = 15.5
        self.p_other = 31

    def calculate(self, p_fusion):
        f_norm = p_fusion / self.p_fusion_ref
        p_mag = f_norm * self.p_mag
        p_cryo = f_norm * self.p_cryo
        p_t_plant = f_norm * self.p_t_plant
        p_other = f_norm * self.p_other
        return p_mag, p_cryo, p_t_plant, p_other


class BalanceOfPlant:
    """
    Balance of plant system for a fusion power reactor

    .. math::
        P_{el}={\\eta}_{BOP}\\Bigg[\\Bigg(\\frac{4}{5}P_{fus}f_{nrgm}-\\
        P_{n_{aux}}-P_{n_{DIV}}+f_{SOL_{rad}}f_{SOL_{ch}}\\Big(\\frac{P_{fus}}{5}+P_{HCD}\\Big)\\Bigg)\\
        \\Big(1+\\frac{f_{p_{BB}}}{1-f_{p_{BB}}}\\Big)
        +\\Bigg(P_{n_{DIV}}+f_{SOL_{rad}}f_{SOL_{ch}}f_{fw}\\Big(\\frac{P_{fus}}{5}+P_{HCD}\\Big)\\Bigg)\\
        \\Big(1+\\frac{f_{p_{DIV}}}{1-f_{p_{DIV}}}\\Big)\\Bigg]

    """  # noqa (W505)

    config: Type[ParameterFrame]
    inputs: dict
    _plotter = BalanceOfPlantPlotter

    # fmt: off
    default_params = [
        ['P_fus_DT', 'D-T fusion power', 1995, 'MW', None, 'PLASMOD'],
        ['P_fus_DD', 'D-D fusion power', 5, 'MW', None, 'PLASMOD'],
        ['P_rad', 'Radiation power', 400, 'MW', None, 'PLASMOD'],
        ['pradfw', 'Fraction of core radiation deposited in the FW', 0.9, 'N/A', None, None],
        ['eta_el_He', 'He compressor electrical efficiency', 0.87, 'N/A', None, 'D.J. Ward, W.E. Han. Results of system studies for DEMO. Report of DEMO study, Task TW6-TRP-002. July 2007'],
        ['eta_isen_He', 'He compressor isentropic efficiency', 0.9, 'N/A', None, 'Fabio Cismondi08/12/16'],
        ['fsolrad', 'SOL radiation fraction', 0.75, 'N/A', None, 'F. Maviglia standard'],
        ['fsolradfw', 'Fraction of SOL radiation deposited in the FW', 0.8, 'N/A', None, 'MC guess'],
        ['fsepchblk', 'SOL power in the form of charged particles', 0.8, 'N/A', None, 'F. Maviglia standard'],
        ['f_fw_a_blk', 'Fraction of alpha and aux power deposited on the blanket FW', 0.91, 'N/A', None, 'Taken from some Bachmann crap'],
        ['eta_el_H2O', 'H2O pump electrical efficiency', 0.87, 'N/A', None, 'F. Cismondi'],
        ['eta_isen_H2O', 'H2O pump isentropic efficiency', 0.99, 'N/A', None, 'F. Cismondi'],
        ['f_pump_H2O_BB', 'BB pumping fraction for WC blanket', 0.004, 'N/A', None, 'F. Cismondi 08/12/16'],
        ['f_pump_H2O_DIV', 'DIV pumping fraction', 0.05, 'N/A', None, 'MC guess x 10-20 lit numbers for BB'],
        ['f_alpha_plasma', 'Fraction of charged particle power deposited in the plasma', 0.95, 'N/A', None, 'PROCESS reference value in 2019']
    ]
    # fmt: on

    def __init__(
        self,
        config,
        fusion_power_strat,
        charged_particle_strat,
        neutron_strat,
        blanket_pump_strat,
        divertor_pump_strat,
        bop_cycle_strat,
        parasitic_load_strat,
    ):
        self.params = config
        self.fusion_power_strat = fusion_power_strat
        self.charged_part_strat = charged_particle_strat
        self.neutron_strat = neutron_strat
        self.blanket_pump_strat = blanket_pump_strat
        self.divertor_pump_strat = divertor_pump_strat
        self.bop_strat = bop_cycle_strat
        self.parasitic_strat = parasitic_load_strat
        self.flow_dict = None

    def build(self):

        p_neutron, p_charged = self.fusion_power_strat.split(
            self.params.P_fus_DT, self.params.P_fus_DD
        )
        p_fusion = self.params.P_fus_DT + self.params.P_fus_DD
        p_radiation = self.params.P_rad
        p_hcd = self.params.P_hcd
        p_hcd_el = self.params.P_hcd_el
        p_separatrix = p_charged - p_radiation - p_hcd
        p_n_blk, p_n_div, p_n_vv, p_n_aux, p_nrgm = self.neutron_strat.split(p_neutron)
        p_rad_sep_blk, p_rad_sep_div, p_rad_sep_aux = self.charged_part_strat.split(
            p_radiation, p_separatrix
        )
        p_rad_sep_fw = p_rad_sep_blk + p_rad_sep_aux

        p_blanket = p_n_blk

        p_blk_pump, p_blk_pump_el = self.blanket_pump_strat.pump(p_blanket)
        p_blanket += p_blk_pump

        p_div = p_n_div
        p_div_pump, p_div_pump_el = self.divertor_pump_strat.pump(p_div)
        p_div += p_div_pump

        eta_bop = self.bop_strat.calculate(p_blanket, p_div)
        total_bop_power = p_blanket + p_div
        p_bop = eta_bop * total_bop_power
        p_bop_loss = total_bop_power - p_bop

        p_mag, p_cryo, p_t_plant, p_other = self.parasitic_strat.calculate(p_fusion)

        p_el_net = (
            p_bop
            - p_hcd_el
            - p_blk_pump_el
            - p_cryo
            - p_mag
            - p_other
            - p_t_plant
            - p_div_pump_el
        )

        self.flow_dict = {
            "Plasma": [p_fusion, p_hcd, -p_neutron, -p_separatrix - p_radiation],
            "H&CD": [p_hcd_el, -p_hcd, -(p_hcd_el - p_hcd)],
            "Neutrons": [p_neutron, p_nrgm, -p_n_blk, -p_n_div, -p_n_vv - p_n_aux],
            "Radiation and \nseparatrix": [
                p_radiation + p_separatrix,
                -p_rad_sep_fw,
                -p_rad_sep_div,
            ],
            "Blanket": [p_n_blk, p_rad_sep_blk, p_blk_pump, p_blk_decayheat, -p_blanket],
            "Divertor": [p_n_div, p_rad_sep_div, -p_div],
            "First wall": [p_rad_sep_fw, -p_rad_sep_aux, -p_rad_sep_blk],
            "BoP": [p_blanket, p_div + p_div_pump, -p_bop_loss, -p_bop],
            "Electricity": [
                p_bop,
                -p_t_plant,
                -p_other,
                -p_cryo,
                -p_mag,
                -p_hcd_el,
                -p_div_pump_el,
                -p_blk_pump_el,
                -p_el_net,
            ],
            "_H&CD loop": [p_hcd_el, -p_hcd_el],
            "_Divertor 2": [p_rad_sep_div, -(p_div - p_n_div)],
            "_H&CD loop 2": [p_hcd_el, -p_hcd_el],
            "_DIV to BOP": [p_div, p_div_pump, -p_div - p_div_pump],
            "_BB coolant loop turn": [
                p_blk_pump_el,
                -p_blk_pump_el + p_blk_pump,
                -p_blk_pump,
            ],
            "_BB coolant loop blanket": [p_blk_pump, -p_blk_pump],
            "_DIV coolant loop turn": [
                p_div_pump_el,
                -p_div_pump_el + p_div_pump,
                -p_div_pump,
            ],
            "_DIV coolant loop divertor": [p_div_pump, -p_div_pump],
        }

    def plot(self, **kwargs):
        plotter = self._plotter(**kwargs)
        return plotter.plot(self.flow_dict)


if __name__ == "__main__":

    neutron_power_strat = NeutronPowerStrategy(
        f_blanket=0.9, f_divertor=0.05, f_vessel=0.04, f_other=0.01
    )
    rad_sep_strat = RadChargedPowerStrategy(
        f_core_rad_fw=0.9,
        f_sol_rad=0.75,
        f_sol_rad_fw=0.8,
        f_sol_ch_fw=0.8,
        f_fw_blk=0.91,
    )
    blanket_pump_strat = HePumping(
        8, 7.5, 300, 500, eta_isentropic=0.9, eta_electric=0.87
    )
    bop_cycle = SuperheatedRankine(500)
    divertor_pump_strat = H2OPumping(f_pump=0.05, eta_isentropic=0.99, eta_electric=0.87)
    parasitic_load_strat = ParasiticLoadStrategy()

    bop = BalanceOfPlant(
        default_params,
        FusionPowerStrategy(),
        charged_particle_strat=rad_sep_strat,
        neutron_strat=neutron_power_strat,
        blanket_pump_strat=blanket_pump_strat,
        divertor_pump_strat=divertor_pump_strat,
        bop_cycle_strat=bop_cycle,
        parasitic_load_strat=parasitic_load_strat,
    )
