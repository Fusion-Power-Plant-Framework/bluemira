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

from bluemira.base.parameter import ParameterFrame
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
        return power + p_pump, p_electric


class H2OPumping(CoolantPumping):
    def __init__(self, f_pump, eta_isentropic, eta_electric):
        self.f_pump = f_pump
        self.eta_isen = eta_isentropic
        self.eta_el = eta_electric

    def pump(self, power):
        p_pump, p_electric = H2O_pumping(power, self.f_pump, self.eta_isen, self.eta_el)
        return power + p_pump, p_electric


class PowerCycleEfficiencyCalc(abc.ABC):
    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def calculate(self, *args) -> float:
        pass


class PredeterminedEfficiency(PowerCycleEfficiencyCalc):
    def __init__(self, efficiency):
        self.efficiency

    def calculate(self) -> float:
        return self.efficiency


class SuperheatedRankine(PowerCycleEfficiencyCalc):
    def __init__(self, bb_t_out):
        self.bb_t_out = bb_t_out

    def calculate(self, p_blanket, p_divertor) -> float:
        return superheated_rankine(p_blanket, p_divertor, self.bb_t_out)


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

    def build(self):
        pass

    def plot(self, figsize=(14, 10), facecolor="k"):
        plotter = self._plotter(figsize, facecolor)
        return self._plotter.plot(self.flow_dict)
