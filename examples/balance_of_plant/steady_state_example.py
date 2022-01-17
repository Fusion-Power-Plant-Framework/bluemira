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
Simple example of a 0-D steady-state balance of plant view.
"""

from bluemira.balance_of_plant.steady_state import (
    BalanceOfPlant,
    H2OPumping,
    HePumping,
    NeutronPowerStrategy,
    ParasiticLoadStrategy,
    PredeterminedEfficiency,
    RadChargedPowerStrategy,
    SuperheatedRankine,
)
from bluemira.base.parameter import ParameterFrame

# fmt: off
default_params = ParameterFrame([
    ['P_fus_DT', 'D-T fusion power', 1995, 'MW', None, 'Input'],
    ['P_fus_DD', 'D-D fusion power', 5, 'MW', None, 'Input'],
    ['P_rad', 'Radiation power', 400, 'MW', None, 'Input'],
    ['P_hcd', "Heating and current drive power", 50, "MW", None, 'Input'],
    ['P_hcd_el', "Heating and current drive electrical power", 150, "MW", None, 'Input'],
    ['P_bb_decay', 'Blanket decay heat', 30, 'MW', None, 'Input'],
])
# fmt: on

neutron_power_strat = NeutronPowerStrategy(
    f_blanket=0.9,
    f_divertor=0.05,
    f_vessel=0.04,
    f_other=0.01,
    energy_multiplication=1.35,
)
rad_sep_strat = RadChargedPowerStrategy(
    f_core_rad_fw=0.9,
    f_sol_rad=0.75,
    f_sol_rad_fw=0.8,
    f_sol_ch_fw=0.8,
    f_fw_blk=0.91,
)
blanket_pump_strat = HePumping(8, 7.5, 300, 500, eta_isentropic=0.9, eta_electric=0.87)
bop_cycle = SuperheatedRankine(bb_t_out=500, delta_t_turbine=20)
divertor_pump_strat = H2OPumping(f_pump=0.05, eta_isentropic=0.99, eta_electric=0.87)
parasitic_load_strat = ParasiticLoadStrategy()

HCPB_bop = BalanceOfPlant(
    default_params,
    rad_sep_strat=rad_sep_strat,
    neutron_strat=neutron_power_strat,
    blanket_pump_strat=blanket_pump_strat,
    divertor_pump_strat=divertor_pump_strat,
    bop_cycle_strat=bop_cycle,
    parasitic_load_strat=parasitic_load_strat,
)
HCPB_bop.build()
HCPB_bop.plot(title="HCPB blanket")


blanket_pump_strat = H2OPumping(0.005, eta_isentropic=0.99, eta_electric=0.87)
bop_cycle = PredeterminedEfficiency(0.33)

WCLL_bop = BalanceOfPlant(
    default_params,
    rad_sep_strat=rad_sep_strat,
    neutron_strat=neutron_power_strat,
    blanket_pump_strat=blanket_pump_strat,
    divertor_pump_strat=divertor_pump_strat,
    bop_cycle_strat=bop_cycle,
    parasitic_load_strat=parasitic_load_strat,
)
WCLL_bop.build()
WCLL_bop.plot(title="WCLL blanket")
