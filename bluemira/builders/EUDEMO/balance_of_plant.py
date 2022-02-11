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
Simple steady-state EU-DEMO balance of plant model
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


def run_power_balance(params):
    neutron_power_strat = NeutronPowerStrategy(
        f_blanket=0.9,
        f_divertor=0.05,
        f_vessel=0.04,
        f_other=0.01,
        energy_multiplication=params.bb_e_mult.value,
    )
    rad_sep_strat = RadChargedPowerStrategy(
        f_core_rad_fw=0.9,
        f_sol_rad=0.75,
        f_sol_rad_fw=0.8,
        f_sol_ch_fw=0.8,
        f_fw_blk=0.91,
    )

    if params.blanket_type.value == "HCPB":
        blanket_pump_strat = HePumping(
            params.bb_p_inlet.value,
            params.bb_p_outlet.value,
            params.bb_t_inlet.value,
            params.bb_t_outlet.value,
            eta_isentropic=params.bb_pump_eta_isen.value,
            eta_electric=params.bb_pump_eta_el.value,
        )
        bop_cycle = SuperheatedRankine(
            bb_t_out=params.bb_t_outlet.value, delta_t_turbine=20
        )
    elif params.blanket_type.value == "WCLL":
        blanket_pump_strat = H2OPumping(
            0.005,
            eta_isentropic=params.bb_pump_eta_isen.value,
            eta_electric=params.bb_pump_eta_el.value,
        )
        bop_cycle = PredeterminedEfficiency(0.33)
    else:
        raise ValueError(f"Unrecognised blanket type {params.blanket_type.value}")

    bop = BalanceOfPlant(params, rad_sep_strat, neutron_power_strat, blanket_pump_strat)
