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

from dataclasses import dataclass
from typing import Dict, Union

import numpy as np

from bluemira.balance_of_plant.steady_state import (
    BalanceOfPlantModel,
    H2OPumping,
    HePumping,
    NeutronPowerStrategy,
    ParasiticLoadStrategy,
    PredeterminedEfficiency,
    RadChargedPowerStrategy,
    SuperheatedRankine,
)

__all__ = ["SteadyStatePowerCycle"]


class EUDEMOReferenceParasiticLoadStrategy(ParasiticLoadStrategy):
    """
    S. Ciattaglia reference point from the mid-2010's
    """

    def __init__(self):
        self.p_fusion_ref = 2037
        self.p_cryo = 44
        self.p_mag = 44
        self.p_t_plant = 15.5
        self.p_other = 31

    def calculate(self, p_fusion):
        """
        Because we were told to do this. Nobody trusts models.
        """
        f_norm = p_fusion / self.p_fusion_ref
        p_mag = f_norm * self.p_mag
        p_cryo = f_norm * self.p_cryo
        p_t_plant = f_norm * self.p_t_plant
        p_other = f_norm * self.p_other
        return p_mag, p_cryo, p_t_plant, p_other


@dataclass
class SteadyStatePowerCycleParams:

    bb_p_inlet: float
    bb_p_outlet: float
    bb_pump_eta_el: float
    bb_pump_eta_isen: float
    bb_t_inlet: float
    bb_t_outlet: float
    blanket_type: str
    div_pump_eta_el: float
    div_pump_eta_isen: float
    e_decay_mult: float
    e_mult: float
    f_core_rad_fw: float
    f_fw_aux: float
    f_sol_ch_fw: float
    f_sol_rad_fw: float
    f_sol_rad: float
    vvpfrac: float


class SteadyStatePowerCycle:
    """
    Calculator for the steady-state power cycle of an EU-DEMO reactor.
    """

    def __init__(self, params: Union[Dict, SteadyStatePowerCycleParams]):
        if isinstance(params, dict):
            self.params = SteadyStatePowerCycleParams(**params)
        elif isinstance(params, SteadyStatePowerCycleParams):
            self.params = params
        else:
            raise TypeError(
                "Invalid type for params. "
                "Must be 'dict' or 'SteadyStatePowerCycleParams',"
                f" found '{type(params).__name__}'."
            )

        self.neutron_power_strat = NeutronPowerStrategy(
            f_blanket=0.9,
            f_divertor=0.05,
            f_vessel=params.vvpfrac,  # TODO: Change this parameter name
            f_other=0.01,
            energy_multiplication=params.e_mult,
            decay_multiplication=params.e_decay_mult,
        )
        self.rad_sep_strat = RadChargedPowerStrategy(
            f_core_rad_fw=params.f_core_rad_fw,
            f_sol_rad=params.f_sol_rad,
            f_sol_rad_fw=params.f_sol_rad_fw,
            f_sol_ch_fw=params.f_sol_ch_fw,
            f_fw_aux=params.f_fw_aux,
        )

        if params.blanket_type == "HCPB":
            self.blanket_pump_strat = HePumping(
                params.bb_p_inlet,
                params.bb_p_outlet,
                params.bb_t_inlet,
                params.bb_t_outlet,
                eta_isentropic=params.bb_pump_eta_isen,
                eta_electric=params.bb_pump_eta_el,
            )
            self.bop_cycle = SuperheatedRankine(
                bb_t_out=params.bb_t_outlet, delta_t_turbine=20
            )
        elif params.blanket_type == "WCLL":
            self.blanket_pump_strat = H2OPumping(
                0.005,
                eta_isentropic=params.bb_pump_eta_isen,
                eta_electric=params.bb_pump_eta_el,
            )
            self.bop_cycle = PredeterminedEfficiency(0.33)
        else:
            raise ValueError(f"Unrecognised blanket type {params.blanket_type}")

        self.divertor_pump_strat = H2OPumping(
            f_pump=0.05,
            eta_isentropic=params.div_pump_eta_isen,
            eta_electric=params.div_pump_eta_el,
        )
        self.parasitic_load_strat = EUDEMOReferenceParasiticLoadStrategy()

    def run(self):
        """Run the balance of plant model."""
        power_cycle = BalanceOfPlantModel(
            self.params,
            rad_sep_strat=self.rad_sep_strat,
            neutron_power_strat=self.neutron_power_strat,
            blanket_pump_strat=self.blanket_pump_strat,
            divertor_pump_strat=self.divertor_pump_strat,
            bop_cycle=self.bop_cycle,
            parasitic_load_strat=self.parasitic_load_strat,
        )
        power_cycle.build()

        flow_dict = power_cycle.flow_dict
        electricity = flow_dict["Electricity"]
        p_el_net = abs(electricity[-1])
        f_recirc = np.sum(np.abs(electricity[1:-1])) / abs(electricity[0])
        eta_ss = p_el_net / (flow_dict["Plasma"][0] + flow_dict["Neutrons"][1])
        return power_cycle, {
            "P_el_net": p_el_net,
            "eta_ss": eta_ss,
            "f_recirc": f_recirc,
        }
