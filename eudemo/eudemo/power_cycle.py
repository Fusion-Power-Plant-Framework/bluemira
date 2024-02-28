# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Simple steady-state EU-DEMO balance of plant model
"""

import enum
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from bluemira.balance_of_plant.steady_state import (
    BalanceOfPlantModel,
    BoPModelParams,
    H2OPumping,
    HePumping,
    NeutronPowerStrategy,
    ParasiticLoadStrategy,
    PredeterminedEfficiency,
    RadChargedPowerStrategy,
    SuperheatedRankine,
)
from bluemira.base.parameter_frame import Parameter, ParameterFrame, make_parameter_frame
from bluemira.codes.interface import BaseRunMode, CodesSolver
from bluemira.codes.interface import CodesTask as Task

__all__ = ["SteadyStatePowerCycleSolver"]


class BlanketType(Enum):
    """Enumeration of blanket types."""

    HCPB = auto()
    WCLL = auto()

    @classmethod
    def _missing_(cls, value: str):
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(
                f"{cls.__name__} has no type {value}"
                f"please select from {*cls._member_names_, }"
            ) from None


@dataclass
class SteadyStatePowerCycleParams(ParameterFrame):
    """
    Steady-state power cycle solver parameter frame
    """

    P_fus_DT: Parameter[float]
    P_fus_DD: Parameter[float]
    P_rad: Parameter[float]
    P_hcd_ss: Parameter[float]
    P_hcd_ss_el: Parameter[float]
    vvpfrac: Parameter[float]
    e_mult: Parameter[float]
    e_decay_mult: Parameter[float]
    f_core_rad_fw: Parameter[float]
    f_sol_rad: Parameter[float]
    f_sol_rad_fw: Parameter[float]
    f_sol_ch_fw: Parameter[float]
    f_fw_aux: Parameter[float]
    blanket_type: Parameter[str]
    bb_p_inlet: Parameter[float]
    bb_p_outlet: Parameter[float]
    bb_t_inlet: Parameter[float]
    bb_t_outlet: Parameter[float]
    bb_pump_eta_isen: Parameter[float]
    bb_pump_eta_el: Parameter[float]
    div_pump_eta_isen: Parameter[float]
    div_pump_eta_el: Parameter[float]


class EUDEMOReferenceParasiticLoadStrategy(ParasiticLoadStrategy):
    """
    S. Ciattaglia reference point from the mid-2010's
    """

    def __init__(self):
        self.p_fusion_ref = 2037e6
        self.p_cryo = 44e6
        self.p_mag = 44e6
        self.p_t_plant = 15.5e6
        self.p_other = 31e6

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


class SteadyStatePowerCycleRunMode(BaseRunMode):
    """Enumeration of the run modes for the steady state power cycle solver"""

    RUN = enum.auto()


class SteadyStatePowerCycleSetup(Task):
    """
    Setup task for the steady-state power cycle model.
    """

    def run(self):
        """
        Run the setup task.
        """
        self.params = make_parameter_frame(self.params, SteadyStatePowerCycleParams)
        params = self.params  # avoid constant 'self' lookup
        # TODO: Get remaining hard-coded values hooked up
        neutron_power_strat = NeutronPowerStrategy(
            f_blanket=0.9,
            f_divertor=0.05,
            f_vessel=params.vvpfrac.value,  # TODO: Change this parameter name
            f_other=0.01,
            energy_multiplication=params.e_mult.value,
            decay_multiplication=params.e_decay_mult.value,
        )
        rad_sep_strat = RadChargedPowerStrategy(
            f_core_rad_fw=params.f_core_rad_fw.value,
            f_sol_rad=params.f_sol_rad.value,
            f_sol_rad_fw=params.f_sol_rad_fw.value,
            f_sol_ch_fw=params.f_sol_ch_fw.value,
            f_fw_aux=params.f_fw_aux.value,
        )
        blanket_type = BlanketType(params.blanket_type.value)
        if blanket_type is BlanketType.HCPB:
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
        elif blanket_type is BlanketType.WCLL:
            blanket_pump_strat = H2OPumping(
                0.005,
                eta_isentropic=params.bb_pump_eta_isen.value,
                eta_electric=params.bb_pump_eta_el.value,
            )
            bop_cycle = PredeterminedEfficiency(0.33)

        divertor_pump_strat = H2OPumping(
            f_pump=0.05,
            eta_isentropic=params.div_pump_eta_isen.value,
            eta_electric=params.div_pump_eta_el.value,
        )
        parasitic_load_strat = EUDEMOReferenceParasiticLoadStrategy()
        return (
            rad_sep_strat,
            neutron_power_strat,
            blanket_pump_strat,
            divertor_pump_strat,
            bop_cycle,
            parasitic_load_strat,
        )


class SteadyStatePowerCycleRun(Task):
    """
    Run task for the steady-state power cycle model.
    """

    def run(self, setup_result):
        """
        Run the run task. (o.O)
        """
        params = make_parameter_frame(self.params, BoPModelParams)
        bop = BalanceOfPlantModel(params, *setup_result)
        bop.build()
        return bop


class SteadyStatePowerCycleTeardown(Task):
    """
    Teardown task for the steady-state power cycle model.
    """

    @staticmethod
    def run(run_result):
        """
        Run the teardown task.
        """
        power_cycle = run_result
        flow_dict = power_cycle.flow_dict
        electricity = flow_dict["Electricity"]
        p_el_net = abs(electricity[-1])
        f_recirc = sum(np.abs(electricity[1:-1])) / abs(electricity[0])
        eta_ss = p_el_net / (flow_dict["Plasma"][0] + flow_dict["Neutrons"][1])
        return power_cycle, {
            "P_el_net": p_el_net,
            "eta_ss": eta_ss,
            "f_recirc": f_recirc,
        }


class SteadyStatePowerCycleSolver(CodesSolver):
    """
    Solver for the steady-state power cycle of an EU-DEMO reactor.
    """

    name = "SteadyStatePowerCycle"
    setup_cls = SteadyStatePowerCycleSetup
    run_cls = SteadyStatePowerCycleRun
    teardown_cls = SteadyStatePowerCycleTeardown
    run_mode_cls = SteadyStatePowerCycleRunMode

    def execute(self):
        """
        Execute the solver.
        """
        power_cycle, result = super().execute(SteadyStatePowerCycleRunMode.RUN)
        self.model = power_cycle
        return result
