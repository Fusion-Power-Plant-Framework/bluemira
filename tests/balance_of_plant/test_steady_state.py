# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from unittest.mock import Mock

import pytest

from bluemira.balance_of_plant.steady_state import (
    BalanceOfPlantModel,
    H2OPumping,
    HePumping,
    RadChargedPowerStrategy,
    SuperheatedRankine,
)


class TestBalanceOfPlantModel:
    def setup_method(self):
        # Mock parameters and strategies with fixed values
        self.params = {
            "P_fus_DT": {"value": 100, "unit": "MW"},
            "P_fus_DD": {"value": 50, "unit": "MW"},
            "P_rad": {"value": 20, "unit": "MW"},
            "P_hcd_ss": {"value": 30, "unit": "MW"},
            "P_hcd_ss_el": {"value": 4, "unit": "MW"},
            "P_n_blanket": {"value": 80, "unit": "MW"},
            "P_n_divertor": {"value": 15, "unit": "MW"},
            "P_n_vessel": {"value": 2, "unit": "MW"},
            "P_n_aux": {"value": 5, "unit": "MW"},
            "P_n_e_mult": {"value": 3, "unit": "MW"},
            "P_n_decay": {"value": 1, "unit": "MW"},
        }

        # Mock strategies (no side effects)
        self.rad_sep_strat = RadChargedPowerStrategy(
            f_core_rad_fw=0.9,
            f_sol_rad=0.75,
            f_sol_rad_fw=0.8,
            f_sol_ch_fw=0.8,
            f_fw_aux=0.09,
        )

        self.blanket_pump_strat = HePumping(
            8e6, 7.5e6, 573.15, 773.15, eta_isentropic=0.9, eta_electric=0.87
        )

        self.divertor_pump_strat = H2OPumping(
            f_pump=0.05, eta_isentropic=0.99, eta_electric=0.87
        )

        self.bop_cycle_strat = SuperheatedRankine(bb_t_out=773.15, delta_t_turbine=20)

        self.parasitic_load_strat = Mock()
        self.parasitic_load_strat.calculate.return_value = (15e6, 8e6, 2e6, 3e6)

    def test_build(self):
        model = BalanceOfPlantModel(
            params=self.params,
            rad_sep_strat=self.rad_sep_strat,
            blanket_pump_strat=self.blanket_pump_strat,
            divertor_pump_strat=self.divertor_pump_strat,
            bop_cycle_strat=self.bop_cycle_strat,
            parasitic_load_strat=self.parasitic_load_strat,
        )
        model.build()

        # Check flow_dict construction
        assert len(model.flow_dict) == 17

        # Sanity check: global delta ≈ 0
        total_flow = sum(sum(flow) for flow in model.flow_dict.values())
        assert total_flow == pytest.approx(0.0, abs=1e-7)
