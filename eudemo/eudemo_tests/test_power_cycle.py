# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later


from bluemira.base.parameter_frame import Parameter
from eudemo.power_cycle import SteadyStatePowerCycleParams, SteadyStatePowerCycleSolver


class TestEUDEMOPowerCycle:
    params = SteadyStatePowerCycleParams(
        Parameter("P_fus_DT", 2000e6, "W", source="test"),
        Parameter("P_fus_DD", 4e6, "W", source="test"),
        Parameter("P_rad", 400e6, "W", source="test"),
        Parameter("P_hcd_ss", 50e6, "W", source="test"),
        Parameter("P_hcd_ss_el", 150e6, "W", source="test"),
        Parameter("vvpfrac", 0.04, "", source="test"),
        Parameter("e_mult", 1.35, "", source="test"),
        Parameter("e_decay_mult", 1.015, "", source="test"),
        Parameter("f_core_rad_fw", 0.3, "", source="test"),
        Parameter("f_sol_rad", 0.2, "", source="test"),
        Parameter("f_sol_rad_fw", 0.9, "", source="test"),
        Parameter("f_sol_ch_fw", 0.2, "", source="test"),
        Parameter("f_fw_aux", 0.05, "", source="test"),
        Parameter("blanket_type", "HCPB", "", source="test"),
        Parameter("bb_p_inlet", 8e6, "Pa", source="test"),
        Parameter("bb_p_outlet", 7.5e6, "Pa", source="test"),
        Parameter("bb_t_inlet", 573, "K", source="test"),
        Parameter("bb_t_outlet", 773, "K", source="test"),
        Parameter("bb_pump_eta_isen", 0.85, "", source="test"),
        Parameter("bb_pump_eta_el", 0.9, "", source="test"),
        Parameter("div_pump_eta_isen", 0.99, "", source="test"),
        Parameter("div_pump_eta_el", 0.98, "", source="test"),
    )

    @classmethod
    def setup_class(cls):
        cls.sspc_solver = SteadyStatePowerCycleSolver(cls.params)
        cls.sspc_result = cls.sspc_solver.execute()

    def test_plotting(self):
        self.sspc_solver.model.plot()

    def test_values(self):
        assert self.sspc_result["P_el_net"] > 0
        assert self.sspc_result["P_el_net"] / 1e6 > 1
