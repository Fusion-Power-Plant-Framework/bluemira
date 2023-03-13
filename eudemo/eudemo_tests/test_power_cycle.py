# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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

import matplotlib.pyplot as plt

from bluemira.base.parameter_frame import Parameter
from eudemo.power_cycle import SteadyStatePowerCycleParams, SteadyStatePowerCycleSolver


class TestEUDEMOPowerCycle:
    params = SteadyStatePowerCycleParams(
        Parameter("P_fus_DT", 2000, "W", source="test"),
        Parameter("P_fus_DD", 4, "W", source="test"),
        Parameter("P_rad", 400, "W", source="test"),
        Parameter("P_hcd_ss", 50, "W", source="test"),
        Parameter("P_hcd_ss_el", 150, "W", source="test"),
        Parameter("vvpfrac", 0.04, "", source="test"),
        Parameter("e_mult", 1.35, "", source="test"),
        Parameter("e_decay_mult", 1.015, "", source="test"),
        Parameter("f_core_rad_fw", 0.3, "", source="test"),
        Parameter("f_sol_rad", 0.2, "", source="test"),
        Parameter("f_sol_rad_fw", 0.9, "", source="test"),
        Parameter("f_sol_rad_ch", 0.2, "", source="test"),
        Parameter("f_fw_aux", 0.05, "", source="test"),
        Parameter("blanket_type", "HCPB", "", source="test"),
        Parameter("bb_p_inlet", 8e6, "Pa", source="test"),
        Parameter("bb_p_outlet", 5e6, "Pa", source="test"),
        Parameter("bb_t_inlet", 573, "K", source="test"),
        Parameter("bb_t_outlet", 873, "K", source="test"),
        Parameter("bb_pump_eta_isen", 0.85, "", source="test"),
        Parameter("bb_pump_eta_el", 0.9, "", source="test"),
        Parameter("div_pump_eta_isen", 0.99, "", source="test"),
        Parameter("div_pump_eta_el", 0.98, "", source="test"),
    )

    def test_solver(self):
        sspc_solver = SteadyStatePowerCycleSolver(self.params)
        sspc_result = sspc_solver.execute()
        sspc_solver.model.plot()
        plt.show()
