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

from bluemira.base.config import ParameterFrame
from bluemira.base.parameter import ParameterMapping
from bluemira.base.solver import NoOpTask, RunMode
from bluemira.codes import interface


class NoOpRunMode(RunMode):
    RUN = 0


class NoOpSolver(interface.CodesSolver):

    name = "MyTestSolver"
    setup_cls = NoOpTask
    run_cls = NoOpTask
    teardown_cls = NoOpTask
    run_mode_cls = NoOpRunMode


class TestCodesSolver:
    def test_modify_mappings_updates_send_recv_values_of_params(self):
        params = ParameterFrame()
        params.add_parameter(
            "param1",
            unit="dimensionless",
            value=1,
            mapping={NoOpSolver.name: ParameterMapping("param1", recv=False, send=True)},
        )
        solver = NoOpSolver(params)

        solver.modify_mappings({"param1": {"recv": True, "send": False}})

        assert solver.params.param1.mapping[solver.name].recv is True
        assert solver.params.param1.mapping[solver.name].send is False
