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

from bluemira.codes.process._inputs import ProcessInputs
from bluemira.codes.process.api import _INVariable


class TestProcessInput:
    def setup_method(self):
        self.df = ProcessInputs()

    def test_to_invariable(self):
        for k, v in self.df.to_invariable().items():
            assert k == v.name
            assert isinstance(v, _INVariable)
            if k not in ["icc", "ixc", "bounds"]:
                assert v.v_type == "Parameter"
            elif k == "icc":
                assert v.v_type == "Constraint Equation"
            elif k == "ixc":
                assert v.v_type == "Iteration Variable"
            elif k == "bounds":
                assert v.v_type == "Bound"
