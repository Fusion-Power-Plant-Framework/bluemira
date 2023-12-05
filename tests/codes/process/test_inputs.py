# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from bluemira.codes.process._inputs import ProcessInputs
from bluemira.codes.process.api import _INVariable


class TestProcessInput:
    def setup_method(self):
        self.df = ProcessInputs()

    def test_to_invariable(self):
        for k, v in self.df.to_invariable().items():
            assert k == v.name
            assert isinstance(v, _INVariable)
            if k not in {"icc", "ixc", "bounds"}:
                assert v.v_type == "Parameter"
            elif k == "icc":
                assert v.v_type == "Constraint Equation"
            elif k == "ixc":
                assert v.v_type == "Iteration Variable"
            elif k == "bounds":
                assert v.v_type == "Bound"
