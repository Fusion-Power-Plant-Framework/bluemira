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

import os
import re
import shutil
import tempfile

import numpy as np
import pytest

from bluemira.utilities.opt_variables import (
    BoundedVariable,
    OptVariables,
    OptVariablesError,
)


class TestBoundedVariable:
    def test_initialisation(self):
        v1 = BoundedVariable("a", 2, 0, 3, descr="test")
        assert v1.name == "a"
        assert v1.value == 2
        assert v1.lower_bound == 0
        assert v1.upper_bound == 3
        assert v1.description == "test"

        v2 = BoundedVariable("b", 0, -1, 1)
        assert v2.name == "b"
        assert v2.value == 0
        assert v2.lower_bound == -1
        assert v2.upper_bound == 1
        assert v2.description is None

        with pytest.raises(OptVariablesError):
            v3 = BoundedVariable("a", 2, 2.5, 3)

    def test_normalised_value(self):
        v1 = BoundedVariable("a", 2, 0, 4)
        assert v1.normalised_value == 0.5

    def test_adjust(self):
        v1 = BoundedVariable("a", 2, 0, 4)
        v1.adjust(value=3, lower_bound=2, upper_bound=4)
        assert v1.value == 3
        assert v1.lower_bound == 2
        assert v1.upper_bound == 4

    def test_adjust_bad(self):
        v1 = BoundedVariable("a", 2, 0, 4)
        with pytest.raises(OptVariablesError):
            v1.adjust(value=0, lower_bound=1, upper_bound=2)
        with pytest.raises(OptVariablesError):
            v1.adjust(lower_bound=2, upper_bound=-1)
        with pytest.raises(OptVariablesError):
            v1.adjust(value=-2, lower_bound=-4, upper_bound=-3)

    def test_adjust_fixed(self):
        v1 = BoundedVariable("a", 2, 0, 4, fixed=True)
        assert v1.value == 2
        with pytest.raises(OptVariablesError):
            v1.adjust(value=3)
        with pytest.raises(OptVariablesError):
            v1.adjust(upper_bound=9)
        with pytest.raises(OptVariablesError):
            v1.value = 3


class TestOptVariables:
    def setup_method(self):
        v1 = BoundedVariable("a", 2, 0, 3)
        v2 = BoundedVariable("b", 0, -1, 1)
        v3 = BoundedVariable("c", -1, -10, 10)
        self.vars = OptVariables([v1, v2, v3])
        v1 = BoundedVariable("a", 2, 0, 3)
        v2 = BoundedVariable("b", 0, -1, 1)
        v3 = BoundedVariable("c", -1, -10, 10)
        self.vars_frozen = OptVariables([v1, v2, v3], frozen=True)

    def test_init(self):
        assert self.vars.n_free_variables == 3
        assert len(self.vars.values) == 3
        assert np.allclose(self.vars.values, np.array([2, 0, -1]))

    def test_add(self):
        v4 = BoundedVariable("d", 4, -4, 6, fixed=False)
        v5 = BoundedVariable("e", 1, -1, 2, fixed=True)
        self.vars.add_variable(v4)
        self.vars.add_variable(v5)
        assert self.vars.n_free_variables == 4
        assert len(self.vars.values) == 5
        assert np.allclose(self.vars.values, np.array([2, 0, -1, 4, 1]))

    def test_remove(self):
        self.vars.remove_variable("c")
        assert len(self.vars.values) == 2
        assert np.allclose(self.vars.values, np.array([2, 0]))

    def test_fix(self):
        self.vars.fix_variable("a", value=100)
        assert self.vars.values[0] == 100
        assert self.vars.n_free_variables == 2
        assert np.allclose(self.vars.values, np.array([100, 0, -1]))

    def test_getitem(self):
        assert self.vars["a"] == self.vars._var_dict["a"]

    def test_frozen(self):
        with pytest.raises(OptVariablesError):
            self.vars_frozen.add_variable(BoundedVariable("new", 0, 0, 0))
        with pytest.raises(OptVariablesError):
            self.vars_frozen.remove_variable("a")

    def test_adjust(self):
        self.vars.adjust_variable("b", fixed=True)
        self.vars_frozen.adjust_variable("b", fixed=True)
        assert self.vars["b"].fixed
        assert self.vars_frozen["b"].fixed

    def test_not_strict_bounds(self):
        v1 = BoundedVariable("a", 2, 0, 3)
        v2 = BoundedVariable("b", 0, -1, 1)
        v3 = BoundedVariable("c", -1, -10, 10)
        opt_vars = OptVariables([v1, v2, v3])

        opt_vars.adjust_variables({"a": {"value": -2}}, strict_bounds=False)

        assert opt_vars["a"].value == -2
        assert opt_vars["a"].lower_bound == -2

    def test_not_strict_bounds2(self):
        v1 = BoundedVariable("a", 2, 0, 3)
        v2 = BoundedVariable("b", 0, -1, 1)
        v3 = BoundedVariable("c", -1, -10, 10)
        opt_vars = OptVariables([v1, v2, v3])

        opt_vars.adjust_variables(
            {"a": {"value": -2, "lower_bound": -1, "upper_bound": 3}},
            strict_bounds=False,
        )
        assert opt_vars["a"].value == -2
        assert opt_vars["a"].lower_bound == -2

        with pytest.raises(OptVariablesError):
            opt_vars.adjust_variables(
                {"a": {"value": -2, "lower_bound": -1, "upper_bound": -3}},
                strict_bounds=False,
            )

    def test_read_write(self):
        tempdir = tempfile.mkdtemp()
        try:
            the_path = os.sep.join([tempdir, "opt_var_test.json"])
            self.vars.to_json(the_path)
            new_vars = OptVariables.from_json(the_path)
            assert new_vars.as_dict() == self.vars.as_dict()
        finally:
            shutil.rmtree(tempdir)

    def test_tabulate_displays_column_values(self):
        v1 = BoundedVariable("a", 2, 0, 3, descr="var a")
        v2 = BoundedVariable("b", 0, -1, 1, fixed=True)
        v3 = BoundedVariable("c", -1, -10, 10)
        opt_vars = OptVariables([v3, v1, v2])

        table = opt_vars.tabulate()

        table_pattern = "\n".join(
            [
                ".*",
                ".*Name.*Value.*Lower Bound.*Upper Bound.*Fixed.*Description.*",
                ".*",
                ".*a.*2.*0.*3.*False.*var a.*",
                ".*",
                ".*b.*0.*-1.*1.*True.* .*",
                ".*",
                ".*c.*-1.*-10.*10.*False.* .*",
                ".*",
            ]
        )
        assert len(table.split("\n")) == 9
        assert re.match(table_pattern, table, flags=re.MULTILINE)
