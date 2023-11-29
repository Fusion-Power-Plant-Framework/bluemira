# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import pytest

from bluemira.base.designer import Designer


class SimpleDesigner(Designer):
    param_cls = None

    def run(self) -> int:
        return 10

    def mock(self) -> int:
        return 11

    def read(self) -> int:
        return 12

    def custom_run_mode(self) -> int:
        return 13


class TestDesigner:
    def test_execute_calls_run_if_no_run_mode_in_build_config(self):
        designer = SimpleDesigner(None, {})

        assert designer.execute() == 10

    @pytest.mark.parametrize(
        ("mode", "output"),
        [("run", 10), ("mock", 11), ("read", 12), ("custom_run_mode", 13)],
    )
    def test_execute_calls_function_given_by_run_mode(self, mode, output):
        designer = SimpleDesigner(None, {"run_mode": mode})

        assert designer.execute() == output

    def test_ValueError_on_init_given_unknown_run_mode(self):
        with pytest.raises(ValueError):  # noqa: PT011
            SimpleDesigner(None, {"run_mode": "not_a_mode"}).execute()
