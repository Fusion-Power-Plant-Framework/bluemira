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

from unittest.mock import MagicMock
import pytest

from bluemira.codes import interface


class EvilDict(dict):
    def __init__(self, *arg, **kw):
        super().__init__(*arg, **kw)

    def pop(self, *args, **kw):
        pass


class TestTask:
    def test_protected_subprocess(self):
        parent = MagicMock()
        parent._run_dir = "./"
        parent.NAME = "TEST"
        task = interface.Task(parent)
        e_dict = EvilDict(shell=True)  # noqa (S604)
        with pytest.raises((FileNotFoundError, TypeError)):
            task._run_subprocess("random command", **e_dict)
        assert e_dict["shell"]
        with pytest.raises(FileNotFoundError):
            task._run_subprocess("random command", shell=e_dict["shell"])  # noqa (S604)
