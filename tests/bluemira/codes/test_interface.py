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

from unittest.mock import MagicMock, Mock

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
        e_dict = EvilDict(shell=True)  # noqa :S604
        with pytest.raises((FileNotFoundError, TypeError)):
            task._run_subprocess("random command", **e_dict)
        assert e_dict["shell"]
        with pytest.raises(FileNotFoundError):
            task._run_subprocess("random command", shell=e_dict["shell"])  # noqa :S604


class TestSetup:

    _remapper_dict = {"a": "a", "b": ["b", "d"], "c": "c"}

    def _remapper(val):  # noqa: N805
        if val == "b":
            return ["b", "d"]
        else:
            return val

    def test_get_new_input_raises_TypeError(self):
        with pytest.raises(TypeError):
            interface.Setup.get_new_inputs(MagicMock(), "hello")

    @pytest.mark.parametrize("remapper", [_remapper_dict, _remapper])
    def test_get_new_input_many_from_one(self, remapper):
        fake_self = MagicMock()
        fake_self._send_mapping = {"a": "b", "b": "c", "c": "d"}
        fake_self._convert_units = lambda x: x
        fake_self.params.get_param = lambda x: x
        inputs = interface.Setup.get_new_inputs(fake_self, remapper)
        assert inputs == {"a": "b", "b": "c", "d": "c", "c": "d"}


class TestFileProgramInterface:
    def test_modify_mappings(self, caplog):
        my_self = MagicMock()
        sr = MagicMock()
        sr.send = False
        sr.recv = True
        my_self.NAME = "TestProgram"
        my_self.params.test_key.mapping = {my_self.NAME: sr}
        my_self.params.test_key2.mapping = {}
        my_self.params.otherkey = Mock(spec=[])  # to raise AttributeError

        interface.FileProgramInterface.modify_mappings(
            my_self, {"test_key2": {"send": False, "recv": True}}
        )
        assert len(caplog.messages) == 1
        interface.FileProgramInterface.modify_mappings(
            my_self, {"otherkey": {"send": False, "recv": True}}
        )
        assert len(caplog.messages) == 2

        assert not my_self.params.test_key.mapping[my_self.NAME].send
        assert my_self.params.test_key.mapping[my_self.NAME].recv

        interface.FileProgramInterface.modify_mappings(
            my_self, {"test_key": {"send": True, "recv": False}}
        )

        assert my_self.params.test_key.mapping[my_self.NAME].send
        assert not my_self.params.test_key.mapping[my_self.NAME].recv
