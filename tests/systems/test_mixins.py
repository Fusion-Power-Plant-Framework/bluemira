# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.
import pytest

"""
Created on Fri Aug  2 07:51:11 2019

@author: matti
"""
import numpy as np
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.shell import Shell
from BLUEPRINT.systems.mixins import UpperPort


class TestPort:
    alpha = np.radians(45)
    beta = 2 * np.radians(90) - alpha
    lil_t = np.tan(alpha / 2) * 0.2
    big_t = (1 / np.tan(alpha) + 1 / np.sin(alpha)) * 0.2
    correct_in = np.array([[6, 8, 8, 6, 6], [-1, -3, 3, 1, -1]])
    correct_out = np.array(
        [
            [5.8, 8.2, 8.2, 5.8, 5.8],
            [-1 - lil_t, -3 - big_t, 3 + big_t, 1 + lil_t, -1 - lil_t],
        ]
    )
    tol = 0.001

    def check_pass(self, port):
        return (port.inner.d2 - self.correct_in < self.tol).all() and (
            port.outer.d2 - self.correct_out < self.tol
        ).all()

    @staticmethod
    def _build_port(x, y, t):
        loop = Loop(x=x, y=y)
        shell = Shell.from_offset(loop, t)
        return UpperPort(shell, 6, 8, 5.8, 10, 0.1)

    def test_good_port(self):
        port = self._build_port([5.5, 10, 10, 5.5], [-0.5, -5, 5, 0.5], 0.2)
        assert self.check_pass(port)

    def test_bad_port(self):
        port = self._build_port([4, 10, 10, 4], [1, -5, 5, -1], 0.2)
        assert self.check_pass(port)

    def test_upper_port_z(self):
        loop = Loop(x=[4, 10, 10, 4], y=[1, -5, 5, -1], z=9.5)
        shell = Shell.from_offset(loop, 0.2)
        port = UpperPort(shell, 6, 8, 5.8, 10, 0.1)
        assert port.inner.z[0] == 9.5


if __name__ == "__main__":
    pytest.main([__file__])
