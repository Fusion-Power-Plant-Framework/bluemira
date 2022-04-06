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
import numpy as np
import pytest

from BLUEPRINT.systems.physicstoolbox import E_DD_fusion, E_DT_fusion, IPB98y2


class TestGCSEPhysics:
    def _msg(self, e, v):
        delta = e - v
        relate = "higher" if delta > 0 else "lower"
        return "E=mc^2 value {0:.2f} MeV {1} than Kikuchi " "reference.".format(
            delta * 1e-6, relate
        )

    def test_DT(self):  # noqa :N802
        e_dt_kikuchi = (3.5 + 14.1) * 1e6
        e, v = E_DT_fusion(), e_dt_kikuchi
        assert np.isclose(e, v, rtol=1e-3), self._msg(e, v)

    def test_DD(self):  # noqa :N802
        e_dd_kikuchi = np.array([1.01 + 3.02, 0.82 + 2.45]) * 1e6
        e, v = E_DD_fusion(), np.average(e_dd_kikuchi)
        assert np.isclose(e, v, rtol=1e-3), self._msg(e, v)


class TestTE:
    @pytest.mark.xfail
    def test_IPB(self):  # noqa :N802
        # This is not working because I never got the right values... couldn't
        # find the scaling law working backwards :/
        i_p = 15
        b_t = 5.3
        p_sep = 0.2 * 400 + 40
        n19 = 9.8
        r_0 = 6.2
        a = 3.1
        kappa = 1.85
        te = IPB98y2(i_p, b_t, p_sep, n19, r_0, a, kappa)
        assert np.isclose(te, 3.7, rtol=1e-3), f"{te:2f}"
