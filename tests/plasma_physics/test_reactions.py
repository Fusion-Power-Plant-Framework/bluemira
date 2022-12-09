# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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

from bluemira.plasma_physics.reactions import E_DD_fusion, E_DT_fusion, reactivity


class TestReactionEnergies:
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


class TestReactivity:
    temp = np.array(
        [0.2, 0.3]
    )  # , 0.4, 0.5, 0.6, 0.7, 0.8, 1., 1.25, 1.3, 1.5, 1.75, 1.8, 2., 2.5, 3., 4., 5., 6., 8., 10., 12., 15., 20., 30., 40., 50.])
    sv_DT = 1e-6 * np.array(
        [1.254e-26, 7.292e-25]
    )  # , 9.344e-24, 5.967e-23, 2.253e-22, 6.740e-22, 1.662e-21, 6.857e-21, 2.546e-20, 3.174e-20,6.923e-20, 1.539e-19, 1.773e-19, 2.977e-19, 8.425e-19, 1.867e-18, 5.974e-18, 1.366e-17, 2.554e-17, 6.222e-17, 1.136e-16, 1.747e-16, 2.74e-16, 4.33e-16, 6.681e-16, 7.998e-16, 8.649e-16])

    @pytest.mark.parametrize("temp_kev, sigmav", np.c_[temp, sv_DT])
    def test_Bosch_Hale_DT(self, temp_kev, sigmav):
        result = reactivity(temp_kev, reaction="D-T", method="Bosch-Hale")
        np.testing.assert_approx_equal(result, sigmav, 8)
