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
    """
    H.-S. Bosch and G.M. Hale 1993 Nucl. Fusion 33 1919
    """

    temp = np.array(
        [
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            1.0,
            1.25,
            1.3,
            1.5,
            1.75,
            1.8,
            2.0,
            2.5,
            3.0,
            4.0,
            5.0,
            6.0,
            8.0,
            10.0,
            12.0,
            15.0,
            20.0,
            30.0,
            40.0,
            50.0,
        ]
    )
    sv_DT = 1e-6 * np.array(
        [
            1.254e-26,
            7.292e-25,
            9.344e-24,
            5.697e-23,
            2.253e-22,
            6.740e-22,
            1.662e-21,
            6.857e-21,
            2.546e-20,
            3.174e-20,
            6.923e-20,
            1.539e-19,
            1.773e-19,
            2.977e-19,
            8.425e-19,
            1.867e-18,
            5.974e-18,
            1.366e-17,
            2.554e-17,
            6.222e-17,
            1.136e-16,
            1.747e-16,
            2.74e-16,
            4.33e-16,
            6.681e-16,
            7.998e-16,
            8.649e-16,
        ]
    )

    sv_DHe3 = 1e-6 * np.array(
        [
            1.414e-35,
            1.033e-32,
            6.537e-31,
            1.241e-29,
            1.166e-28,
            6.960e-28,
            3.032e-27,
            3.057e-26,
            2.590e-25,
            3.708e-25,
            1.317e-24,
            4.813e-24,
            6.053e-24,
            1.399e-23,
            7.477e-23,
            2.676e-22,
            1.710e-21,
            6.377e-21,
            1.739e-20,
            7.504e-20,
            2.126e-19,
            4.715e-19,
            1.175e-18,
            3.482e-18,
            1.363e-17,
            3.160e-17,
            5.554e-17,
        ]
    )

    sv_DD_He3p = 1e-6 * np.array(
        [
            4.482e-28,
            2.004e-26,
            2.168e-25,
            1.169e-24,
            4.200e-24,
            1.162e-23,
            2.681e-23,
            9.933e-23,
            3.319e-22,
            4.660e-22,  # OMG
            8.284e-22,
            1.713e-21,
            1.948e-21,
            3.110e-21,
            7.905e-21,
            1.602e-20,
            4.447e-20,
            9.128e-20,
            1.573e-19,
            3.457e-19,
            6.023e-19,
            9.175e-19,
            1.481e-18,
            2.603e-18,
            5.271e-18,
            8.235e-18,
            1.133e-17,
        ]
    )

    sv_DD_Tp = 1e-6 * np.array(
        [
            4.640e-28,
            2.071e-26,
            2.237e-25,
            1.204e-24,
            4.321e-24,
            1.193e-23,
            2.751e-23,
            1.017e-22,
            3.387e-22,
            4.143e-22,
            8.431e-22,
            1.739e-21,
            1.976e-21,
            3.150e-21,
            7.969e-21,
            1.608e-20,
            4.428e-20,
            9.024e-20,
            1.545e-19,
            3.354e-19,
            5.781e-19,
            8.723e-19,
            1.390e-18,
            2.399e-18,
            4.728e-18,
            7.249e-18,
            9.838e-18,
        ]
    )

    @pytest.mark.parametrize("temp_kev, sigmav", np.c_[temp, sv_DT])
    def test_Bosch_Hale_DT(self, temp_kev, sigmav):
        result = reactivity(temp_kev, reaction="D-T", method="Bosch-Hale")
        np.testing.assert_allclose(result, sigmav, rtol=0.0025, atol=0)

    @pytest.mark.parametrize("temp_kev, sigmav", np.c_[temp, sv_DHe3])
    def test_Bosch_Hale_DHe(self, temp_kev, sigmav):
        result = reactivity(temp_kev, reaction="D-He3", method="Bosch-Hale")
        np.testing.assert_allclose(result, sigmav, rtol=0.003, atol=0)

    @pytest.mark.parametrize("temp_kev, sigmav", np.c_[temp, sv_DD_He3p])
    def test_Bosch_Hale_DD_He3p(self, temp_kev, sigmav):
        result = reactivity(temp_kev, reaction="D-D1", method="Bosch-Hale")
        np.testing.assert_allclose(result, sigmav, rtol=0.003, atol=0)

    @pytest.mark.parametrize("temp_kev, sigmav", np.c_[temp, sv_DD_Tp])
    def test_Bosch_Hale_DD_Tp(self, temp_kev, sigmav):
        result = reactivity(temp_kev, reaction="D-D2", method="Bosch-Hale")
        np.testing.assert_allclose(result, sigmav, rtol=0.0035, atol=0)
