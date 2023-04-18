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
import numpy as np

from eudemo.blanket.panelling._paneller import norm_tangents


def test_tangent_returns_tangent_vectors():
    # fmt: off
    xz = np.array([
        [0.34558419, 0.82161814, 0.33043708, -1.30315723, 0.90535587, 0.44637457,
         -0.53695324, 0.5811181, 0.3645724, 0.2941325, 0.02842224, 0.54671299],
        [-0.73645409, -0.16290995, -0.48211931, 0.59884621, 0.03972211, -0.29245675,
         -0.78190846, -0.25719224, 0.00814218, -0.27560291, 1.29406381, 1.00672432],
    ])
    # fmt: on

    tngnt = norm_tangents(xz)

    # fmt: off
    expected = np.array([
        [0.63866332, -0.05945048, -0.94133318, 0.74046041, 0.8910329, -0.86890311,
         0.96741701, 0.75207387, -0.9979486, -0.25290957, 0.19325713, 0.87458661],
        [0.7694863, 0.99823126, 0.33747866, 0.67209998, -0.45393874, -0.49498221,
         0.25318831, 0.65907882, -0.06402027, 0.96748992, 0.98114814, -0.48486932],
    ])
    # fmt: on
    np.testing.assert_array_almost_equal(tngnt, expected, decimal=7)
