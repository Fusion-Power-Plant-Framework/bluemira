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

from bluemira.base.constants import raw_uc
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.tools import make_polygon
from bluemira.magnetostatics.polyhedral_prism import polyhedral_discretised_sources
from bluemira.magnetostatics.trapezoidal_prism import TrapezoidalPrismCurrentSource


def test_benchmark():
    """
    Verification test.

    Benchmarked against cube using trapezoidal current source with 45 deg offset.
    """
    # Babic and Aykel example (single trapezoidal prism)
    source = TrapezoidalPrismCurrentSource(
        np.array([0, 0, 0]),
        np.array([0, 0, 4]),
        np.array([0, 1, 0]),
        np.array([1, 0, 0]),
        0.5,
        0.5,
        0,
        0,
        1e6,
    )
    points1 = Coordinates(
        {"x": [0.5, 0.5, -0.5, -0.5], "y": [0.5, -0.5, -0.5, 0.5], "z": [0, 0, 0, 0]}
    )
    wire = make_polygon(points1, label="wire", closed=True)
    source2 = polyhedral_discretised_sources(
        np.array([0, 0, 1]),
        np.array([1, 1, 0]),
        wire,
        4,
        0,
        0,
        1e6,
        200,
    )
    field = source.field(2, 0, 4)
    field2 = source2.field(2, 0, 4)
    abs_field = raw_uc(np.sqrt(sum(field**2)), "T", "mT")  # Field in mT
    abs_field2 = raw_uc(np.sqrt(sum(field2**2)), "T", "mT")  # Field in mT
    # Assume truncated last digit and not rounded...
    field_5decimals = np.trunc(abs_field * 10**5) / 10**5
    field_5decimals2 = np.trunc(abs_field2 * 10**5) / 10**5
    assert field_5decimals == field_5decimals2

    # Test singularity treatments:
    field = source.field(np.sqrt(0.5 * 1.0**2), 0, 2)
    field2 = source2.field(np.sqrt(0.5 * 1.0**2), 0, 2)
    abs_field = raw_uc(np.sqrt(sum(field**2)), "T", "mT")  # Field in mT
    abs_field2 = raw_uc(np.sqrt(sum(field2**2)), "T", "mT")  # Field in mT
    # Assume truncated last digit and not rounded...
    field_5decimals = np.trunc(abs_field * 10**5) / 10**5
    field_5decimals2 = np.trunc(abs_field2 * 10**5) / 10**5
    assert field_5decimals == field_5decimals2
