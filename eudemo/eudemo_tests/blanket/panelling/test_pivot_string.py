# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from pathlib import Path

import numpy as np
import pytest

from bluemira.geometry.parameterisations import PrincetonD
from eudemo.blanket.panelling._pivot_string import make_pivoted_string

DATA_DIR = Path(Path(__file__).parent, "data")


class TestMakePivotedString:
    def test_returns_points_matching_snapshot(self):
        """
        This tests that the function returns the same thing as the
        equivalent class method in BLUEPRINT.

        The data for this test was generated using the
        BLUEPRINT.geometry.stringgeom.String class from BLUEPRINT.

        The code used to generate the test data:

        .. code-block:: python

            from BLUEPRINT.geometry.stringgeom import String
            from bluemira.geometry.parameterisations import PrincetonD

            shape = PrincetonD().create_shape()
            points = shape.discretize()
            s = String(points, angle=20, dx_min=0.5, dx_max=2.5)

            np.save("panelling_ref_data.npy", s.new_points)

        Using bluemira 437a1c10, and BLUEPRINT e3fb8d1c.
        """
        boundary = PrincetonD({
            "x1": {"value": 4},
            "x2": {"value": 14},
            "dz": {"value": 0},
        }).create_shape()
        boundary_points = boundary.discretize().T

        new_points, _ = make_pivoted_string(
            boundary_points, max_angle=20, dx_min=0.5, dx_max=2.5
        )

        ref_data = np.load(Path(DATA_DIR, "panelling_ref_data.npy"))
        np.testing.assert_almost_equal(new_points, ref_data)

    @pytest.mark.parametrize("dx_max", [0, 0.5, 0.9999])
    def test_ValueError_given_dx_min_gt_dx_max(self, dx_max):
        boundary = PrincetonD({
            "x1": {"value": 4},
            "x2": {"value": 14},
            "dz": {"value": 0},
        }).create_shape()
        boundary_points = boundary.discretize().T

        with pytest.raises(ValueError):  # noqa: PT011
            make_pivoted_string(boundary_points, max_angle=20, dx_min=1, dx_max=dx_max)
