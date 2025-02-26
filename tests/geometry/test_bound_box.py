# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.geometry.bound_box import BoundingBox
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import PolySpline
from bluemira.geometry.tools import boolean_cut, make_polygon, revolve_shape


class TestBoundingBox:
    rng = np.random.default_rng()

    def test_null(self):
        x, y, z = np.zeros(100), np.zeros(100), np.zeros(100)
        xb, yb, zb = BoundingBox.from_xyz(x, y, z).get_box_arrays()
        assert np.all(xb == 0)
        assert np.all(yb == 0)
        assert np.all(zb == 0)

    def test_random(self):
        x, y, z = self.rng.random(100), self.rng.random(100), self.rng.random(100)
        args = self.rng.integers(0, 100, 8)
        x[args] = np.array([-2, -2, -2, -2, 2, 2, 2, 2])
        y[args] = np.array([-2, -2, 2, 2, 2, -2, -2, 2])
        z[args] = np.array([-2, 2, -2, 2, -2, 2, -2, 2])
        xb, yb, zb = BoundingBox.from_xyz(x, y, z).get_box_arrays()

        assert np.allclose(xb, np.array([-2, -2, -2, -2, 2, 2, 2, 2]))
        assert np.allclose(yb, np.array([-2, -2, 2, 2, -2, -2, 2, 2]))
        assert np.allclose(zb, np.array([-2, 2, -2, 2, -2, 2, -2, 2]))


class TestHardBoundingBox:
    ps = PolySpline({
        "bottom": {"value": 0.509036},
        "flat": {"value": 1},
        "height": {"value": 10.1269},
        "lower": {"value": 0.2},
        "tilt": {"value": 19.6953},
        "top": {"value": 0.46719},
        "upper": {"value": 0.326209},
        "x1": {"value": 5},
        "x2": {"value": 11.8222},
        "z2": {"value": -0.170942},
    })

    cut_box = BluemiraFace(
        make_polygon({"x": [0, 15, 15, 0], "z": [-7, -7, -5, -5], "y": 0}, closed=True)
    )
    wire = boolean_cut(ps.create_shape(), cut_box)[0]
    temp = wire.deepcopy()
    temp.close()
    solid = revolve_shape(BluemiraFace(temp), degree=159)

    @pytest.mark.xfail(reason="Bounding box is not precise")
    def test_bad_bounding_box(self):
        assert np.isclose(self.wire.bounding_box.z_min, -5.0)

    def test_opt_bounding_box(self):
        bb = self.wire.optimal_bounding_box
        assert np.isclose(bb.z_min, -5.0)

    def test_opt_bounding_box_solid(self):
        solid = self.solid.deepcopy()
        vertices, indices = solid._tessellate(1.0)
        bb = self.solid.optimal_bounding_box
        vertices2, indices2 = solid._tessellate(1.0)
        assert np.isclose(bb.z_min, -5.0)
        # Test that bounding box via tesselation did not modify properties
        np.testing.assert_allclose(vertices, vertices2)
        np.testing.assert_allclose(indices, indices2)
        vertices3, _ = solid._tessellate(0.01)
        assert vertices3.shape != vertices2.shape
