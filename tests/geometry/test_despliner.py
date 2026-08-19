# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import numpy as np
import pytest

from bluemira.geometry.despliner import has_splines
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import (
    extrude_shape,
    interpolate_bspline,
    make_bezier,
    make_bsplinesurface,
    make_polygon,
)


@pytest.fixture
def unit_box():
    face = BluemiraFace(
        make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
            closed=True,
        )
    )
    return extrude_shape(face, [0, 0, 1])


@pytest.fixture
def bspline_surface_extruded():
    poles = np.array([
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
    ])
    surface = make_bsplinesurface(
        poles,
        mults_u=[2, 2],
        mults_v=[2, 2],
        knot_vector_u=[0.0, 1.0],
        knot_vector_v=[0.0, 1.0],
        degree_u=1,
        degree_v=1,
        weights=np.ones((2, 2)),
        periodic=True,
        check_rational=True,
    )
    return extrude_shape(surface, [0, 1, 0])


@pytest.fixture
def bspline_xz_extruded():
    points = {
        "x": [0.0, 1.0, 0.0, -1.0, 0.0],
        "y": 0,
        "z": [1.0, 0.0, -1.0, 0.0, 1.0],
    }
    return extrude_shape(
        BluemiraFace(interpolate_bspline(points, closed=True)),
        [0, 1, 0],
    )


@pytest.fixture
def bspline_xy_extruded():
    points = {
        "x": [0.0, 1.0, 0.0, -1.0, 0.0],
        "y": [1.0, 0.0, -1.0, 0.0, 1.0],
        "z": 0,
    }
    return extrude_shape(
        BluemiraFace(interpolate_bspline(points, closed=True)),
        [0, 0, 1],
    )


@pytest.fixture
def bezier_yz_extruded():
    points = {
        "x": 0,
        "y": [0.0, 1.0, 0.0, -1.0, 0.0],
        "z": [1.0, 0.0, -1.0, 0.0, 1.0],
    }
    return extrude_shape(
        BluemiraFace(make_bezier(points, closed=True)),
        [1, 0, 0],
    )


@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("unit_box", False),
        ("bspline_surface_extruded", True),
        ("bspline_xz_extruded", True),
        ("bspline_xy_extruded", True),
        ("bezier_yz_extruded", True),
    ],
)
def test_has_splines(request, fixture, expected):
    assert has_splines(request.getfixturevalue(fixture)) is expected
