# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import pytest

from bluemira.geometry.despliner import has_splines
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import (
    extrude_shape,
    interpolate_bspline,
    make_bezier,
    make_polygon,
)


def make_unit_box():
    face = BluemiraFace(
        make_polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], closed=True)
    )
    return extrude_shape(face, [0, 0, 1])


def make_bspline_solid(points, vec, closed):
    return extrude_shape(BluemiraFace(interpolate_bspline(points, closed)), vec)


def make_bezier_solid(points, vec, closed):
    return extrude_shape(BluemiraFace(make_bezier(points, closed)), vec)


@pytest.mark.parametrize(
    ("shape_func", "args", "expected"),
    [
        (make_unit_box, (), False),
        (
            make_bspline_solid,
            (
                {
                    "x": [0.0, 1.0, 0.0, -1.0, 0.0],
                    "y": 0,
                    "z": [1.0, 0.0, -1.0, 0.0, 1.0],
                },
                [0, 1, 0],
                True,
            ),
            True,
        ),
        (
            make_bspline_solid,
            (
                {
                    "x": [0.0, 1.0, 0.0, -1.0, 0.0],
                    "y": [1.0, 0.0, -1.0, 0.0, 1.0],
                    "z": 0,
                },
                [0, 0, 1],
                True,
            ),
            True,
        ),
        (
            make_bezier_solid,
            (
                {
                    "x": 0,
                    "y": [0.0, 1.0, 0.0, -1.0, 0.0],
                    "z": [1.0, 0.0, -1.0, 0.0, 1.0],
                },
                [1, 0, 0],
                True,
            ),
            True,
        ),
    ],
)
def test_has_splines(shape_func, args, expected):
    assert has_splines(shape_func(*args)) is expected
