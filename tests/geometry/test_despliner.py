# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.base.components import Component, PhysicalComponent
from bluemira.geometry.despliner import (
    check_solid_has_revolution,
    check_solid_has_splines,
    despline_xz_component,
)
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import (
    extrude_shape,
    interpolate_bspline,
    make_bezier,
    make_bsplinesurface,
    make_polygon,
    revolve_shape,
)
from bluemira.geometry.wire import BluemiraWire


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


@pytest.mark.cadquery_only
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
def test_check_solid_has_splines(request, fixture, expected):
    """
    Test check_solid_has_splines() is correctly identifying
    solids with and without splines
    """
    assert check_solid_has_splines(request.getfixturevalue(fixture)) is expected


@pytest.fixture
def solid_with_surfaces_of_revolution():
    curve = make_bezier({
        "x": [1.0, 1.5, 2.0],
        "y": 0,
        "z": [-1.0, 0.0, 1.0],
    })

    closure = make_polygon({
        "x": [2.0, 0.0, 0.0, 1.0],
        "y": 0,
        "z": [1.0, 1.0, -1.0, -1.0],
    })

    face = BluemiraFace(BluemiraWire([curve, closure]))

    return revolve_shape(
        face,
        base=(0, 0, 0),
        direction=(0, 0, 1),
        degree=360,
    )


@pytest.mark.cadquery_only
@pytest.mark.parametrize(
    ("fixture", "expected"),
    [
        ("solid_with_surfaces_of_revolution", True),
        ("bspline_surface_extruded", False),
        ("bspline_xz_extruded", False),
        ("bspline_xy_extruded", False),
        ("bezier_yz_extruded", False),
    ],
)
def test_check_solid_has_revolution(request, fixture, expected):
    """
    Test check_solid_has_splines() is correctly identifying
    solids with and without splines
    """
    assert check_solid_has_revolution(request.getfixturevalue(fixture)) is expected


@pytest.fixture
def splined_d_shape_component():
    """Create a component containing a D-shaped face with a spline edge."""
    straight = make_polygon(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 1],
            [0, 0, 1],
        ],
        closed=False,
    )

    spline = interpolate_bspline([
        [0, 0, 1],
        [-0.2, 0, 0.75],
        [-0.3, 0, 0.5],
        [-0.2, 0, 0.25],
        [0, 0, 0],
    ])

    boundary = BluemiraWire([*straight.edges, spline])
    boundary.close()
    face = BluemiraFace(boundary)

    xz_component = Component(
        "xz",
        children=[
            PhysicalComponent(
                name="d_shape",
                shape=face,
            )
        ],
    )

    xyz_component = Component(
        "xyz",
        children=[
            PhysicalComponent(
                name="d_shape",
                shape=revolve_shape(
                    face,
                    base=(0, 0, 0),
                    direction=(0, 0, 1),
                    degree=360.0,
                ),
            )
        ],
    )

    return Component(
        "test_component",
        children=[xz_component, xyz_component],
    )


@pytest.mark.cadquery_only
@pytest.mark.parametrize(
    "discretisation",
    [20, 25, 50],
)
def test_despline_xz_component(splined_d_shape_component, discretisation):
    """
    Test despline_xz_component() is correctly desplining
    xz face with splines
    """
    desplined_component = despline_xz_component(
        component=splined_d_shape_component, discretisation=discretisation
    )

    desplined_xz_face = desplined_component.get_component("xz").get_component_properties(
        "shape"
    )
    assert check_solid_has_splines(desplined_xz_face) is False

    xz_boundaries = desplined_xz_face.boundary

    for wire in xz_boundaries:
        assert len(wire.vertexes.T) == discretisation + 1
