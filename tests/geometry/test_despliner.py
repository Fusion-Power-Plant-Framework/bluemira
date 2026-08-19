# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import numpy as np
import pytest

from bluemira.base.components import Component, PhysicalComponent
from bluemira.geometry.despliner import create_desplined_component_360, has_splines
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
    """
    Test has_splines() is correctly identifying
    solids with and without splines
    """
    assert has_splines(request.getfixturevalue(fixture)) is expected


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


@pytest.mark.parametrize(
    "discretisation",
    [20, 25, 50],
)
def test_create_desplined_component(splined_d_shape_component, discretisation):
    """
    Test create_desplined_component() is correctly desplining
    solids with splines
    """
    desplined_component = create_desplined_component_360(
        inp_component=splined_d_shape_component, discretisation=discretisation
    )

    assert (
        has_splines(
            desplined_component.get_component("xyz").get_component_properties("shape")
        )
        is False
    )

    xz_boundaries = (
        desplined_component
        .get_component("xz")
        .get_component_properties("shape")
        .boundary
    )
    for wire in xz_boundaries:
        assert len(wire.vertexes.T) == discretisation + 1
