# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest
from matproplib.library.beryllium import Be12Ti
from matproplib.library.tungsten import PlanseeTungsten

from bluemira.base.components import Component, PhysicalComponent
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import (
    interpolate_bspline,
    make_polygon,
    revolve_shape,
)
from bluemira.geometry.wire import BluemiraWire
from bluemira.radiation_transport.generalised_neutronics.geometry import (
    NeutronicsGeometryManager,
)


def line(a, b):
    return make_polygon({
        "x": [a[0], b[0]],
        "y": [0, 0],
        "z": [a[1], b[1]],
    })


def spline(points):
    points = np.asarray(points)

    return interpolate_bspline(
        Coordinates({
            "x": points[:, 0],
            "y": np.zeros(len(points)),
            "z": points[:, 1],
        })
    )


@pytest.fixture
def nontouching_pair():
    """Two non-touching splineless components."""

    # Solid 1
    xz_face_1 = BluemiraFace(
        make_polygon(
            [
                (0.4, 0.0, 0.5),
                (0.8, 0.0, 0.5),
                (0.8, 0.0, 1.5),
                (0.4, 0.0, 1.5),
            ],
            closed=True,
        ),
        label="xz face 1",
    )

    xyz_shape_1 = revolve_shape(
        xz_face_1,
        base=(0, 0, 0),
        direction=(0, 0, 1),
        degree=360.0,
    )

    xz_comp_1 = Component(
        "xz",
        children=[
            PhysicalComponent(
                name="xz_1_phys",
                shape=xz_face_1,
                material=PlanseeTungsten(),
            )
        ],
    )

    xyz_comp_1 = Component(
        "xyz",
        children=[
            PhysicalComponent(
                name="xyz_1_phys",
                shape=xyz_shape_1,
                material=PlanseeTungsten(),
            )
        ],
    )

    component_1 = Component("Solid_1")
    component_1.add_children([xz_comp_1, xyz_comp_1])

    # Solid 2
    xz_face_2 = BluemiraFace(
        make_polygon(
            [
                (1.0, 0.0, 0.5),
                (1.6, 0.0, 0.5),
                (1.6, 0.0, 1.5),
                (1.0, 0.0, 1.5),
            ],
            closed=True,
        ),
        label="xz face 2",
    )

    xyz_shape_2 = revolve_shape(
        xz_face_2,
        base=(0, 0, 0),
        direction=(0, 0, 1),
        degree=360.0,
    )

    xz_comp_2 = Component(
        "xz",
        children=[
            PhysicalComponent(
                name="xz_2_phys",
                shape=xz_face_2,
                material=Be12Ti(),
            )
        ],
    )

    xyz_comp_2 = Component(
        "xyz",
        children=[
            PhysicalComponent(
                name="xyz_2_phys",
                shape=xyz_shape_2,
                material=Be12Ti(),
            )
        ],
    )

    component_2 = Component("Solid_2")
    component_2.add_children([xz_comp_2, xyz_comp_2])

    return component_1, component_2


@pytest.fixture
def nontouching_splined_pair():
    """Two non-touching components with splined boundaries."""

    p_left = [
        (0.40, 0.50),
        (0.44, 0.75),
        (0.46, 1.00),
        (0.44, 1.25),
        (0.40, 1.50),
    ]

    # Outer boundary of Solid_1
    p_middle = [
        (0.80, 0.50),
        (0.84, 0.75),
        (0.86, 1.00),
        (0.84, 1.25),
        (0.80, 1.50),
    ]

    # Inner boundary of Solid_2.
    # There is a gap between this and p_middle.
    p_shared = [
        (0.94, 0.50),
        (0.91, 0.75),
        (0.90, 1.00),
        (0.91, 1.25),
        (0.94, 1.50),
    ]

    p_right = [
        (1.60, 0.50),
        (1.54, 0.75),
        (1.50, 1.00),
        (1.54, 1.25),
        (1.60, 1.50),
    ]

    # Solid 1
    xz_face_1 = BluemiraFace(
        BluemiraWire([
            spline(p_left),
            line(p_left[-1], p_middle[-1]),
            spline(p_middle[::-1]),
            line(p_middle[0], p_left[0]),
        ]),
        label="xz face 1",
    )

    xyz_shape_1 = revolve_shape(
        xz_face_1,
        base=(0, 0, 0),
        direction=(0, 0, 1),
        degree=360.0,
    )

    xz_comp_1 = Component(
        "xz",
        children=[
            PhysicalComponent(
                name="xz_1_phys",
                shape=xz_face_1,
                material=PlanseeTungsten(),
            )
        ],
    )

    xyz_comp_1 = Component(
        "xyz",
        children=[
            PhysicalComponent(
                name="xyz_1_phys",
                shape=xyz_shape_1,
                material=PlanseeTungsten(),
            )
        ],
    )

    component_1 = Component("Solid_1")
    component_1.add_children([xz_comp_1, xyz_comp_1])

    # Solid 2
    xz_face_2 = BluemiraFace(
        BluemiraWire([
            spline(p_shared),
            line(p_shared[-1], p_right[-1]),
            spline(p_right[::-1]),
            line(p_right[0], p_shared[0]),
        ]),
        label="xz face 2",
    )

    xyz_shape_2 = revolve_shape(
        xz_face_2,
        base=(0, 0, 0),
        direction=(0, 0, 1),
        degree=360.0,
    )

    xz_comp_2 = Component(
        "xz",
        children=[
            PhysicalComponent(
                name="xz_2_phys",
                shape=xz_face_2,
                material=Be12Ti(),
            )
        ],
    )

    xyz_comp_2 = Component(
        "xyz",
        children=[
            PhysicalComponent(
                name="xyz_2_phys",
                shape=xyz_shape_2,
                material=Be12Ti(),
            )
        ],
    )

    component_2 = Component("Solid_2")
    component_2.add_children([xz_comp_2, xyz_comp_2])

    return component_1, component_2


@pytest.fixture
def touching_splined_pair():
    """Two components sharing the same splined boundary."""

    p_left = [
        (0.40, 0.50),
        (0.44, 0.75),
        (0.46, 1.00),
        (0.44, 1.25),
        (0.40, 1.50),
    ]

    # Exact shared spline between Solid_1 and Solid_2
    p_shared = [
        (0.94, 0.50),
        (0.91, 0.75),
        (0.90, 1.00),
        (0.91, 1.25),
        (0.94, 1.50),
    ]

    p_right = [
        (1.60, 0.50),
        (1.54, 0.75),
        (1.50, 1.00),
        (1.54, 1.25),
        (1.60, 1.50),
    ]

    # Solid 1
    xz_face_1 = BluemiraFace(
        BluemiraWire([
            spline(p_left),
            line(p_left[-1], p_shared[-1]),
            spline(p_shared[::-1]),
            line(p_shared[0], p_left[0]),
        ]),
        label="xz face 1",
    )

    xyz_shape_1 = revolve_shape(
        xz_face_1,
        base=(0, 0, 0),
        direction=(0, 0, 1),
        degree=360.0,
    )

    xz_comp_1 = Component(
        "xz",
        children=[
            PhysicalComponent(
                name="xz_1_phys",
                shape=xz_face_1,
                material=PlanseeTungsten(),
            )
        ],
    )

    xyz_comp_1 = Component(
        "xyz",
        children=[
            PhysicalComponent(
                name="xyz_1_phys",
                shape=xyz_shape_1,
                material=PlanseeTungsten(),
            )
        ],
    )

    component_1 = Component("Solid_1")
    component_1.add_children([xz_comp_1, xyz_comp_1])

    # Solid 2
    xz_face_2 = BluemiraFace(
        BluemiraWire([
            spline(p_shared),
            line(p_shared[-1], p_right[-1]),
            spline(p_right[::-1]),
            line(p_right[0], p_shared[0]),
        ]),
        label="xz face 2",
    )

    xyz_shape_2 = revolve_shape(
        xz_face_2,
        base=(0, 0, 0),
        direction=(0, 0, 1),
        degree=360.0,
    )

    xz_comp_2 = Component(
        "xz",
        children=[
            PhysicalComponent(
                name="xz_2_phys",
                shape=xz_face_2,
                material=Be12Ti(),
            )
        ],
    )

    xyz_comp_2 = Component(
        "xyz",
        children=[
            PhysicalComponent(
                name="xyz_2_phys",
                shape=xyz_shape_2,
                material=Be12Ti(),
            )
        ],
    )

    component_2 = Component("Solid_2")
    component_2.add_children([xz_comp_2, xyz_comp_2])

    return component_1, component_2


@pytest.mark.cadquery_only
class TestNeutronicsGeometryManager:
    """Tests for NeutronicsGeometryManager."""

    @pytest.mark.parametrize(
        ("pair_fixture", "index", "expected"),
        [
            ("touching_splined_pair", 0, ["Solid_2"]),
            ("touching_splined_pair", 1, ["Solid_1"]),
            ("nontouching_splined_pair", 0, []),
            ("nontouching_splined_pair", 1, []),
            ("nontouching_pair", 0, []),
            ("nontouching_pair", 1, []),
        ],
    )
    def test_check_touching(
        self,
        pair_fixture,
        index,
        expected,
        request,
    ):
        components = request.getfixturevalue(pair_fixture)

        assert (
            NeutronicsGeometryManager.check_touching(
                components[index],
                components,
                overlap_tolerance=1e-6,
            )
            == expected
        )

    @pytest.mark.parametrize(
        "pair_fixture",
        [
            "touching_splined_pair",
            "nontouching_splined_pair",
            "nontouching_pair",
        ],
    )
    def test_from_list_of_components(self, pair_fixture, request, monkeypatch):
        components = request.getfixturevalue(pair_fixture)

        manager = NeutronicsGeometryManager.from_list_of_components(
            components=list(components),
            discretisations=[10, 10],
            overlap_tolerance=1e-6,
            gap_tolerance=1e-2,
        )

        assert isinstance(manager, NeutronicsGeometryManager)
        assert len(manager.get_all_components()) == len(components)

        warnings = []

        monkeypatch.setattr(
            "bluemira.base.look_and_feel.bluemira_warn",
            warnings.append,
        )

        manager.inspect_xyz_overlaps(tolerance=1e-10)

        assert not warnings
