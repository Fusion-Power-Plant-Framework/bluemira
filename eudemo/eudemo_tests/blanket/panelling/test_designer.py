# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from functools import cache
from unittest import mock

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bluemira.display import plot_2d
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import (
    boolean_cut,
    find_clockwise_angle_2d,
    make_polygon,
    signed_distance_2D_polygon,
    slice_shape,
    split_wire,
)
from bluemira.geometry.wire import BluemiraWire
from eudemo.blanket.panelling import PanellingDesigner
from eudemo.ivc.wall_silhouette_parameterisation import WallPolySpline


def cut_wire_below_z(wire: BluemiraWire, proportion: float) -> BluemiraWire:
    """Cut a wire below z that is 'proportion' of the height of the wire."""
    bbox = wire.bounding_box
    z_cut_coord = proportion * (bbox.z_max - bbox.z_min) + bbox.z_min
    cutting_box = np.array([
        [bbox.x_min - 1, 0, bbox.z_min - 1],
        [bbox.x_min - 1, 0, z_cut_coord],
        [bbox.x_max + 1, 0, z_cut_coord],
        [bbox.x_max + 1, 0, bbox.z_min - 1],
        [bbox.x_min - 1, 0, bbox.z_min - 1],
    ])
    pieces = boolean_cut(wire, [make_polygon(cutting_box, closed=True)])
    return pieces[np.argmax([p.center_of_mass[2] for p in pieces])]


@cache
def make_cut_johner():
    """
    Make a wall shape and cut it below a (fictional) x-point.

    As this is for testing, we just use a JohnerLCFS with a slightly
    larger radius than default, then cut it below a z-coordinate that
    might be the x-point in an equilibrium.
    """
    johner_wire = JohnerLCFS(var_dict={"r_0": {"value": 10.5}}).create_shape()
    return cut_wire_below_z(johner_wire, 1 / 4)


@cache
def make_cut_polyspline():
    wall_wire = WallPolySpline().create_shape()
    return cut_wire_below_z(wall_wire, 1 / 4)


@cache
def make_mock_panels_johner():
    params = {
        "fw_a_max": {"value": 30, "unit": "degrees"},
        "fw_dL_min": {"value": 0.1, "unit": "m"},
    }
    boundary = make_cut_johner()
    return PanellingDesigner(params, wall_boundary=boundary).mock()


@cache
def make_panels_johner():
    params = {
        "fw_a_max": {"value": 30, "unit": "degrees"},
        "fw_dL_min": {"value": 0.1, "unit": "m"},
    }
    boundary = make_cut_johner()
    return PanellingDesigner(params, wall_boundary=boundary).run()


def coords_xz_to_polygon(coords: np.ndarray) -> BluemiraWire:
    coords_3d = np.zeros((3, coords.shape[1]))
    coords_3d[0] = coords[0]
    coords_3d[2] = coords[1]
    return make_polygon(coords_3d)


def cut_polygon_vertically(shape: BluemiraWire) -> tuple[BluemiraWire, BluemiraWire]:
    """Cut a polygon either side of a vertical line through it's centre."""
    centre_x = shape.center_of_mass[0]
    cutting_plane = BluemiraPlane((centre_x, 0, 0), (1, 0, 0))
    slice_points = slice_shape(shape, cutting_plane)
    return split_wire(shape, slice_points[0], tolerance=1e-8)


class TestPanellingDesigner:
    @pytest.mark.parametrize("max_angle", [30, 50])
    @pytest.mark.parametrize(
        "shape",
        [
            make_cut_johner(),
            make_cut_polyspline(),
            *cut_polygon_vertically(make_cut_johner()),
            *cut_polygon_vertically(make_cut_polyspline()),
        ],
        ids=[
            "johner",
            "polyspline",
            "johner_ib",
            "johner_ob",
            "polyspline_ib",
            "polyspline_ob",
        ],
    )
    def test_all_angles_lt_max_angle(self, max_angle, shape):
        params = {
            "fw_a_max": {"value": max_angle, "unit": "degrees"},
            "fw_dL_min": {"value": 0, "unit": "m"},
        }

        designer = PanellingDesigner(params, shape)
        panel_edges: np.ndarray = designer.run()

        _, ax = plt.subplots()
        plot_2d(shape, show=False, ax=ax)
        ax.plot(panel_edges[0], panel_edges[1], "--x", linewidth=0.5, color="r")

        panel_vecs = np.diff(panel_edges).T
        angles = [
            find_clockwise_angle_2d(panel_vecs[i], panel_vecs[i + 1])
            for i in range(len(panel_vecs) - 1)
        ]
        assert (
            np.less_equal(angles, max_angle) | np.isclose(angles, max_angle, rtol=1e-3)
        ).all()

    @pytest.mark.parametrize("dl_min", [0.1, 0.9])
    @pytest.mark.parametrize(
        "shape",
        [
            make_cut_johner(),
            make_cut_polyspline(),
            *cut_polygon_vertically(make_cut_johner()),
            *cut_polygon_vertically(make_cut_polyspline()),
        ],
        ids=[
            "johner",
            "polyspline",
            "johner_ib",
            "johner_ob",
            "polyspline_ib",
            "polyspline_ob",
        ],
    )
    def test_panel_lengths_gt_min_length(self, dl_min, shape):
        params = {
            "fw_a_max": {"value": 40, "unit": "degrees"},
            "fw_dL_min": {"value": dl_min, "unit": "m"},
        }

        panel_edges = PanellingDesigner(params, shape).run()

        _, ax = plt.subplots()
        plot_2d(shape, show=False, ax=ax)
        ax.plot(panel_edges[0], panel_edges[1], "--x", linewidth=0.5, color="r")

        lengths = np.sqrt(np.sum(np.diff(panel_edges.T, axis=0) ** 2, axis=1))
        abs_tol = 1e-3
        assert np.all(lengths >= dl_min - abs_tol)

    @pytest.mark.parametrize(
        "panel_edges",
        [make_mock_panels_johner(), make_panels_johner()],
        ids=["mock", "run"],
    )
    def test_panels_fully_enclose_wall_boundary(self, panel_edges):
        poly_panels = coords_xz_to_polygon(panel_edges).discretise()
        boundary = make_cut_johner()
        signed_dists = signed_distance_2D_polygon(
            poly_panels.xz.T, boundary.discretise().xz.T
        )
        np.testing.assert_array_less(signed_dists, 0 + 1e-6)

    @pytest.mark.parametrize(
        "panel_edges",
        [make_mock_panels_johner(), make_panels_johner()],
        ids=["mock", "run"],
    )
    def test_end_of_panels_at_end_of_boundary(self, panel_edges):
        poly_panels = coords_xz_to_polygon(panel_edges)
        boundary = make_cut_johner()
        np.testing.assert_allclose(poly_panels.start_point(), boundary.start_point())
        np.testing.assert_allclose(poly_panels.end_point(), boundary.end_point())

    @mock.patch("eudemo.blanket.panelling._designer.bluemira_warn")
    def test_returns_guess_and_warning_given_infeasible_problem(self, warn_mock):
        params = {
            "fw_a_max": {"value": 35, "unit": "degrees"},
            # this minimum panel length is way too high for us to get a
            # feasible solution
            "fw_dL_min": {"value": 10, "unit": "m"},
        }
        shape = make_cut_johner()

        panel_edges = PanellingDesigner(params, shape).run()

        _, ax = plt.subplots()
        plot_2d(shape, show=False, ax=ax)
        ax.plot(panel_edges[0], panel_edges[1], "--x", linewidth=0.5, color="r")

        # test that we at least get a solution that encloses the boundary
        poly_panels = coords_xz_to_polygon(panel_edges).discretise()
        signed_dists = signed_distance_2D_polygon(
            poly_panels.xz.T, shape.discretise().xz.T
        )
        np.testing.assert_array_less(signed_dists, 0 + 1e-6)
        # expect at least one warning that the problem was not solvable
        assert warn_mock.call_count >= 1
        assert any(
            "no feasible solution found" in args[0] for args in warn_mock.call_args
        )
