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

import os

import numpy as np
import pytest

from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.tools import boolean_cut, find_clockwise_angle_2d, make_polygon
from bluemira.geometry.wire import BluemiraWire
from eudemo.ivc._paneller import make_pivoted_string
from eudemo.ivc.panelling import PanellingDesigner

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def cut_wire_below_z(wire: BluemiraWire, proportion: float) -> BluemiraWire:
    """Cut a wire below the z-coordinate that is 'proportion' of the height of the wire."""
    bbox = wire.bounding_box
    z_cut_coord = proportion * (bbox.z_max - bbox.z_min) + bbox.z_min
    cutting_box = np.array(
        [
            [bbox.x_min - 1, 0, bbox.z_min - 1],
            [bbox.x_min - 1, 0, z_cut_coord],
            [bbox.x_max + 1, 0, z_cut_coord],
            [bbox.x_max + 1, 0, bbox.z_min - 1],
            [bbox.x_min - 1, 0, bbox.z_min - 1],
        ]
    )
    pieces = boolean_cut(wire, [make_polygon(cutting_box, closed=True)])
    return pieces[np.argmax([p.center_of_mass[2] for p in pieces])]


def make_cut_johner():
    """
    Make a wall shape and cut it below a (fictional) x-point.

    As this is for testing, we just use a JohnerLCFS with a slightly
    larger radius than default, then cut it below a z-coordinate that
    might be the x-point in an equilibrium.
    """
    johner_wire = JohnerLCFS(var_dict={"r_0": {"value": 10.5}}).create_shape()
    return cut_wire_below_z(johner_wire, 1 / 4)


def make_cut_polyspline():
    from eudemo.ivc.wall_silhouette_parameterisation import WallPolySpline

    wall_wire = WallPolySpline().create_shape()
    return cut_wire_below_z(wall_wire, 1 / 4)


class TestPanellingDesigner:
    @classmethod
    def setup_class(cls):
        cls.wall_boundary = make_cut_johner()

    @pytest.mark.parametrize("mode", ["mock", "run"])
    @pytest.mark.parametrize("max_angle", [25, 50])
    @pytest.mark.parametrize(
        "shape", [make_cut_johner(), make_cut_polyspline()], ids=["johner", "polyspline"]
    )
    def test_all_angles_lt_max_angle(self, mode, max_angle, shape):
        params = {
            "fw_a_max": {"value": max_angle, "unit": "degrees"},
            "fw_dL_min": {"value": 0, "unit": "m"},
            "fw_dL_max": {"value": 2.5, "unit": "m"},
        }
        designer = PanellingDesigner(params, shape)

        panel_edges: np.ndarray = getattr(designer, mode)()

        import matplotlib.pyplot as plt

        from bluemira.display import plot_2d

        panel_vecs = np.diff(panel_edges).T
        angles = []
        lengths = []
        for i in range(len(panel_vecs) - 1):
            angles.append(find_clockwise_angle_2d(panel_vecs[i], panel_vecs[i + 1]))
            lengths.append(np.linalg.norm(panel_vecs[i] - panel_vecs[i + 1]))
        print(f"angles: {angles}")
        print(f"lengths: {lengths}")

        _, ax = plt.subplots()
        plot_2d(shape, show=False, ax=ax)
        ax.plot(panel_edges[0], panel_edges[1], "--x", linewidth=0.5, color="r")
        plt.show()

        assert np.less_equal(angles, max_angle).all() or np.isclose(angles, 0).all()

    @pytest.mark.parametrize("dl_min, dl_max", [[0, 5], [0.1, 2], [1, 3]])
    @pytest.mark.parametrize(
        "shape", [make_cut_johner(), make_cut_polyspline()], ids=["johner", "polyspline"]
    )
    def test_panel_lengths_in_specified_range(self, dl_min, dl_max, shape):
        params = {
            "fw_a_max": {"value": 90, "unit": "degrees"},
            "fw_dL_min": {"value": dl_min, "unit": "m"},
            "fw_dL_max": {"value": dl_max, "unit": "m"},
        }
        build_config = {"algorithm": "COBYLA", "opt_conditions": {"ftol_rel": 1e-5}}
        panel_edges = PanellingDesigner(params, shape, build_config).run()

        import matplotlib.pyplot as plt

        from bluemira.display import plot_2d

        lengths = np.sqrt(np.sum(np.diff(panel_edges.T, axis=0) ** 2, axis=1))
        print(f"lengths: {lengths}")

        _, ax = plt.subplots()
        plot_2d(shape, show=False, ax=ax)
        ax.plot(panel_edges[0], panel_edges[1], "--x", linewidth=0.5, color="r")
        plt.show()

        abs_tol = 1e-3
        assert np.all(lengths >= dl_min - abs_tol)
        assert np.all(lengths <= dl_max + abs_tol)

    def test_panels_fully_enclose_wall_boundary(self):
        pass

    def test_end_of_panels_at_end_of_boundary(self):
        pass


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
        boundary = PrincetonD(
            {"x1": {"value": 4}, "x2": {"value": 14}, "dz": {"value": 0}}
        ).create_shape()
        boundary_points = boundary.discretize().T

        new_points = make_pivoted_string(
            boundary_points, max_angle=20, dx_min=0.5, dx_max=2.5
        )

        ref_data = np.load(os.path.join(DATA_DIR, "panelling_ref_data.npy"))
        np.testing.assert_almost_equal(new_points, ref_data)

    @pytest.mark.parametrize("dx_max", [0, 0.5, 0.9999])
    def test_ValueError_given_dx_min_gt_dx_max(self, dx_max):
        boundary = PrincetonD(
            {"x1": {"value": 4}, "x2": {"value": 14}, "dz": {"value": 0}}
        ).create_shape()
        boundary_points = boundary.discretize().T

        with pytest.raises(ValueError):
            make_pivoted_string(boundary_points, max_angle=20, dx_min=1, dx_max=dx_max)
