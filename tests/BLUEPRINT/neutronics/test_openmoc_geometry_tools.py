# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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
import pytest

openmoc = pytest.importorskip("openmoc")

from BLUEPRINT.neutronics.openmoc_geometry_tools import (  # noqa :E402
    PlaneHelper,
    calc_triangle_centroid,
    evaluate_point_against_plane,
    get_halfspace,
    get_normalised_plane_properties,
    get_plane_properties,
    get_plane_properties_from_points,
)


class TestPointsAndPlanes:
    """
    A test class for points and planes functionality
    """

    point1 = [0.2, 0.9]
    point2 = [-0.2, 0.0]
    point3 = [5.0, 2.0]

    plane1 = openmoc.Plane(A=0.9, B=-0.4, C=0.0, D=0.18)
    plane2 = openmoc.XPlane(x=5.0)
    plane3 = openmoc.YPlane(y=-3.0)

    def test_plane_properties_from_points(self):
        """
        Test that plane properties can be retrieved from points
        """
        plane_props = get_plane_properties_from_points(self.point1, self.point2)

        assert len(plane_props) == 4
        assert np.allclose(plane_props, (0.9, -0.4, 0, 0.18))

    def test_plane_properties(self):
        """
        Test that the plane properties can be retrieved from a plane
        """
        plane_props = get_plane_properties(self.plane1)

        assert len(plane_props) == 4
        assert np.allclose(plane_props, (0.9, -0.4, 0, 0.18))

    def test_plane_properties_xplane(self):
        """
        Test that the plane properties can be retrieved from an x-plane
        """
        plane_props = get_plane_properties(self.plane2)

        assert len(plane_props) == 4
        assert np.allclose(plane_props, (1.0, 0.0, 0, -5.0))

    def test_plane_properties_yplane(self):
        """
        Test that the plane properties can be retrieved from a y-plane
        """
        plane_props = get_plane_properties(self.plane3)

        assert len(plane_props) == 4
        assert np.allclose(plane_props, (0.0, 1.0, 0, 3.0))

    def test_normalised_plane_properties(self):
        """
        Test that the plane properties can be retrieved from a plane
        """
        norm_plane_props = get_normalised_plane_properties(self.plane1)

        assert len(norm_plane_props) == 2
        assert np.allclose(norm_plane_props, (2.25, 0.45))

    def test_normalised_plane_properties_xplane(self):
        """
        Test that the plane properties can be retrieved from a plane
        """
        norm_plane_props = get_normalised_plane_properties(self.plane2)

        assert len(norm_plane_props) == 2
        assert np.allclose(norm_plane_props, (float("inf"), 5.0))

    def test_normalised_plane_properties_yplane(self):
        """
        Test that the plane properties can be retrieved from a plane
        """
        norm_plane_props = get_normalised_plane_properties(self.plane3)

        assert len(norm_plane_props) == 2
        assert np.allclose(norm_plane_props, (0.0, -3.0))

    def test_evaluate_point_against_plane(self):
        """
        Test that a point can be evaluated against a plane
        """
        value = evaluate_point_against_plane(self.plane1, self.point3)

        assert np.isclose(value, 3.88)

    def test_get_halfspace(self):
        """
        Test that a point can be evaluated against a plane
        """
        value = get_halfspace(self.plane1, self.point3)

        assert np.isclose(value, +1)


class TestPlaneHelper:
    """
    A class to test the PlaneHelper caching functionality
    """

    def test_add_new_plane(self):
        helper = PlaneHelper()
        plane = openmoc.Plane(A=7.0, B=-5.0, C=0.0, D=4.1)
        helper.add_plane(plane)

        assert len(helper.planes) == 1
        assert plane in helper.planes.values()

    def test_find_new_plane(self):
        helper = PlaneHelper()
        plane1 = openmoc.Plane(A=-1.0, B=3.0, C=0.0, D=2.0)
        helper.add_plane(plane1)
        plane2 = helper.find_plane(plane1)

        assert len(helper.planes) == 1
        assert plane2 is plane1

    def test_coplanar(self):
        helper = PlaneHelper()
        plane1 = openmoc.Plane(A=3.0, B=2.0, C=0.0, D=4.0)
        helper.add_plane(plane1)
        plane2 = openmoc.Plane(A=1.5, B=1.0, C=0.0, D=2.0)
        helper.add_plane(plane2)
        plane3 = helper.find_plane(plane2)

        assert len(helper.planes) == 1
        assert plane3 is plane1

    def test_coplanar_x0(self):
        helper = PlaneHelper()
        plane1 = openmoc.Plane(A=1.0, B=0.0, C=0.0, D=4.0)
        helper.add_plane(plane1)
        plane2 = openmoc.Plane(A=2.0, B=0.0, C=0.0, D=8.0)
        helper.add_plane(plane2)
        plane3 = helper.find_plane(plane2)

        assert len(helper.planes) == 1
        assert plane3 is plane1

    def test_coplanar_y0(self):
        helper = PlaneHelper()
        plane1 = openmoc.Plane(A=0.0, B=1.0, C=0.0, D=4.0)
        helper.add_plane(plane1)
        plane2 = openmoc.Plane(A=0.0, B=2.0, C=0.0, D=8.0)
        helper.add_plane(plane2)
        plane3 = helper.find_plane(plane2)

        assert len(helper.planes) == 1
        assert plane3 is plane1


class TestMeshing:
    """
    A class to test routines for meshing
    """

    def test_calc_triangle_centroid(self):
        point1 = [0.2, 0.9]
        point2 = [-0.2, 0.0]
        point3 = [5.0, 2.0]

        centroid = calc_triangle_centroid(point1, point2, point3)

        assert len(centroid) == 2
        assert np.allclose(centroid, (5.0 / 3.0, 2.9 / 3.0))
