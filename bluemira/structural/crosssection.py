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

"""
Objects and tools for calculating cross-sectional properties
"""
from __future__ import annotations

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from bluemira.structural.material import StructuralMaterial

from copy import deepcopy

import numba as nb
import numpy as np

from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_circle, make_polygon
from bluemira.structural.constants import NEAR_ZERO
from bluemira.structural.error import StructuralError


@nb.jit(nopython=True, cache=True)
def _calculate_properties(y, z):
    """
    Calculate cross-sectional properties for arbitrary polygons.
    """
    q_zz, q_yy, i_zz, i_yy, i_zy = 0, 0, 0, 0, 0

    for i in range(len(y) - 1):  # zip is slow in numba
        y1, y2 = y[i], y[i + 1]
        z1, z2 = z[i], z[i + 1]
        d_area = y1 * z2 - y2 * z1
        q_zz += (y2 + y1) * d_area
        q_yy += (z2 + z1) * d_area
        i_zz += (y1**2 + y1 * y2 + y2**2) * d_area
        i_yy += (z1**2 + z1 * z2 + z2**2) * d_area
        i_zy += (z1 * y2 + 2 * z1 * y1 + 2 * z2 * y2 + z2 * y1) * d_area
    return q_zz / 6, q_yy / 6, i_zz / 12, i_yy / 12, i_zy / 24


def _transform_properties(izz, iyy, izy, alpha):
    """
    Transform second moments of area for a rotation about the centroid.

    \t:math:`I_{uu}=(I_{xx}+I_{yy})/2+[(I_{xx}-I_{yy})/2]cos(2\\alpha)-I_{xy}sin(2\\alpha)`

    \t:math:`I_{vv}=(I_{xx}+I_{yy})/2-[(I_{xx}-I_{yy})/2]cos(2\\alpha)+I_{xy}sin(2\\alpha)`

    \t:math:`I_{uv}=[(I_{xx}-I_{yy})/2]sin(2\\alpha)+I_{xy}cos(2\\alpha)`
    """
    # We need to clip the cos and sin terms for near-zero values, because they
    # are about to be multiplied with E (often very big!)
    cos2alpha = np.cos(2 * alpha)
    sin2alpha = np.sin(2 * alpha)

    if np.abs(cos2alpha) < NEAR_ZERO:
        cos2alpha = 0.0
    if np.abs(sin2alpha) < NEAR_ZERO:
        sin2alpha = 0.0

    i_uu = 0.5 * (izz + iyy) + 0.5 * (izz - iyy) * cos2alpha - izy * sin2alpha
    i_vv = 0.5 * (izz + iyy) - 0.5 * (izz - iyy) * np.cos(2 * alpha) + izy * sin2alpha
    i_uv = 0.5 * (izz - iyy) * sin2alpha + izy * cos2alpha
    return i_uu, i_vv, i_uv


class CrossSection:
    """
    Base class for a structural cross-section of a 1-D beam.
    """

    __slots__ = (
        "area",
        "i_yy",
        "i_zz",
        "i_zy",
        "ei_yy",
        "ei_zz",
        "ei_zy",
        "j",
        "area_sy",
        "area_sz",
        "ry",
        "rz",
        "qyy",
        "qzz",
        "centroid_geom",
        "geometry",
        "y",
        "z",
    )

    def __init__(self):
        pass

    def make_geometry(self, *args, **kwargs):
        """
        Make a BluemiraFace object for the CrossSection.
        """
        raise NotImplementedError

    def plot(self, ax=None):
        """
        Plot the CrossSection
        """
        self.geometry.plot(ax=ax, points=True)

    def rotate(self, angle: float):
        """
        Rotate the CrossSection about its centroid.

        Parameters
        ----------
        angle:
            The angle to rotate the CrossSection by [degrees]
        """
        alpha = np.deg2rad(angle)

        try:  # A CrossSection will either have Inn or EInn properties
            i_uu, i_vv, i_uv = _transform_properties(
                self.i_zz, self.i_yy, self.i_zy, alpha
            )
            self.i_zz = i_uu
            self.i_yy = i_vv
            self.i_zy = i_uv
        except AttributeError:
            pass

        try:
            ei_uu, ei_vv, ei_uv = _transform_properties(
                self.ei_zz, self.ei_yy, self.ei_zy, alpha
            )
            self.ei_zz = ei_uu
            self.ei_yy = ei_vv
            self.ei_zy = ei_uv
        except AttributeError:
            pass

        if isinstance(self.geometry, list):
            for geometry in self.geometry:
                geometry.rotate(base=self.centroid, direction=(1, 0, 0), degree=angle)
        else:
            self.geometry.rotate(base=self.centroid, direction=(1, 0, 0), degree=angle)

    def translate(self, vector: np.ndarray):
        """
        Translate the CrossSection. Should not affect its properties. Note that
        CrossSections are defined in the y-z plane.

        Parameters
        ----------
        vector:
            The translation vector.
        """
        self.geometry.translate(vector)

    @property
    def centroid(self):
        """
        Centroid of the cross-section geometry
        """
        return self.geometry.center_of_mass


class RectangularBeam(CrossSection):
    """
    Rectangular beam cross-section.

    Parameters
    ----------
    width:
        The width of the beam
    height:
        The height of the beam
    """

    __slots__ = ()

    def __init__(self, width: float, height: float):
        super().__init__()
        self.area = width * height
        self.i_zz = width**3 * height / 12
        self.i_yy = width * height**3 / 12
        self.i_zy = 0.0
        self.j = self.calc_torsion(width, height)
        self.ry = np.sqrt(self.i_yy / self.area)
        self.rz = np.sqrt(self.i_zz / self.area)
        self.qyy = 0  # Centred about (0, 0)
        self.qzz = 0  # Centred about (0, 0)
        self.make_geometry(width, height)

    @staticmethod
    def calc_torsion(width: float, height: float) -> float:
        """
        Estimate the torsional constant of the rectangular beam.

        Notes
        -----
        Young, W and Budynas, R: Roark's Formulas for Stress and Strain

        \t:math:`J\\approx ab^3(\\dfrac{16}{3}-3.36\\dfrac{b}{a}(1-\\dfrac{b^4}{12a^4}))`
        """
        if width >= height:
            a = width / 2
            b = height / 2
        else:
            a = height / 2
            b = width / 2

        return a * b**3 * (16 / 3 - 3.36 * (b / a) * (1 - b**4 / (12 * a**4)))

    def make_geometry(self, width: float, height: float):
        """
        Make a BluemiraFace for the RectangularBeam cross-section.
        """
        w = 0.5 * width
        h = 0.5 * height
        self.y = np.array([-w, w, w, -w, -w])
        self.z = np.array([-h, -h, h, h, -h])
        polygon = BluemiraFace(
            make_polygon(
                {
                    "x": 0,
                    "y": self.y,
                    "z": self.z,
                }
            )
        )
        self.geometry = polygon


class CircularBeam(CrossSection):
    """
    Circular beam cross-section

    Parameters
    ----------
    radius:
        The radius of the circular cross-section
    n_discr:
        Number of points to discretise to when plotting
    """

    __slots__ = ()

    def __init__(self, radius: float, n_discr: int = 30):
        super().__init__()
        self.area = np.pi * radius**2
        self.i_zz = np.pi * radius**4 / 4
        self.i_yy = np.pi * radius**4 / 4
        self.i_zy = 0.0
        self.j = np.pi * radius**4 / 2
        self.ry = radius / 2
        self.rz = radius / 2
        self.qyy = 0  # Centred about (0, 0)
        self.qzz = 0  # Centred about (0, 0)
        circle = make_circle(radius, center=(0, 0, 0), axis=(1, 0, 0))
        self.geometry = BluemiraFace(circle)
        self.y, self.z = circle.discretize(ndiscr=n_discr).yz


class CircularHollowBeam(CrossSection):
    """
    Circular hollow beam cross-section

    Parameters
    ----------
    r_inner:
        The inner radius of the hollow circular cross-section
    r_outer:
        The outer radius of the hollow circular cross-section
    n_discr:
        Number of points to discretise to when plotting
    """

    __slots__ = ()

    def __init__(self, r_inner: float, r_outer: float, n_discr: int = 30):
        super().__init__()
        self.area = np.pi * (r_outer**2 - r_inner**2)
        self.i_zz = np.pi / 4 * (r_outer**4 - r_inner**4)
        self.i_yy = np.pi / 4 * (r_outer**4 - r_inner**4)
        self.i_zy = 0.0
        self.j = np.pi / 2 * (r_outer**4 - r_inner**4)
        self.ry = np.sqrt((r_outer**2 + r_inner**2) / 4)
        self.rz = np.sqrt((r_outer**2 + r_inner**2) / 4)
        self.qyy = 0  # Centred about (0, 0)
        self.qzz = 0  # Centred about (0, 0)

        inner = make_circle(r_inner, center=(0, 0, 0), axis=(1, 0, 0))
        outer = make_circle(r_outer, center=(0, 0, 0), axis=(1, 0, 0))
        self.geometry = BluemiraFace([outer, inner])
        self.y, self.z = np.concatenate(
            [outer.discretize(ndiscr=n_discr).yz, inner.discretize(ndiscr=n_discr).yz],
            axis=1,
        )


class IBeam(CrossSection):
    """
    Generic, symmetric I-beam cross-section

    Parameters
    ----------
    base:
        I-beam base width
    depth:
        I-beam depth
    web:
        I-beam web thickness
    flange:
        I-beam flange thickness
    """

    __slots__ = ()

    def __init__(self, base: float, depth: float, flange: float, web: float):
        super().__init__()
        self.check_dimensions(base, depth, flange, web)
        h = depth - 2 * flange
        b, d, t, s = base, depth, web, flange
        self.area = b * d - h * (b - t)
        self.i_yy = (b * d**3 - h**3 * (b - t)) / 12
        self.i_zz = (2 * s * b**3 + h * t**3) / 12
        self.i_zy = 0.0
        self.j = (2 * b * t**3 + (d - s) * t**3) / 3
        self.ry = np.sqrt(self.i_yy / self.area)
        self.rz = np.sqrt(self.i_zz / self.area)
        self.qyy = 0  # Centred about (0, 0)
        self.qzz = 0  # Centred about (0, 0)
        self.make_geometry(base, depth, flange, web)

    @staticmethod
    def check_dimensions(base: float, depth: float, flange: float, web: float):
        """
        Edge case eradication
        """
        if (
            (depth - 2 * flange <= 0)
            or (base - web <= 0)
            or (base <= 0)
            or (depth <= 0)
            or (flange <= 0)
            or (web <= 0)
        ):
            raise StructuralError("I-beam dimensions don't make sense.")

    def make_geometry(self, base: float, depth: float, flange: float, web: float):
        """
        Make a BluemiraFace for the IBeam cross-section.
        """
        b, d, f, w = 0.5 * base, 0.5 * depth, flange, 0.5 * web
        self.y = np.array([-b, b, b, w, w, b, b, -b, -b, -w, -w, -b, -b])
        self.z = np.array(
            [
                -d,
                -d,
                -d + f,
                -d + f,
                d - f,
                d - f,
                d,
                d,
                d - f,
                d - f,
                -d + f,
                -d + f,
                -d,
            ]
        )
        self.geometry = BluemiraFace(make_polygon({"x": 0, "y": self.y, "z": self.z}))


class AnalyticalCrossSection(CrossSection):
    """
    Analytical formulation for a polygonal cross-section. Torsional properties
    less accurate. Faster as based on analytical calculation of cross-sectional
    properties, as opposed to FE.

    Parameters
    ----------
    geometry:
        The geometry for the polygonal cross-section
    n_discr:
        Number of points to discretise to when plotting
    j_opt_var:
        Torsional constant estimation parameter from optimisation

    Notes
    -----
    All cross-section properties calculated exactly (within reason), except for
    the torsional constant J, which is approximated using St Venant's approach.
    The j_opt_var for fitting the J value must be determined based on suitable
    finite element analyses.

    If the geometry has any holes in it, they will be treated as holes.
    """

    __slots__ = ()

    def __init__(
        self, geometry: BluemiraFace, n_discr: int = 100, j_opt_var: float = 14.123
    ):
        super().__init__()
        self.geometry = deepcopy(geometry)
        self.area = area = self.geometry.area
        self.y, self.z = (
            self.geometry.boundary[0].discretize(ndiscr=n_discr, byedges=True).yz
        )

        q_zz_o, q_yy_o, i_zz_o, i_yy_o, i_zy_o = _calculate_properties(self.y, self.z)

        if len(self.geometry.boundary) > 1:
            # Cut out any holes in the face
            for wire in self.geometry.boundary[1:]:
                y, z = wire.discretize(ndiscr=n_discr, byedges=True).yz
                q_zz_i, q_yy_i, i_zz_i, i_yy_i, i_zy_i = _calculate_properties(y, z)

                q_zz_o -= q_zz_i
                q_yy_o -= q_yy_i
                i_zz_o -= i_zz_i
                i_yy_o -= i_yy_i
                i_zy_o -= i_zy_i
                self.y = np.append(self.y, y)
                self.z = np.append(self.z, z)

        cy = q_zz_o / area
        cz = q_yy_o / area
        self.centroid_geom = (cy, cz)

        self.i_yy = i_yy_o - area * cz**2
        self.i_zz = i_zz_o - area * cy**2
        self.i_zy = i_zy_o - area * cz * cy
        self.qyy = q_yy_o
        self.qzz = q_zz_o
        self.ry = np.sqrt(self.i_yy / area)
        self.rz = np.sqrt(self.i_zz / area)

        # OK so there is no cute general polygon form for J... need FE!
        # St Venant approach
        self.j = self.area**4 / (j_opt_var * (i_yy_o + i_zz_o))


class AnalyticalCompositeCrossSection(CrossSection):
    """
    A cross-section object for composite structural beam.

    When making a composite cross-section, we need to add material properties
    in order to effectively weight the cross-sectional properties.

    Cross-sectional properties are calculated analytical relations, and are
    therefore much faster than an FE approach. For simple cross-sections, the
    properties are all identical except for J, where a fitting on similar
    shapes must be carried out, following St. Venant's method.

    This somewhat modifies the API when getting properties...

    Parameters
    ----------
    geometry:
        The ordered list of geometries making up the cross-section
    materials:
        The ordered list of Materials to use for the geometry
    """

    __slots__ = ("ea", "nu", "gj", "rho")

    def __init__(self, geometry: BluemiraFace, materials: List[StructuralMaterial]):
        super().__init__()
        self.geometry = deepcopy(geometry)

        n = len(geometry.boundary)

        if len(materials) != n:
            raise StructuralError(f"Need {n} materials for this geometry.")

        outer = AnalyticalCrossSection(geometry.boundary)
        inners = []
        for wire in geometry.boundary[1:]:
            face = BluemiraFace(wire)
            inners.append(AnalyticalCrossSection(face))

        cross_sections = [outer]
        cross_sections.extend(inners)

        e_values = np.array([mat["E"] for mat in materials])
        g_values = np.array([mat["G"] for mat in materials])
        rho_values = np.array([mat["rho"] for mat in materials])
        areas = np.array([xs.area for xs in cross_sections])
        i_yy_values = np.array([xs.i_yy for xs in cross_sections])
        i_zz_values = np.array([xs.i_zz for xs in cross_sections])
        i_zy_values = np.array([xs.i_zy for xs in cross_sections])

        self.area = np.sum(areas)
        self.ea = np.dot(e_values, areas)
        self.ei_yy = np.dot(e_values, i_yy_values)
        self.ei_zz = np.dot(e_values, i_zz_values)
        self.ei_zy = np.dot(e_values, i_zy_values)
        ga = np.dot(g_values, areas)
        rhoa = np.dot(rho_values, areas)

        self.nu = self.ea / (2 * ga) - 1
        self.ry = np.sqrt(self.ei_yy / self.ea)
        self.rz = np.sqrt(self.ei_zz / self.ea)

        self.gj = ga / self.area * n * sum(xs.j for xs in cross_sections)
        self.rho = rhoa / self.area
