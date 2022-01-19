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

"""
Objects and tools for calculating cross-sectional properties
"""

from copy import deepcopy

import numba as nb
import numpy as np

from bluemira.geometry._deprecated_loop import Loop
from bluemira.geometry._deprecated_tools import get_control_point, segment_lengths
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_circle, make_polygon
from bluemira.structural.constants import NEAR_ZERO
from bluemira.structural.error import StructuralError


def _get_min_length(coordinates):
    return np.min(segment_lengths(coordinates.x, coordinates.y, coordinates.z))


def _get_max_length(coordinates):
    return np.max(segment_lengths(coordinates.x, coordinates.y, coordinates.z))


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
        i_zz += (y1 ** 2 + y1 * y2 + y2 ** 2) * d_area
        i_yy += (z1 ** 2 + z1 * z2 + z2 ** 2) * d_area
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

    def __init__(self):
        # Initialise properties
        self.area = None
        self.i_yy = None
        self.i_zz = None
        self.i_zy = None
        self.ei_yy = None
        self.ei_zz = None
        self.ei_zy = None
        self.j = None
        self.area_sy = None
        self.area_sz = None
        self.ry = None
        self.rz = None
        self.qyy = None
        self.qzz = None
        self.centroid = None
        self.centroid_geom = None
        self.ea = None
        self.gj = None
        self.nu = None
        self.rho = None

        self.geometry: BluemiraFace = None

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

    def rotate(self, angle):
        """
        Rotate the CrossSection about its centroid.

        Parameters
        ----------
        angle: float
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
        except TypeError:
            pass

        try:
            ei_uu, ei_vv, ei_uv = _transform_properties(
                self.ei_zz, self.ei_yy, self.ei_zy, alpha
            )
            self.ei_zz = ei_uu
            self.ei_yy = ei_vv
            self.ei_zy = ei_uv
        except TypeError:
            pass

        if isinstance(self.geometry, list):
            for geometry in self.geometry:
                geometry.rotate(base=geometry.centroid, axis=(1, 0, 0), degree=angle)
        else:
            self.geometry.rotate(
                base=self.geometry.centroid, axis=(1, 0, 0), degree=angle
            )

    def translate(self, vector):
        """
        Translate the CrossSection. Should not affect its properties. Note that
        CrossSections are defined in the y-z plane.

        Parameters
        ----------
        vector: iterable(3)
            The translation vector.
        """
        self.geometry.translate(vector)

    @property
    def centroid(self):
        return self.geometry.center_of_mass


class RectangularBeam(CrossSection):
    """
    Rectangular beam cross-section.

    Parameters
    ----------
    width: float
        The width of the beam
    height: float
        The height of the beam
    """

    def __init__(self, width, height):
        super().__init__()
        self.area = width * height
        self.i_zz = width ** 3 * height / 12
        self.i_yy = width * height ** 3 / 12
        self.i_zy = 0.0
        self.j = self.calc_torsion(width, height)
        self.ry = np.sqrt(self.i_yy / self.area)
        self.rz = np.sqrt(self.i_zz / self.area)
        self.qyy = 0  # Centred about (0, 0)
        self.qzz = 0  # Centred about (0, 0)
        self.width = width
        self.height = height
        self.make_geometry()

    @staticmethod
    def calc_torsion(width, height):
        """
        Estimate the torsional constant of the rectangular beam.

        Notes
        -----
        Young, W and Budynas, R: Roark's Formulas for Stress and Strain
        """
        if width >= height:
            a = width / 2
            b = height / 2
        else:
            a = height / 2
            b = width / 2

        return a * b ** 3 * (16 / 3 - 3.36 * (b / a) * (1 - b ** 4 / (12 * a ** 4)))

    def make_geometry(self):
        """
        Make a BluemiraFace for the RectangularBeam cross-section.
        """
        width = self.width
        height = self.height
        polygon = BluemiraFace(
            make_polygon(
                {
                    "x": 0,
                    "y": [-width / 2, width / 2, width / 2, -width / 2, -width / 2],
                    "z": [-height / 2, -height / 2, height / 2, height / 2, -height / 2],
                }
            )
        )
        self.geometry = polygon


class CircularBeam(CrossSection):
    """
    Circular beam cross-section

    Parameters
    ----------
    radius: float
        The radius of the circular cross-section
    """

    def __init__(self, radius):
        super().__init__()
        self.area = np.pi * radius ** 2
        self.i_zz = np.pi * radius ** 4 / 4
        self.i_yy = np.pi * radius ** 4 / 4
        self.i_zy = 0.0
        self.j = np.pi * radius ** 4 / 2
        self.ry = radius / 2
        self.rz = radius / 2
        self.geometry = BluemiraFace(
            make_circle(radius, center=(0, 0, 0), axis=(1, 0, 0))
        )


class CircularHollowBeam(CrossSection):
    """
    Circular hollow beam cross-section

    Parameters
    ----------
    r_inner: float
        The inner radius of the hollow circular cross-section
    r_outer: float
        The outer radius of the hollow circular cross-section
    """

    def __init__(self, r_inner, r_outer):
        super().__init__()
        self.area = np.pi * (r_outer ** 2 - r_inner ** 2)
        self.i_zz = np.pi / 4 * (r_outer ** 4 - r_inner ** 4)
        self.i_yy = np.pi / 4 * (r_outer ** 4 - r_inner ** 4)
        self.i_zy = 0.0
        self.j = np.pi / 2 * (r_outer ** 4 - r_inner ** 4)
        self.ry = np.sqrt((r_outer ** 2 + r_inner ** 2) / 4)
        self.rz = np.sqrt((r_outer ** 2 + r_inner ** 2) / 4)

        inner = make_circle(r_inner, center=(0, 0, 0), axis=(1, 0, 0))
        outer = make_circle(r_outer, center=(0, 0, 0), axis=(1, 0, 0))
        self.geometry = BluemiraFace([outer, inner])


class IBeam(CrossSection):
    """
    Generic, symmetric I-beam cross-section

    Parameters
    ----------
    base: float
        I-beam base width
    depth: float
        I-beam depth
    web: float
        I-beam web thickness
    flange:float
        I-beam flange thickness
    """

    def __init__(self, base, depth, flange, web):
        super().__init__()
        self.check_dimensions(base, depth, flange, web)
        h = depth - 2 * flange
        self.area = self.calc_area(base, depth, h, web)
        self.i_yy = self.calc_i_yy(base, depth, h, web)
        self.i_zz = self.calc_i_zz(base, h, web, flange)
        self.i_zy = 0.0
        self.j = self.calc_torsion(base, depth, web, flange)
        self.ry = np.sqrt(self.i_yy / self.area)
        self.rz = np.sqrt(self.i_zz / self.area)
        self.make_geometry(base, depth, flange, web)

    @staticmethod
    def check_dimensions(base, depth, flange, web):
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

    def make_geometry(self, base, depth, flange, web):
        """
        Make a BluemiraFace for the IBeam cross-section.
        """
        y = [
            -base / 2,
            base / 2,
            base / 2,
            web / 2,
            web / 2,
            base / 2,
            base / 2,
            -base / 2,
            -base / 2,
            -web / 2,
            -web / 2,
            -base / 2,
            -base / 2,
        ]
        z = [
            -depth / 2,
            -depth / 2,
            -depth / 2 + flange,
            -depth / 2 + flange,
            depth / 2 - flange,
            depth / 2 - flange,
            depth / 2,
            depth / 2,
            depth / 2 - flange,
            depth / 2 - flange,
            -depth / 2 + flange,
            -depth / 2 + flange,
            -depth / 2,
        ]
        self.geometry = BluemiraFace(make_polygon({"x": 0, "y": y, "z": z}))

    @staticmethod
    def calc_area(b, d, h, t):
        """
        Calculate the area of the Ibeam.
        """
        return b * d - h * (b - t)

    @staticmethod
    def calc_i_zz(b, h, t, s):
        """
        Calculate the zz second moment of area of the Ibeam.

        \t:math:`\\int\\int y^2 dydz`
        """
        return (2 * s * b ** 3 + h * t ** 3) / 12

    @staticmethod
    def calc_i_yy(b, d, h, t):
        """
        Calculate the yy second moment of area of the Ibeam.

        \t:math:`\\int\\int z^2 dydz`
        """
        return (b * d ** 3 - h ** 3 * (b - t)) / 12

    @staticmethod
    def calc_torsion(b, d, t, s):
        """
        Calculate the torsional constant of the Ibeam.
        """
        return (2 * b * t ** 3 + (d - s) * t ** 3) / 3


def get_coordinate_point_facets(coordinates):
    facets = [[i, i + 1] for i in range(len(coordinates) - 1)]
    facets.append([0, len(coordinates) - 1])
    return coordinates.yz, facets


class RapidCustomCrossSection(CrossSection):
    """
    Analytical formulation for a polygonal cross-section. Torsional properties
    less accurate. Faster as based on analytical calculation of cross-sectional
    properties, as opposed to FE.

    Parameters
    ----------
    geometry: BluemiraFace
        The geometry of the CrossSection
    """

    def __init__(self, geometry, opt_var=3e8):
        super().__init__()
        self.geometry = deepcopy(geometry)

        self.area = self.geometry.area

        y_l, z_l = self.geometry.y, self.geometry.z
        q_zz, q_yy, i_zz, i_yy, i_zy = _calculate_properties(y_l, z_l)

        self.i_yy = i_yy - self.area * self.centroid[2] ** 2
        self.i_zz = i_zz - self.area * self.centroid[1] ** 2
        self.i_zy = i_zy - self.area * self.centroid[1] * self.centroid[2]
        self.qyy = q_yy
        self.qzz = q_zz
        self.ry = np.sqrt(self.i_yy / self.area)
        self.rz = np.sqrt(self.i_zz / self.area)

        # OK so there is no cute general polygon form for J... need FE!
        # St Venant approach
        self.j = self.area ** 4 / (opt_var * (i_yy + i_zz))


class RapidCustomHollowCrossSection(CrossSection):
    """
    Analytical formulation for a polygonal cross-section. Torsional properties
    less accurate. Faster as based on analytical calculation of cross-sectional
    properties, as opposed to FE.

    Parameters
    ----------
    geometry: Shell
        The Shell for the polygonal cross-section

    Notes
    -----
    All cross-section properties calculated exactly (within reason), except for
    the torsional constant J, which is approximated using St Venant's approach.
    The j_opt_var for fitting the J value must be determined based on suitable
    finite element analyses.
    """

    def __init__(self, shell, j_opt_var=14.123):
        super().__init__()
        area = shell.area
        self.geometry = deepcopy(shell)

        self.area = area
        self.centroid = self.geometry.inner.centroid

        y_l, z_l = self.geometry.outer.y, self.geometry.outer.z
        q_zz_o, q_yy_o, i_zz_o, i_yy_o, i_zy_o = _calculate_properties(y_l, z_l)

        y_l, z_l = self.geometry.inner.y, self.geometry.inner.z
        q_zz_i, q_yy_i, i_zz_i, i_yy_i, i_zy_i = _calculate_properties(y_l, z_l)

        q_zz = q_zz_o - q_zz_i
        q_yy = q_yy_o - q_yy_i
        i_zz = i_zz_o - i_zz_i
        i_yy = i_yy_o - i_yy_i
        i_zy = i_zy_o - i_zy_i

        cy = q_zz / area
        cz = q_yy / area
        self.centroid_geom = (cy, cz)

        self.i_yy = i_yy - area * cz ** 2
        self.i_zz = i_zz - area * cy ** 2
        self.i_zy = i_zy - area * cz * cy
        self.qyy = q_yy
        self.qzz = q_zz
        self.ry = np.sqrt(self.i_yy / area)
        self.rz = np.sqrt(self.i_zz / area)

        # OK so there is no cute general polygon form for J... need FE!
        # St Venant approach
        self.j = self.area ** 4 / (j_opt_var * (i_yy + i_zz))


class AnalyticalCrossSection(CrossSection):
    """
    Analytical formulation for a polygonal cross-section. Torsional properties
    less accurate. Faster as based on analytical calculation of cross-sectional
    properties, as opposed to FE.

    Parameters
    ----------
    geometry: BluemiraFace
        The geometry for the polygonal cross-section

    Notes
    -----
    All cross-section properties calculated exactly (within reason), except for
    the torsional constant J, which is approximated using St Venant's approach.
    The j_opt_var for fitting the J value must be determined based on suitable
    finite element analyses.

    If the geometry has any holes in it, they will be treated as holes.
    """

    def __init__(self, geometry, n_discr=100, j_opt_var=14.123):
        super().__init__()
        self.geometry = deepcopy(geometry)
        self.area = area = self.geometry.area
        y, z = self.geometry.boundary[0].discretize(ndiscr=n_discr, byedges=True).yz

        q_zz_o, q_yy_o, i_zz_o, i_yy_o, i_zy_o = _calculate_properties(y, z)

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

        cy = q_zz_o / area
        cz = q_yy_o / area
        self.centroid_geom = (cy, cz)

        self.i_yy = i_yy_o - area * cz ** 2
        self.i_zz = i_zz_o - area * cy ** 2
        self.i_zy = i_zy_o - area * cz * cy
        self.qyy = q_yy_o
        self.qzz = q_zz_o
        self.ry = np.sqrt(self.i_yy / area)
        self.rz = np.sqrt(self.i_zz / area)

        # OK so there is no cute general polygon form for J... need FE!
        # St Venant approach
        self.j = self.area ** 4 / (j_opt_var * (i_yy_o + i_zz_o))


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
    geometry: BluemiraFace
        The ordered list of geometries making up the cross-section
    materials: List[Material]
        The ordered list of Materials to use for the geometry
    """

    def __init__(self, geometry, materials):
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

        self.area = 0
        self.ea = 0
        ga = 0
        self.ei_yy = 0
        self.ei_zz = 0
        self.ei_zy = 0

        for xs, mat in zip(cross_sections, materials):
            self.area += xs.area
            e, g = mat["E"], mat["G"]
            self.ea += e * xs.area
            ga += g * xs.area
            self.ei_yy += e * xs.i_yy
            self.ei_zz += e * xs.i_zz
            self.ei_zy += e * xs.i_zy

        self.nu = self.ea / (2 * ga) - 1
        self.ry = np.sqrt(self.ei_yy / self.ea)
        self.rz = np.sqrt(self.ei_zz / self.ea)

        self.gj = ga / self.area * n * sum(xs.j for xs in cross_sections)
