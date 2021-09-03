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
import numpy as np
import numba as nb
from copy import deepcopy
from sectionproperties.pre.sections import CustomSection, MergedSection
from sectionproperties.pre.pre import Material as SPMaterial
from sectionproperties.analysis.cross_section import CrossSection as _CrossSection
from BLUEPRINT.geometry.loop import Loop, MultiLoop
from BLUEPRINT.geometry.shell import Shell
from BLUEPRINT.geometry.geomtools import circle_seg, get_control_point
from BLUEPRINT.base.error import BeamsError
from BLUEPRINT.beams.constants import NEAR_ZERO


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
    Base class for a structurual cross-section of a 1-D beam.
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

        self.geometry = None

    def make_loop(self, *args, **kwargs):
        """
        Make a Loop object for the CrossSection.
        """
        raise NotImplementedError

    def plot(self, ax=None):
        """
        Plot the CrossSection
        """
        self.geometry.plot(ax=ax, points=True)

    def copy(self):
        """
        Get a deep copy of the CrossSection.
        """
        return deepcopy(self)

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

        cy, cz = self.centroid
        if isinstance(self.geometry, list):
            for geometry in self.geometry:
                geometry.rotate(angle, p1=[0, cy, cz], p2=[1, cy, cz])
        else:
            self.geometry.rotate(angle, p1=[0, cy, cz], p2=[1, cy, cz])

    def translate(self, vector):
        """
        Translate the CrossSection. Should not affect its properties. Note that
        CrossSections are defined in the y-z plane.

        Parameters
        ----------
        vector: iterable(3)
            The translation vector.
        """
        self.geometry.translate(vector, update=True)
        self.centroid[0] += vector[1]
        self.centroid[1] += vector[2]


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
        self.make_loop()

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

    def make_loop(self):
        """
        Make a Loop for the RectangularBeam cross-section.
        """
        width = self.width
        height = self.height
        loop = Loop(
            y=[-width / 2, width / 2, width / 2, -width / 2, -width / 2],
            z=[-height / 2, -height / 2, height / 2, height / 2, -height / 2],
        )
        self.geometry = loop
        self.centroid = self.geometry.centroid


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
        self.centroid = 0.0, 0.0

        y, z = circle_seg(radius, self.centroid, npoints=20)
        loop = Loop(y=y, z=z)
        loop.close()
        self.geometry = loop


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
        self.centroid = 0.0, 0.0

        y1, z1 = circle_seg(r_inner, self.centroid, npoints=20)
        y2, z2 = circle_seg(r_outer, self.centroid, npoints=20)
        inner = Loop(y=y1, z=z1)
        inner.close()
        outer = Loop(y=y2, z=z2)
        outer.close()
        self.geometry = Shell(inner, outer)


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
        self.make_loop(base, depth, flange, web)

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
            raise BeamsError("I-beam dimensions don't make sense.")

    def make_loop(self, base, depth, flange, web):
        """
        Make a Loop for the IBeam cross-section.
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
        self.geometry = Loop(y=y, z=z)
        self.centroid = self.geometry.centroid

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


def loop_to_point_facet(loop):
    """
    Convert a Loop object to a sectionproperties list, list combo

    Parameters
    ----------
    loop: BLUEPRINT::geometry:Loop
        The Loop to convert

    Returns
    -------
    points: [[float, float], ...]
        The list of point [x, y] lists
    facets: [[int, int], ...]
        The list of node id [i, j] lists
    """
    points = loop.d2.T.tolist()
    facets = [[i, i + 1] for i in range(len(loop) - 1)]

    # Closed loops (CrossSections always closed)
    facets.append([0, len(loop) - 1])
    return points, facets


def mat_to_mat(material):
    """
    Converts a BLUEPRINT Material object to a sectionproperties Material
    """
    return SPMaterial(
        "Dummy",
        material["E"],
        material["nu"],
        yield_strength=0,
        color=np.random.rand(3),
    )


class CustomCrossSection(CrossSection):
    """
    A polygonic cross-section implementation for those more complicated shapes.

    Parameters
    ----------
    loop: BLUEPRINT::geometry::Loop object
        The loop of the CrossSection
    """

    def __init__(self, loop, hollow=False):
        super().__init__()
        self._set_loop(loop)
        geometry = self.make_geometry(self.geometry, hollow)
        self._calc_properties_fe(geometry)

    def _set_loop(self, loop):
        loop = loop.copy()  # Detach CrossSection loop from its input geometry object
        if loop.__class__.__name__ == "Loop":
            if not loop.closed:
                loop.close()
        if not loop.plan_dims == ["y", "z"]:
            loop = self._rebase_loop(loop)
        self.geometry = loop

    @staticmethod
    def _rebase_loop(loop):
        a, b = loop.plan_dims
        typ = loop.__class__.__name__
        if typ == "Loop":
            y, z = loop[a], loop[b]
            return Loop(y=y, z=z)
        elif typ == "Shell":
            yi, zi = loop.inner[a], loop.inner[b]
            yo, zo = loop.outer[a], loop.outer[b]
            return Shell(Loop(y=yi, z=zi), Loop(y=yo, z=zo))
        else:
            raise ValueError

    def make_geometry(self, loop, hollow=False):
        """
        Make the geometry of the CustomCrossSection
        """
        typ = loop.__class__.__name__
        if typ == "Loop":
            return self._make_loop_geometry(loop)
        elif typ == "Shell":
            return self._make_shell_geometry(loop, hollow)
        else:
            raise ValueError

    @staticmethod
    def _make_loop_geometry(loop):
        """
        Makes a sectionproperties Geometry object from a Loop
        """
        points, facets = loop_to_point_facet(loop)
        cp = get_control_point(loop)
        geometry = CustomSection(points, facets, [], [cp])
        mesh_size = loop.get_min_length()
        # Keep the mesh size in the geometry
        geometry.mesh_size = mesh_size
        return geometry

    @staticmethod
    def _make_shell_geometry(shell, hollow=False):
        """
        Makes a sectionproperties Geometry object from a Shell
        """
        points, facets = loop_to_point_facet(shell.outer)
        p2, f2 = loop_to_point_facet(shell.inner)
        f2 = np.array(f2, dtype=np.int32) + len(shell.outer)
        f2 = f2.tolist()
        points.extend(p2)
        facets.extend(f2)
        if hollow:
            holes = [get_control_point(shell.inner)]
            control = [get_control_point(shell)]
        else:
            holes = []
            control = [get_control_point(shell)]
        # NOTE: This will not make a hole in the section, but make two
        # individual mesh regions
        geometry = CustomSection(points, facets, holes, control)

        l_1 = shell.inner.get_min_length()
        l_2 = shell.outer.get_min_length()
        mesh_size = min([l_1, l_2])

        geometry.mesh_size = mesh_size
        return geometry

    def _calc_properties_fe(self, geometry):
        section = self._make_section(geometry)
        section.calculate_geometric_properties()
        area, ixx, iyy, ixy, j, phi = section.calculate_frame_properties()
        self.area = area
        self.i_yy = ixx
        self.i_zz = iyy
        self.i_zy = ixy
        self.j = j
        self.centroid = [section.section_props.cx, section.section_props.cy]
        self.ry = section.section_props.rx_c
        self.rz = section.section_props.ry_c
        self.qyy = section.section_props.qx
        self.qzz = section.section_props.qy

        # Keep this in case needed (CAN'T COPY OR PICKLE)
        # self._section = section

    @staticmethod
    def _make_section(geometry):
        # Although recommended, this can break small crosssection meshes
        geometry.clean_geometry()

        # Mesh the cross-seciton geometry with a reasonable resolution
        mesh = geometry.create_mesh([geometry.mesh_size / 4])
        return _CrossSection(geometry, mesh)


class RapidCustomCrossSection(CrossSection):
    """
    Analytical formulation for a polygonal cross-section. Torsional properties
    less accurate. Faster as based on analytical calculation of cross-sectional
    properties, as opposed to FE.

    Parameters
    ----------
    loop: Loop
        The Loop for the polygonal cross-section
    """

    def __init__(self, loop, opt_var=3e8):
        super().__init__()
        self.geometry = loop.copy()

        self.area = self.geometry.area
        self.centroid = self.geometry.centroid

        y_l, z_l = self.geometry.y, self.geometry.z
        q_zz, q_yy, i_zz, i_yy, i_zy = _calculate_properties(y_l, z_l)

        self.i_yy = i_yy - self.area * self.centroid[1] ** 2
        self.i_zz = i_zz - self.area * self.centroid[0] ** 2
        self.i_zy = i_zy - self.area * self.centroid[0] * self.centroid[1]
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
    shell: Shell
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
        self.geometry = shell.copy()

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


class MultiCrossSection(CrossSection):
    """
    A multiple cross-section class. Do not use for CrossSections that are touching.

    Parameters
    ----------
    x_sections: List[CrossSection]
        The list of CrossSections from which to make a MultiCrossSection
    centroid: iterable(2)
        The cz, cy centroid of the MultiCrossSection
    """

    def __init__(self, x_sections, centroid):
        super().__init__()
        self.geometry = MultiLoop([xs.geometry for xs in x_sections], stitch=False)
        self.centroid = centroid

        area = 0.0
        qyy = 0.0
        qzz = 0.0
        i_yy = 0.0
        i_zz = 0.0
        i_zy = 0.0
        for xs in x_sections:
            dz = abs(self.centroid[1] - xs.centroid[1])
            dy = abs(self.centroid[0] - xs.centroid[0])
            area += xs.area
            qyy += xs.area * dz
            qzz += xs.area * dy
            # Parallel axis theorem
            i_yy += xs.i_yy + xs.area * dz ** 2
            i_zz += xs.i_zz + xs.area * dy ** 2
        self.area = area
        self.qyy = qyy
        self.qzz = qzz
        self.i_yy = i_yy
        self.i_zz = i_zz
        self.i_zy = i_zy

        self.ry = np.sqrt(self.i_yy / self.area)
        self.rz = np.sqrt(self.i_zz / self.area)

        # This is probably a bad idea... but should be OK for disconnected sections
        self.j = sum([xs.j for xs in x_sections])


class CompositeCrossSection(CustomCrossSection):
    """
    A cross-section object for composite structural.

    When making a composite cross-section, we need to add material properties
    in order to effectively weight the cross-sectional properties.

    Cross-sectional properties are calculated using finite element analysis.

    This somewhat modifies the API when getting properties...

    Parameters
    ----------
    loops: List[Loop]
        The ordered list of Loops making up the geometry
    materials: List[Material]
        The ordered list of Materials to use for the geometry
    """

    def __init__(self, loops, materials):
        if len(loops) != len(materials):
            raise BeamsError(
                "We need the same number of materials and x-sections to build"
                "a composite."
            )
        # Need to store this information to retrieve stresses in the various
        # parts of the CompositeCrossSection
        self._set_loops(loops)
        self.material = materials
        self._calc_density(loops, materials)

        geometries = [self.make_geometry(loop) for loop in loops]
        materials = [mat_to_mat(material) for material in materials]

        if len(geometries) == 1:
            geometries = geometries[0]

        self._geometries = geometries
        self._materials = materials
        self._calc_properties_fe(geometries, materials)
        self._correct_axis()

    def _set_loops(self, loops):
        new_loops = []
        for loop in loops:
            loop = loop.copy()
            if loop.__class__.__name__ == "Loop":
                if not loop.closed:
                    loop.close()
            if not loop.plan_dims == ["y", "z"]:
                loop = self._rebase_loop(loop)
            new_loops.append(loop)
        self.geometry = new_loops

    @staticmethod
    def _make_section(geometries, materials):
        geometry = MergedSection(geometries)
        mesh_sizes = [g.mesh_size for g in geometries]

        geometry.clean_geometry()
        mesh = geometry.create_mesh(mesh_sizes)

        return _CrossSection(geometry, mesh, materials=materials)

    def _calc_properties_fe(self, geometries, materials):
        section = self._make_section(geometries, materials)
        section.calculate_geometric_properties()
        # area is not correctly calculated for composites (returns EA)
        _, ixx, iyy, ixy, j, phi = section.calculate_frame_properties()

        sec_props = section.section_props

        self.qyy = sec_props.qx
        self.qzz = sec_props.qy
        self.i_yy = ixx
        self.i_zz = iyy
        self.i_zy = ixy
        self.j = j
        self.centroid = [sec_props.cx, sec_props.cy]
        self.ry = sec_props.rx_c
        self.rz = sec_props.ry_c

        self.ea = sec_props.ea
        self.ei_yy = sec_props.ixx_c
        self.ei_zz = sec_props.iyy_c
        self.ei_zy = sec_props.ixy_c
        self.gj = sec_props.j / (2 * (1 + sec_props.nu_eff))
        self.nu = sec_props.nu_eff

        # self._section = section

    def _correct_axis(self):
        """
        Correct the cross-sectional properties so that they are about the
        prescribed centroid of the shape (0, 0).
        """
        if self.geometry[0].__class__.__name__ == "Shell":
            area1 = self.geometry[0].area
            centroid = self.geometry[0].centroid
            reference = self.centroid
            e1 = self.material[0]["E"]
            e2 = self.material[1]["E"]
            dy = centroid[0] - reference[0]
            dz = centroid[1] - reference[1]

            self.ei_yy -= area1 * e1 * dz ** 2
            self.ei_zz -= area1 * e1 * dy ** 2

            area2 = self.geometry[1].area
            centroid2 = self.geometry[1].centroid
            dy = centroid2[0] - reference[0]
            dz = centroid2[1] - reference[1]

            self.ei_yy -= area2 * e2 * dz ** 2
            self.ei_zz -= area2 * e2 * dy ** 2

    def _calc_density(self, loops, materials):
        """
        Calculates the cross-sectional area-weighted density of the CompositeCS
        """
        areas = np.array([loop.area for loop in loops])
        densities = np.array([mat["rho"] for mat in materials])
        self.area = np.sum(areas)
        self.rho = np.dot(areas, densities) / np.sum(areas)

    def plot(self, ax=None):
        """
        Plot the CustomCrossSection and the underlying FE mesh used to
        calculate its properties.
        """
        section = self._make_section(self._geometries, self._materials)
        section.plot_mesh(materials=True)


class AnalyticalShellComposite(CompositeCrossSection):
    """
    A cross-section object for composite structural.

    When making a composite cross-section, we need to add material properties
    in order to effectively weight the cross-sectional properties.

    Cross-sectional properties are calculated analytical relations, and are
    therefore much fast than the FE approach. For simple cross-sections, the
    properties are all identical except for J, where a fitting on similar
    shapes must be carried out, following St. Venant's method.

    This somewhat modifies the API when getting properties...

    Parameters
    ----------
    shell: Shell
        The ordered list of Loops making up the geometry
    materials: List[Material]
        The ordered list of Materials to use for the geometry
    """

    def __init__(self, shell, materials):

        if len(materials) != 2:
            raise BeamsError("Need 2 materials for a DuploRectangleComposite.")

        # Need to store this information to retrieve stresses in the various
        # parts of the CompositeCrossSection
        loops = [shell.outer.copy(), shell.inner.copy()]
        self._set_loops(loops)
        self.material = materials
        self._calc_density([shell, shell.inner], materials)

        loops.sort(key=lambda x: x.area)

        inner, outer = loops

        a = inner.get_max_length() / 2
        b = inner.get_min_length() / 2
        # Use analytical relation for a rectangle torsion constant
        j_inner = a * b ** 3 * (16 / 3 - 3.36 * (b / a) * (1 - b ** 4 / (12 * a ** 4)))

        inner = RapidCustomCrossSection(inner)
        inner.j = j_inner

        outer = RapidCustomHollowCrossSection(shell)

        self._calculate_composite_properties(inner, outer)

    def rotate(self, angle):
        """
        Rotate the AnalyticalShellComposite cross-section about its centroid.

        Parameters
        ----------
        angle: float
            The angle to rotate the CrossSection by [degrees]
        """
        loops = self.geometry
        loops.sort(key=lambda x: x.area)

        inner, outer = loops

        a = inner.get_max_length() / 2
        b = inner.get_min_length() / 2
        # Use analytical relation for a rectangle torsion constant
        j_inner = a * b ** 3 * (16 / 3 - 3.36 * (b / a) * (1 - b ** 4 / (12 * a ** 4)))

        inner = RapidCustomCrossSection(inner)
        inner.j = j_inner
        outer = RapidCustomHollowCrossSection(Shell(*loops))
        inner.rotate(angle)
        outer.rotate(angle)
        self._calculate_composite_properties(inner, outer)

    def _calculate_composite_properties(self, inner, outer):
        e_inner, e_outer = self.material[1]["E"], self.material[0]["E"]
        g_inner, g_outer = self.material[1]["G"], self.material[0]["G"]

        ea = e_inner * inner.area + e_outer * outer.area
        ga = g_inner * inner.area + g_outer * outer.area

        nu_eff = ea / (2 * ga) - 1

        eiyy = inner.i_yy * e_inner + outer.i_yy * e_outer
        eizz = inner.i_zz * e_inner + outer.i_zz * e_outer
        eizy = inner.i_zy * e_inner + outer.i_zy * e_outer

        self.centroid = inner.centroid

        self.ea = ea
        self.nu = nu_eff

        self.ei_yy = eiyy
        self.ei_zz = eizz
        self.ei_zy = eizy
        self.ry = np.sqrt(eiyy / ea)
        self.rz = np.sqrt(eizz / ea)

        # Area weighted G modulus
        g = (g_inner * inner.area + g_outer * outer.area) / (inner.area + outer.area)
        # Still eyeballing this one...
        self.gj = g * (outer.j + inner.j) * 2


if __name__ == "__main__":
    from BLUEPRINT import test

    test(plotting=True)
