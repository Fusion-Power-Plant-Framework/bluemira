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
Some useful mixin classes for systems
"""
import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from BLUEPRINT.base.error import SystemsError
from BLUEPRINT.geometry.geomtools import get_boundary, lineq
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.shell import Shell

TOLERANCE = 1e-5


class OnionRing:
    """
    Mixin class for generating port geometries.
    """

    @staticmethod
    def _bulletproofclock(loop):
        """
        Verifica que el Loop esta bien orientado
        """
        xy = loop.d2
        i = loop.argmin([min(xy[0]), min(xy[1])])
        if i != 0:
            loop.reorder(i, 1)

    def _generate_xz_plot_loops(self):
        """
        Isso ai depende totalmente da orientação do Loope. O indicio 0 precisa
        de estar abaixo a la esquerda
        """

        def append(x, z, xr, zr, loop, i, j):
            x.append(loop[i][1 + j][0])
            z.append(loop[i][1 + j][2])
            xr.append(loop[i][0 + j][0])
            zr.append(loop[i][0 + j][2])
            return x, z, xr, zr

        for n, p in self.plugs.items():
            j = 0 if "Upper" in n else 2  # cheap xy xz only square
            x, z = [], []
            xr, zr = [], []
            x, z, xr, zr = append(x, z, xr, zr, p, 0, j)
            for i, layer in enumerate(p[1:]):
                x, z, xr, zr = append(x, z, xr, zr, p, i, j)
            x, z, xr, zr = append(x, z, xr, zr, p, -1, j)
            x = x + xr[::-1]
            z = z + zr[::-1]
            self.geom[n] = Loop(x=x, y=0, z=z)
            self.geom[n].close()
        return [self.geom[key] for key in self.xz_plot_loop_names]


class Port(Shell):
    """
    Port base class.
    """

    def m_c_b(self):
        """
        Line intersect parameters.
        """
        x_ib = min(self.inner[self.plan_dims[0]])
        if self.inner[self.plan_dims[1]][3] < 0:
            y_ib = -min(abs(self.inner[self.plan_dims[1]]))
        else:
            y_ib = min(abs(self.inner[self.plan_dims[1]]))
        x_ob = max(self.inner[self.plan_dims[0]])
        y_ob = max(self.inner[self.plan_dims[1]])
        m, c = lineq([x_ib, y_ib], [x_ob, y_ob])
        beta = np.arctan((y_ib - y_ob) / (x_ib - x_ob))
        return m, c, beta

    @property
    def plan_dims(self):
        """
        Planar coordinates of the Port.
        """
        return self.inner.plan_dims


class UpperPort(Port):
    """
    Upper port object.
    """

    def __init__(self, shell, x_in_min, x_in_max, x_out_min, x_out_max, l_min):
        self.inner = shell.inner
        self.outer = shell.outer

        self.m, self.c, self.beta = self.m_c_b()
        self._check_closed()
        if not hasattr(self, "thickness"):
            self.thickness = self._get_thickness()

        if x_out_min > x_in_min - self.thickness:
            bluemira_warn("Poorly specified port limits. Fixing.")
            x_in_min = x_out_min + self.thickness
        if x_out_max < x_in_max + self.thickness:
            bluemira_warn("Poorly specified port limits. Fixing.")
            x_in_max = x_out_max - self.thickness
        self.triangular_correction(l_min)
        self.inner_undershoot(x_in_min)
        self.inner_overshoot(x_in_max)
        self.outer_undershoot(x_out_min)
        self.outer_overshoot(x_out_max)

    def _get_thickness(self):
        if "x" in self.plan_dims:
            tk = abs(max(self.inner.x) - max(self.outer.x))
        elif "y" in self.plan_dims:
            tk = abs(max(self.inner.y) - max(self.outer.y))
        return tk

    def _check_closed(self):
        if not self.inner.closed:
            self.inner.close()
        if not self.outer.closed:
            self.outer.close()

    def triangular_correction(self, min_inner_length):
        """
        Correct for triangular ports.
        """
        if self.inner.y[3] < min_inner_length / 2:
            tk = self._get_thickness()
            x_in = (min_inner_length / 2 - self.c) / self.m
            x = np.array([x_in, self.inner.x[1], self.inner.x[2], x_in])
            y = np.array(
                [
                    -min_inner_length / 2,
                    self.inner.y[1],
                    self.inner.y[2],
                    min_inner_length / 2,
                ]
            )
            coords = dict(zip(self.plan_dims, [x, y]))
            coords[self._get_3rd_dim()] = self.inner[self._get_3rd_dim()][:-1]
            loop = Loop(**coords)
            loop.close()
            self.inner = loop
            self.outer = Loop.offset(self.inner, tk)

    def inner_undershoot(self, min_inner_x):
        """
        Check for inner undershoot.
        """
        if min(self.inner.x) < min_inner_x:
            self.inner = self.inner.trim([min_inner_x, 100], [min_inner_x, -100])
            tk = self._get_thickness()
            d = min_inner_x - tk
            self.outer = self.outer.trim([d, 100], [d, -100])

    def inner_overshoot(self, max_inner_x):
        """
        Check for inner overshoot.
        """
        tk = self._get_thickness()
        if max(self.inner.x) > max_inner_x:
            self.inner = self.inner.trim(
                [max_inner_x, -100], [max_inner_x, 100], method="closest"
            )
            d = max_inner_x + tk
            self.outer = self.outer.trim([d, 100], [d, -100], method="closest")

    def outer_undershoot(self, min_outer_x):
        """
        Check for outer undershoot.
        """
        tk = self._get_thickness()
        if min(self.outer.x) < min_outer_x:
            self.outer = self.outer.trim([min_outer_x, 100], [min_outer_x, -100])
            d = min_outer_x + tk
            self.inner = self.inner.trim([d, 100], [d, -100], method="furthest")

    def outer_overshoot(self, max_outer_x):
        """
        Check for outer overshoot.
        """
        tk = self._get_thickness()
        if max(self.outer.x) > max_outer_x:
            self.outer = self.outer.trim(
                [max_outer_x, 100], [max_outer_x, -100], method="closest"
            )
            d = max_outer_x - tk
            self.inner = self.inner.trim([d, 100], [d, -100], method="closest")


class Meshable:
    """
    Mixin for classes that can be meshed
    """

    def generate_cross_sections(
        self,
        mesh_sizes=None,
        geometry_names=None,
        min_length=None,
        min_angle=None,
        verbose=True,
    ):
        """
        Generate cross-section objects for this system.

        This cleans the points forming the geometries based on the `min_length`
        and `min_angle` using the :func:`~BLUEPRINT.geometry.geomtools.clean_loop_points`
        algorithm.

        For each requested geometry, clean points are then fed into a sectionproperties
        `CustomSection` object, with corresponding facets and control point. The
        geometry is cleaned, using the sectionproperties `clean_geometry` method, before
        creating a mesh and loading the mesh and geometry into a sectionproperties
        `CrossSection`.

        Also provides the points, facets, control points, and holes are used to generate
        the `CrossSection` objects. Points and facets correspond to the boundary of the
        geometries that have been used to generate the `CrossSection` objects.

        Parameters
        ----------
        mesh_sizes : List[float], optional.
            A list of maximum element areas corresponding to each region
            within the cross-section geometry, by default None.
            If None then the minimium length between nodes on the geometry
            are used.
        geometry_names : List[str], optional
            A list of names in the system's geometry dictionary to generate
            cross-sections for.
            If None then all geometries for analysis in the X-Z plane will be
            used, by default None.
        min_length : float, optional
            The minimum length [m] by which any two points should be separated,
            by default None.
        min_angle : float, optional
            The minimum angle [°] between any three points, beyond which points are not
            removed by cleaning even if they lie within min_length, by default None.
        verbose : bool, optional
            Determines if verbose mesh cleaning output should be provided,
            by default True.

        Returns
        -------
        List[`sectionproperties.analysis.cross_section.CrossSection`]
            The `CrossSection` objects representing the system.
        points : List[float, float]
            The points bounding the geometries used to generate the `CrossSection`
            objects.
        facets : List[int, int]
            The facets bounding the geometries used to generate the `CrossSection`
            objects.
        control_points : List[float, float]
            The control points used to generate the `CrossSection` objects.
        holes : List[float, float]
            The holes used to generate the `CrossSection` objects.
        """
        if not callable(getattr(self, "_generate_xz_plot_loops", None)):
            raise SystemsError(
                "Meshable.generate_cross_sections can only be called on classes that implement the _generate_xz_plot_loops method."
            )
        if not getattr(self, "xz_plot_loop_names", None):
            raise SystemsError(
                "Meshable.generate_cross_sections can only be called on classes that implement the xz_plot_loop_names property."
            )

        geometries = self._generate_xz_plot_loops()
        if geometry_names:
            xz_names = self.xz_plot_loop_names
            geometries = [geometries[xz_names.index(name)] for name in geometry_names]
        cross_sections = []
        points = []
        facets = []
        control_points = []
        holes = []
        polygons = []
        for geometry in geometries:
            (geom_cross_section, clean_geom,) = geometry.generate_cross_section(
                mesh_sizes, min_length, min_angle, verbose
            )

            cross_sections += [geom_cross_section]
            points += clean_geom.get_points()
            facets += clean_geom.get_closed_facets(start=len(facets))
            control_points += [clean_geom.get_control_point()]
            holes += [clean_geom.get_hole()]

            polygons += [clean_geom.as_shpoly()]
        points, facets = get_boundary(polygons)
        return cross_sections, points, facets, control_points, holes
