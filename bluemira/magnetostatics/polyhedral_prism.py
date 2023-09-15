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
Analytical expressions for the field inside an arbitrarily shaped winding pack
with arbitrarily shaped cross-section, following equations as described in:


"""
import math
import sys

import matplotlib.collections as col
import matplotlib.path as pltpath
import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.constants import MU_0, MU_0_2PI
from bluemira.base.look_and_feel import bluemira_error
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import distance_to, make_polygon
from bluemira.magnetostatics.baseclass import (
    ArbitraryCrossSectionCurrentSource,
    SourceGroup,
)
from bluemira.magnetostatics.biot_savart import BiotSavartFilament
from bluemira.magnetostatics.tools import process_xyz_array
from bluemira.magnetostatics.trapezoidal_prism import TrapezoidalPrismCurrentSource
from bluemira.utilities.plot_tools import Plot3D

__all__ = ["PolyhedralPrismCurrentSource"]


def trap_dist(theta, pos, min_pos, vec):
    """
    func to produce distance between centre and trap end
    """
    dy = np.dot((pos - min_pos), vec)
    dz = dy * np.tan(theta)
    return dz


class PolyhedralPrismCurrentSource(ArbitraryCrossSectionCurrentSource):
    """
    prism current source
    """

    def __init__(
        self,
        origin,
        ds,
        normal,
        t_vec,
        trap_vec,
        n,
        length,
        width,
        alpha,
        beta,
        current,
        nrows,
        coords=None,
    ):
        """
        initialisation
        """
        self.origin = origin
        if coords == None:
            self.n = n
        else:
            self.n = np.shape(coords)[1]
        self.normal = normal / np.linalg.norm(normal)
        self.length = np.linalg.norm(ds)
        self.dcm = np.array([t_vec, ds / self.length, normal])
        self.length = length
        self.width = width
        self.theta = 2 * np.pi / self.n
        self.nrows = nrows
        vec = trap_vec
        self.trap_vec = vec / np.linalg.norm(vec)
        perp_vec = np.cross(normal, self.trap_vec)
        self.perp_vec = perp_vec / np.linalg.norm(perp_vec)
        self.theta_l = np.deg2rad(beta)
        self.theta_u = np.deg2rad(alpha)
        self.points = self._calc_points(coords)
        self.area = self._cross_section_area()
        self.J = current / self.area
        self.sources = self._segmentation_setup(self.nrows)

    def _cross_section_area(self):
        """
        Function to calculate cross sectional area of prism
        """

        points = self.points[0]
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        coords = Coordinates({"x": x, "y": y, "z": z})
        wire = make_polygon(coords, closed=True)
        face = BluemiraFace(boundary=wire)
        return face.area

    def _shape_min_max(self, points, vector):
        """
        Function to calculate min and max points of prism cross section along vector
        """
        vals = []
        for p in points:
            vals += [np.dot(p, vector)]
        pmin = points[vals.index(min(vals)), :]
        pmax = points[vals.index(max(vals)), :]
        return pmin, pmax

    def _calc_points(self, coords=None):
        """
        Function to calculate all the points of the prism in local coords
        """
        if coords == None:
            c_points = []
            for i in range(self.n + 1):
                c_points += [
                    np.array(
                        [
                            round(self.width * np.sin(i * self.theta), 10),
                            round(self.width * np.cos(i * self.theta), 10),
                            0,
                        ]
                    )
                ]
            c_points = np.vstack([np.array(c_points)])
        else:
            points = coords.T
            c_points = []
            for p in points:
                c_points += [np.array(p)]
            c_points += [np.array(points[0, :])]
            c_points = np.vstack([np.array(c_points)])
        boundl, boundu = self._shape_min_max(c_points, self.trap_vec)

        l_points = []
        u_points = []

        for p in c_points:
            dz_l = trap_dist(self.theta_l, p, boundl, self.trap_vec)
            l_points += [
                np.array([p[0], p[1], round(p[2] - 0.5 * self.length - dz_l, 10)])
            ]
            dz_u = trap_dist(self.theta_u, p, boundl, self.trap_vec)
            u_points += [
                np.array([p[0], p[1], round(p[2] + 0.5 * self.length + dz_u, 10)])
            ]
        l_points = np.vstack([np.array(l_points)])
        u_points = np.vstack([np.array(u_points)])
        points = [c_points] + [l_points] + [u_points]
        # add lines between cuts
        for i in range(self.n):
            points += [np.vstack([l_points[i], u_points[i]])]
        p_array = []
        for p in points:
            p_array.append(self._local_to_global(p))

        return np.array(p_array, dtype=object)

    def _segmentation_setup(self, nrows):
        """
        Function to break up shape into series of segments of trapezoidal shapes
        Method of segmentation is to bound the central line of each segsment
        with the edge of the prism, with top of first segment (and bot of last
        segment) matching the top (and bot) vertex.
        n is number of segments
        """
        points = self.points[0]
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        coords = Coordinates({"x": x, "y": y, "z": z})
        main_wire = make_polygon(coords, closed=True)
        par_min, par_max = self._shape_min_max(points, self.trap_vec)
        b = np.dot(par_max - par_min, self.trap_vec) / nrows
        perp_min, perp_max = self._shape_min_max(points, self.perp_vec)
        perp_dist = np.dot(perp_max - perp_min, self.perp_vec)
        sources = SourceGroup([])
        c_area = 0
        for i in range(nrows):
            d = i * b + b / 2
            c = par_min + d * self.trap_vec
            u = c + perp_dist * self.perp_vec
            l = c - perp_dist * self.perp_vec
            x = np.array([l[0], u[0]])
            y = np.array([l[1], u[1]])
            z = np.array([l[2], u[2]])
            coords = Coordinates({"x": x, "y": y, "z": z})
            wire = make_polygon(coords, closed=False)
            dist, vectors = distance_to(main_wire, wire)
            if np.round(dist, 4) > 0.0:
                print("no intersect between line and wire")
            else:
                p1 = np.array(vectors[0][0])
                p2 = np.array(vectors[1][0])
                o = np.multiply(0.5, (p1 + p2))
                width = np.linalg.norm(p2 - p1)
                area = width * b
                c_area += area
                current = self.J * area
                dz_l = trap_dist(self.theta_l, o, par_min, self.trap_vec)
                dz_u = trap_dist(self.theta_u, o, par_min, self.trap_vec)
                length = self.length + dz_l + dz_u
                source = TrapezoidalPrismCurrentSource(
                    o,
                    length * self.normal,
                    self.perp_vec,
                    self.trap_vec,
                    b / 2,
                    width / 2,
                    self.theta_u,
                    self.theta_l,
                    current,
                )
                sources.add_to_group([source])
        self.seg_area = c_area
        return sources

    @process_xyz_array
    def field(self, x, y, z):
        point = np.array([x, y, z])
        Bx, By, Bz = self.sources.field(*point)
        return Bx, By, Bz
