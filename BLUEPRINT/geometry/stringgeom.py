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
A cheeky jagged edge shape paramterisation - used in firstwallprofile
"""
from itertools import count

import matplotlib.pyplot as plt
import numpy as np


class String:
    """
    Simon McIntosh's String class for simplifying a Loop based on minimum
    turning angle and minimum and maximum segment lengths.

    Parameters
    ----------
    points: np.array(N, 2or 3)
        String of X, Y, Z point coordinates
    angle: float
        Maximum turning angle [degree]
    dx_min: float
        Minimum segment length [m]
    dx_max: float
        Maximum segment length
    verbose: bool (default = False)
        Print output
    """

    def __init__(self, points, angle=10, dx_min=0, dx_max=np.inf, verbose=False):
        # Constructors
        self.new_points = None
        self.index = None
        self.n = None

        self.points = points  # string of points
        self.ndim = np.shape(self.points)[1]
        self.angle = angle  # maximum turning angle [degrees]
        self.dx_min = dx_min  # minimum pannel length
        self.dx_max = dx_max  # maximum pannel length
        self.space(verbose=verbose)

    def space(self, verbose=False, **kwargs):
        """
        Option to override init defaults with kwargs
        """
        self.angle = kwargs.get("angle", self.angle)
        self.dx_min = kwargs.get("dx_min", self.dx_min)
        self.dx_max = kwargs.get("dx_max", self.dx_max)

        t_vector = self.points[1:] - self.points[:-1]  # tangent vector
        t_vec_norm = np.linalg.norm(t_vector, axis=1)
        t_vec_norm[t_vec_norm == 0] = 1e-36  # protect zero division
        median_dt = np.median(t_vec_norm)  # average step length
        t_vector /= t_vec_norm.reshape(-1, 1) * np.ones((1, np.shape(t_vector)[1]))

        self.new_points = np.zeros(np.shape(self.points))

        self.index = np.zeros(len(self.points), dtype=int)
        delta_x, delta_turn = np.zeros(len(self.points)), np.zeros(len(self.points))
        self.new_points[0] = self.points[0]
        to, po = t_vector[0], self.points[0]
        k = count(1)
        for i, (p, t) in enumerate(zip(self.points[1:], t_vector)):
            c = np.cross(to, t)
            c_mag = np.linalg.norm(c)
            dx = np.linalg.norm(p - po)  # segment length
            if (
                c_mag > np.sin(self.angle * np.pi / 180) and dx > self.dx_min
            ) or dx + median_dt > self.dx_max:  # store
                j = next(k)
                self.new_points[j] = self.points[i]  # pivot point
                self.index[j] = i + 1  # pivot index
                delta_x[j - 1] = dx  # pannel length
                delta_turn[j - 1] = np.arcsin(c_mag) * 180 / np.pi
                to, po = t, p  # update
        if dx > self.dx_min:
            j = next(k)
            delta_x[j - 1] = dx  # last segment length
        else:
            delta_x[j - 1] += dx  # last segment length
        self.new_points[j] = p  # replace / append last point
        self.index[j] = i + 1  # replace / append last point
        self.n = j + 1  # reduced point number
        self.new_points = self.new_points[: j + 1]  # trim
        self.index = self.index[: j + 1]  # trim
        delta_x = delta_x[:j]  # trim

        if verbose:
            print(f"\nturning angle: {self.angle:1.2f}")
            print(
                f"minimum pannel length: {np.min(delta_x):1.2f}, set: {self.dx_min:1.2f}"
            )
            print(
                f"maximum pannel length: { np.max(delta_x):1.2f}, set: {self.dx_max:1.2f}"
            )

            print(
                f"points input: {len(self.points)}, simplified: {len(self.new_points)}\n"
            )

    def plot(self, projection="2D", aspect=1):
        """
        Plot the String.
        """
        fig = plt.gcf()
        fig_width = fig.get_figwidth()
        fig_height = fig_width / aspect
        fig.set_figheight(fig_height)

        if projection == "3D":
            ax = fig.gca(projection="3d")
            ax.plot(
                self.new_points[:, 0], self.new_points[:, 1], self.new_points[:, 2], "o-"
            )
            ax.plot(self.points[:, 0], self.points[:, 1], self.points[:, 2], "-")
        else:
            ax = fig.gca()
            ax.plot(self.points[:, 0], self.points[:, 1], "-")
            ax.plot(self.new_points[:, 0], self.new_points[:, 1], "o-", ms=15)

        ax.set_axis_off()
        ax.set_aspect("equal")

        if projection == "3D":
            bb, zo = 7, 5
            ax.set_xlim([-bb, bb])
            ax.set_ylim([-bb, bb])
            ax.set_zlim([-bb + zo, bb + zo])
