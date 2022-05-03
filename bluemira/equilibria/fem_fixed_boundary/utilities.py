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

"""Module to support the fem_fixed_boundary implementation"""

from typing import Callable

import dolfin
import matplotlib.pyplot as plt
import numpy as np
import scipy
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.tools import interpolate_bspline


def b_coil_axis(r, z, pz, curr):
    """
    Return the module of the magnetic field of a coil (of radius r and centered in
    (0, z)) calculated on a point on the coil axis at a distance pz from the
    axis origin.
    """
    return 4 * np.pi * 1e-7 * curr * r**2 / (r**2 + (pz - z) ** 2) ** 1.5 / 2.0


def _convert_const_to_dolfin(value):
    """Convert a constant value to a dolfin function"""
    if isinstance(value, (int, float)):
        return dolfin.Constant(float)
    else:
        raise ValueError("Value must be integer or float.")


class ScalarSubFunc(dolfin.UserExpression):
    """
    Create a dolfin UserExpression from a set of functions defined in the subdomains

    Parameters
    ----------
    func_list: Iterable[int, float, callable]
        list of functions to be interpolated into the subdomains. Int and float values
        are considered as constant functions. Any other callable function must return
        a single value.
    mark_list: Iterable[int]
        list of markers that identify the subdomain in which the respective functions
        of func_list must to be applied.
    subdomains: dolfin.cpp.mesh.MeshFunctionSizet
        the whole subdomains mesh function
    """

    def __init__(self, func_list, mark_list=None, subdomains=None, **kwargs):
        super().__init__(**kwargs)
        self.functions = self.check_functions(func_list)
        self.markers = mark_list
        self.subdomains = subdomains

    def check_functions(self, functions):
        if isinstance(functions, (int, float, Callable)):
            return [functions]
        if isinstance(functions, list):
            if all(isinstance(f, (int, float, Callable)) for f in functions):
                return functions
        raise ValueError(
            "Accepted functions are instance of (int, float, Callable)"
            "or alist of them."
        )

    def eval_cell(self, values, x, cell):
        """Evaluate the value on each cell"""
        if self.markers is None or self.subdomains is None:
            func = self.functions[0]
        else:
            m = self.subdomains[cell.index]
            if m in self.markers:
                index = np.where(np.array(self.markers) == m)
                func = self.functions[index[0][0]]
            else:
                func = 0
        if callable(func):
            values[0] = func(x)
        elif isinstance(func, (int, float)):
            values[0] = func
        else:
            raise ValueError(f"{func} is not callable or is not a constant")

    def value_shape(self):
        """
        Value_shape function (necessary for a UserExpression)
        https://fenicsproject.discourse.group/t/problems-interpolating-a-userexpression-and-plotting-it/1303
        """
        return ()


def plot_scalar_field(x, y, z, levels=20, axis=None, contour=True, tofill=True,
                      **kwargs):

    cntr = None
    cntrf = None

    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot()

    if not kwargs:
        kwargs = {"linewidths": 2, "colors": "k"}

    # # ----------
    # # Tricontour
    # # ----------
    # # Directly supply the unordered, irregularly spaced coordinates
    # # to tricontour.
    # opts = {'linewidths': 0.5, 'colors':'k'}
    if contour:
        cntr = axis.tricontour(x, y, z, levels=levels, **kwargs)

    if tofill:
        cntrf = axis.tricontourf(x, y, z, levels=levels, cmap="RdBu_r")
        plt.gcf().colorbar(cntrf, ax=axis)

    plt.gca().set_aspect("equal")

    return axis, cntr, cntrf


def plot_profile(x, prof, var_name, var_unit):
    """
    Plot profile
    """
    fig, ax = plt.subplots()
    ax.plot(x, prof)
    ax.set(xlabel="x (-)", ylabel=var_name + " (" + var_unit + ")")
    ax.grid()
    plt.show()


class Solovev:
    def __init__(self, R0, a, kappa, delta, A1, A2):
        self.R0 = R0
        self.a = a
        self.kappa = kappa
        self.delta = delta
        self.A1 = A1
        self.A2 = A2
        self._findParams()

    def _findParams(self):
        ri = self.R0 - self.a
        ro = self.R0 + self.a
        rt = self.R0 - self.delta * self.a
        zt = self.kappa * self.a

        m = np.array(
            [
                [1.0, ri**2, ri**4, ri**2 * np.log(ri)],
                [1.0, ro**2, ro**4, ro**2 * np.log(ro)],
                [
                    1.0,
                    rt**2,
                    rt**2 * (rt**2 - 4 * zt**2),
                    rt**2 * np.log(rt) - zt**2,
                ],
                [0.0, 2.0, 4 * (rt**2 - 2 * zt**2), 2 * np.log(rt) + 1.0],
            ]
        )

        b = np.array(
            [
                [-(ri**4) / 8.0, 0],
                [-(ro**4) / 8.0, 0.0],
                [-(rt**4) / 8.0, +(zt**2) / 2.0],
                [-(rt**2) / 2.0, 0.0],
            ]
        )
        b = b * np.array([self.A1, self.A2])
        b = np.sum(b, axis=1)

        self.coeff = scipy.linalg.solve(m, b)
        print(f"Solovev coefficients: {self.coeff}")

    def psi(self, points):
        if len(points.shape) == 1:
            points = np.array([points])

        c = lambda x: np.array(
            [
                1.0,
                x[0] ** 2,
                x[0] ** 2 * (x[0] ** 2 - 4 * x[1] ** 2),
                x[0] ** 2 * np.log(x[0]) - x[1] ** 2,
                (x[0] ** 4) / 8.0,
                -(x[1] ** 2) / 2.0,
            ]
        )

        m = np.concatenate((self.coeff, np.array([self.A1, self.A2])))

        return [np.sum(c(x) * m) * 2 * np.pi for x in points]

    def plot_psi(self, ri, zi, dr, dz, nr, nz, levels=20, axis=None, tofill=True):
        r = np.linspace(ri, ri + dr, nr)
        z = np.linspace(zi, zi + dz, nz)
        rv, zv = np.meshgrid(r, z)
        points = np.vstack([rv.ravel(), zv.ravel()]).T
        psi = self.psi(points)
        cplot = plot_scalar_field(
            points[:, 0], points[:, 1], psi, levels=levels, axis=axis, tofill=tofill
        )
        return cplot + (points, psi)



def calculate_plasma_shape_params(points, data, levels):
    R_geo = np.zeros(len(levels))
    kappa = np.zeros(len(levels))
    delta = np.zeros(len(levels))

    axis, cntr, _ = plot_scalar_field(
        points[:, 0],
        points[:, 1],
        data,
        levels=levels,
        axis=None,
        tofill=False,
    )
    plt.show()

    for i in range(len(cntr.collections)):
        vertices = cntr.collections[i].get_paths()[0].vertices
        x = vertices.T[0]
        y = x*0
        z = vertices.T[1]
        vertices = Coordinates({'x': x, 'y': y, 'z': z})
        wire = interpolate_bspline(vertices,"psi_95", True)
        interp_points = wire.discretize(1000)

        ind_z_max = np.argmax(interp_points.z)
        PU = interp_points.T[ind_z_max]
        ind_z_min = np.argmin(interp_points.z)
        PL = interp_points.T[ind_z_min]
        ind_x_max = np.argmax(interp_points.x)
        PO = interp_points.T[ind_x_max]
        ind_x_min = np.argmin(interp_points.x)
        PI = interp_points.T[ind_x_min]

        # geometric center of a magnetic flux surface
        R_geo[i] = (PO[0] + PI[0])/2

        # elongation
        a = (PO[0] - PI[0])/2
        b = (PU[2] - PL[2])/2
        if a == 0:
            kappa[i] = 1
        else:
            kappa[i] = b/a

        # triangularity
        c = R_geo[i] - PL[0]
        d = R_geo[i] - PU[0]
        if a == 0:
            delta[i] = 0
        else:
            delta[i] = (c + d) / 2 / a

        return R_geo, kappa, delta
