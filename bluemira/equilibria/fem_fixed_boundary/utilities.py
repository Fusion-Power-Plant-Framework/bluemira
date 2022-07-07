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
        """Check if the argument is a function or a list of fuctions"""
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


def plot_scalar_field(
    x, y, data, levels=20, axis=None, contour=True, tofill=True, **kwargs
):
    """
    Plot a scalar field

    Parameters
    ----------
    x: Iterable
        x coordinate of the points in which
    """
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
        cntr = axis.tricontour(x, y, data, levels=levels, **kwargs)

    if tofill:
        cntrf = axis.tricontourf(x, y, data, levels=levels, cmap="RdBu_r")
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


from matplotlib._tri import TriContourGenerator
from matplotlib.tri.triangulation import Triangulation


def get_tricontours(x, z, array, value):
    """
    Get the contours of a value in a triangular set of points.

    Parameters
    ----------
    x: np.array(n, m)
        The x value array
    z: np.array(n, m)
        The z value array
    array: np.array(n, m)
        The value array
    value: float
        The value of the desired contour in the array

    Returns
    -------
    value_loop: np.array(ni, mi)
        The points of the value contour in the array
    """

    tri = Triangulation(x, z)
    tcg = TriContourGenerator(tri.get_cpp_triangulation(), array)
    return tcg.create_contour(value)[0][0]


def calculate_plasma_shape_params_new(points, psi, levels):
    """
    Calculate the plasma parameters (Rgeo, kappa, delta) for a given magnetic
    isoflux.

    Parameters
    ----------
    points: Iterable
        2D points on which psi has been calculated
    psi: Iterable
        scalar value of the plasma magnetic poloidal flux at points
    levels: Iterable
        values that identify the isoflux curve at which the plasma parameters are
        calculated

    Returns
    -------
    r_geo: Iterable
        array of averaged radial coordinate of the isoflux curves
    kappa:
        array of the kappa value of the isoflux curves
    delta:
        array of the delta value of the isoflux curves
    """
    r_geo = np.zeros(len(levels))
    kappa = np.zeros(len(levels))
    delta = np.zeros(len(levels))

    tri = Triangulation(points[:, 0], points[:, 1])
    tcg = TriContourGenerator(tri.get_cpp_triangulation(), psi)

    for i, value in enumerate(levels):
        vertices = tcg.create_contour(value)[0][0]
        x = vertices.T[0]
        y = x * 0
        z = vertices.T[1]
        vertices = Coordinates({"x": x, "y": y, "z": z})
        wire = interpolate_bspline(vertices, "psi_95", True)
        interp_points = wire.discretize(1000)

        ind_z_max = np.argmax(interp_points.z)
        pu = interp_points.T[ind_z_max]
        ind_z_min = np.argmin(interp_points.z)
        pl = interp_points.T[ind_z_min]
        ind_x_max = np.argmax(interp_points.x)
        po = interp_points.T[ind_x_max]
        ind_x_min = np.argmin(interp_points.x)
        pi = interp_points.T[ind_x_min]

        # geometric center of a magnetic flux surface
        r_geo[i] = (po[0] + pi[0]) / 2

        # elongation
        a = (po[0] - pi[0]) / 2
        b = (pu[2] - pl[2]) / 2
        if a == 0:
            kappa[i] = 1
        else:
            kappa[i] = b / a

        # triangularity
        c = r_geo[i] - pl[0]
        d = r_geo[i] - pu[0]
        if a == 0:
            delta[i] = 0
        else:
            delta[i] = (c + d) / 2 / a

    return r_geo, kappa, delta


def calculate_plasma_shape_params(points, psi, levels):
    """
    Calculate the plasma parameters (Rgeo, kappa, delta) for a given magnetic
    isoflux.

    Parameters
    ----------
    points: Iterable
        2D points on which psi has been calculated
    psi: Iterable
        scalar value of the plasma magnetic poloidal flux at points
    levels: Iterable
        values that identify the isoflux curve at which the plasma parameters are
        calculated

    Returns
    -------
    r_geo: Iterable
        array of averaged radial coordinate of the isoflux curves
    kappa:
        array of the kappa value of the isoflux curves
    delta:
        array of the delta value of the isoflux curves
    """
    r_geo = np.zeros(len(levels))
    kappa = np.zeros(len(levels))
    delta = np.zeros(len(levels))

    axis, cntr, _ = plot_scalar_field(
        points[:, 0],
        points[:, 1],
        psi,
        levels=levels,
        axis=None,
        tofill=False,
    )
    plt.show()

    for i in range(len(cntr.collections)):
        vertices = cntr.collections[i].get_paths()[0].vertices
        check = get_tricontours(points[:, 0], points[:, 1], psi, levels[i])
        print(np.average(vertices.T[0]))
        print(np.average(check.T[0]))
        x = vertices.T[0]
        y = x * 0
        z = vertices.T[1]
        vertices = Coordinates({"x": x, "y": y, "z": z})
        wire = interpolate_bspline(vertices, "psi_95", True)
        interp_points = wire.discretize(1000)

        ind_z_max = np.argmax(interp_points.z)
        pu = interp_points.T[ind_z_max]
        ind_z_min = np.argmin(interp_points.z)
        pl = interp_points.T[ind_z_min]
        ind_x_max = np.argmax(interp_points.x)
        po = interp_points.T[ind_x_max]
        ind_x_min = np.argmin(interp_points.x)
        pi = interp_points.T[ind_x_min]

        # geometric center of a magnetic flux surface
        r_geo[i] = (po[0] + pi[0]) / 2

        # elongation
        a = (po[0] - pi[0]) / 2
        b = (pu[2] - pl[2]) / 2
        if a == 0:
            kappa[i] = 1
        else:
            kappa[i] = b / a

        # triangularity
        c = r_geo[i] - pl[0]
        d = r_geo[i] - pu[0]
        if a == 0:
            delta[i] = 0
        else:
            delta[i] = (c + d) / 2 / a

        return r_geo, kappa, delta
