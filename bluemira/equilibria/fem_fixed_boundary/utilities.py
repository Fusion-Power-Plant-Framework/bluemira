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
from matplotlib._tri import TriContourGenerator
from matplotlib.tri.triangulation import Triangulation
from scipy.optimize import minimize

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.utilities.tools import is_num


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
    x: np.array(n, m)
        x coordinate array
    z: np.array(n, m)
        z coordinate array
    data: np.array(n, m)
        value array
    levels: int
        Number of contour levels to plot
    axis: Optional[Axis]
        axis onto which to plot
    contour: bool
        Whether or not to plot contour lines
    tofill: bool
        Whether or not to plot filled contours

    Returns
    -------
    axis: Axis
        Matplotlib axis on which the plot ocurred
    """
    cntr = None
    cntrf = None

    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot()
    else:
        fig = plt.gcf()

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
        fig.colorbar(cntrf, ax=axis)

    axis.set_xlabel("x [m]")
    axis.set_ylabel("z [m]")
    axis.set_aspect("equal")

    return axis


def plot_profile(x, prof, var_name, var_unit):
    """
    Plot profile
    """
    fig, ax = plt.subplots()
    ax.plot(x, prof)
    ax.set(xlabel="x (-)", ylabel=var_name + " (" + var_unit + ")")
    ax.grid()
    plt.show()


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
    value: Union[float, Iterable]
        The value of the desired contour in the array

    Returns
    -------
    value_loop: List[np.array(ni, mi)]
        The points of the value contour in the array
    """
    tri = Triangulation(x, z)
    tcg = TriContourGenerator(tri.get_cpp_triangulation(), array)
    if is_num(value):
        value = [value]

    contours = []
    for val in value:
        contours.append(tcg.create_contour(val)[0][0])

    return contours


def calculate_plasma_shape_params(psi_norm_func, mesh, psi_norm, plot=False):
    """
    Calculate the plasma parameters (r_geo, kappa, delta) for a given magnetic
    isoflux using optimisation.

    Parameters
    ----------
    psi_norm_func: callable
        Function to calculate normalised psi
    mesh: dolfin.Mesh
        Mesh object to use to estimate extrema prior to optimisation
    psi_norm: float
        Normalised psi value for which to calculate the shape parameters
    plot: bool
        Whether or not to plot

    Returns
    -------
    r_geo: float
        Geometric major radius of the flux surface at psi_norm
    kappa: float
        Elongation of the flux surface at psi_norm
    delta: float
        Triangularity of the flux surface at psi_norm
    """
    points = mesh.coordinates()
    psi_norm_array = [psi_norm_func(x) for x in points]

    contour = get_tricontours(points[:, 0], points[:, 1], psi_norm_array, psi_norm)[0]
    x, z = contour.T
    ind_z_max = np.argmax(z)
    pu = contour[ind_z_max]
    ind_z_min = np.argmin(z)
    pl = contour[ind_z_min]
    ind_x_max = np.argmax(x)
    po = contour[ind_x_max]
    ind_x_min = np.argmin(x)
    pi = contour[ind_x_min]

    search_range = mesh.hmax()

    def f_constrain_p95(x):
        """
        Constraint function for points on the psi_norm surface.
        """
        return psi_norm_func(x) - psi_norm

    def find_extremum(func, x0):
        """
        Extremum finding using constrained optimisation
        """
        # TODO: Replace scipy minimize with something a little more robust
        bounds = [(xi - search_range, xi + search_range) for xi in x0]
        result = minimize(
            func,
            x0,
            constraints=({"fun": f_constrain_p95, "type": "eq"}),
            method="SLSQP",
            bounds=bounds,
            options={"disp": False, "ftol": 1e-10, "maxiter": 1000},
        )
        if not result.success:
            bluemira_warn("Flux surface extremum finding failing:\n" f"{result.message}")

        return result.x

    pi_opt = find_extremum(lambda x: x[0], pi)

    po_opt = find_extremum(lambda x: -x[0], po)

    pl_opt = find_extremum(lambda x: x[1], pl)

    pu_opt = find_extremum(lambda x: -x[1], pu)

    if plot:
        from dolfin import plot  # noqa

        _, ax = plt.subplots()
        plot(mesh)
        ax.tricontour(points[:, 0], points[:, 1], psi_norm_array)
        ax.plot(x, z, color="r")
        ax.plot(*po, marker="o", color="r")
        ax.plot(*pi, marker="o", color="r")
        ax.plot(*pu, marker="o", color="r")
        ax.plot(*pl, marker="o", color="r")

        ax.plot(*po_opt, marker="o", color="b")
        ax.plot(*pi_opt, marker="o", color="b")
        ax.plot(*pu_opt, marker="o", color="b")
        ax.plot(*pl_opt, marker="o", color="b")
        ax.set_aspect("equal")
        plt.show()

    pi, po, pu, pl = pi_opt, po_opt, pu_opt, pl_opt
    # geometric center of a magnetic flux surface
    r_geo = 0.5 * (po_opt[0] + pi_opt[0])

    # elongation
    a = 0.5 * (po_opt[0] - pi_opt[0])
    b = 0.5 * (pu_opt[1] - pl_opt[1])
    if a == 0:
        kappa = 1
    else:
        kappa = b / a

    # triangularity
    c = r_geo - pl_opt[0]
    d = r_geo - pu_opt[0]
    if a == 0:
        delta = 0
    else:
        delta = 0.5 * (c + d) / a

    return r_geo, kappa, delta
