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

"""Module to support the fem_fixed_boundary implementation"""

from typing import Callable, Iterable, List, Optional, Tuple, Union

import dolfin
import matplotlib.pyplot as plt
import numpy as np
import scipy
from matplotlib._tri import TriContourGenerator
from matplotlib.axes._axes import Axes
from matplotlib.tri.triangulation import Triangulation
from scipy.interpolate import interp1d

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.utilities.optimiser import Optimiser, approx_derivative
from bluemira.utilities.tools import is_num


def b_coil_axis(r, z, pz, curr):
    """
    Return the module of the magnetic field of a coil (of radius r and centered in
    (0, z)) calculated on a point on the coil axis at a distance pz from the
    axis origin.

    TODO: add equation
    """
    return 4 * np.pi * 1e-7 * curr * r**2 / (r**2 + (pz - z) ** 2) ** 1.5 / 2.0


def _convert_const_to_dolfin(value: float):
    """Convert a constant value to a dolfin function"""
    if not isinstance(value, (int, float)):
        raise ValueError("Value must be integer or float.")

    return dolfin.Constant(value)


class ScalarSubFunc(dolfin.UserExpression):
    """
    Create a dolfin UserExpression from a set of functions defined in the subdomains

    Parameters
    ----------
    func_list: Union[Iterable[Union[float, Callable]], float, Callable]
        list of functions to be interpolated into the subdomains. Int and float values
        are considered as constant functions. Any other callable function must return
        a single value.
    mark_list: Iterable[int]
        list of markers that identify the subdomain in which the respective functions
        of func_list must to be applied.
    subdomains: dolfin.cpp.mesh.MeshFunctionSizet
        the whole subdomains mesh function
    """

    def __init__(
        self,
        func_list: Union[Iterable[Union[float, Callable]], float, Callable],
        mark_list: Optional[Iterable[int]] = None,
        subdomains: Optional[dolfin.cpp.mesh.MeshFunctionSizet] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.functions = self.check_functions(func_list)
        self.markers = mark_list
        self.subdomains = subdomains

    def check_functions(
        self,
        functions: Union[Iterable[Union[float, Callable]], float, Callable],
    ) -> Iterable[Union[float, Callable]]:
        """Check if the argument is a function or a list of fuctions"""
        if not isinstance(functions, Iterable):
            functions = [functions]
        if all(isinstance(f, (float, Callable)) for f in functions):
            return functions
        raise ValueError(
            "Accepted functions are instance of (int, float, Callable)"
            "or a list of them."
        )

    def eval_cell(self, values: List, x: float, cell):
        """Evaluate the value on each cell"""
        if self.markers is None or self.subdomains is None:
            func = self.functions[0]
        else:
            m = self.subdomains[cell.index]
            func = (
                self.functions[np.where(np.array(self.markers) == m)[0][0]]
                if m in self.markers
                else 0
            )
        if callable(func):
            values[0] = func(x)
        elif isinstance(func, (int, float)):
            values[0] = func
        else:
            raise ValueError(f"{func} is not callable or is not a constant")

    def value_shape(self) -> Tuple:
        """
        Value_shape function (necessary for a UserExpression)
        https://fenicsproject.discourse.group/t/problems-interpolating-a-userexpression-and-plotting-it/1303
        """
        return ()


def plot_scalar_field(
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
    levels: int = 20,
    ax: Optional[Axes] = None,
    contour: bool = True,
    tofill: bool = True,
    **kwargs,
) -> Tuple[Axes, Union[Axes, None], Union[Axes, None]]:
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
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    defaults = {"linewidths": 2, "colors": "k"}
    contour_kwargs = {**defaults, **kwargs}

    cntr = None
    cntrf = None

    if contour:
        cntr = ax.tricontour(x, y, data, levels=levels, **contour_kwargs)

    if tofill:
        cntrf = ax.tricontourf(x, y, data, levels=levels, cmap="RdBu_r")
        fig.colorbar(cntrf, ax=ax)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal")

    return ax, cntr, cntrf


def plot_profile(
    x: np.ndarray,
    prof: np.ndarray,
    var_name: str,
    var_unit: str,
    ax: Optional[Axes] = None,
    show: bool = True,
):
    """
    Plot profile
    """
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(x, prof)
    ax.set(xlabel="x (-)", ylabel=f"{var_name} ({var_unit})")
    ax.grid()
    if show:
        plt.show()


def get_tricontours(
    x: np.ndarray, z: np.ndarray, array: np.ndarray, value: Union[float, Iterable]
) -> List[np.ndarray]:
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
    return [tcg.create_contour(val)[0][0] for val in value]


def find_flux_surface(psi_norm_func, psi_norm, mesh=None, n_points=100):
    """
    Find a flux surface in the psi_norm function precisely by normalised psi value.

    Parameters
    ----------
    psi_norm_func: Callable[[np.ndarray], float]
        Function to calculate normalised psi
    mesh: dolfin.Mesh
        Mesh object to use to estimate extrema prior to optimisation
    psi_norm: float
        Normalised psi value for which to find the flux surface
    mesh: Optional[dolfin.Mesh]
        Mesh object to use to estimate the flux surface
        If None, reasonable guesses are used.
    n_points: int
        Number of points along the flux surface

    Returns
    -------
    points: np.ndarray
        x, z coordinates of the flux surface
    """
    x_axis, z_axis = find_magnetic_axis(lambda x: -psi_norm_func(x), mesh=mesh)

    if mesh:
        search_range = mesh.hmax()
        mpoints = mesh.coordinates()
        psi_norm_array = [psi_norm_func(x) for x in mpoints]
        contour = get_tricontours(
            mpoints[:, 0], mpoints[:, 1], psi_norm_array, psi_norm
        )[0]
        d_guess = abs(np.max(contour[0, :]) - x_axis) - search_range
        bounds = [(d_guess - 2 * search_range, d_guess + 2 * search_range)]

        def get_next_guess(points, distances, i):  # noqa: U100
            return np.hypot(points[0, i - 1] - x_axis, points[1, i - 1] - z_axis)

    else:
        d_guess = 0.5
        bounds = [(0.1, None)]

        def get_next_guess(points, distances, i):  # noqa: U100
            return distances[i]

    def psi_norm_match(x):
        return abs(psi_norm_func(x) - psi_norm)

    def theta_line(d, theta_i):
        return float(x_axis + d * np.cos(theta_i)), float(z_axis + d * np.sin(theta_i))

    def psi_line_match(d, *args):
        return psi_norm_match(theta_line(d, args[0]))

    theta = np.linspace(0, 2 * np.pi, n_points - 1, endpoint=False, dtype=float)
    points = np.zeros((2, n_points), dtype=float)
    distances = np.zeros(n_points)
    for i in range(len(theta)):
        result = scipy.optimize.minimize(
            psi_line_match,
            x0=d_guess,
            args=(theta[i]),
            bounds=bounds,
            method="SLSQP",
            options={"disp": False, "ftol": 1e-14, "maxiter": 1000},
        )
        points[:, i] = theta_line(result.x, theta[i])
        distances[i] = result.x
        d_guess = get_next_guess(points, distances, i)

    points[:, -1] = points[:, 0]

    return points


def calculate_plasma_shape_params(
    psi_norm_func: Callable[[np.ndarray], np.ndarray],
    mesh: dolfin.Mesh,
    psi_norm: float,
    plot: bool = False,
) -> Tuple[float, float, float]:
    """
    Calculate the plasma parameters (r_geo, kappa, delta) for a given magnetic
    isoflux using optimisation.

    Parameters
    ----------
    psi_norm_func: Callable[[np.ndarray], float]
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
    ind_z_min = np.argmin(z)
    ind_x_max = np.argmax(x)
    ind_x_min = np.argmin(x)

    pu = contour[ind_z_max]
    pl = contour[ind_z_min]
    po = contour[ind_x_max]
    pi = contour[ind_x_min]

    search_range = mesh.hmax()

    def f_constrain_p95(x: np.ndarray) -> np.ndarray:
        """
        Constraint function for points on the psi_norm surface.
        """
        return psi_norm_func(x) - psi_norm

    def find_extremum(
        func: Callable[[np.ndarray], np.ndarray], x0: Iterable[float]
    ) -> np.ndarray:
        """
        Extremum finding using constrained optimisation
        """
        # TODO: Replace scipy minimize with something a little more robust
        bounds = [(xi - search_range, xi + search_range) for xi in x0]
        result = scipy.optimize.minimize(
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
    pl_opt = find_extremum(lambda x: x[1], pl)

    po_opt = find_extremum(lambda x: -x[0], po)
    pu_opt = find_extremum(lambda x: -x[1], pu)

    if plot:
        _, ax = plt.subplots()
        dolfin.plot(mesh)
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
    kappa = 1 if a == 0 else b / a

    # triangularity
    c = r_geo - pl_opt[0]
    d = r_geo - pu_opt[0]
    delta = 0 if a == 0 else 0.5 * (c + d) / a

    return r_geo, kappa, delta


def find_magnetic_axis(psi_func, mesh=None):
    """
    Find the magnetic axis in the poloidal flux map.

    Parameters
    ----------
    psi_func: Callable[[np.ndarray], float]
        Function to return psi at a given point
    mesh: Optional[dolfin.Mesh]
        Mesh object to use to estimate magnetic axis prior to optimisation
        If None, a reasonable guess is made.

    Returns
    -------
    mag_axis: np.ndarray
        Position vector (2) of the magnetic axis [m]
    """
    optimiser = Optimiser(
        "SLSQP", 2, opt_conditions={"ftol_abs": 1e-6, "max_eval": 1000}
    )

    if mesh:
        points = mesh.coordinates()
        psi_array = [psi_func(x) for x in points]
        psi_max_arg = np.argmax(psi_array)

        x0 = points[psi_max_arg]
        search_range = mesh.hmax()
        lower_bounds = x0 - search_range
        upper_bounds = x0 + search_range
    else:
        x0 = np.array([0.1, 0.0])
        lower_bounds = np.array([0, -2.0])
        upper_bounds = np.array([20.0, 2.0])

    def maximise_psi(x, grad):
        result = -psi_func(x)
        if grad.size > 0:
            grad[:] = approx_derivative(
                lambda x: -psi_func(x),
                x,
                f0=result,
                bounds=[lower_bounds, upper_bounds],
            )

        return result

    optimiser.set_objective_function(maximise_psi)

    optimiser.set_lower_bounds(lower_bounds)
    optimiser.set_upper_bounds(upper_bounds)
    x_star = optimiser.optimise(x0)
    return np.array(x_star, dtype=float)


def _interpolate_profile(
    x: np.ndarray, profile_data: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Interpolate profile data"""
    return interp1d(x, profile_data, kind="linear", fill_value="extrapolate")
