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
from matplotlib._tri import TriContourGenerator
from matplotlib.axes._axes import Axes
from matplotlib.tri import Triangulation
from scipy.interpolate import interp1d

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.utilities.error import ExternalOptError
from bluemira.utilities.opt_problems import OptimisationConstraint, OptimisationObjective
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

    results = []
    for val in value:
        contour = tcg.create_contour(val)[0]
        if len(contour) > 0:
            results.append(contour[0])
        else:
            from bluemira.base.look_and_feel import bluemira_warn

            bluemira_warn(f"No tricontour found for {val=}")
    return results


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
        d_guess = np.array([abs(np.max(contour[0, :]) - x_axis) - search_range])

        def lower_bound(x):
            return max(0.001, x - search_range)

        def upper_bound(x):
            return x + search_range

    else:
        d_guess = np.array([0.5])

        def lower_bound(x):
            return 0.1

        def upper_bound(x):
            return np.inf

    def psi_norm_match(x):
        return abs(psi_norm_func(x) - psi_norm)

    def theta_line(d, theta_i):
        return float(x_axis + d * np.cos(theta_i)), float(z_axis + d * np.sin(theta_i))

    def psi_line_match(d, grad, theta):
        result = psi_norm_match(theta_line(d, theta))
        if grad.size > 0:
            grad[:] = approx_derivative(
                lambda x: psi_norm_match(theta_line(x, theta)),
                d,
                f0=result,
                bounds=[lower_bound(d), upper_bound(d)],
            )

        return result

    theta = np.linspace(0, 2 * np.pi, n_points - 1, endpoint=False, dtype=float)
    points = np.zeros((2, n_points), dtype=float)
    distances = np.zeros(n_points)
    for i in range(len(theta)):
        optimiser = Optimiser(
            "SLSQP", 1, opt_conditions={"ftol_abs": 1e-14, "max_eval": 1000}
        )
        optimiser.set_lower_bounds(lower_bound(d_guess))
        optimiser.set_upper_bounds(upper_bound(d_guess))
        optimiser.set_objective_function(
            OptimisationObjective(psi_line_match, f_objective_args={"theta": theta[i]})
        )
        result = optimiser.optimise(d_guess)

        points[:, i] = theta_line(result, theta[i])
        distances[i] = result
        d_guess = result

    points[:, -1] = points[:, 0]

    return points


def _f_max_radius(x, grad):
    result = -x[0]
    if grad.size > 0:
        grad[0] = -1.0
        grad[1] = 0.0
    return result


def _f_min_radius(x, grad):
    result = x[0]
    if grad.size > 0:
        grad[0] = 1.0
        grad[1] = 0.0
    return result


def _f_max_vert(x, grad):
    result = -x[1]
    if grad.size > 0:
        grad[0] = 0.0
        grad[1] = -1.0
    return result


def _f_min_vert(x, grad):
    result = x[1]
    if grad.size > 0:
        grad[0] = 0.0
        grad[1] = 1.0
    return result


def _f_constrain_psi_norm(
    constraint: np.ndarray,
    x: np.ndarray,
    grad: np.ndarray,
    psi_norm_func=None,
    lower_bounds=None,
    upper_bounds=None,
) -> np.ndarray:
    """
    Constraint function for points on the psi_norm surface.
    """
    result = psi_norm_func(x)
    constraint[:] = result
    if grad.size > 0:
        grad[:] = approx_derivative(
            psi_norm_func,
            x,
            f0=result,
            bounds=[lower_bounds, upper_bounds],
            method="3-point",
        )
    return np.array([result])


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

    def f_psi_norm(x):
        return psi_norm_func(x) - psi_norm

    points = mesh.coordinates()
    psi_norm_array = [psi_norm_func(x) for x in points]

    contour = get_tricontours(points[:, 0], points[:, 1], psi_norm_array, psi_norm)[0]
    x, z = contour.T

    pu = contour[np.argmax(z)]
    pl = contour[np.argmin(z)]
    po = contour[np.argmax(x)]
    pi = contour[np.argmin(x)]

    search_range = mesh.hmax()

    def find_extremum(
        func: Callable[[np.ndarray], np.ndarray], x0: Iterable[float]
    ) -> np.ndarray:
        """
        Extremum finding using constrained optimisation
        """
        lower_bounds = x0 - search_range
        upper_bounds = x0 + search_range
        # NOTE: COBYLA appears to do a better job here, as it seems that the
        # NLOpt implementation of SLSQP really requires a feasible starting
        # solution, which is not so readily available with this tight equality
        # constraint. The scipy SLSQP implementation apparently does not require
        # such a good starting solution. Neither SLSQP nor COBYLA can guarantee
        # convergence without a feasible starting point.
        optimiser = Optimiser(
            "COBYLA", 2, opt_conditions={"ftol_abs": 1e-10, "max_eval": 1000}
        )
        optimiser.set_objective_function(func)
        optimiser.set_lower_bounds(lower_bounds)
        optimiser.set_upper_bounds(upper_bounds)

        f_constraint = OptimisationConstraint(
            _f_constrain_psi_norm,
            f_constraint_args={
                "psi_norm_func": f_psi_norm,
                "lower_bounds": lower_bounds,
                "upper_bounds": upper_bounds,
            },
            constraint_type="equality",
        )

        optimiser.add_eq_constraints(f_constraint, tolerance=1e-10)
        try:
            x_star = optimiser.optimise(x0)
        except ExternalOptError as e:
            bluemira_warn(
                f"calculate_plasma_shape_params::find_extremum failing at {x0}, defaulting to mesh value: {e}"
            )
            x_star = x0
        return x_star

    pi_opt = find_extremum(_f_min_radius, pi)
    pl_opt = find_extremum(_f_min_vert, pl)
    po_opt = find_extremum(_f_max_radius, po)
    pu_opt = find_extremum(_f_max_vert, pu)

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
