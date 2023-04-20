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

import os
from typing import Callable, Iterable, List, Optional, Tuple, Union

import dolfin
import matplotlib.pyplot as plt
import numpy as np
from dolfin import BoundaryMesh, Mesh, Vertex
from matplotlib._tri import TriContourGenerator
from matplotlib.axes._axes import Axes
from matplotlib.tri import Triangulation
from scipy.interpolate import interp1d

from bluemira.base.components import PhysicalComponent
from bluemira.base.constants import EPS
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.flux_surfaces import ClosedFluxSurface
from bluemira.geometry.coordinates import Coordinates
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf
from bluemira.utilities.error import ExternalOptError
from bluemira.utilities.opt_problems import OptimisationConstraint, OptimisationObjective
from bluemira.utilities.optimiser import Optimiser, approx_derivative
from bluemira.utilities.tools import is_num


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
            bluemira_warn(f"No tricontour found for {val=}")
            results.append(None)
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


def get_mesh_boundary(mesh):
    """
    Retrieve the boundary of the mesh, as an ordered set of coordinates.

    Parameters
    ----------
    mesh: dolfin.Mesh
        Mesh for which to retrieve the exterior boundary

    Returns
    -------
    xbdry: np.ndarray
        x coordinates of the boundary
    zbdry: np.ndarray
        z coordinates of the boundary
    """
    boundary = BoundaryMesh(mesh, "exterior")
    edges = boundary.cells()
    check_edge = np.ones(boundary.num_edges())

    index = 0
    temp_edge = edges[index]
    sorted_v = []
    sorted_v.append(temp_edge[0])

    for i in range(len(edges) - 1):
        temp_v = [v for v in temp_edge if v not in sorted_v][0]
        sorted_v.append(temp_v)
        check_edge[index] = 0
        connected = np.where(edges == temp_v)[0]
        index = [e for e in connected if check_edge[e] == 1][0]
        temp_edge = edges[index]

    points_sorted = []
    for v in sorted_v:
        points_sorted.append(Vertex(boundary, v).point().array())
    points_sorted = np.array(points_sorted)
    return points_sorted[:, 0], points_sorted[:, 1]


def get_flux_surfaces_from_mesh(
    mesh,
    psi_norm_func: callable,
    x_1d: Optional[np.ndarray] = None,
    nx: Optional[int] = None,
):
    """
    Get a list of flux surfaces from a mesh and normalised psi callable.

    Parameters
    ----------
    mesh: Mesh
        Mesh for which to extract the flux surfaces
    psi_norm_func: callable
        Callable for psi_norm on the mesh
    x_1d: Optional[np.ndarray]
        Array of 1-D normalised psi_values [0..1]. If None, nx will
        define a linearly spaced vector.
    nx: Optional[int]
        Number of points to linearly space along [0..1]. If x_1d is
        defined, not used.

    Returns
    -------
    x_1d: np.ndarray
        The 1-D normalised psi_values for which flux surfaces could be
        retrieved.
    flux_surfaces: List[ClosedFluxSurface]
        The list of closed flux surfaces

    Notes
    -----
    x_1d is returned, as it is not always possible to return a flux surface for
    small values of normalised psi.
    """
    if x_1d is None:
        if nx is None:
            raise ValueError("Please input either x_1d: np.ndarray or nx: int.")
        else:
            x_1d = np.linspace(0, 1, nx)
    else:
        if nx is not None:
            bluemira_warn("x_1d and nx specified, discarding nx.")

    mesh_points = mesh.coordinates()
    x = mesh_points[:, 0]
    z = mesh_points[:, 1]
    psi_norm_data = np.array([psi_norm_func(p) for p in mesh_points])

    index = []
    flux_surfaces = []
    for i, xi in enumerate(x_1d):
        if np.isclose(xi, 1.0, rtol=0, atol=EPS):
            path = get_mesh_boundary(mesh)
            fs = Coordinates({"x": path[0], "z": path[1]})
            fs.close()
            flux_surfaces.append(ClosedFluxSurface(fs))
        else:
            path = get_tricontours(x, z, psi_norm_data, xi)[0]
            if path is not None:
                fs = Coordinates({"x": path.T[0], "z": path.T[1]})
                fs.close()
                flux_surfaces.append(ClosedFluxSurface(fs))
            else:
                index.append(i)

    n = len(index)
    for xi in range(n):
        x_1d = np.delete(x_1d, index[xi])

    return x_1d, flux_surfaces


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
    points = mesh.coordinates()
    psi_norm_array = [psi_norm_func(x) for x in points]

    contour = get_tricontours(points[:, 0], points[:, 1], psi_norm_array, psi_norm)[0]
    x, z = contour.T

    pu = contour[np.argmax(z)]
    pl = contour[np.argmin(z)]
    po = contour[np.argmax(x)]
    pi = contour[np.argmin(x)]

    if plot:
        _, ax = plt.subplots()
        dolfin.plot(mesh)
        ax.tricontour(points[:, 0], points[:, 1], psi_norm_array)
        ax.plot(x, z, color="r")
        ax.plot(*po, marker="o", color="r")
        ax.plot(*pi, marker="o", color="r")
        ax.plot(*pu, marker="o", color="r")
        ax.plot(*pl, marker="o", color="r")

        ax.set_aspect("equal")
        plt.show()

    # geometric center of a magnetic flux surface
    r_geo = 0.5 * (po[0] + pi[0])

    # elongation
    a = 0.5 * (po[0] - pi[0])
    b = 0.5 * (pu[1] - pl[1])
    kappa = 1 if a == 0 else b / a

    # triangularity
    c = r_geo - pl[0]
    d = r_geo - pu[0]
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


def cell_distance(cell, refine_point, distance):
    """Find the cells that are located in the vicinity of a point"""
    # Get the center of the cell
    cell_center = cell.midpoint()[:]

    # Calculate the distance between the cell center and the refinement point
    d = np.linalg.norm(cell_center - np.array(refine_point))

    # Refine the cell if it is close to the refinement point
    if d < distance:
        return True
    else:
        return False


def refine_mesh(mesh, refine_point, distance, num_levels=1):
    """Refine the mesh around a reference point"""
    for level in range(num_levels):
        cell_markers = dolfin.MeshFunction("bool", mesh, mesh.topology().dim())
        cell_markers.set_all(False)
        for cell in dolfin.cells(mesh):
            if cell_distance(cell, refine_point, distance):
                cell_markers[cell.index()] = True
        mesh = dolfin.refine(mesh, cell_markers)

    return mesh


def create_mesh(
    plasma: PhysicalComponent,
    directory: str,
    mesh_filename: str,
    mesh_name_msh: str,
) -> Mesh:
    """
    Create mesh
    """
    meshing.Mesh(meshfile=os.path.join(directory, mesh_name_msh))(plasma)
    msh_to_xdmf(mesh_name_msh, dimensions=(0, 2), directory=directory)
    return import_mesh(mesh_filename, directory=directory, subdomains=True)[0]
