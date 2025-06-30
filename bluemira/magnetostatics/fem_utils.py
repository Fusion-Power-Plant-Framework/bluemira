# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Finite element method utilities
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING
from unittest.mock import patch

import gmsh
import matplotlib.pyplot as plt
import matplotlib.tri as tr
import numpy as np
import ufl
from dolfinx import cpp, geometry, plot
from dolfinx.fem import (
    Constant,
    Expression,
    Function,
    assemble_scalar,
    create_interpolation_data,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.io import gmshio
from dolfinx.mesh import entities_to_geometry
from dolfinx.plot import vtk_mesh
from mpi4py import MPI
from petsc4py import PETSc

from bluemira.base.look_and_feel import bluemira_debug, bluemira_warn

if TYPE_CHECKING:
    import numpy.typing as npt
    from dolfinx.mesh import Mesh

old_m_to_m = gmshio.model_to_mesh


def convert_to_points_array(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """
    Convert points to array

    Returns
    -------
    x:
        Array of points.
    """
    x = np.asarray(x)
    if len(x.shape) == 1:
        if len(x) == 2:  # noqa: PLR2004
            x = np.asarray([x[0], x[1], 0])
        x = np.asarray([x])
    if x.shape[1] == 2:  # noqa: PLR2004
        x = np.asarray([x[:, 0], x[:, 1], np.zeros(x.shape[0])]).T
    return x


def model_to_mesh(
    model: type[gmsh.model] | None = None,
    comm=MPI.COMM_WORLD,
    rank: int = 0,
    gdim: int | Sequence[int] = 3,
    **kwargs,
):
    """
    Convert gmsh model to dolfinx mesh.

    Returns
    -------
    result:
        Dolfinx mesh.
    labels:
        Associated names.

    Notes
    -----
    Patches dolfinx.io.gmshio.model_to_mesh to allow non sequential dimensions
    """
    if isinstance(gdim, Sequence):
        dimensions = gdim
        gdim = len(dimensions)
    else:
        dimensions = np.arange(2)[:gdim]

    if model is None:
        model = gmsh.model

    labels = {
        model.getPhysicalName(dim, tag): (dim, tag)
        for dim, tag in model.getPhysicalGroups()
    }

    extr_geometry = functools.partial(
        extract_geometry, gmshio.extract_geometry, dimensions
    )
    with patch("dolfinx.io.gmshio.extract_geometry", new=extr_geometry):
        result = old_m_to_m(model, comm, rank, gdim, **kwargs)
    return result, labels


def extract_geometry(
    func: Callable[[type[gmsh.model]], np.ndarray],
    dimensions: Iterable[int],
    model: type[gmsh.model],
):
    """
    Extract model geometry

    Designed to call dolfinx.io.gmshio.extract_geometry but patch for non
    sequential dimensions

    Returns
    -------
    x:
        Extracted model geometry.
    """
    x = func(model)
    if any(dimensions != np.arange(len(dimensions))):
        return x[:, dimensions]
    return x


def calc_bb_tree(mesh: Mesh, padding: float = 0.0) -> geometry.BoundingBoxTree:
    """
    Calculate the BoundingBoxTree of a dolfinx mesh

    Returns
    -------
    :
        Bounding box tree.
    """
    return geometry.bb_tree(mesh, mesh.topology.dim, padding=padding)


class BluemiraFemFunction(Function):
    """A supporting class that extends the BluemiraFemFunction implementing
    the __call__ function to return the interpolated function value on the specified
    points.

    Notes
    -----
    In dolfinx v.0.5.0 it seems not to be possible to get a function value in a point
    without doing all the operations incorporated in eval_f. In case this functionality
    is integrated in most recent version of dolfinx, this class can be removed and
    replaced simply by BluemiraFemFunction.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._bb_tree = calc_bb_tree(self.function_space.mesh)

    def interpolate(self, u, cells=None):
        """Interpolate function and cache bb_tree"""
        if cells is None:
            mesh = self.function_space.mesh
            cell_map = mesh.topology.index_map(mesh.topology.dim)
            num_cells_on_proc = cell_map.size_local + cell_map.num_ghosts
            cells = np.arange(num_cells_on_proc, dtype=np.int32)

        if hasattr(u, "function_space"):
            nmm = create_interpolation_data(
                self.function_space,
                u.function_space,
                cells,
                padding=1e-8,
            )
            super().interpolate_nonmatching(u, cells, interpolation_data=nmm)
        else:
            super().interpolate(u, cells)
        calc_bb_tree(self.function_space.mesh)

    def __call__(self, points: np.ndarray):
        """
        Call function

        Parameters
        ----------
        points:
            points at which function is evaluated

        Returns
        -------
        res:
            interpolated function value for new_points
        new_points:
            points for which interpolation was possible

        Notes
        -----
        Overwrite of the call function such that the behaviour is similar to the one
        expected for a Callable.
        """
        return self._eval_new(points)[0]

    def _eval_new(self, points: np.ndarray | list):
        """
        Supporting function for __call__

        Parameters
        ----------
        points:
            points at which function is evaluated

        Returns
        -------
        res:
            interpolated function value for new_points
        new_points:
            points for which interpolation was possible

        """
        # initial_shape = points.shape
        res, new_points = (
            np.squeeze(a) for a in eval_f(self, convert_to_points_array(points))
        )
        if res.size == 1:
            return res[()], new_points[()]
        return res, new_points


def closest_point_in_mesh(mesh: Mesh, points: np.ndarray) -> np.ndarray:
    """Calculate closest point in mesh

    Parameters
    ----------
    mesh:
        mesh
    points:
        points at which a function is evaluated

    Returns
    -------
    new_points:
        closest point(s) in a mesh


    TODO hopefully remove in dolfinx >0.7.1
    """
    shape = points.shape
    points = convert_to_points_array(points)

    tdim = mesh.topology.dim
    num_entities_local = (
        mesh.topology.index_map(tdim).size_local
        + mesh.topology.index_map(tdim).num_ghosts
    )
    # _colliding_entity_bboxes = geometry.compute_collisions_points(tree, points)
    geom_dofs = entities_to_geometry(
        mesh,
        dim=tdim,
        entities=geometry.compute_closest_entity(
            tree=geometry.bb_tree(mesh, tdim),
            midpoint_tree=geometry.create_midpoint_tree(
                mesh, tdim, np.arange(num_entities_local, dtype=np.int32)
            ),
            mesh=mesh,
            points=points,
        ),
        permute=False,
    )

    dist = []
    min_arg_dist = []

    # NOTE: compute_distance_gjk must to be applied point to point
    for i, p in enumerate(points):
        temp = np.array([
            np.linalg.norm(geometry.compute_distance_gjk(p, mesh.geometry.x[dof]))
            for dof in geom_dofs[i]
        ])
        dist.append(temp)
        min_arg_dist.append(np.argmin(temp))

    new_points = np.array([
        p
        - geometry.compute_distance_gjk(
            p, mesh.geometry.x[geom_dofs[i][min_arg_dist[i]]]
        )
        for i, p in enumerate(points)
    ])

    if len(shape) == 1:
        new_points = new_points[0]
    return new_points


def calculate_area(
    mesh: Mesh, boundaries: object | None = None, tag: int | None = None
) -> float:
    """
    Calculate the area of a sub-domain

    Parameters
    ----------
    mesh:
        mesh of the FEM model
    boundaries:
        boundaries mesh tags
    tag:
        subdomain tag

    Returns
    -------
    :
        area of the subdomain
    """
    return integrate_f(Constant(mesh, PETSc.ScalarType(1)), mesh, boundaries, tag)


def integrate_f(
    f: Constant | BluemiraFemFunction,
    mesh: Mesh,
    boundaries=None,
    tag: int | None = None,
) -> float:
    """
    Calculate the integral of a function on the specified sub-domain

    Parameters
    ----------
    function:
        function to be integrated in the subdomain
    mesh:
        mesh of the FEM model
    boundaries:
        boundaries mesh tags
    tag:
        subdomain tag (default None). When None,
        the integral is made on the whole mesh domain.

    Returns
    -------
    :
        area of the subdomain
    """
    dx = (
        ufl.dx
        if boundaries is None
        else ufl.Measure("dx", subdomain_data=boundaries, domain=mesh)
    )
    return (
        assemble_scalar(form(f * dx))
        if tag is None
        else assemble_scalar(form(f * dx(tag)))
    )


# Plotting method for scalar field.
# Not really clear how to manage pyvista plots. Left here just as reference
# for a better future implementation.
@contextmanager
def pyvista_plot_show_save(filename: str = "field.svg"):
    """Show or save figure from pyvista

    Yields
    ------
    :
        the pyvista plotter
    """
    import pyvista  # noqa: PLC0415

    if pyvista.OFF_SCREEN:
        pyvista.start_xvfb()
    plotter = pyvista.Plotter()

    try:
        yield plotter
    finally:
        if pyvista.OFF_SCREEN:
            plotter.screenshot(filename)
        else:
            plotter.show()


def plot_fem_scalar_field(field: BluemiraFemFunction, filename: str = "field.svg"):
    """
    Plot a scalar field given by the dolfinx function "field" in pyvista or in a file.

    Parameters
    ----------
    field:
        function to be plotted
    filename:
        file in which the plot is stored

    Notes
    -----
    if pyvista.OFF_SCREEN is False the plot is shown on the screen, otherwise
    the file with the plot is saved.
    """
    import pyvista  # noqa: PLC0415

    with pyvista_plot_show_save(filename) as plotter:
        V = field.function_space  # noqa: N806
        degree = V.ufl_element().degree()
        field_grid = pyvista.UnstructuredGrid(*vtk_mesh(V.mesh if degree == 0 else V))
        field_grid.point_data["Field"] = field.x.array
        field_grid.set_active_scalars("Field")
        warp = field_grid.warp_by_scalar("Field", factor=1)
        _actor = plotter.add_mesh(warp, show_edges=True)


def error_L2(  # noqa: N802
    uh: BluemiraFemFunction,
    u_ex: BluemiraFemFunction | Expression,
    degree_raise: int = 0,
) -> float:
    """
    Calculate the L2 error norm between two functions.
    This method has been taken from https://jsdokken.com/dolfinx-tutorial/chapter4/convergence.html.

    Parameters
    ----------
    uh:
        first function
    u_ex:
        second function or an expression (in case, for example, of exact solution)
    degree_raise:
        increase of polynomial degree with respect uh degree
        (useful when comparing with an exact solution)

    Returns
    -------
    :
        integral error
    """
    # Create higher order function space
    degree = uh.function_space.ufl_element().degree()
    family = uh.function_space.ufl_element().family()
    mesh = uh.function_space.mesh
    W = functionspace(mesh, (family, degree + degree_raise))  # noqa: N806
    # Interpolate approximate solution
    u_W = BluemiraFemFunction(W)  # noqa: N806
    u_W.interpolate(uh)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = BluemiraFemFunction(W)  # noqa: N806
    if not isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = Expression(u_ex, W.element.interpolation_points())
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)

    # Compute the error in the higher order function space
    e_W = BluemiraFemFunction(W)  # noqa: N806
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    error = form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)

    return np.sqrt(error_global)


def eval_f(function: Function, points: np.ndarray) -> tuple[np.ndarray, ...]:
    """
    Evaluate the value of a dolfinx function in the specified points

    Parameters
    ----------
    function:
        reference function
    points:
        points on which the function shall be calculated


    Returns
    -------
    values:
        values of the function at the specified points
    points_on_proc:
        specified points

    """
    mesh = function.function_space.mesh

    bb_tree = function._bb_tree if hasattr(function, "_bb_tree") else calc_bb_tree(mesh)
    cells = []
    points_on_proc = []

    # points = closest_point_in_mesh(mesh, points)
    # Find cells whose bounding-box collide with the points
    cell_candidates = geometry.compute_collisions_points(bb_tree, points)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points)
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
        else:
            closest_point = closest_point_in_mesh(mesh, np.array([point]))
            temp_cell_candidates = geometry.compute_collisions_points(
                bb_tree, closest_point
            )
            temp_colliding_cells = geometry.compute_colliding_cells(
                mesh, temp_cell_candidates, closest_point
            )
            if (tcc := temp_colliding_cells.links(0)).size > 0:
                points_on_proc.append(closest_point[0])
                cells.append(tcc[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    values = np.array(function.eval(points_on_proc, cells)).reshape(points.shape[0], -1)

    if points.shape != points_on_proc.shape:
        bluemira_debug(
            "Some points cannot be interpolated (no colliding cells have been found "
            f"for {points.shape[0] - points_on_proc.shape[0]} points)."
            f"Original points: {points.shape} - "
            f"Interpolated points: {points_on_proc.shape}"
        )

    return values, points_on_proc


def plot_scalar_field(
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
    levels: int | np.ndarray = 20,
    ax: plt.Axes | None = None,
    *,
    contour: bool = True,
    tofill: bool = True,
    **kwargs,
) -> dict[str, plt.Axes | None]:
    """
    Plot a scalar field from numpy arrays.

    Parameters
    ----------
    x:
        x coordinate array
    z:
        z coordinate array
    data:
        value array
    levels:
        Number of contour levels to plot or values for contour levels
    axis:
        axis onto which to plot
    contour:
        Whether or not to plot contour lines
    tofill:
        Whether or not to plot filled contours

    Returns
    -------
    :
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

    triang = tr.Triangulation(x, y)

    # plot only triangles with sidelength smaller some max_radius
    triangles = triang.triangles
    # Mask off unwanted triangles.
    xtri = x[triangles] - np.roll(x[triangles], 1, axis=1)
    ytri = y[triangles] - np.roll(y[triangles], 1, axis=1)
    _maxi = np.max(np.sqrt(xtri**2 + ytri**2), axis=1)

    if contour:
        cntr = ax.tricontour(triang, data, levels=levels, **contour_kwargs)

    if tofill:
        cntrf = ax.tricontourf(triang, data, levels=levels, cmap="RdBu_r")
        fig.colorbar(cntrf, ax=ax)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal")

    return {"ax": ax, "cntr": cntr, "cntrf": cntrf}


def read_from_msh(
    filename: str,
    comm=MPI.COMM_WORLD,
    rank: int = 0,
    gdim: int | tuple = 3,
    partitioner=None,
):
    """Wraps `dolfinx.io.gmshio.read_from_msh` to patch dimensional reading


    Parameters
    ----------
    filename: Name of ``.msh`` file.
    comm: MPI communicator to create the mesh on.
    rank: Rank of ``comm`` responsible for reading the ``.msh`` file.
    gdim: Geometric dimension of the mesh

    Returns
    -------
    ``(mesh, cell_tags, facet_tags), labels`` with meshtags for
    associated physical groups for cells and facets.

    """
    with patch("dolfinx.io.gmshio.model_to_mesh", new=model_to_mesh):
        return gmshio.read_from_msh(filename, comm, rank, gdim, partitioner)


@dataclass
class Association:
    """Mesh associations (density current, boundaries tag, target current)"""

    v: BluemiraFemFunction | Callable | float
    tag: int
    Itot: float | None = None


def create_j_function(
    mesh: Mesh, cell_tags, values: list[Association], eltype: tuple = ("DG", 0)
) -> BluemiraFemFunction:
    """
    Create the dolfinx current density function for the whole domain given a set of
    current density for each sub-domain.

    Parameters
    ----------
    mesh: Mesh
        mesh of the FEM model
    cell_tags:
        mesh cell tags
    values:
        list of association (density current, boundaries tag, target current).
        If target current is not None, the applied current density is rescaled
        in order to obtain the total target current in the subdomain identified
        by the boundaries tag.
    eltype:
        a tuple of type (element family, element degree)

    Returns
    -------
    J:
        a dolfinx function on the function space (mesh, eltype) with the values of
        the density current to be applied at each cell

    Raises
    ------
    ValueError
        value not callable or a number

    Notes
    -----
    If multiple functions are defined on the same subdomain, the contributions
    of each function are summed up.
    """
    function_space = functionspace(mesh, eltype)

    unique_tags = np.unique(cell_tags.values)
    J = BluemiraFemFunction(function_space)  # noqa: N806

    temp = BluemiraFemFunction(function_space)

    for value in values:
        if value.tag in unique_tags:
            cells = cell_tags.find(value.tag)
            dofs = locate_dofs_topological(function_space, 2, cells)

            if isinstance(value.v, BluemiraFemFunction | Callable):
                temp.interpolate(value.v, cells)
            elif isinstance(value.v, int | float):
                temp.x.array[dofs] += value.v
            else:
                raise ValueError(
                    f"{value.v} is not a number, Callable or BluemiraFemFunction object"
                )
            if value.Itot is not None:
                factor = value.Itot / integrate_f(temp, mesh, cell_tags, value.tag)
                temp.x.array[dofs] *= factor

            J.x.array[dofs] += temp.x.array[dofs]
        else:
            bluemira_warn(f"Tag {value.tag} is not in boundaries")
    return J


def compute_B_from_Psi(
    psi: BluemiraFemFunction, eltype: tuple, eltype1: tuple | None = None
) -> BluemiraFemFunction:
    """
    Compute the magnetic flux density given the magnetic flux function
    for an axisymmetric problem.

    Parameters
    ----------
    psi:
        the magnetic flux function in the mesh domain
    eltype:
        Element type identified (e.g. ("P", 1)) for the magnetic flux density function

    Returns
    -------
    B:
        Magnetic flux density function in the mesh domain
    """
    mesh = psi.function_space.mesh
    W0 = functionspace(mesh, (*eltype, (mesh.geometry.dim,)))  # noqa: N806
    B0 = BluemiraFemFunction(W0)

    x_0 = ufl.SpatialCoordinate(mesh)[0]

    B_expr = Expression(
        ufl.as_vector((
            -psi.dx(1) / (2 * np.pi * x_0),
            psi.dx(0) / (2 * np.pi * x_0),
        )),
        W0.element.interpolation_points(),
    )

    B0.interpolate(B_expr)

    if eltype1 is not None:
        W = functionspace(mesh, (*eltype1, (mesh.geometry.dim,)))  # noqa: N806
        B = BluemiraFemFunction(W)
        B.interpolate(B0)
    else:
        B = B0

    return B


def plot_meshtags(
    mesh: Mesh,
    meshtags: cpp.mesh.MeshTags_float64 | cpp.mesh.MeshTags_int32 | None = None,
    filename: str = "meshtags.svg",
):
    """
    Plot dolfinx mesh with markers using pyvista.

    Parameters
    ----------
    mesh:
        Mesh
    meshtags:
        Mesh tags
    filename:
        Full path for plot save
    """
    # Create VTK mesh
    import pyvista  # noqa: PLC0415

    cells, types, x = plot.vtk_mesh(mesh)
    grid = pyvista.UnstructuredGrid(cells, types, x)

    # Attach the cells tag data to the pyvita grid
    if meshtags is not None:
        grid.cell_data["Marker"] = meshtags.values
        grid.set_active_scalars("Marker")

    with pyvista_plot_show_save(filename) as plotter:
        plotter.add_mesh(grid, show_edges=True, show_scalar_bar=False)
        plotter.view_xy()
