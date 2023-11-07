import functools
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
from unittest.mock import patch

import matplotlib.pyplot as plt
import matplotlib.tri as tr
import numpy as np
import pyvista
import ufl
from dolfinx import cpp, geometry, plot
from dolfinx.fem import (
    Constant,
    Expression,
    Function,
    assemble_scalar,
    form,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.io import gmshio
from dolfinx.mesh import Mesh
from dolfinx.plot import vtk_mesh
from mpi4py import MPI
from petsc4py import PETSc

from bluemira.base.look_and_feel import bluemira_error, bluemira_warn

old_m_to_m = gmshio.model_to_mesh


def convert_to_points_array(x):
    x = np.array(x)
    if len(x.shape) == 1:
        if len(x) == 2:
            x = np.array([x[0], x[1], 0])
        x = np.array([x])
    if x.shape[1] == 2:
        x = np.array([x[:, 0], x[:, 1], x[:, 0] * 0]).T
    return x


def model_to_mesh(model, comm, rank: int, gdim: Union[int, Iterable[int]] = 3, **kwargs):
    if isinstance(gdim, Iterable):
        dimensions = gdim
        gdim = len(dimensions)
    else:
        dimensions = np.arange(2)[:gdim]

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


def extract_geometry(func, dimensions, model):
    x = func(model)
    if any(dimensions != np.arange(len(dimensions))):
        return x[:, dimensions]
    return x


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

    def __call__(self, points: np.ndarray):
        """
        Call function

        Notes
        -----
        Overwrite of the call function such that the behaviour is similar to the one
        expected for a Callable.
        """
        return self._eval_new(points)[0]

    def _eval_new(self, points: Union[np.ndarray, List]):
        """
        Supporting function for __call__
        """
        # initial_shape = points.shape
        points = convert_to_points_array(points)
        res, new_points = eval_f(self, points)
        if len(res.shape) == 1:
            res = res[0]
            new_points = new_points[0]
        elif res.shape[1] == 1:
            res = res.reshape(-1)
            if res.shape[0] == 1:
                res = res[0]
                new_points = new_points[0]
        elif res.shape[0] == 1:  # and len(initial_shape) == 1:
            res = res[0]
            new_points = new_points[0]

        return res, new_points


def closest_point_in_mesh(mesh, points):
    points = convert_to_points_array(points)

    closest_points = []

    tdim = mesh.topology.dim
    tree = geometry.bb_tree(mesh, tdim)
    num_entities_local = (
        mesh.topology.index_map(tdim).size_local
        + mesh.topology.index_map(tdim).num_ghosts
    )
    entities = np.arange(num_entities_local, dtype=np.int32)
    midpoint_tree = geometry.create_midpoint_tree(mesh, tdim, entities)
    closest_entities = geometry.compute_closest_entity(tree, midpoint_tree, mesh, points)
    _colliding_entity_bboxes = geometry.compute_collisions_points(tree, points)
    mesh_geom = mesh.geometry.x
    geom_dofs = cpp.mesh.entities_to_geometry(
        mesh._cpp_object, tdim, np.atleast_2d(closest_entities), False
    )
    mesh_nodes = mesh_geom[geom_dofs][0]
    for p in points:
        displacement = geometry.compute_distance_gjk(p, mesh_nodes)
        closest_points.append(p - displacement)
    return np.array(closest_points)


def calculate_area(mesh: Mesh, boundaries: object, tag: Optional[int] = None) -> float:
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
    area of the subdomain
    """
    f = Constant(mesh, PETSc.ScalarType(1))

    return integrate_f(f, mesh, boundaries, tag)


def integrate_f(
    f: BluemiraFemFunction, mesh: Mesh, boundaries=None, tag: Optional[int] = None
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
def plot_fem_scalar_field(field: BluemiraFemFunction, filename: str = "field.png"):
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
    plotter = pyvista.Plotter()
    V = field.function_space
    degree = V.ufl_element().degree()
    field_grid = pyvista.UnstructuredGrid(*vtk_mesh(V.mesh if degree == 0 else V))
    field_grid.point_data["Field"] = field.x.array
    field_grid.set_active_scalars("Field")
    warp = field_grid.warp_by_scalar("Field", factor=1)
    _actor = plotter.add_mesh(warp, show_edges=True)
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        _field_fig = plotter.screenshot(filename)


def error_L2(
    uh: BluemiraFemFunction,
    u_ex: Union[BluemiraFemFunction, Expression],
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
    integral error
    """
    # Create higher order function space
    degree = uh.function_space.ufl_element().degree()
    family = uh.function_space.ufl_element().family()
    mesh = uh.function_space.mesh
    W = functionspace(mesh, (family, degree + degree_raise))
    # Interpolate approximate solution
    u_W = BluemiraFemFunction(W)
    u_W.interpolate(uh)

    # Interpolate exact solution, special handling if exact solution
    # is a ufl expression or a python lambda function
    u_ex_W = BluemiraFemFunction(W)
    if not isinstance(u_ex, ufl.core.expr.Expr):
        u_expr = Expression(u_ex, W.element.interpolation_points())
        u_ex_W.interpolate(u_expr)
    else:
        u_ex_W.interpolate(u_ex)

    # Compute the error in the higher order function space
    e_W = BluemiraFemFunction(W)
    e_W.x.array[:] = u_W.x.array - u_ex_W.x.array

    # Integrate the error
    error = form(ufl.inner(e_W, e_W) * ufl.dx)
    error_local = assemble_scalar(error)
    error_global = mesh.comm.allreduce(error_local, op=MPI.SUM)

    return np.sqrt(error_global)


def eval_f(
    function: Function, points: np.ndarray, check: str = "warn"
) -> Tuple[np.ndarray, ...]:
    """
    Evaluate the value of a dolfinx function in the specified points

    Parameters
    ----------
    function:
        reference function
    points:
        points on which the function shall be calculated
    check:
        ["off", "warn", "error"]


    Returns
    -------
    the values of the function in the specified points

    """
    # TODO(je-cook) get rid of check strings
    mesh = function.function_space.mesh
    bb_tree = geometry.bb_tree(mesh, mesh.topology.dim)
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
            point = closest_point_in_mesh(mesh, np.array([point]))
            temp_cell_candidates = geometry.compute_collisions_points(bb_tree, point)
            temp_colliding_cells = geometry.compute_colliding_cells(
                mesh, temp_cell_candidates, point
            )
            points_on_proc.append(point[0])
            cells.append(temp_colliding_cells.links(0)[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    values = np.array(function.eval(points_on_proc, cells)).reshape(points.shape[0], -1)

    if check != "off":
        print_func = bluemira_warn if check == "warn" else bluemira_error
        if points.shape != points_on_proc.shape:
            print_func(
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
    levels: Union[int, np.ndarray] = 20,
    ax: Optional[plt.Axes] = None,
    contour: bool = True,
    tofill: bool = True,
    **kwargs,
) -> Dict[str, Union[plt.Axes, None]]:
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
    comm,
    rank: int = 0,
    gdim: Union[int, tuple] = 3,
    partitioner=None,
):
    """Wraps `dolfinx.io.gmshio.read_from_msh` to patch dimensional reading


    Parameters
    ----------
        filename: Name of ``.msh`` file.
        comm: MPI communicator to create the mesh on.
        rank: Rank of ``comm`` responsible for reading the ``.msh``
            file.
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

    v: Union[BluemiraFemFunction, Callable, float]
    tag: int
    Itot: Union[float, None] = None


def create_j_function(
    mesh: Mesh,
    cell_tags,
    values: List[Association],
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

    Returns
    -------
    a dolfinx function ("DG", 0) with the values of the density current to be applied at
    each cell

    Notes
    -----
    If multiple functions are defined on the same subdomain,
    the contributions of each function are summed up.
    """
    function_space = functionspace(mesh, ("DG", 0))

    unique_tags = np.unique(cell_tags.values)
    J = BluemiraFemFunction(function_space)  # noqa: N806

    temp = BluemiraFemFunction(function_space)

    for value in values:
        if value.tag in unique_tags:
            cells = cell_tags.find(value.tag)
            dofs = locate_dofs_topological(function_space, 2, cells)

            if isinstance(value.v, (BluemiraFemFunction, Callable)):
                temp.interpolate(value.v, cells)
            elif isinstance(value.v, (int, float)):
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
            raise Warning(f"Tag {value.tag} is not in boundaries")
    return J


def compute_B_from_Psi(
    psi: BluemiraFemFunction, eltype: tuple, eltype1: Optional[tuple] = None
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

    Return
    ------
    Magnetic flux density function in the mesh domain
    """
    mesh = psi.function_space.mesh
    eltype = (*eltype, (mesh.geometry.dim,))
    W0 = functionspace(mesh, eltype)  # noqa: N806
    B0 = BluemiraFemFunction(W0)

    # x = ufl.SpatialCoordinate(mesh)

    B_expr = Expression(
        ufl.as_vector(
            (
                -psi.dx(1) / (2 * np.pi * r),
                psi.dx(0) / (2 * np.pi * r),
            )
        ),
        W0.element.interpolation_points(),
    )

    B0.interpolate(B_expr)

    if eltype1 is not None:
        W = functionspace(mesh, eltype)  # noqa: N806
        B = BluemiraFemFunction(W)
        B.interpolate(B0)
    else:
        B = B0

    return B


def plot_meshtags(
    mesh: Mesh,
    meshtags: Optional[Union[cpp.mesh.MeshTags_float64, cpp.mesh.MeshTags_int32]] = None,
    filename: str = "meshtags.png",
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
    cells, types, x = plot.vtk_mesh(mesh)
    grid = pyvista.UnstructuredGrid(cells, types, x)

    # Attach the cells tag data to the pyvita grid
    if meshtags is not None:
        grid.cell_data["Marker"] = meshtags.values
        grid.set_active_scalars("Marker")

    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True, show_scalar_bar=False)
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        plotter.screenshot(filename)