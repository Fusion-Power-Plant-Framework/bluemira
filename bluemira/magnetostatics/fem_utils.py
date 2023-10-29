from typing import Optional
import dolfinx.fem
from ufl import as_vector, SpatialCoordinate
from typing import List, Callable
import matplotlib.tri as tr

import dolfinx.fem
from typing import Union, Tuple

import ufl
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx.fem import (
    Expression,
    Function,
    FunctionSpace,
    VectorFunctionSpace,
    assemble_scalar,
    form,
)
from dolfinx import geometry
from matplotlib.pyplot import Axes
import matplotlib.pyplot as plt
import dolfinx.plot as plot
import pyvista

import bluemira.base.look_and_feel

def convert_to_points_array(x):
    x = np.array(x)
    if len(x.shape) == 1:
        if len(x) == 2:
            x = np.array([x[0], x[1], 0])
        x = np.array([x])
    if x.shape[1] == 2:
        x = np.array([x[:, 0], x[:, 1], x[:, 0]*0]).T
    return x

class BluemiraFemFunction(Function):
    """A supporting class that extends the BluemiraFemFunction implementing
    the __call__ function to return the interpolated function value on the specified
    points.

    Note
    ----
        In dolfinx v.0.5.0 it seems not to be possible to get a function value in a point
        without doing all the operations incorporated in eval_f. In case this functionality
        is integrated in most recent version of dolfinx, this class can be removed and
        replaced simply by BluemiraFemFunction.
    """
    def __call__(self, points: np.array):
        """
        Call function

        Notes
        -----
        Overwrite of the call function such that the behaviour is similar to the one expected
        for a Callable.
        """
        res, _ = self._eval_new(points)
        return res

    def _eval_new(self, points: Union[np.array, List]):
        """
        Supporting function for __call__
        """
        initial_shape = points.shape
        points = convert_to_points_array(points)
        res, new_points = eval_f(self, points)
        if len(res.shape) == 1:
            res = res[0]
            new_points = new_points[0]
        else:
            if res.shape[1] == 1:
                res = res.reshape(-1)
                if res.shape[0] == 1:
                    res = res[0]
                    new_points = new_points[0]
            elif res.shape[0] == 1 and len(initial_shape) == 1:
                res = res[0]
                new_points = new_points[0]

        return res, new_points


def closest_point_in_mesh(mesh, points):
    points = convert_to_points_array(points)

    closest_points = []

    tdim = mesh.topology.dim
    tree = dolfinx.geometry.BoundingBoxTree(mesh, tdim)
    num_entities_local = mesh.topology.index_map(tdim).size_local + mesh.topology.index_map(tdim).num_ghosts
    entities = np.arange(num_entities_local, dtype=np.int32)
    midpoint_tree = dolfinx.geometry.create_midpoint_tree(mesh, tdim, entities)
    closest_entities = dolfinx.geometry.compute_closest_entity(tree, midpoint_tree, mesh, points)
    colliding_entity_bboxes = dolfinx.geometry.compute_collisions(tree, points)
    mesh_geom = mesh.geometry.x
    geom_dofs = dolfinx.cpp.mesh.entities_to_geometry(mesh, tdim, [closest_entities], False)
    mesh_nodes = mesh_geom[geom_dofs][0]
    for p in points:
        displacement = dolfinx.geometry.compute_distance_gjk(p, mesh_nodes)
        closest_points.append(p - displacement)
    return np.array(closest_points)

def plot_meshtags(
    mesh: dolfinx.mesh.Mesh,
    meshtags: Union[dolfinx.cpp.mesh.MeshTags_float64, dolfinx.cpp.mesh.MeshTags_int32],
    filename: str = "meshtags.png",
):
    """
    Plot dolfinx mesh with markers using pyvista.

    Parameters
    ----------
    mesh: dolfinx.mesh.Mesh
        Mesh
    meshtags: Union[dolfinx.cpp.mesh.MeshTags_float64, dolfinx.cpp.mesh.MeshTags_int32]
        Mesh tags
    filename: str
        Full path for plot save
    """

    # Create VTK mesh
    cells, types, x = plot.create_vtk_mesh(mesh)
    grid = pyvista.UnstructuredGrid(cells, types, x)

    # Attach the cells tag data to the pyvita grid
    grid.cell_data["Marker"] = meshtags.values
    grid.set_active_scalars("Marker")

    plotter = pyvista.Plotter()
    plotter.add_mesh(grid, show_edges=True, show_scalar_bar=False)
    plotter.view_xy()
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        plotter.screenshot(filename)


def calculate_area(
    mesh: dolfinx.mesh.Mesh, boundaries: object, tag: int = None
) -> object:
    """
    Calculate the area of a sub-domain

    Parameters
    ----------
    mesh: dolfinx.mesh.Mesh
        mesh of the FEM model
    boundaries:
        boundaries mesh tags
    tag:
        subdomain tag

    Returns
    -------
    float:
        area of the subdomain
    """
    f = dolfinx.fem.Constant(mesh, PETSc.ScalarType(1))

    return integrate_f(f, mesh, boundaries, tag)


def integrate_f(
    f: BluemiraFemFunction, mesh: dolfinx.mesh.Mesh, boundaries = None, tag: int = None
):
    """
    Calculate the integral of a function on the specified sub-domain

    Parameters
    ----------
    function: BluemiraFemFunction
        function to be integrated in the subdomain
    mesh: dolfinx.mesh.Mesh
        mesh of the FEM model
    boundaries:
        boundaries mesh tags
    tag: int
        subdomain tag (default None). When None, the integral is made on the whole mesh domain.

    Returns
    -------
    float:
        area of the subdomain
    """
    if boundaries is None:
        dx = ufl.dx
    else:
        dx = ufl.Measure("dx", subdomain_data=boundaries, domain=mesh)
    if tag is None:
        integral = assemble_scalar(form(f * dx))
    else:
        integral = assemble_scalar(form(f * dx(tag)))

    return integral


# Plotting method for scalar field.
# Not really clear how to manage pyvista plots. Left here just as reference
# for a better future implementation.
def plot_fem_scalar_field(field: BluemiraFemFunction, filename: str = "field.png"):
    """
    Plot a scalar field given by the dolfinx function "field" in pyvista or in a file.

    Parameters
    ----------
    field: BluemiraFemFunction
        function to be plotted
    filename: str
        file in which the plot is stored

    Note
    ----
    if pyvista.OFF_SCREEN is False the plot is shown on the screen, otherwise
    the file with the plot is saved.
    """
    from dolfinx.plot import create_vtk_mesh
    import pyvista

    plotter = pyvista.Plotter()
    V = field.function_space
    degree = V.ufl_element().degree()
    if degree == 0:
        field_grid = pyvista.UnstructuredGrid(*create_vtk_mesh(V.mesh))
    else:
        field_grid = pyvista.UnstructuredGrid(*create_vtk_mesh(V))
    field_grid.point_data["Field"] = field.x.array
    field_grid.set_active_scalars("Field")
    warp = field_grid.warp_by_scalar("Field", factor=1)
    actor = plotter.add_mesh(warp, show_edges=True)  # noqa F841
    if not pyvista.OFF_SCREEN:
        plotter.show()
    else:
        field_fig = plotter.screenshot(filename)  # noqa F841


def error_L2(
    uh: BluemiraFemFunction,
    u_ex: Union[BluemiraFemFunction, Expression],
    degree_raise: int = 0,
):
    """
    Calculate the L2 error norm between two functions.
    This method has been taken from https://jsdokken.com/dolfinx-tutorial/chapter4/convergence.html.

    Parameters
    ----------
    uh: BluemiraFemFunction
        first function
    u_ex: Union[BluemiraFemFunction, dolfinx.fem.Expression]
        second function or an expression (in case, for example, of exact solution)
    degree_raise: int
        increase of polynomial degree with respect uh degree (useful when comparing with an exact solution)

    Returns
    -------
    float:
        integral error
    """
    # Create higher order function space
    degree = uh.function_space.ufl_element().degree()
    family = uh.function_space.ufl_element().family()
    mesh = uh.function_space.mesh
    W = FunctionSpace(mesh, (family, degree + degree_raise))
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


def eval_f(function: Function, points: np.array, check: str="warn"):
    """
    Evaluate the value of a dolfinx function in the specified points

    Parameters
    ----------
    function: reference function
    points: points on which the function shall be calculated
    check: ["off", "warn", "error"]


    Returns
    -------
    array:
        the values of the function in the specified points

    """
    mesh = function.function_space.mesh
    bb_tree = geometry.BoundingBoxTree(mesh, mesh.topology.dim)
    cells = []
    points_on_proc = []

    # points = closest_point_in_mesh(mesh, points)
    # Find cells whose bounding-box collide with the points
    cell_candidates = geometry.compute_collisions(bb_tree, points)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(mesh, cell_candidates, points)
    for i, point in enumerate(points):
        if len(colliding_cells.links(i)) > 0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
        else:
            point = closest_point_in_mesh(mesh, np.array([point]))
            temp_cell_candidates = geometry.compute_collisions(bb_tree, point)
            temp_colliding_cells = geometry.compute_colliding_cells(mesh, temp_cell_candidates, point)
            points_on_proc.append(point[0])
            cells.append(temp_colliding_cells.links(0)[0])

    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    values = np.array(function.eval(points_on_proc, cells)).reshape(points.shape[0], -1)

    if check == "warn":
        if points.shape != points_on_proc.shape:
            bluemira.base.look_and_feel.bluemira_warn(
                f"Some points cannot be interpolated (no colliding cells have been found "
                f"for {points.shape[0] - points_on_proc.shape[0]} points)."
                f"Original points: {points.shape} - Interpolated points: {points_on_proc.shape}"
            )
    elif check == "error":
        if points.shape != points_on_proc.shape:
            bluemira.base.look_and_feel.bluemira_error(
                f"Some points cannot be interpolated (no colliding cells have been found "
                f"for {points.shape[0] - points_on_proc.shape[0]} points)."
                f"Original points: {points.shape} - Interpolated points: {points_on_proc.shape}"
            )

    return values, points_on_proc


def plot_scalar_field(
    x: np.ndarray,
    y: np.ndarray,
    data: np.ndarray,
    levels: Union[int, np.array] = 20,
    ax: Optional[Axes] = None,
    contour: bool = True,
    tofill: bool = True,
    **kwargs,
) -> Tuple[Axes, Union[Axes, None], Union[Axes, None]]:
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
    maxi = np.max(np.sqrt(xtri**2 + ytri**2), axis=1)

    if contour:
        cntr = ax.tricontour(triang, data, levels=levels, **contour_kwargs)

    if tofill:
        cntrf = ax.tricontourf(triang, data, levels=levels, cmap="RdBu_r")
        fig.colorbar(cntrf, ax=ax)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_aspect("equal")

    plot_info = {"ax": ax, "cntr": cntr, "cntrf": cntrf}

    return plot_info

# FEM functions
def create_j_function(
    mesh: dolfinx.mesh.Mesh,
    cell_tags,
    values: List[
        Union[
            Tuple[float, int, Union[float, None]],
            Tuple[Callable, int, Union[float, None]],
            Tuple[BluemiraFemFunction, int, Union[float, None]],
        ]
    ],
):
    """
    Create the dolfinx current density function for the whole domain given a set of current density
    for each sub-domain.

    Parameters
    ----------
    mesh: dolfinx.mesh.Mesh
        mesh of the FEM model
    cell_tags:
        mesh cell tags
    values: List[Union[Tuple[float, int, Union[float, None]],
                       Tuple[Callable, int, Union[float, None]],
                       Tuple[BluemiraFemFunction, int, Union[float, None]]]]
        list of association (density current, boundaries tag, target current). If target current is not None,
        the applied current density is rescaled in order to obtain the total target current in the subdomain identified
        by the boundaries tag.

    Returns
    -------
    BluemiraFemFunction:
        a dolfinx function ("DG", 0) with the values of the density current to be applied at each cell

    Note
    ----
        If multiple functions are defined on the same subdomain, the contributions of each function are summed up.
    """
    function_space = FunctionSpace(mesh, ("DG", 0))

    unique_tags = np.unique(cell_tags.values)
    J = BluemiraFemFunction(function_space)
    J.x.set(0)

    temp = BluemiraFemFunction(function_space)
    temp.x.set(0)

    for value in values:
        if len(value) == 2:
            v = value[0]
            tag = value[1]
            Itot = None
        elif len(value) == 3:
            v = value[0]
            tag = value[1]
            Itot = value[2]

        if tag in unique_tags:
            cells = cell_tags.find(tag)
            dofs = dolfinx.fem.locate_dofs_topological(function_space, 2, cells)

            if isinstance(v, (BluemiraFemFunction, Callable)):
                temp.interpolate(v, cells)
            elif isinstance(v, (int, float)):
                temp.x.array[dofs] += v
            else:
                raise ValueError(
                    f"{v} is not a number, Callable or BluemiraFemFunction object"
                )
            if Itot is not None:
                factor = Itot / integrate_f(temp, mesh, cell_tags, tag)
                temp.x.array[dofs] *= factor

            J.x.array[dofs] += temp.x.array[dofs]
            temp.x.set(0)
        else:
            raise Warning(f"Tag {tag} is not in boundaries")
    return J


def compute_B_from_Psi(Psi: BluemiraFemFunction, eltype: tuple, eltype1: tuple = None):
    """
    Compute the magnetic flux density given the magnetic flux function for an axisymmetric problem.

    Parameters
    ----------
    Psi: Funcion
        the magnetic flux function in the mesh domain
    eltype: tuple
        Element type identified (e.g. ("CG", 1)) for the magnetic flux density function

    Return
    ------
    BluemiraFemFunction:
        Magnetic flux density function in the mesh domain
    """
    mesh = Psi.function_space.mesh

    W0 = VectorFunctionSpace(mesh, eltype)
    B0 = BluemiraFemFunction(W0)

    x = SpatialCoordinate(mesh)

    r = x[0]

    B_expr = Expression(
        as_vector(
            (
                -Psi.dx(1) / (2 * np.pi * r),
                Psi.dx(0) / (2 * np.pi * r),
            )
        ),
        W0.element.interpolation_points(),
    )

    B0.interpolate(B_expr)

    if eltype1 is not None:
        W = VectorFunctionSpace(mesh, eltype)
        B = BluemiraFemFunction(W)
        B.interpolate(B0)
    else:
        B = B0

    return B


def plot_meshtags(
    mesh: dolfinx.mesh.Mesh,
    meshtags: Union[dolfinx.cpp.mesh.MeshTags_float64, dolfinx.cpp.mesh.MeshTags_int32] = None,
    filename: str = "meshtags.png",
):
    """
    Plot dolfinx mesh with markers using pyvista.

    Parameters
    ----------
    mesh: dolfinx.mesh.Mesh
        Mesh
    meshtags: Union[dolfinx.cpp.mesh.MeshTags_float64, dolfinx.cpp.mesh.MeshTags_int32]
        Mesh tags
    filename: str
        Full path for plot save
    """

    # Create VTK mesh
    cells, types, x = plot.create_vtk_mesh(mesh)
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
