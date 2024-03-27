# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Create csg geometry by converting from bluemira geometry objects made of wires."""
# ruff: noqa: PLR2004, D105

from __future__ import annotations

from collections import abc
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np
import openmc
from matplotlib import pyplot as plt  # for debugging
from numpy import typing as npt

from bluemira.geometry.constants import EPS_FREECAD
from bluemira.geometry.error import GeometryError
from bluemira.geometry.tools import make_circle_arc_3P
from bluemira.neutronics.params import BlanketLayers
from bluemira.neutronics.radial_wall import CellWalls, Vertices

if TYPE_CHECKING:
    from bluemira.neutronics.make_pre_cell import PreCellArray
    from bluemira.neutronics.params import TokamakThicknesses


def is_strictly_monotonically_increasing(series):
    """Check if a series is strictly monotonically increasing"""  # or decreasing
    diff = np.diff(series)
    return all(diff > 0)  # or all(diff<0)


# VERY ugly solution. Maybe tack this into the openmc universe instead?
hangar = {}  # it's called a hangar because it's where the planes are parked ;)


def plot_surfaces(surfaces_list: List[openmc.Surface]):
    """
    Plot a list of surfaces in matplotlib.
    """
    ax = plt.axes()
    # ax.set_aspect(1.0) # don't do this as it makes the plot hard to read.
    for i, surface in enumerate(surfaces_list):
        plot_coords(surface, color_num=i)
    ax.legend()
    ax.set_ylim([-10, 10])
    ax.set_xlim([-10, 10])


def plot_coords(surface: openmc.Surface, color_num: int):
    """
    In the range [-10, 10], plot the RZ cross-section of the ZCylinder/ZPlane/ZCone.
    """
    if isinstance(surface, openmc.ZCylinder):
        plt.plot(
            [surface.x0, surface.x0],
            [-10, 10],
            label=f"{surface.id}: {surface.name}",
            color=f"C{color_num}",
        )
    elif isinstance(surface, openmc.ZPlane):
        plt.plot(
            [-10, 10],
            [surface.z0, surface.z0],
            label=f"{surface.id}: {surface.name}",
            color=f"C{color_num}",
        )
    elif isinstance(surface, openmc.ZCone):
        intercept = surface.z0
        slope = 1 / np.sqrt(surface.r2)

        def equation_pos(x):
            return slope * np.array(x) + intercept

        def equation_neg(x):
            return -slope * np.array(x) + intercept

        y_pos, y_neg = equation_pos([-10, 10]), equation_neg([-10, 10])
        plt.plot(
            [-10, 10],
            y_pos,
            label=f"{surface.id}: {surface.name} (upper)",
            color=f"C{color_num}",
        )
        plt.plot(
            [-10, 10],
            y_neg,
            label=f"{surface.id}: {surface.name} (lower)",
            linestyle="--",
            color=f"C{color_num}",
        )


def surface_from_2points(
    point1: npt.NDArray,
    point2: npt.NDArray,
    surface_id: Optional[int] = None,
    name: str = "",
) -> Optional[Union[openmc.Surface, openmc.model.ZConeOneSided]]:
    """
    Create either a cylinder, a cone, or a surface from 2 points using only the
    rz coordinates of any two points on it.

    Parameters
    ----------
    point1, point2: ndarray of shape (2,)
        any two non-trivial (i.e. cannot be the same) points on the rz cross-section of
        the surface, each containing the r and z coordinates
    surface_id, name:
        see openmc.Surface

    Returns
    -------
    surface: openmc.surface.Surface, None
        if the two points provided are redundant: don't return anything, as this is a
        single point pretending to be a surface. This will come in handy for handling the
        creation of BlanketCells made with 3 surfaces rather than 4.
    """
    dr, dz = point2 - point1
    if np.isclose(dr, 0, rtol=0, atol=EPS_FREECAD):
        _r = point1[0]
        if np.isclose(dz, 0, rtol=0, atol=EPS_FREECAD):
            return None
            # raise GeometryError(
            #     "The two points provided aren't distinct enough to "
            #     "uniquely identify a surface!"
            # )
        return openmc.ZCylinder(r=_r, surface_id=surface_id, name=name)
    if np.isclose(dz, 0, rtol=0, atol=EPS_FREECAD):
        _z = point1[-1]
        z_plane = openmc.ZPlane(z0=_z, surface_id=surface_id, name=name)
        hangar[_z] = z_plane
        return hangar[_z]
    slope = dz / dr
    z_intercept = -slope * point1[0] + point1[-1]
    return openmc.ZCone(z0=z_intercept, r2=slope**-2, surface_id=surface_id, name=name)


def torus_from_3points(
    point1: npt.NDArray,
    point2: npt.NDArray,
    point3: npt.NDArray,
    surface_id: Optional[int] = None,
    name: str = "",
) -> openmc.ZTorus:
    """
    Make a circular torus centered on the z-axis using 3 points.
    All 3 points should lie on the RZ plane AND the surface of the torus simultaneously.

    Parameters
    ----------
    point1, point2, point3: ndarray of shape (2,)
        RZ coordinates of the 3 points on the surface of the torus.
    surface_id, name:
        See openmc.Surface
    """
    point1 = point1[0], 0, point1[-1]
    point2 = point2[0], 0, point2[-1]
    point3 = point3[0], 0, point3[-1]
    circle = make_circle_arc_3P(point1, point2, point3)
    cad_circle = circle.shape.OrderedEdges[0].Curve
    center, radius = cad_circle.Center, cad_circle.Radius
    return torus_from_circle(
        [center[0], center[-1]], radius, surface_id=surface_id, name=name
    )


def torus_from_circle(
    center: Sequence[float],
    minor_radius: float,
    surface_id: Optional[int] = None,
    name: str = "",
):
    """
    Make a circular torus centered on the z-axis.
    The circle would lie on the RZ plane AND the surface of the torus simultaneously.

    Parameters
    ----------
    minor_radius
        Radius of the cross-section circle, which forms the minor radius of the torus.
    center
        Center of the cross-section circle, which forms the center of the torus.
    surface_id, name:
        See openmc.Surface
    """
    return z_torus(
        [center[0], center[-1]], minor_radius, surface_id=surface_id, name=name
    )


def z_torus(
    center: npt.NDArray,
    minor_radius: float,
    surface_id: Optional[int] = None,
    name: str = "",
) -> openmc.ZTorus:
    """
    A circular torus centered on the z-axis.
    The center refers to the major radius and it's z coordinate.

    Parameters
    ----------
    center: ndarray of shape (2,)
        The center of the torus' RZ plane cross-section
    minor_radius: ndarray of shape (2,)
        minor radius of the torus

    Returns
    -------
    openmc.ZTorus
    """
    major_radius, height = center[0], center[-1]
    return openmc.ZTorus(
        z0=height,
        a=major_radius,
        b=minor_radius,
        c=minor_radius,
        surface_id=surface_id,
        name=name,
    )


def torus_from_5points(
    point1: npt.NDArray[float],
    point2: npt.NDArray[float],
    point3: npt.NDArray[float],
    point4: npt.NDArray[float],
    point5: npt.NDArray[float],
    surface_id: Optional[int] = None,
    name: str = "",
) -> openmc.ZTorus:
    """
    Make an elliptical torus centered on the z-axis using 5 points.
    All 5 points should lie on the RZ plane AND the surface of the torus simultaneously.
    Semi-major/semi-minor axes of the ellipse must be aligned to the R/Z (or Z/R) axes.

    Parameters
    ----------
    point1, point2, point3: ndarray of shape (2,)
        RZ coordinates of the 3 points on the surface of the torus.
    surface_id, name:
        See openmc.Surface
    """
    raise NotImplementedError(
        "The maths of determining where the ellipse center should"
        "be is yet to be worked out."
    )
    center, vertical_radius, horizontal_radius = (point1, point2, point3, point4, point5)
    major_radius, height = center[0], center[-1]
    return openmc.ZTorus(
        z0=height,
        a=major_radius,
        b=vertical_radius,
        c=horizontal_radius,
        surface_id=surface_id,
        name=name,
    )


def choose_halfspace(
    surface: openmc.Surface, choice_points: npt.NDArray
) -> openmc.Halfspace:
    """
    Simply take the centroid point of all of the choice_points, and choose the
    corresponding half space

    Parameters
    ----------
    surface
        an openmc surface

    Returns
    -------
    openmc.Halfspace
    """
    pt = np.mean(choice_points, axis=0)
    value = surface.evaluate(pt)
    if value > 0:
        return +surface
    if value < 0:
        return -surface
    raise GeometryError("Centroid point is directly on the surface")


def choose_plane_cylinders(
    surface: Union[openmc.ZPlane, openmc.ZCylinder], choice_points: npt.NDArray
) -> openmc.Halfspace:
    """choose_region for ZPlane and ZCylinder."""
    x, y, z = np.array(choice_points).T
    values = surface.evaluate([x, y, z])
    threshold = EPS_FREECAD
    if isinstance(surface, openmc.ZCylinder):
        threshold = 2 * EPS_FREECAD * surface.r + EPS_FREECAD**2

    if all(values >= -threshold):
        return +surface
    if all(values <= threshold):
        return -surface

    raise GeometryError(f"There are points on both sides of this {type(surface)}!")


def choose_region_cone(
    surface: openmc.ZCone, choice_points: npt.NDArray
) -> openmc.Region:
    """
    choose_region for ZCone.
    When reading this function's code, bear in mind that a Z cone can be separated into
    3 parts:
        A. the upper cone (evaluates to negative),
        B. outside of the cone (evaluates to positive),
        C. the lower cone (evaluates to negative).
    We have to account for the following cases:
    -------------------------------------
    | upper cone | outside | lower cone |
    -------------------------------------
    |    Y      |     N    |      N     |
    |    Y      |     Y    |      N     |
    |    N      |     Y    |      N     |
    |    N      |     Y    |      Y     |
    |    N      |     N    |      Y     |
    -------------------------------------
    All other cases should raise an error.
    The tricky part to handle is the floating point precision problem:
        it's possible that th every point used to create the cone does not lie on the
        cone/ lies on the wrong side of the cone.
        Hence the first step is to shink the choice_points by 0.5% towards the centroid.

    Returns
    -------
    region
        openmc.Region, specifically openmc.Halfspace or openmc.Union of openmc.Halfspace
    """
    # shrink to avoid floating point number comparison imprecision issues
    centroid = np.mean(choice_points, axis=0)
    choice_points = (choice_points + 0.01 * centroid) / 1.01
    x, y, z = np.array(choice_points).T
    values = surface.evaluate([x, y, z])
    middle = values > 0
    if all(middle):  # exist outside of cone
        return +surface

    z_dist = z - surface.z0
    upper_cone = np.logical_and(~middle, z_dist > 0)
    lower_cone = np.logical_and(~middle, z_dist < 0)
    # upper_not_cone = np.logical_and(middle, z_dist > 0)
    # lower_not_cone = np.logical_and(middle, z_dist < 0)

    if all(upper_cone):
        # everything in the upper cone.
        plane = find_suitable_z_plane(  # the highest we can cut is at the lowest z.
            surface.z0,
            [surface.z0 - EPS_FREECAD, min(z) - EPS_FREECAD],
            1000 + surface.id,
            f"Ambiguity plane for cone {surface.id}",
        )
        return -surface & +plane
    if all(lower_cone):
        # everything in the lower cone
        plane = find_suitable_z_plane(  # the lowest we can cut is at the highest z.
            surface.z0,
            [max(z) + EPS_FREECAD, surface.z0 + EPS_FREECAD],
            1000 + surface.id,
            f"Ambiguity plane for cone {surface.id}",
        )
        return -surface & -plane
    if all(np.logical_or(upper_cone, lower_cone)):
        raise GeometryError(
            "Both cones have vertices lying inside! Cannot compute a contiguous "
            "region that works for both. Check if polygon is convex?"
        )
    plane = find_suitable_z_plane(  # In this rare case, make its own plane.
        surface.z0,
        [surface.z0 + EPS_FREECAD, surface.z0 - EPS_FREECAD],
        1000 + surface.id,
        f"Ambiguity plane for cone {surface.id}",
    )
    if all(np.logical_or(upper_cone, middle)):
        return +surface | +plane
    if all(np.logical_or(lower_cone, middle)):
        return +surface | -plane
    raise GeometryError("Can't have points in upper cone, lower cone AND outside!")


def choose_region(surface: openmc.Surface, choice_points: npt.NDArray) -> openmc.Region:
    """
    Calculate the correct side of the region such that all vertices of the polygon are
    situated within the required region.

    Parameters
    ----------
    surface
        :class:`openmc.surface.Surface`
    choice_points: np.ndarray of shape (N, 3)
        a list of points representing the vertices of a convex polygon in RZ plane

    Returns
    -------
    region
        openmc.Region, specifically openmc.Halfspace or openmc.Union of openmc.Halfspace
    """
    # switch case: blue
    if isinstance(surface, (openmc.ZPlane, openmc.ZCylinder)):
        return choose_plane_cylinders(surface, choice_points)

    if isinstance(surface, openmc.ZCone):
        return choose_region_cone(surface, choice_points)
    raise NotImplementedError(
        f"Surface {type(surface)} is not ready for use in the axis-symmetric case!"
    )


def find_suitable_z_plane(
    z0: float,
    z_range: Optional[Iterable[float]] = None,
    surface_id: Optional[int] = None,
    name: str = "",
    **kwargs,
):
    """Find a suitable z from the hangar, or create a new one if no matches are found."""
    if z_range:
        z_min, z_max = min(z_range), max(z_range)
        for key in hangar:
            if z_min <= key <= z_max:
                return hangar[key]  # return the first match
    hangar[z0] = openmc.ZPlane(z0=z0, surface_id=surface_id, name=name, **kwargs)
    return hangar[z0]


def flatten_region(region: openmc.Region) -> openmc.Intersection:
    """Expand the expression partially using the rule of associativity"""
    return openmc.Intersection(get_expression(region).values())


def get_expression(region: openmc.Region) -> Dict[str, openmc.Region]:
    """Get a dictionary of all of the elements that shall be intersected in the end."""
    if isinstance(region, openmc.Halfspace):  # termination condition
        return {region.side + str(region.surface.id): region}
    if isinstance(region, openmc.Intersection):
        final_intersection = {}
        for _r in region:
            final_intersection.update(get_expression(_r))
        return final_intersection
    return {str(region): region}


class BlanketCell(openmc.Cell):
    """
    A generic blanket cell that forms the base class for the five specialized types of
    blanket cells.

    It's a special case of openmc.Cell, in that it has 3 to 4 surfaces
    (mandatory surfaces: exterior_surface, ccw_surface, cw_surface;
    optional surface: interior_surface), and it is more wieldy because we don't have to
    specify the relevant half-space for each surface; instead the corneres of the cell
    is provided by the user, such that the appropriate regions are chosen.
    """

    def __init__(
        self,
        exterior_surface: openmc.Surface,
        ccw_surface: openmc.Surface,
        cw_surface: openmc.Surface,
        interior_surface: Optional[openmc.Surface],
        vertices: Vertices,
        cell_id=None,
        name="",
        fill=None,
    ):
        """
        Create the openmc.Cell from 3 to 4 surfaces and an example point.

        Parameters
        ----------
        exterior_surface
            Surface on the exterior side of the cell
        ccw_surface
            Surface on the ccw wall side of the cell
        cw_surface
            Surface on the cw wall side of the cell
        vertices
            A list of points. Could be 2D or 3D.
        interior_surface
            Surface on the interior side of the cell
        cell_id
            see :class:`openmc.Cell`
        name
            see :class:`openmc.Cell`
        fill
            see :class:`openmc.Cell`
        """
        self.exterior_surface = exterior_surface
        self.ccw_surface = ccw_surface
        self.cw_surface = cw_surface
        self.interior_surface = interior_surface
        self.vertices = vertices

        _surfaces_list = [exterior_surface, ccw_surface, cw_surface, interior_surface]

        vertices_array = vertices.to_3D().to_array()
        intersection_region = openmc.Intersection([
            choose_region(surface, vertices_array)
            for surface in _surfaces_list
            if surface is not None
        ])
        final_region = flatten_region(intersection_region)  # can lead to speed up

        super().__init__(cell_id=cell_id, name=name, fill=fill, region=final_region)


class BlanketCellStack(abc.Sequence):
    """
    A stack of openmc.Cells, first cell is closest to the interior and last cell is
    closest to the exterior. They should all be situated at the same poloidal angle.
    """

    def __init__(self, cell_stack: List[BlanketCell]):
        """
        The shared surface between adjacent cells must be the SAME one, i.e. same id and
        hash, not just identical.
        They must share the SAME counter-clockwise surface and the SAME clockwise surface
        (left and right side surfaces of the stack, for the stack pointing straight up).

        Because the bounding_box function is INCORRECT, we can't perform a check on the
        bounding box to confirm that the stack is linearly increasing/decreasing in xyz
        bounds.

        Variables
        ---------
        fw_cell: FirstWallCell
        bz_cell: Optional[BreedingZoneCell]
        mnfd_cell: ManifoldCell
        vv_cell: VacuumVesselCell
        """
        self.cell_stack = cell_stack
        for int_cell, ext_cell in zip(cell_stack[:-1], cell_stack[1:]):
            if int_cell.exterior_surface is not ext_cell.interior_surface:
                raise ValueError("Expected a contiguous stack of cells!")

        # store each respecitve cell under their corresponding attribute names.
        cell_dict = self.ascribe_iterable_quantity_to_layer(self.cell_stack)
        self.sf_cell = cell_dict[BlanketLayers.Surface.name]
        self.fw_cell = cell_dict[BlanketLayers.FirstWall.name]
        self.bz_cell = cell_dict[BlanketLayers.BreedingZone.name]
        self.mnfd_cell = cell_dict[BlanketLayers.Manifold.name]
        self.vv_cell = cell_dict[BlanketLayers.VacuumVessel.name]

    def __len__(self) -> int:
        return self.cell_stack.__len__()

    def __getitem__(self, index_or_slice) -> Union[List[BlanketCell], BlanketCell]:
        return self.cell_stack.__getitem__(index_or_slice)

    def __repr__(self) -> str:
        return super().__repr__().replace(" at ", f" of {len(self)} BlanketCells at ")

    @staticmethod
    def ascribe_iterable_quantity_to_layer(quantity: Sequence) -> Dict:
        """
        Given an iterable of length 4 or 5, convert it into a dictionary such that
        each of the 4/5 quantities corresponds to a layer.
        """
        if len(quantity) == 4:
            return {
                BlanketLayers.Surface.name: quantity[0],
                BlanketLayers.FirstWall.name: quantity[1],
                BlanketLayers.BreedingZone.name: None,
                BlanketLayers.Manifold.name: quantity[2],
                BlanketLayers.VacuumVessel.name: quantity[3],
            }
        if len(quantity) == 5:
            return {
                BlanketLayers.Surface.name: quantity[0],
                BlanketLayers.FirstWall.name: quantity[1],
                BlanketLayers.BreedingZone.name: quantity[2],
                BlanketLayers.Manifold.name: quantity[3],
                BlanketLayers.VacuumVessel.name: quantity[4],
            }
        raise NotImplementedError(
            "Expected 4 or 5 layers to the blanket, where"
            " the surface, first wall, manifold, and vacuum vessel are mandatory,"
            " and the breeding zone is optional."
        )

    @property
    def interior_surface(self):  # noqa: D102
        return self.sf_cell.interior_surface

    @property
    def exterior_surface(self):  # noqa: D102
        return self.vv_cell.exterior_surface

    @property
    def ccw_surface(self):  # noqa: D102
        return self.sf_cell.ccw_surface

    @property
    def cw_surface(self):  # noqa: D102
        return self.sf_cell.cw_surface

    @property
    def interfaces(self):
        """All of the internal radial surfaces, (i.e. not exposed to plasma or air,)
        arranged from innermost to outermost.
        """
        if not hasattr(self, "_interfaces"):
            self._interfaces = [cell.interior_surface for cell in self.cell_stack]
            self._interfaces.append(self.cell_stack[-1].exterior_surface)
        return self._interfaces

    @classmethod
    def from_surfaces(
        cls,
        ccw_surf: openmc.Surface,
        cw_surf: openmc.Surface,
        layer_interfaces: List[openmc.Surface],
        vertices: Iterable[Iterable[float]],
        fill_dict: dict[str, openmc.Material],
        cell_ids: Optional[Iterable[int]] = None,
        cell_names: Optional[Iterable[int]] = None,
    ):
        """Create a stack of cells from a collection of openmc.Surfaces."""
        num_cell_in_stack = len(vertices)
        if not cell_ids:
            cell_ids = [None] * num_cell_in_stack
        if not cell_names:
            cell_names = [""] * num_cell_in_stack

        cell_type_at_num = {
            v: k
            for k, v in cls.ascribe_iterable_quantity_to_layer(
                range(num_cell_in_stack)
            ).items()
        }
        cell_stack = []
        for _cell_num, (int_surf, ext_surf, verts, _id, _name) in enumerate(
            zip(
                layer_interfaces[:-1],
                layer_interfaces[1:],
                vertices,
                cell_ids,
                cell_names,
                # strict=True # TODO: uncomment when we move to Python 3.10
            )
        ):
            cell_stack.append(
                BlanketCell(
                    ext_surf,
                    ccw_surf,
                    cw_surf,
                    int_surf,
                    verts,
                    _id,
                    _name,
                    fill_dict[cell_type_at_num[_cell_num]],
                )
            )
        return cls(cell_stack)


class BlanketCellArray(abc.Sequence):
    """
    An array of BlanketCellStack. Interior and exterior curve are both assumed convex.

    Parameters
    ----------
    blanket_cell_array
        a list of BlanketCellStack

    Variables
    ---------
    poloidal_surfaces: List[openmc.Surface]
        a list of surfaces radiating from (approximately) the gyrocenter to the entrance.
    radial_surfaces: List[List[openmc.Surface]]
        a list of lists of surfaces. Each list is the layer interface of a a stack's
    """

    def __init__(self, blanket_cell_array: List[BlanketCellStack]):
        """
        Create array from a list of BlanketCellStack
        """
        self.blanket_cell_array = blanket_cell_array
        self.poloidal_surfaces = [self[0].ccw_surface]
        self.radial_surfaces = []
        for i, this_stack in enumerate(self):
            self.poloidal_surfaces.append(this_stack.cw_surface)
            self.radial_surfaces.append(this_stack.interfaces)

            # check neighbouring cells share the same lateral surface
            if i != len(self) - 1:
                next_stack = self[i + 1]
                if this_stack.cw_surface is not next_stack.ccw_surface:
                    raise GeometryError(
                        f"Neighbouring BlanketCellStack [{i}] and "
                        f"[{i + 1}] are not aligned!"
                    )

    def __len__(self) -> int:
        return self.blanket_cell_array.__len__()

    def __getitem__(
        self, index_or_slice
    ) -> Union[List[BlanketCellStack], BlanketCellStack]:
        return self.blanket_cell_array.__getitem__(index_or_slice)

    def __add__(self, other_array) -> BlanketCellArray:
        return BlanketCellArray(self.blanket_cell_array + other_array.blanket_cell_array)

    def __repr__(self) -> str:
        return (
            super().__repr__().replace(" at ", f" of {len(self)} BlanketCellStacks at ")
        )

    @staticmethod
    def check_cut_point_ordering(
        cut_point_series: npt.NDArray[float], direction_vector: npt.NDArray[float]
    ):
        """
        Parameters
        ----------
        cut_point_series: np.ndarray of shape (M+1, 2)
            where M = number of cells in the blanket cell stack (i.e. number of layers
            in the blanket). Each point has two dimensions
        direction_vector:
            direction that these points are all supposed to go towards.
        """
        projections = np.dot(np.array(cut_point_series)[:, [0, -1]], direction_vector)
        if not is_strictly_monotonically_increasing(projections):
            raise GeometryError(
                "Some surfaces crosses over each other within the cell stack!"
            )

    @classmethod
    def from_pre_cell_array(
        cls,
        pre_cell_array: PreCellArray,
        material_dict: Dict[str, openmc.Material],
        thicknesses: TokamakThicknesses,
    ) -> BlanketCellArray:
        """
        Create a BlanketCellArray from a
        :class:`~bluemira.neutronics.make_pre_cell.PreCellArray`.
        This method assumes itself is the first method to be run to create cells in the
        :class:`~openmc.Universe.`
        """
        cell_walls = CellWalls.from_pre_cell_array(pre_cell_array)

        find_suitable_z_plane(
            min(cell_walls[:, :, -1].flatten()) - EPS_FREECAD,
            surface_id=999,
            boundary_type="vacuum",
            name="Blanket bottom",
        )
        find_suitable_z_plane(
            max(cell_walls[:, :, -1].flatten()) + EPS_FREECAD,
            surface_id=1000,
            boundary_type="vacuum",
            name="Blanket top",
        )
        innermost_cyl = openmc.ZCylinder(  # noqa: F841
            min(abs(cell_walls[:, :, 0].flatten())) - EPS_FREECAD,
            surface_id=1999,
            boundary_type="vacuum",
        )
        outermost_cyl = openmc.ZCylinder(  # noqa: F841
            max(abs(cell_walls[:, :, 0].flatten())) + EPS_FREECAD,
            surface_id=2000,
            boundary_type="vacuum",
        )
        # left wall
        ccw_surf = surface_from_2points(
            *cell_walls[0],
            surface_id=9,
            name="Blanket cell wall 0 (renamed 9 as openmc does not support 0"
            "indexing).",
        )

        cell_array = []
        for i, (pre_cell, cw_wall) in enumerate(zip(pre_cell_array, cell_walls[1:])):
            # right wall
            cw_surf = surface_from_2points(
                *cw_wall, surface_id=10 * (i + 1), name=f"Blanket cell wall {i + 1}"
            )

            # getting the interfaces between layers.
            interior_wire = pre_cell.cell_walls.starts
            cw_wall_cuts, ccw_wall_cuts = [interior_wire[0]], [interior_wire[1]]
            vertices, cell_ids = [], []
            surf_stack = [
                surface_from_2points(
                    *interior_wire,
                    surface_id=10 * i + 1,
                    name=f"plasma-facing inner surface {i}",
                )
            ]
            # switching logic to choose inboard vs outboard. TODO: functionalize
            if cw_wall[0, 0] < thicknesses.inboard_outboard_transition_radius:
                thickness_here = thicknesses.inboard.extended_prefix_sums()[1:]
            else:
                thickness_here = thicknesses.outboard.extended_prefix_sums()[1:]

            for j, (interface_height, _height_type) in enumerate([
                (0.005, "thick"),
                *[(value, "frac") for value in thickness_here],
            ]):
                # switching logic to choose thickness vs fraction. TODO: functionalize or
                # make prettier in params.py
                if _height_type == "thick":
                    points = pre_cell.get_cell_wall_cut_points_by_thickness(
                        interface_height
                    )
                elif _height_type == "frac":
                    points = pre_cell.get_cell_wall_cut_points_by_fraction(
                        interface_height
                    )
                else:
                    raise RuntimeError
                surf_stack.append(
                    surface_from_2points(*points, surface_id=10 * i + (j + 2))
                )
                vertices.append(
                    Vertices(points[0], points[1], cw_wall_cuts[-1], ccw_wall_cuts[-1])
                )
                cell_ids.append(10 * i + j)
                cw_wall_cuts.append(points[0]), ccw_wall_cuts.append(points[1])
            surf_stack[-1].name = f"vacuum vessel outer surface {i}"

            cls.check_cut_point_ordering(cw_wall_cuts, pre_cell.normal_to_interior)
            cls.check_cut_point_ordering(ccw_wall_cuts, pre_cell.normal_to_interior)

            # TODO: implement curved exterior in the future.
            exterior_curve_comp = pre_cell.exterior_wire.shape.OrderedEdges
            if len(exterior_curve_comp) != 1 or not exterior_curve_comp[
                0
            ].Curve.TypeId.startswith("Part::GeomLine"):
                raise NotImplementedError(
                    "Not ready to make curved-line cross-sections yet!"
                )

            stack = BlanketCellStack.from_surfaces(
                ccw_surf=ccw_surf,
                cw_surf=cw_surf,
                layer_interfaces=surf_stack,
                vertices=vertices,
                fill_dict=material_dict,
                cell_ids=cell_ids,
            )
            cell_array.append(stack)
            ccw_surf = cw_surf  # right wall -> left wall, as we shift right.

        return cls(cell_array)

    def make_plasma_void_region(self) -> openmc.Region:
        """
        Create the region that is enclosed by the first wall.
        We assume that this region is convex. Would fail silently if not.

        Returns
        -------
        plasma_void_upper
            an instance of openmc.Region
        """
        first_wall_surfaces = [radial_stack[0] for radial_stack in self.radial_surfaces]
        joining_points = [
            np.insert(cell_stack[0].vertices.interior_end, 1, 0, axis=-1)
            for cell_stack in self
        ]

        plasma_void_upper = openmc.Intersection(
            choose_region(surf, joining_points) for surf in first_wall_surfaces
        )

        return flatten_region(plasma_void_upper)


class DivertorCell(openmc.Cell):
    """
    A generic Divertor cell forming either the inner target's, outer target's, or
    dome's surface or bulk.
    """

    pass


class DivertorCellStack(abc.Sequence):
    """
    A stack of DivertorCells (openmc.Cells), first cell is closest to the interior and
    last cell is closest to the exterior. They should all be situated on the same
    poloidal angle.
    """

    pass


class DivertorCellArray(abc.Sequence):
    """Turn the divertor into a cell array"""

    pass
