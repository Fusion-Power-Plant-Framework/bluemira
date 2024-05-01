# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Create csg geometry by converting from bluemira geometry objects made of wires. All units
in this module are in SI (distrance:[m]) unless otherwise specified by the docstring.
"""

from __future__ import annotations

from itertools import pairwise
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import openmc

from bluemira.geometry.constants import EPS_FREECAD
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.error import GeometryError
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.tools import make_circle_arc_3P, make_polygon, revolve_shape
from bluemira.geometry.wire import BluemiraWire
from bluemira.neutronics.constants import DTOL_CM, to_cm, to_cm3, to_m
from bluemira.neutronics.params import BlanketLayers
from bluemira.neutronics.radial_wall import (
    CellWalls,
    polygon_revolve_signed_volume,
)
from bluemira.neutronics.wires import CircleInfo, StraightLineInfo, WireInfoList

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    import numpy.typing as npt

    from bluemira.neutronics.make_materials import MaterialsLibrary
    from bluemira.neutronics.make_pre_cell import (
        DivertorPreCell,
        DivertorPreCellArray,
        PreCell,
        PreCellArray,
    )
    from bluemira.neutronics.params import DivertorThickness, TokamakDimensions

# Found to work by trial and error. I'm sorry.
SHRINK_DISTANCE = 0.0005  # [m] = 0.05cm = 0.5 mm


def is_monotonically_increasing(series):
    """Check if a series is monotonically increasing"""  # or decreasing
    return all(np.diff(series) >= -EPS_FREECAD)  # or all(diff<0)


def plot_surfaces(surfaces_list: list[openmc.Surface]):
    """
    Plot a list of surfaces in matplotlib.
    """
    ax = plt.axes()
    # ax.set_aspect(1.0) # don't do this as it makes the plot hard to read.
    for i, surface in enumerate(surfaces_list):
        plot_coords(surface, color_num=i)
    ax.legend()
    ax.set_ylim([-1000, 1000])
    ax.set_xlim([-1000, 1000])


def plot_coords(surface: openmc.Surface, color_num: int):
    """
    In the range [-1000, 1000], plot the RZ cross-section of the ZCylinder/ZPlane/ZCone.
    """
    if isinstance(surface, openmc.ZCylinder):
        plt.plot(
            [surface.x0, surface.x0],
            [-1000, 1000],
            label=f"{surface.id}: {surface.name}",
            color=f"C{color_num}",
        )
    elif isinstance(surface, openmc.ZPlane):
        plt.plot(
            [-1000, 1000],
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

        y_pos, y_neg = equation_pos([-1000, 1000]), equation_neg([-1000, 1000])
        plt.plot(
            [-1000, 1000],
            y_pos,
            label=f"{surface.id}: {surface.name} (upper)",
            color=f"C{color_num}",
        )
        plt.plot(
            [-1000, 1000],
            y_neg,
            label=f"{surface.id}: {surface.name} (lower)",
            linestyle="--",
            color=f"C{color_num}",
        )


def get_depth_values(
    pre_cell: PreCell, blanket_dimensions: TokamakDimensions
) -> npt.NDArray[np.float64]:
    """
    Choose the depth values that this pre-cell is suppose to use, according to where it
    is physically positioned (hence is classified as inboard or outboard).

    Parameters
    ----------
    pre_cell
        :class:`~PreCell` to be classified as either inboard or outboard
    blanket_dimensions
        :class:`bluemira.neutronics.params.TokamakDimensions` recording the
        dimensions of the blanket in SI units (unit: [m]).

    Returns
    -------
    depth_series
        a series of floats corresponding to the N-1 interfaces between the N layers.
        Each float represents how deep into the blanket (i.e. how many [m] into the
        first wall we need to drill, from the plasma facing surface) to hit that
        interface layer.
    """
    if check_inboard_outboard(pre_cell, blanket_dimensions):
        return blanket_dimensions.inboard.get_interface_depths()
    return blanket_dimensions.outboard.get_interface_depths()


def check_inboard_outboard(
    pre_cell: PreCell, blanket_dimensions: TokamakDimensions
) -> bool:
    """If this pre-cell is an inboard, return True.
    Otherwise, this pre-cell belongs to outboard, return False
    """
    cell_reference_radius = pre_cell.vertex[0].mean()
    return cell_reference_radius < blanket_dimensions.inboard_outboard_transition_radius


def torus_from_3points(
    point1: npt.NDArray[np.float64],
    point2: npt.NDArray[np.float64],
    point3: npt.NDArray[np.float64],
    surface_id: int | None = None,
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
    surface_id: int | None = None,
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
    center: npt.NDArray[np.float64],
    minor_radius: float,
    surface_id: int | None = None,
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
    major_radius, height, minor_radius = to_cm([center[0], center[-1], minor_radius])
    return openmc.ZTorus(
        z0=height,
        a=major_radius,
        b=minor_radius,
        c=minor_radius,
        surface_id=surface_id,
        name=name,
    )


def torus_from_5points(
    point1: npt.NDArray[np.float64],
    point2: npt.NDArray[np.float64],
    point3: npt.NDArray[np.float64],
    point4: npt.NDArray[np.float64],
    point5: npt.NDArray[np.float64],
    surface_id: int | None = None,
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
    pt = np.mean(to_cm(choice_points), axis=0)
    value = surface.evaluate(pt)
    if value > 0:
        return +surface
    if value < 0:
        return -surface
    raise GeometryError("Centroid point is directly on the surface")


def choose_plane_cylinders(
    surface: openmc.ZPlane | openmc.ZCylinder, choice_points: npt.NDArray
) -> openmc.Halfspace:
    """
    choose a side of the Halfspace in the region of ZPlane and ZCylinder.

    Parameters
    ----------
    surface
        :class:`openmc.surface.Surface` of a openmc.ZPlane or openmc.ZCylinder
    choice_points: np.ndarray of shape (N, 3)
        a list of points representing the vertices of a convex polygon in RZ plane

    Returns
    -------
    region: openmc.Halfspace
        a Halfspace of the provided surface that the points exists on.
    """
    x, y, z = np.array(to_cm(choice_points)).T
    values = surface.evaluate([x, y, z])
    threshold = DTOL_CM
    if isinstance(surface, openmc.ZCylinder):
        threshold = 2 * DTOL_CM * surface.r + DTOL_CM**2

    if all(values >= -threshold):
        return +surface
    if all(values <= threshold):
        return -surface

    raise GeometryError(f"There are points on both sides of this {type(surface)}!")


# simplfying openmc.Intersection by associativity
def flat_intersection(region_list: Iterable[openmc.Region]) -> openmc.Intersection:
    """
    Get the flat intersection of an entire list of regions.
    e.g. (a (b c)) becomes (a b c)
    """
    possibly_redundant_reg = openmc.Intersection(region_list)
    return openmc.Intersection(intersection_dictionary(possibly_redundant_reg).values())


def intersection_dictionary(region: openmc.Region) -> dict[str, openmc.Region]:
    """Get a dictionary of all of the elements that shall be intersected together,
    applying the rule of associativity
    """
    if isinstance(region, openmc.Halfspace):  # termination condition
        return {region.side + str(region.surface.id): region}
    if isinstance(region, openmc.Intersection):
        final_intersection = {}
        for _r in region:
            final_intersection.update(intersection_dictionary(_r))
        return final_intersection
    return {str(region): region}


# simplfying openmc.Union by associativity
def flat_union(region_list: Iterable[openmc.Region]) -> openmc.Union:
    """
    Get the flat union of an entire list of regions.
    e.g. (a | (b | c)) becomes (a | b | c)
    """
    possibly_redundant_reg = openmc.Union(region_list)
    return openmc.Union(union_dictionary(possibly_redundant_reg).values())


# TODO: Raise issue/papercut to check if simplifying the boolean expressions can yield
# speedup or not, and if so, we should attempt to simplify it further.
# E.g. the expression (-1 ((-1107 -1) | -1108)) can be simplified to (-1107 | -1108) -1;
# And don't even get me started on how much things can get simplified when ~ is involved.
# It is possible that boolean expressions get condensed appropriately before getting
# parsed onto openmc. I can't tell either way.


def union_dictionary(region: openmc.Region) -> dict[str, openmc.Region]:
    """Get a dictionary of all of the elements that shall be unioned together,
    applying the rule of associativity
    """
    if isinstance(region, openmc.Halfspace):  # termination condition
        return {region.side + str(region.surface.id): region}
    if isinstance(region, openmc.Union):
        final_intersection = {}
        for _r in region:
            final_intersection.update(union_dictionary(_r))
        return final_intersection
    return {str(region): region}


class BluemiraNeutronicsCSG:
    """Container for CSG planes to enable reuse of planes, very eco friendly"""

    def __init__(self):
        # it's called a hangar because it's where the planes are parked ;)
        self.hangar = {}

    def surface_from_2points(
        self,
        point1: npt.NDArray[np.float64],
        point2: npt.NDArray[np.float64],
        surface_id: int | None = None,
        name: str = "",
    ) -> openmc.Surface | openmc.model.ZConeOneSided | None:
        """
        Create either a cylinder, a cone, or a surface from 2 points using only the
        rz coordinates of any two points on it.

        Parameters
        ----------
        point1, point2:
            any two non-trivial (i.e. cannot be the same) points on the rz cross-section
            of the surface, each containing the r and z coordinates
            Units: [m]
        surface_id, name:
            see openmc.Surface

        Returns
        -------
        surface: openmc.surface.Surface, None
            if the two points provided are redundant: don't return anything, as this is a
            single point pretending to be a surface. This will come in handy for handling
            the creation of BlanketCells made with 3 surfaces rather than 4.
        """
        point1, point2 = to_cm((point1, point2))
        dr, dz = point2 - point1
        if np.isclose(dr, 0, rtol=0, atol=DTOL_CM):
            _r = point1[0]
            if np.isclose(dz, 0, rtol=0, atol=DTOL_CM):
                return None
                # raise GeometryError(
                #     "The two points provided aren't distinct enough to "
                #     "uniquely identify a surface!"
                # )
            return openmc.ZCylinder(r=_r, surface_id=surface_id, name=name)
        if np.isclose(dz, 0, rtol=0, atol=DTOL_CM):
            _z = point1[-1]
            z_plane = openmc.ZPlane(z0=_z, surface_id=surface_id, name=name)
            self.hangar[_z] = z_plane
            return self.hangar[_z]
        slope = dz / dr
        z_intercept = -slope * point1[0] + point1[-1]
        return openmc.ZCone(
            z0=z_intercept, r2=slope**-2, surface_id=surface_id, name=name
        )

    def surface_from_straight_line(
        self,
        straight_line_info: StraightLineInfo,
        surface_id: int | None = None,
        name: str = "",
    ):
        """Create a surface to match the straight line info provided."""
        start_end = np.array(straight_line_info[:2])[:, ::2]
        return self.surface_from_2points(*start_end, surface_id=surface_id, name=name)

    def surfaces_from_info_list(self, wire_info_list: WireInfoList, name: str = ""):
        """
        Create a list of surfaces using a list of wire infos.

        Parameters
        ----------
        wire_info_list
            List of wires
        name
            This name will be *reused* across all of the surfaces created in this list.
        """
        surface_list = []
        for wire in wire_info_list.info_list:
            info = wire.key_points
            plane_cone_cylinder = self.surface_from_straight_line(info, name=name)
            if isinstance(info, CircleInfo):
                torus = torus_from_circle(info.center, info.radius, name=name)
                # will need the openmc.Union of these two objects later.
                surface_list.append((plane_cone_cylinder, torus))
            else:
                surface_list.append((plane_cone_cylinder,))
        return tuple(surface_list)

    def find_suitable_z_plane(
        self,
        z0: float,
        z_range: Iterable[float] | None = None,
        surface_id: int | None = None,
        name: str = "",
        **kwargs,
    ):
        """
        Find a suitable z from the hangar, or create a new one if no matches are found.

        Parameters
        ----------
        z0:
            The height of the plane, if we need to create it. Unit: [m]
        z_range:
            If we a suitable z-plane already exists, then we only accept it if it lies
            within this range of z. Unit: [m]
        surface_id, name:
            See openmc.Surface
        """
        if z_range is not None:
            z_min, z_max = min(z_range), max(z_range)
            for key in self.hangar:
                if z_min <= key <= z_max:
                    return self.hangar[key]  # return the first match
        self.hangar[z0] = openmc.ZPlane(
            z0=to_cm(z0), surface_id=surface_id, name=name, **kwargs
        )
        return self.hangar[z0]

    def choose_region_cone(
        self,
        surface: openmc.ZCone,
        choice_points: npt.NDArray,
        *,
        control_id: bool = False,
    ) -> openmc.Region:
        """
        choose the region for a ZCone.
        When reading this function's code, bear in mind that a Z cone can be separated
        into 3 parts:
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
            Hence the first step is to shink the choice_points by 0.5%
            towards the centroid.

        Parameters
        ----------
        surface: opnemc.ZCone
            where all points are expected to be excluded from at least one of its two
            one-sided cones.
        choice_points:
            An array of points that, after choosing the appropriate region,
            should all lie in the chosen region.
        control_id:
            When an ambiguity plane is needed, we ned to create a surface. if
            control_id = True, then this would force the surface to be created with
            id = 1000 + the id of the cone. This is typically only used so that
            we have full control of (and easily understandable records of)
            every surfaces' ID.
            Thus elsewhere in the code, most other classes/methods turns control_id
            on when cell_ids are also provided (proving intention on controlling IDs
            of OpenMC objects).

        Returns
        -------
        region
            openmc.Region, specifically (openmc.Halfspace) or
            (openmc.Union of 2 openmc.Halfspaces)
        """
        # shrink to avoid floating point number comparison imprecision issues.
        # Especially important when the choice point sits exactly on the surface.
        centroid = np.mean(choice_points, axis=0)
        step_dir = centroid - choice_points
        unit_step_dir = (step_dir.T / np.linalg.norm(step_dir, axis=1)).T
        choice_points += SHRINK_DISTANCE * unit_step_dir

        # # Alternative
        # choice_points = (choice_points + 0.01 * centroid) / 1.01
        # take one step towards the centroid = 0.1 cm

        x, y, z = np.array(to_cm(choice_points)).T
        values = surface.evaluate([x, y, z])
        middle = values > 0
        if all(middle):  # exist outside of cone
            return +surface

        z_dist = z - surface.z0
        upper_cone = np.logical_and(~middle, z_dist > 0)
        lower_cone = np.logical_and(~middle, z_dist < 0)
        # upper_not_cone = np.logical_and(middle, z_dist > 0)
        # lower_not_cone = np.logical_and(middle, z_dist < 0)

        # all of the following cases requires a plane to be made.
        if control_id:
            surface_id = 1000 + surface.id
        else:
            surface_id = max(openmc.Surface.used_ids)
            surface_id += 1000 + surface_id % 1000

        if all(upper_cone):
            # everything in the upper cone.
            plane = (
                self.find_suitable_z_plane(  # the highest we can cut is at the lowest z.
                    to_m(surface.z0),
                    to_m([surface.z0 - DTOL_CM, min(z) - DTOL_CM]),
                    surface_id=surface_id,
                    name=f"Ambiguity plane for cone {surface.id}",
                )
            )
            return -surface & +plane
        if all(lower_cone):
            # everything in the lower cone
            plane = (
                self.find_suitable_z_plane(  # the lowest we can cut is at the highest z.
                    to_m(surface.z0),
                    to_m([max(z) + DTOL_CM, surface.z0 + DTOL_CM]),
                    surface_id=surface_id,
                    name=f"Ambiguity plane for cone {surface.id}",
                )
            )
            return -surface & -plane
        if all(np.logical_or(upper_cone, lower_cone)):
            raise GeometryError(
                "Both cones have vertices lying inside! Cannot compute a contiguous "
                "region that works for both. Check if polygon is convex?"
            )
        plane = self.find_suitable_z_plane(  # In this rare case, make its own plane.
            to_m(surface.z0),
            to_m([surface.z0 + DTOL_CM, surface.z0 - DTOL_CM]),
            surface_id=surface_id,
            name=f"Ambiguity plane for cone {surface.id}",
        )
        if all(np.logical_or(upper_cone, middle)):
            return +surface | +plane
        if all(np.logical_or(lower_cone, middle)):
            return +surface | -plane
        raise GeometryError("Can't have points in upper cone, lower cone AND outside!")

    def choose_region(
        self,
        surface: openmc.Surface | tuple[openmc.Surface, openmc.ZTorus | None],
        vertices_array: npt.NDArray,
        *,
        control_id: bool = False,
    ) -> openmc.Region:
        """
        Pick the correct region of the surface that includes all of the points in
        vertices_array.

        Parameters
        ----------
        surface
            Either a :class:`openmc.Surface`, or a 1-tuple or 2-tuple of
            :class:`openmc.Surface`. If it is a tuple, the first element is always a
            :class:`openmc.ZPlane`/:class:`openmc.ZCone`/:class:`openmc.ZCylinder`;
            the second element (if present) is always :class:`openmc.ZTorus`.
        vertices_array
            array of shape (?, 3), that the final region should include.
        control_id
            Passed as argument onto
            :func:`~bluemira.neutronics.make_csg.choose_region_cone`

        Returns
        -------
            An openmc.Region built from surface provided and includes all of these
        """
        # switch case: blue
        if isinstance(surface, openmc.ZPlane | openmc.ZCylinder):
            return choose_plane_cylinders(surface, vertices_array)

        if isinstance(surface, openmc.ZCone):
            return self.choose_region_cone(
                surface, vertices_array, control_id=control_id
            )

        if isinstance(surface, tuple):
            chosen_first_region = self.choose_region(
                surface[0], vertices_array, control_id=control_id
            )
            # cone, or cylinder, or plane, or (cone | ambiguity_surface)
            if len(surface) == 1:
                return chosen_first_region
            # -surface[1] = inside of torus
            return flat_union((chosen_first_region, -surface[1]))
        raise NotImplementedError(
            f"Surface {type(surface)} is not ready for use in the axis-symmetric case!"
        )

    def region_from_surface_series(
        self,
        series_of_surfaces: Sequence[
            openmc.Surface | tuple[openmc.Surface, openmc.ZTorus | None] | None
        ],
        vertices_array: npt.NDArray,
        *,
        control_id: bool = False,
    ) -> openmc.Intersection:
        """
        Switch between choose_region() and choose_region_from_tuple_of_surfaces()
        depending on the type of each element in the series_of_surfaces.

        Parameters
        ----------
        series_of_surfaces
            Each of them can be a None, a 1-tuple of surface, a 2-tuple of surfaces, or a
            surface. For the last 3 options, see
            :func:`~bluemira.neutronics.make_csg.choose_region` for more.

        vertices_array
            array of shape (?, 3), where every single point should be included by, or at
            least on the edge of the returned Region.
        control_id
            Passed as argument onto
            :func:`~bluemira.neutronics.make_csg.choose_region_cone`

        Returns
        -------
        intersection_region: openmc.Region
            openmc.Intersection of a list of
            [(openmc.Halfspace) or (openmc.Union of openmc.Halfspace)]
        """
        intersection_regions = []
        for surface in series_of_surfaces:
            if surface is None:
                continue
            intersection_regions.append(
                self.choose_region(surface, vertices_array, control_id=control_id)
            )
        return flat_intersection(intersection_regions)


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
        interior_surface: openmc.Surface | None,
        vertices: Coordinates,
        csg: BluemiraNeutronicsCSG,
        cell_id: int | None = None,
        name: str = "",
        fill: openmc.Material | None = None,
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
        self.vertex = vertices
        self.csg = csg

        super().__init__(
            cell_id=cell_id,
            name=name,
            fill=fill,
            region=self.csg.region_from_surface_series(
                [exterior_surface, ccw_surface, cw_surface, interior_surface],
                self.vertex.xyz.T,
                control_id=bool(cell_id),
            ),
        )

        self.volume = to_cm3(polygon_revolve_signed_volume(vertices.xz.T))
        if self.volume <= 0:
            raise GeometryError("Wrong ordering of vertices!")


class BlanketCellStack:
    """
    A stack of openmc.Cells, first cell is closest to the interior and last cell is
    closest to the exterior. They should all be situated at the same poloidal angle.
    """

    def __init__(self, cell_stack: list[BlanketCell]):
        """
        The shared surface between adjacent cells must be the SAME one, i.e. same id and
        hash, not just identical.
        They must share the SAME counter-clockwise surface and the SAME clockwise surface
        (left and right side surfaces of the stack, for the stack pointing straight up).

        Because the bounding_box function is INCORRECT, we can't perform a check on the
        bounding box to confirm that the stack is linearly increasing/decreasing in xyz
        bounds.

        Series of cells
        ---------
        SurfaceCell
        FirstWallCell
        BreedingZoneCell
        ManifoldCell
        VacuumVesselCell
        """
        self.cell_stack = cell_stack
        for int_cell, ext_cell in pairwise(cell_stack):
            if int_cell.exterior_surface is not ext_cell.interior_surface:
                raise ValueError("Expected a contiguous stack of cells!")

    def __len__(self) -> int:
        """Number of cells in stack"""
        return len(self.cell_stack)

    def __getitem__(self, index_or_slice) -> list[BlanketCell] | BlanketCell:
        """Get cell from stack"""
        return self.cell_stack[index_or_slice]

    def __repr__(self) -> str:
        """String representation"""
        return (
            super()
            .__repr__()
            .replace(" at ", f" of {len(self.cell_stack)} BlanketCells at ")
        )

    @staticmethod
    def check_cut_point_ordering(
        cut_point_series: npt.NDArray[np.float64],
        direction_vector: npt.NDArray[np.float64],
        location_msg: str = "",
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
        direction = direction_vector / np.linalg.norm(direction_vector)
        projections = np.dot(np.array(cut_point_series)[:, [0, -1]], direction)
        if not is_monotonically_increasing(projections):
            raise GeometryError(f"Some surfaces crosses over each other! {location_msg}")

    @property
    def interior_surface(self):
        """Get interior surface"""
        return self.cell_stack[0].interior_surface

    @property
    def exterior_surface(self):
        """Get exterior surface"""
        return self.cell_stack[-1].exterior_surface

    @property
    def ccw_surface(self):
        """Get counter clockwise surface"""
        return self.cell_stack[0].ccw_surface

    @property
    def cw_surface(self):
        """Get clockwise surface"""
        return self.cell_stack[0].cw_surface

    @property
    def interfaces(self):
        """
        All of the radial surfaces, including the innermost (exposed to plasma) and
        outermost (facing vacuum vessel); arranged in that order (from innermost to
        outermost).
        """
        if not hasattr(self, "_interfaces"):
            self._interfaces = [cell.interior_surface for cell in self.cell_stack]
            self._interfaces.append(self.cell_stack[-1].exterior_surface)
        return self._interfaces

    def get_overall_region(
        self, csg: BluemiraNeutronicsCSG, *, control_id: bool = False
    ) -> openmc.Region:
        """
        Calculate the region covering the entire cell stack.

        Parameters
        ----------
        control_id
            Passed as argument onto
            :func:`~bluemira.neutronics.make_csg.region_from_surface_series`
        """
        return csg.region_from_surface_series(
            [
                self.exterior_surface,
                self.ccw_surface,
                self.cw_surface,
                self.interior_surface,
            ],
            np.vstack((
                self.cell_stack[0].vertex.xz.T[(1, 2),],
                self.cell_stack[-1].vertex.xz.T[(3, 0),],
            )),
            control_id=control_id,
        )

    @classmethod
    def from_pre_cell(
        cls,
        pre_cell: PreCell,
        ccw_surface: openmc.Surface,
        cw_surface: openmc.Surface,
        depth_series: Sequence,
        csg: BluemiraNeutronicsCSG,
        fill_lib: dict[str, openmc.Material],
        *,
        inboard: bool,
        blanket_stack_num: int | None = None,
    ):
        """
        Create a CellStack using a precell and TWO surfaces that sandwiches that precell.

        Parameters
        ----------
        pre_cell
            An instance of :class:`~PreCell`
        ccw_surf
            An instance of :class:`openmc.surface.Surface`
        cw_surf
            An instance of :class:`openmc.surface.Surface`
        depth_series
            a series of floats corresponding to the N-2 interfaces between the N-1
            layers, whereas the N-th layer is the vacuum vessel (and the pre-cell has
            already stored the thickness for that).
            Each float represents how deep into the blanket (i.e. how many [cm] into the
            first wall we need to drill, from the plasma facing surface) to hit that
            interface layer.
        fill_lib
            :class:`~MaterialsLibrary` so that it separates into .inboard, .outboard,
            .divertor, .tf_coil_windings, etc.
        inboard
            boolean denoting whether this cell is inboard or outboard
        blanket_stack_num
            An optional number indexing the current stack. Used for labelling.

        """
        # check exterior wire is correct
        ext_curve_comp = pre_cell.exterior_wire.shape.OrderedEdges
        if len(ext_curve_comp) != 1:
            raise TypeError("Incorrect type of BluemiraWire parsed in.")
        if not ext_curve_comp[0].Curve.TypeId.startswith("Part::GeomLine"):
            raise NotImplementedError("Not ready to make curved-line cross-section yet!")

        # 1. Calculate cut points required to make the surface stack, without actually
        #    creating the surfaces.
        # shape (M+1, 2, 2)
        wall_cut_pts = np.asarray([
            pre_cell.cell_walls.starts,
            *(
                pre_cell.get_cell_wall_cut_points_by_thickness(interface_depth)
                for interface_depth in depth_series
            ),
            np.array([pre_cell.vv_point[:, 0], pre_cell.vv_point[:, 1]]),
            pre_cell.cell_walls.ends,
        ])
        # 1.1 perform sanity check
        directions = np.diff(pre_cell.cell_walls, axis=1)  # shape (2, 1, 2)
        dirs = directions[:, 0, :]
        i = blanket_stack_num or "(unspecified)"
        cls.check_cut_point_ordering(
            wall_cut_pts[:, 0],
            dirs[0],
            location_msg=f"Occuring in cell stack {i}'s CCW wall",
        )
        cls.check_cut_point_ordering(
            wall_cut_pts[:, 1],
            dirs[1],
            location_msg=f"Occuring in cell stack {i}'s CW wall",
        )

        # 2. Accumulate the corners of each cell.
        vertices = [
            Coordinates({
                "x": np.asarray([out[1, 0], inn[1, 0], inn[0, 0], out[0, 0]]),
                "z": np.asarray([out[1, 1], inn[1, 1], inn[0, 1], out[0, 1]]),
            })
            for inn, out in pairwise(wall_cut_pts)
        ]

        # shape (M, 2, 2)
        projection_ccw = wall_cut_pts[:, 0] @ dirs[0] / np.linalg.norm(dirs[0])
        projection_cw = wall_cut_pts[:, 1] @ dirs[1] / np.linalg.norm(dirs[1])
        layer_mask = np.array(
            [
                not (ccw_depth <= DTOL_CM and cw_depth <= DTOL_CM)
                for (ccw_depth, cw_depth) in zip(
                    np.diff(projection_ccw), np.diff(projection_cw), strict=True
                )
            ],
            dtype=bool,
        )  # shape (M,)

        # 3. Choose the ID of the stack's surfaces and cells.
        if blanket_stack_num is not None:
            # Note: all IDs must be natural number, i.e. integer > 0.
            # So we're using an indexing scheme that starts from 1.
            # len=M
            cell_ids = [10 * blanket_stack_num + v + 1 for v in range(len(vertices))]
            # left (ccw_surface) surface had already been created, and since our indexing
            # scheme starts from 1, therefore we're using +2 in the following line.
            # len=M+1
            surface_ids = [
                10 * blanket_stack_num + v + 2 for v in range(len(wall_cut_pts))
            ]
        else:
            cell_ids = [None] * len(vertices)  # len=M
            surface_ids = [None] * len(wall_cut_pts)  # len=M+1

        # 4. create the actual openmc.Surfaces and Cells.
        int_surf = (
            csg.surface_from_2points(
                *wall_cut_pts[0],
                surface_id=surface_ids[0],
                name=f"plasma-facing surface of blanket cell stack {i}",
            )
            if pre_cell.interior_wire
            else None
        )  # account for the case.

        cell_stack = []
        # k = range(0, M - layer_mask == False)
        for k, points in enumerate(wall_cut_pts[1:][layer_mask]):
            j = k + 1  # = range(1, M+1)
            cell_type = BlanketLayers(j)
            if j > 1:
                int_surf.name = (
                    f"{cell_type}-{cell_type.name} "
                    f"interface boundary of blanket cell stack {i}"
                )
            ext_surf = csg.surface_from_2points(
                *points,
                surface_id=surface_ids[j],  # up to M+1
            )

            cell_stack.append(
                BlanketCell(
                    exterior_surface=ext_surf,
                    ccw_surface=ccw_surface,
                    cw_surface=cw_surface,
                    interior_surface=int_surf,
                    vertices=vertices[k],  # up to M
                    csg=csg,
                    cell_id=cell_ids[k],  # up to M
                    name=f"{cell_type.name} of blanket cell stack {i}",
                    fill=fill_lib.match_blanket_material(cell_type, inboard=inboard),
                )
            )
            int_surf = ext_surf
        int_surf.name = "vacuum-vessel-facing surface"

        return cls(cell_stack)


class BlanketCellArray:
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

    def __init__(
        self, blanket_cell_array: list[BlanketCellStack], csg: BluemiraNeutronicsCSG
    ):
        """
        Create array from a list of BlanketCellStack
        """
        self.blanket_cell_array = blanket_cell_array
        self.poloidal_surfaces = [self.blanket_cell_array[0].ccw_surface]
        self.radial_surfaces = []
        self.csg = csg
        for i, this_stack in enumerate(self.blanket_cell_array):
            self.poloidal_surfaces.append(this_stack.cw_surface)
            self.radial_surfaces.append(this_stack.interfaces)

            # check neighbouring cells share the same lateral surface
            if i != len(self.blanket_cell_array) - 1:
                next_stack = self.blanket_cell_array[i + 1]
                if this_stack.cw_surface is not next_stack.ccw_surface:
                    raise GeometryError(
                        f"Neighbouring BlanketCellStack [{i}] and "
                        f"[{i + 1}] are not aligned!"
                    )

    def __len__(self) -> int:
        """Number of cell stacks"""
        return len(self.blanket_cell_array)

    def __getitem__(self, index_or_slice) -> list[BlanketCellStack] | BlanketCellStack:
        """Get cell stack"""
        return self.blanket_cell_array[index_or_slice]

    def __repr__(self) -> str:
        """String representation"""
        return (
            super()
            .__repr__()
            .replace(" at ", f" of {len(self.blanket_cell_array)} BlanketCellStacks at ")
        )

    def exterior_vertices(self) -> npt.NDArray:
        """
        Returns all of the tokamak's poloidal cross-section's outside corners'
        coordinates, in 3D.

        Returns
        -------
        exterior_vertices: npt.NDArray of shape (N+1, 3)
            Arranged clockwise (inboard to outboard).
        """
        exterior_vertices = [self.blanket_cell_array[0][-1].vertex.xyz[:, 3]]
        exterior_vertices.extend(
            stack[-1].vertex.xyz[:, 0] for stack in self.blanket_cell_array
        )
        return np.array(exterior_vertices)

    def interior_vertices(self) -> npt.NDArray:
        """
        Returns all of the tokamak's poloidal cross-section's inside corners'
        coordinates, in 3D.

        Parameters
        ----------
        interior_vertices: npt.NDArray of shape (N+1, 3)
            Arranged clockwise (inboard to outboard).
        """
        return np.asarray([
            self.blanket_cell_array[0][0].vertex.xyz[:, 2],
            *(stack[0].vertex.xyz[:, 1] for stack in self.blanket_cell_array),
        ])

    def interior_surfaces(self) -> list[openmc.Surface]:
        """
        Get all of the innermost (plasm-facing) surface.
        Runs clockwise.
        """
        return [surf_stack[0] for surf_stack in self.radial_surfaces]

    def exterior_surfaces(self) -> list[openmc.Surface]:
        """
        Get all of the outermost (vacuum-vessel-facing) surface.
        Runs clockwise.
        """
        return [surf_stack[-1] for surf_stack in self.radial_surfaces]

    def exclusion_zone(self, *, control_id: bool = False) -> openmc.Region:
        """
        Get the exclusion zone AWAY from the plasma.
        Usage: plasma_region = openmc.Union(..., ~self.exclusion_zone(), ...)
        Assumes that all of the panels (interior surfaces) together forms a convex hull.

        Parameters
        ----------
        control_id
            Passed as argument onto
            :func:`~bluemira.neutronics.make_csg.region_from_surface_series`.
        """
        return openmc.Union([
            self.csg.region_from_surface_series(
                [stack.cw_surface, stack.ccw_surface, stack.interior_surface],
                np.vstack((
                    stack[0].vertex.xz.T[(1, 2),],
                    stack[-1].vertex.xz.T[(3, 0),],
                )),
                control_id=control_id,
            )
            for stack in self.blanket_cell_array
        ])

    @classmethod
    def from_pre_cell_array(
        cls,
        pre_cell_array: PreCellArray,
        material_library: MaterialsLibrary,
        blanket_dimensions: TokamakDimensions,
        csg: BluemiraNeutronicsCSG,
        *,
        control_id: bool = False,
    ) -> BlanketCellArray:
        """
        Create a BlanketCellArray from a
        :class:`~bluemira.neutronics.make_pre_cell.PreCellArray`.
        This method assumes itself is the first method to be run to create cells in the
        :class:`~openmc.Universe.`

        Parameters
        ----------
        pre_cell_array
            PreCellArray
        material_library
            :class:`~MaterialsLibrary` so that it separates into .inboard, .outboard,
            .divertor, .tf_coil_windings, etc.
        blanket_dimensions
            :class:`bluemira.neutronics.params.TokamakDimensions` recording the
            dimensions of the blanket in SI units (unit: [m]).
        control_id
            Passed as argument onto
            :func:`~bluemira.neutronics.make_csg.region_from_surface_series`.
        """
        cell_walls = CellWalls.from_pre_cell_array(pre_cell_array)

        # left wall
        ccw_surf = csg.surface_from_2points(
            *cell_walls[0],
            surface_id=1 if control_id else None,
            name="Blanket cell wall 0",
        )
        cell_array = []
        for i, (pre_cell, cw_wall) in enumerate(
            zip(pre_cell_array.pre_cells, cell_walls[1:], strict=True)
        ):
            # right wall
            cw_surf = csg.surface_from_2points(
                *cw_wall,
                surface_id=1 + 10 * (i + 1) if control_id else None,
                name=f"Blanket cell wall of blanket cell stack {i + 1}",
            )

            stack = BlanketCellStack.from_pre_cell(
                pre_cell,
                ccw_surf,
                cw_surf,
                get_depth_values(pre_cell, blanket_dimensions),
                csg=csg,
                fill_lib=material_library,
                inboard=check_inboard_outboard(pre_cell, blanket_dimensions),
                blanket_stack_num=i if control_id else None,
            )
            cell_array.append(stack)
            ccw_surf = cw_surf

        return cls(cell_array, csg)


class DivertorCell(openmc.Cell):
    """
    A generic Divertor cell forming either the (inner target's/outer target's/
    dome's) (surface/ bulk).
    """

    def __init__(
        self,
        exterior_surfaces: list[tuple[openmc.Surface]],
        cw_surface: openmc.Surface,
        ccw_surface: openmc.Surface,
        interior_surfaces: list[tuple[openmc.Surface]],
        exterior_wire: WireInfoList,
        interior_wire: WireInfoList,
        csg: BluemiraNeutronicsCSG,
        subtractive_region: openmc.Region | None = None,
        cell_id: int | None = None,
        name: str = "",
        fill: openmc.Material | None = None,
    ):
        """Create a cell from exterior_surface"""
        self.exterior_surfaces = exterior_surfaces
        self.cw_surface = cw_surface
        self.ccw_surface = ccw_surface
        self.interior_surfaces = interior_surfaces
        self.exterior_wire = exterior_wire
        self.interior_wire = interior_wire
        self.csg = csg

        region = self.csg.region_from_surface_series(
            [
                self.cw_surface,
                self.ccw_surface,
                *self.exterior_surfaces,
                *self.interior_surfaces,
            ],
            self.get_all_vertices(),
            control_id=bool(cell_id),
        )

        if subtractive_region:
            region &= ~subtractive_region
        super().__init__(cell_id=cell_id, name=name, fill=fill, region=region)
        self.volume = self.get_volume()

    @property
    def outline(self):
        """
        Make the outline into a BluemiraWire. This method is created solely for the
        purpose of calculating the volume.

        This is slow but it is accurate and works well.
        """
        if not hasattr(self, "_outline"):
            wire_list = [
                self.interior_wire.restore_to_wire(),
                self.exterior_wire.restore_to_wire(),
            ]
            # make the connecting wires so that BluemiraWire doesn't moan about
            # having a discontinuous wires.
            i = self.interior_wire.get_3D_coordinates()
            e = self.exterior_wire.get_3D_coordinates()
            if not np.array_equal(e[-1], i[0]):
                wire_list.insert(0, make_polygon([e[-1], i[0]]))
            if not np.array_equal(i[-1], e[0]):
                wire_list.insert(-1, make_polygon([i[-1], e[0]]))
            self._outline = BluemiraWire(wire_list)
        return self._outline

    def get_volume(self):
        """
        Get the volume using the BluemiraWire of its own outline.
        """
        half_solid = BluemiraSolid(revolve_shape(self.outline))
        cm3_volume = to_cm3(half_solid.volume * 2)
        if cm3_volume <= 0:
            raise GeometryError("Volume (as calculated by FreeCAD) is negative!")
        return cm3_volume

    def get_all_vertices(self) -> npt.NDArray:
        """
        Get all of the vertices of this cell, which should help us find its convex hull.
        """
        return np.concatenate([
            self.exterior_wire.get_3D_coordinates(),
            self.interior_wire.get_3D_coordinates(),
        ])

    def exclusion_zone(
        self, *, away_from_plasma: bool = True, control_id: bool = False
    ) -> openmc.Region:
        """
        Get the exclusion zone of a semi-CONVEX cell.

        This can only be validly used:
            If away_from_plasma=True, then the interior side of the cell must be convex.
            If away_from_plasma=False, then the exterior_side of the cell must be convex.
        Usage: next_cell_region = flat_intersection(..., ~this_cell.exclusion_zone())

        Parameters
        ----------
        control_id
            Passed as argument onto
            :func:`~bluemira.neutronics.make_csg.region_from_surface_series`
        """
        _surfaces = [self.cw_surface, self.ccw_surface]
        if away_from_plasma:
            _surfaces.extend(self.interior_surfaces)
        else:
            _surfaces.extend(self.exterior_surfaces)
        return self.csg.region_from_surface_series(
            _surfaces, self.get_all_vertices(), control_id=control_id
        )


class DivertorCellStack:
    """
    A CONVEX object! i.e. all its exterior points together should make a convex hull.
    A stack of DivertorCells (openmc.Cells), first cell is closest to the interior and
    last cell is closest to the exterior. They should all be situated on the same
    poloidal angle.
    """

    def __init__(
        self, divertor_cell_stack: list[DivertorCell], csg: BluemiraNeutronicsCSG
    ):
        self.cell_stack = divertor_cell_stack
        self.csg = csg
        # This check below is invalid because of how we subtract region instead.
        # for int_cell, ext_cell in pairwise(self.cell_stack):
        #     if int_cell.exterior_surfaces is not ext_cell.interior_surfaces:
        #         raise ValueError("Expected a contiguous stack of cells!")

    @property
    def interior_surfaces(self):
        """Get interior surfaces"""
        return self.cell_stack[0].interior_surfaces

    @property
    def exterior_surfaces(self):
        """Get exterior surfaces"""
        return self.cell_stack[-1].exterior_surfaces

    @property
    def ccw_surface(self):
        """Get counter clockwise surface"""
        return self.cell_stack[-1].ccw_surface

    @property
    def cw_surface(self):
        """Get clockwise surface"""
        return self.cell_stack[-1].cw_surface

    @property
    def exterior_wire(self):
        """Alias to find the outermost cell's exterior_wire"""
        return self.cell_stack[-1].exterior_wire

    @property
    def interior_wire(self):
        """Alias to find the innermost cell's interior_wire"""
        return self.cell_stack[0].interior_wire

    @property
    def interfaces(self):
        """
        All of the radial surfaces, including the innermost (exposed to plasma) and
        outermost (facing the vacuum vessel); arranged in that order (from innermost to
        outermost).
        """
        if not hasattr(self, "_interfaces"):
            self._interfaces = [cell.interior_surfaces for cell in self.cell_stack]
            self._interfaces.append(self.cell_stack[-1].exterior_surfaces)
        return self._interfaces  # list of list of (1- or 2-tuple of) surfaces.

    def __len__(self) -> int:
        """Length of DivertorCellStack"""
        return len(self.cell_stack)

    def __getitem__(self, index_or_slice) -> list[DivertorCell] | DivertorCell:
        """Get item for DivertorCellStack"""
        return self.cell_stack[index_or_slice]

    def __repr__(self) -> str:
        """String representation"""
        return super().__repr__().replace(" at ", f" of {len(self)} DivertorCells at ")

    def get_all_vertices(self) -> npt.NDArray:
        """
        Returns
        -------
        vertices_array
            shape = (N+M, 3)
        """
        return np.concatenate([
            self.interior_wire.get_3D_coordinates(),
            self.exterior_wire.get_3D_coordinates(),
        ])

    def get_overall_region(self, *, control_id: bool = False) -> openmc.Region:
        """
        Get the region that this cell-stack encompasses.

        Parameters
        ----------
        control_id
            Passed as argument onto
            :func:`~bluemira.neutronics.make_csg.region_from_surface_series`
        """
        return self.csg.region_from_surface_series(
            [
                self.cw_surface,
                self.ccw_surface,
                *self.interior_surfaces,
                *self.exterior_surfaces,
            ],
            self.get_all_vertices(),
            control_id=control_id,
        )

    @classmethod
    def from_divertor_pre_cell(
        cls,
        divertor_pre_cell: DivertorPreCell,
        cw_surface: openmc.Surface,
        ccw_surface: openmc.Surface,
        material_library: MaterialsLibrary,
        csg: BluemiraNeutronicsCSG,
        armour_thickness: float = 0,
        stack_num: str | int = "",
    ) -> DivertorCellStack:
        """
        Create a stack from a single pre-cell and two poloidal surfaces sandwiching it.

        Parameters
        ----------
        stack_num:
            A string or number to identify the cell stack by.
        """
        # I apologize that this is still hard to read.
        # I'm trying to make cell_stack a 3-element list if armour_thickness>0,
        # but a 2-element list if armour_thickness==0.

        outermost_wire = divertor_pre_cell.exterior_wire
        outermost_surf = csg.surfaces_from_info_list(outermost_wire)
        outer_wire = divertor_pre_cell.vv_wire
        outer_surf = csg.surfaces_from_info_list(outer_wire)
        if armour_thickness == 0:
            inner_wire = divertor_pre_cell.interior_wire
            inner_surf = csg.surfaces_from_info_list(inner_wire)
        else:
            inner_wire = divertor_pre_cell.offset_interior_wire(armour_thickness)
            inner_surf = csg.surfaces_from_info_list(inner_wire)
            innermost_wire = divertor_pre_cell.interior_wire
            innermost_surf = csg.surfaces_from_info_list(innermost_wire)
        cell_stack = [
            # The middle cell is the only cell guaranteed to be convex.
            # Therefore it is the first cell to be made.
            DivertorCell(
                # surfaces: ext, cw, ccw, int.
                outer_surf,
                cw_surface,
                ccw_surface,
                inner_surf,
                # wires: ext, int.
                outer_wire,
                inner_wire,
                csg=csg,
                name=f"Bulk of divertor in diver cell stack {stack_num}",
                fill=material_library.divertor_mat,
            ),
        ]
        cell_stack.append(
            DivertorCell(
                # surfaces: ext, cw, ccw, int.
                outermost_surf,
                cw_surface,
                ccw_surface,
                inner_surf,
                # wires: ext, int.
                outermost_wire,
                outer_wire.reverse(),
                csg=csg,
                subtractive_region=cell_stack[0].exclusion_zone(away_from_plasma=False),
                name="Vacuum Vessel behind the divertor in divertor cell stack"
                f"{stack_num}",
                fill=material_library.div_fw_mat,
            )
        )
        # Unfortunately, this does mean that the vacuum vessel has a larger ID than the
        # divertor cassette.
        if armour_thickness > 0:
            # exterior of bulk becomes the interior of surface cell.
            cell_stack.insert(
                0,
                DivertorCell(
                    # surfaces: ext, cw, ccw, int.
                    # Same ext surfaces as before.
                    # It'll be handled by subtractive_region later.
                    outer_surf,
                    cw_surface,
                    ccw_surface,
                    innermost_surf,
                    # wires: ext, int.
                    inner_wire.reverse(),
                    innermost_wire,
                    csg=csg,
                    name=f"Divertor armour in divertor cell stack {stack_num}",
                    # subtract away everything in the first cell.
                    subtractive_region=cell_stack[0].exclusion_zone(
                        away_from_plasma=True
                    ),
                    fill=material_library.div_sf_mat,
                ),
            )
            # again, unfortunately, this does mean that the surface armour cell has the
            # largest ID.
        return cls(cell_stack, csg)


class DivertorCellArray:
    """Turn the divertor into a cell array"""

    def __init__(self, cell_array: list[DivertorCellStack]):
        """Create array from a list of DivertorCellStack."""
        self.cell_array = cell_array
        self.poloidal_surfaces = [self.cell_array[0].cw_surface]
        self.radial_surfaces = []
        # check neighbouring cells have the same cell stack.
        for i, this_stack in enumerate(self.cell_array):
            self.poloidal_surfaces.append(this_stack.ccw_surface)
            self.radial_surfaces.append(this_stack.interfaces)

            # check neighbouring cells share the same lateral surface
            if i != len(self.cell_array) - 1:
                next_stack = self.cell_array[i + 1]
                if this_stack.ccw_surface is not next_stack.cw_surface:
                    raise GeometryError(
                        f"Neighbouring DivertorCellStack {i} and {i + 1} are expected to"
                        " share the same poloidal wall."
                    )

    def __len__(self) -> int:
        """Length of DivertorCellArray"""
        return len(self.cell_array)

    def __getitem__(self, index_or_slice) -> list[DivertorCellStack] | DivertorCellStack:
        """Get item for DivertorCellArray"""
        return self.cell_array[index_or_slice]

    def __repr__(self) -> str:
        """String representation"""
        return (
            super().__repr__().replace(" at ", f" of {len(self)} DivertorCellStacks at")
        )

    def interior_surfaces(self) -> list[openmc.Surface]:
        """
        Get all of the innermost (plasm-facing) surface.
        Runs clockwise.
        """
        return [surf_stack[0] for surf_stack in self.radial_surfaces]

    def exterior_surfaces(self) -> list[openmc.Surface]:
        """
        Get all of the outermost (vacuum-vessel-facing) surface.
        Runs clockwise.
        """
        return [surf_stack[-1] for surf_stack in self.radial_surfaces]

    def exterior_vertices(self) -> npt.NDArray:
        """
        Returns all of the tokamak's poloidal cross-section's outside corners'
        coordinates, in 3D.

        Returns
        -------
        exterior_vertices: npt.NDArray of shape (N+1, 3)
            Arranged counter-clockwise (inboard to outboard).
        """
        return np.concatenate([
            stack.exterior_wire.get_3D_coordinates()[::-1] for stack in self.cell_array
        ])

    def interior_vertices(self) -> npt.NDArray:
        """
        Returns all of the tokamak's poloidal cross-section's inside corners'
        coordinates, in 3D.

        Parameters
        ----------
        interior_vertices: npt.NDArray of shape (N+1, 3)
            Arranged counter-clockwise (inboard to outboard).
        """
        return np.concatenate([
            stack.interior_wire.get_3D_coordinates() for stack in self.cell_array
        ])

    def exclusion_zone(self, *, control_id: bool = False) -> openmc.Region:
        """
        Get the exclusion zone AWAY from the plasma.
        Usage: plasma_region = openmc.Union(..., ~self.exclusion_zone(), ...)
        Assumes every single cell-stack is made of an interior surface which itself forms
        a convex hull.

        Parameters
        ----------
        control_id
            Passed as argument onto
            :func:`~bluemira.neutronics.make_csg.region_from_surface_series`
        """
        return openmc.Union([
            stack[0].exclusion_zone(away_from_plasma=True, control_id=control_id)
            for stack in self.cell_array
        ])

    @classmethod
    def from_pre_cell_array(
        cls,
        pre_cell_array: DivertorPreCellArray,
        material_library: MaterialsLibrary,
        divertor_thickness: DivertorThickness,
        csg: BluemiraNeutronicsCSG,
        override_start_end_surfaces: tuple[openmc.Surface, openmc.Surface] | None = None,
    ) -> DivertorCellArray:
        """
        Create the entire divertor from the pre-cell array.

        Parameters
        ----------
        pre_cell_array
            The array of divertor pre-cells.
        material_library
            container of openmc.Material
        divertor_thickness
            A parameter :class:`bluemira.neutronics.params.DivertorThickness`. For now it
            only has one scalar value stating how thick the divertor armour should be.
        override_start_end_surfaces
            openmc.Surfaces that would be used as the first cw_surface and last
            ccw_surface
        """
        stack_list = []

        def get_final_surface() -> openmc.Surface:
            """Generate the final surface on-the-fly so that it gets the correct id."""
            if override_start_end_surfaces:
                return override_start_end_surfaces[-1]
            return csg.surface_from_straight_line(
                pre_cell_array[-1].ccw_wall[-1].key_points
            )

        if override_start_end_surfaces:
            cw_surf = override_start_end_surfaces[0]
        else:
            cw_surf = csg.surface_from_straight_line(
                pre_cell_array[0].cw_wall[0].key_points
            )
        for i, dpc in enumerate(pre_cell_array):
            if i == (len(pre_cell_array) - 1):
                ccw_surf = get_final_surface()
            else:
                ccw_surf = csg.surface_from_straight_line(dpc.ccw_wall[-1].key_points)
            stack_list.append(
                DivertorCellStack.from_divertor_pre_cell(
                    dpc,
                    cw_surf,
                    ccw_surf,
                    material_library,
                    csg,
                    divertor_thickness.surface,
                    i + 1,
                )
            )
            cw_surf = ccw_surf
        return cls(stack_list)

    def get_hollow_merged_cells(self) -> list[openmc.Cell]:
        """
        Returns a list of cells (unnamed, unspecified-ID) where each corresponds to a
        cell-stack.
        """
        return [
            openmc.Cell(region=stack.get_overall_region()) for stack in self.cell_array
        ]


class TFCoils:
    """Turn the divertor into a cell array"""

    def __init__(self, tf_coils: list[openmc.Cell]):
        """Create array from a list of openmc.Cell."""
        self.tf_coils = tf_coils

    def __len__(self) -> int:
        """Get number of tf coil cells"""
        return len(self.tf_coils)

    def __getitem__(self, index_or_slice) -> list[openmc.Cell] | openmc.Cell:
        """Get tf coil cell"""
        return self.tf_coils[index_or_slice]

    def __repr__(self) -> str:
        """String representation"""
        return (
            super()
            .__repr__()
            .replace(" at ", f" of {len(self.tf_coils)} openmc.Cells of tf-coils at")
        )
