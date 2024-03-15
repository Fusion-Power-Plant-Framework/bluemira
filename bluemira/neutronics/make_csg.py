# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Create csg geometry by converting from bluemira geometry objects made of wires."""
# ruff: noqa: PLR2004, D105

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Union

import numpy as np
import openmc
from numpy import typing as npt

from bluemira.geometry.constants import EPS_FREECAD
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.error import GeometryError
from bluemira.geometry.tools import make_circle_arc_3P
from bluemira.neutronics.params import BlanketLayers
from bluemira.neutronics.radial_wall import CellWalls

if TYPE_CHECKING:
    from bluemira.geometry.wire import BluemiraWire
    from bluemira.neutronics.make_pre_cell import PreCellArray
    from bluemira.neutronics.params import TokamakThicknesses


def is_strictly_monotonically_increasing(series):
    """Check if a series is strictly monotonically increasing"""  # or decreasing
    diff = np.diff(series)
    return all(diff > 0)  # or all(diff<0)


def surface_from_2points(
    point1: npt.NDArray,
    point2: npt.NDArray,
    surface_id: Optional[int] = None,
    name: str = "",
) -> Optional[openmc.Surface]:
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
        if np.isclose(dz, 0, rtol=0, atol=EPS_FREECAD):
            # raise GeometryError(
            #     "The two points provided aren't distinct enough to "
            #     "uniquely identify a surface!"
            # )
            return None
        return openmc.ZCylinder(r=point1[0], surface_id=surface_id, name=name)
    if np.isclose(dz, 0, rtol=0, atol=EPS_FREECAD):
        return openmc.ZPlane(z0=point1[-1], surface_id=surface_id, name=name)
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
    return torus_from_BMWire_circle(circle, surface_id=surface_id, name=name)


def torus_from_BMWire_circle(
    bmwire_circle: BluemiraWire, surface_id: Optional[int] = None, name: str = ""
):
    """
    Make a circular torus centered on the z-axis matching the circle provided in a
    bluemirawire.
    All 3 points should lie on the RZ plane AND the surface of the torus simultaneously.

    Parameters
    ----------
    bmwire_circle
        A BluemiraWire that is made of only a single circle/ arc of circle.
    surface_id, name:
        See openmc.Surface
    """
    if len(bmwire_circle.shape.OrderedEdges) != 1:
        raise ValueError("Expected a BluemiraWire made of only one (1) wire: a circle.")
    cad_circle = bmwire_circle.shape.OrderedEdges[0].Curve
    center = cad_circle.Center[0], cad_circle.Center[-1]
    minor_radius = cad_circle.Radius
    return z_torus(center, minor_radius, surface_id=surface_id, name=name)


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
    point1: npt.NDArray,
    point2: npt.NDArray,
    point3: npt.NDArray,
    point4: npt.NDArray,
    point5: npt.NDArray,
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
    surface: openmc.Surface, choice_point: Iterable[float]
) -> openmc.Halfspace:
    """
    Parameters
    ----------
    surface
        A surface that we want to choose a side of
    choice_point
        A point on the region of the side of the surface that we want.

    Returns
    -------
    openmc.surface.Halfspace
    """
    value = surface.evaluate(choice_point)
    if np.sign(value) == +1:
        return +surface
    if np.sign(value) == -1:
        return -surface
    raise ValueError("Choice point is located exactly on the surface!")


class BlanketCell(openmc.Cell):
    """
    A generic blanket cell that forms the base class for the five specialized types of
    blanket cells.

    It's a special case of openmc.Cell, in that it has 3 to 4 surfaces (mandatory
    surfaces: exterior_surface, ccw_surface, cw_surface; optional: interior_surface), and
    it is more wieldy because we don't have to specify the relevant half-space for each
    surface; instead an example_point (any point inside the cell will do) is provided by
    the user and that would choose the appropriate half-space.
    """

    def __init__(
        self,
        exterior_surface: openmc.Surface,
        ccw_surface: openmc.Surface,
        cw_surface: openmc.Surface,
        example_point: Union[Iterable[float], Coordinates],
        interior_surface: Optional[openmc.Surface] = None,
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
        example_point
            Any arbitrary point inside the cell. Could be a simple xz coordiante, or a
            :class:`bluemira.geometry.coordinates.Coordinates` object.
        interior_surface
            Surface on the interior side of the cell
        cell_id
            see :class:`openmc.Cell`
        name
            see :class:`openmc.Cell`
        fill
            see :class:`openmc.Cell`
        """
        self.interior_surface = interior_surface
        self.exterior_surface = exterior_surface
        self.ccw_surface = ccw_surface
        self.cw_surface = cw_surface

        if not isinstance(example_point, Coordinates):
            example_point = Coordinates([example_point[0], 0, example_point[-1]])
        self.example_point = example_point

        region = (
            choose_halfspace(self.exterior_surface, example_point)
            & choose_halfspace(self.ccw_surface, example_point)
            & choose_halfspace(self.cw_surface, example_point)
        )
        if self.interior_surface:
            region &= choose_halfspace(self.interior_surface, example_point)
        super().__init__(cell_id=cell_id, name=name, fill=fill, region=region)

    #     _surfaces_list = [self.exterior_surface, self.ccw_surface, self.cw_surface]
    #     if self.interior_surface:
    #         _surfaces_list.append(self.interior_surface)
    #     self.cell = openmc_cell_from_surfaces_and_point(
    #                 _surfaces_list,
    #                 self.example_point,
    #                 cell_id=cell_id,
    #                 name=name,
    #                 fill=fill)

    # def __len__(self):
    #     return 1

    # def __iter__(self):
    #     return iter([self.cell,])

    # def __getitem__(self, index):
    #     if index==0:
    #         return self.cell
    #     else:
    #         raise IndexError(f"Only one cell is available in {str(self)}")


def openmc_cell_from_surfaces_and_point(
    surfaces: Iterable[openmc.Surface],
    example_point: Iterable[float],
    cell_id=None,
    name="",
    fill=None,
) -> openmc.Cell:
    """
    Create from a list of surfaces and a point

    Parameters
    ----------
    surfaces:
        List of openmc.surface
    example_point:
        Any arbitrary point inside the cell. Could be a simple pair of xz coordiante, or
        a :class:`bluemira.geometry.coordinates.Coordinates` object.
    cell_id
        see :class:`openmc.Cell`
    name
        see :class:`openmc.Cell`
    fill
        see :class:`openmc.Cell`
    """
    if not isinstance(example_point, Coordinates):
        example_point = Coordinates([example_point[0], 0, example_point[-1]])
    region = openmc.Intersection([choose_halfspace(s, example_point) for s in surfaces])
    return openmc.Cell(cell_id=cell_id, name=name, fill=fill, region=region)


class BlanketCellStack:
    """
    A stack of openmc.Cells, stacking from the inboard direction towards the outboard
    direction. They should all be situated at the same poloidal angle.
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

    def __iter__(self):
        return self.cell_stack.__iter__()

    def __repr__(self) -> str:
        return super().__repr__().replace(" at ", f" of {len(self)} BlanketCells at ")

    @staticmethod
    def ascribe_iterable_quantity_to_layer(quantity: Iterable) -> Dict:
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
        """All of the radial surfaces, arranged from innermost to outermost."""
        if not hasattr(self, "_interfaces"):
            self._interfaces = [cell.interior_surface for cell in self.cell_stack]
            self._interfaces.append(self.exterior_surface)
        return self._interfaces

    @classmethod
    def from_surfaces(
        cls,
        ccw_surf: openmc.Surface,
        cw_surf: openmc.Surface,
        layer_interfaces: List[openmc.Surface],
        example_points: Iterable[Iterable[float]],
        fill_dict: dict[str, openmc.Material],
        cell_ids: Optional[Iterable[int]] = None,
        cell_names: Optional[Iterable[int]] = None,
    ):
        """Create a stack of cells from a collection of openmc.Surfaces."""
        num_cell_in_stack = len(example_points)
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
        for _cell_num, (int_surf, ext_surf, eg_point, _id, _name) in enumerate(
            zip(
                layer_interfaces[:-1],
                layer_interfaces[1:],
                example_points,
                cell_ids,
                cell_names,
                # strict=True # TODO: uncomment in Python 3.10
            )
        ):
            cell_stack.append(
                BlanketCell(
                    ext_surf,
                    ccw_surf,
                    cw_surf,
                    eg_point,
                    int_surf,
                    _id,
                    _name,
                    fill_dict[cell_type_at_num[_cell_num]],
                )
            )
        return cls(cell_stack)


class BlanketCellArray:
    """
    An array of BlanketCellStack

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

    def __setitem__(
        self,
        index_or_slice,
        new_blanket_cell_stack: Union[List[BlanketCellStack], BlanketCellStack],
    ):
        raise NotImplementedError(
            "The content of this class is not intended to be " "changed on-the-fly!"
        )

    def __iter__(self):
        return self.blanket_cell_array.__iter__()

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
        :class:`~bluemira.neutronics.make_pre_cell.PreCellArray` .
        """
        cell_walls = CellWalls.from_pre_cell_array(pre_cell_array)
        # left wall
        ccw_surf = surface_from_2points(
            *cell_walls[0], surface_id=0, name="Blanket cell wall 0"
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
            example_points, cell_ids = [], []
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
                # switching logic to choose thickness vs fraction. TODO: functionalize
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
                eg_pt = (
                    points[0] + points[1] + cw_wall_cuts[-1] + ccw_wall_cuts[-1]
                ) / 4
                example_points.append(eg_pt)
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
                    "Not ready to make curved-line cross-sections " "yet!"
                )

            stack = BlanketCellStack.from_surfaces(
                ccw_surf=ccw_surf,
                cw_surf=cw_surf,
                layer_interfaces=surf_stack,
                example_points=example_points,
                cell_ids=cell_ids,
                fill_dict=material_dict,
            )
            cell_array.append(stack)
            ccw_surf = cw_surf  # right wall -> left wall, as we shift right.

        return cls(cell_array)
