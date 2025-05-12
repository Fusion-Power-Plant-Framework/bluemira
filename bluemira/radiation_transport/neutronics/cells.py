# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Cut bluemira wires into pre-cells."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from itertools import chain, pairwise
from types import MappingProxyType
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon as mpl_Polygon
from numpy import typing as npt

from bluemira.display import plot_2d, show_cad
from bluemira.geometry.constants import EPS_FREECAD
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.tools import (
    boolean_cut,
    extrude_shape,
    make_polygon,
    polygon_revolve_signed_volume,
    revolve_shape,
)
from bluemira.geometry.wire import BluemiraWire
from bluemira.radiation_transport.neutronics.error import (
    CSGGeometryError,
    CSGGeometryValidationError,
    check_if_read_only_attr_has_been_set,
)
from bluemira.radiation_transport.neutronics.parent_child import ParentLinkable
from bluemira.radiation_transport.neutronics.validation import (
    check_cell_wall_alignment,
    check_cells_are_neighbours_by_face,
    check_stacks_are_neighbours,
    check_vertices_ordering,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes


def cyclic_sequence(iterable: Iterable) -> tuple:
    """Turn the iterable into a list of len==len(iterable)+1, where the final element is
    the first element of the iterable.

    Returns
    -------
    :
        (0,1,2,...,100,0)
    """
    t = tuple(iterable)
    return (*t, t[0])


class CSGReactor(Sequence):
    """A collection of pre-cell stacks, arranged in the clockwise (cw) direction."""

    def __init__(
        self,
        cell_stacks: Sequence[CellStack],
        *,
        validate: bool = True,
        link_children: bool = True,
    ):
        """
        Parameters
        ----------
        cell_stack:
            the collection of cell stacks.
        validate:
            validate the geometry (i.e. check if neighbouring cell stacks are indeed
            neighbouring, oriented correctly).
        link_children:
            Create the cell walls between neighbouring cell stacks, such that they can be
            reference by the relevant stacks. Also set itself as the parent of each
            stack.
        """
        self.cell_stacks = cell_stacks
        if validate:
            self.validate_stacks_contiguity(self.cell_stacks)
        self._contiguity_validated = validate
        if link_children:
            self.set_cell_stacks_properties(self.cell_stacks, self)
        # these are extra regions trackers, won't be directly used.
        self.subtracted_regions = []
        self.added_regions = []

    def add_region(self, region):
        for stack in self.cell_stacks:
            stack.add_region(region)
        self.added_regions.append(region)

    def subtract_region(self, region):
        for stack in self.cell_stacks:
            stack.subtract_region(region)
        self.subtracted_regions.append(region)

    def get_edge_points(self) -> npt.NDArray:
        """Get a list of sample points on the edge of the volume of the cell,
        such that if these points are included in the CSG volume, the rest of the cell is
        guaranteed to also be included in the CSG volume.
        """
        return [*self.in_face.samples, *self.ex_face.samples]

    @property
    def csg_regions(self):
        return [
            *chain(*[stack.csg_regions[0] for stack in self.cell_stacks]),
            *self.added_regions,
        ], self.subtracted_regions

    def __getitem__(self, index) -> CellStack | CSGReactor:
        if isinstance(index, slice):
            return CSGReactor(
                self.cell_stacks[index], validate=False, link_children=False
            )
        return self.cell_stacks[index]

    def __len__(self) -> int:
        return len(self.cell_stacks)

    @staticmethod
    def validate_stacks_contiguity(cell_stacks: Sequence[CellStack]) -> None:
        """
        Raises
        ------
        CSGGeometryValidationError
            Thrown if stacks are not ordered correctly (cw) and neighbouring.
        """
        for stack_ccw, stack_cw in pairwise(cyclic_sequence(cell_stacks)):
            check_stacks_are_neighbours(stack_ccw, stack_cw)

    @staticmethod
    def set_cell_stacks_properties(
        cell_stacks: Sequence[CellStack], parent: CSGReactor | None
    ) -> None:
        """
        Raises
        ------
        ReadOnlyAttributeError
            Thrown when a cell stack's parent has already been set, but 'parent' is still
            parsed here.
        """
        for stack_ccw, stack_cw in pairwise(cyclic_sequence(cell_stacks)):
            if parent:
                stack_ccw.parent = parent
            stack_ccw.cw_wall = stack_cw.ccw_wall = CellWall([
                stack_ccw.cw_in,
                stack_ccw.cw_ex,
            ])

    @property
    def in_faces(self) -> list[RadialInterface]:
        """Interior surfaces"""
        return [stack.in_face for stack in self.cell_stacks]

    @property
    def ex_faces(self) -> list[RadialInterface]:
        """Exterior surfaces"""
        return [stack.ex_face for stack in self.cell_stacks]

    @property
    def volumes(self) -> list[list[np.float64]]:
        return [stack.volumes for stack in self.cell_stacks]

    @property
    def total_volume(self):
        return sum(sum(vlist) for vlist in self.volumes)

    @property
    def outlines(self) -> list[list[BluemiraWire]]:
        return [stack.outline for stack in self.cell_stacks]

    @property
    def solids(self) -> list[list[BluemiraSolid]]:
        return [stack.solids for stack in self.cell_stacks]

    def plot_2d(self, *args, **kwargs) -> Axes:
        return plot_2d(chain(*self.outlines), *args, **kwargs)

    def show_cad(self, *args, show_half_only: bool = False, **kwargs) -> Axes:
        solids = chain(*self.solids)
        if show_half_only:
            bounding_boxes = [sol.bounding_box for sol in solids]
            limits = np.array([
                (bb.x_min, bb.x_max, bb.y_min, bb.z_min, bb.z_max)
                for bb in bounding_boxes
            ])
            x_min = limits[:, 0].min() - EPS_FREECAD * 10
            x_max = limits[:, 1].max() + EPS_FREECAD * 10
            y_min = limits[:, 2].min() - EPS_FREECAD * 10
            z_min = limits[:, 3].min() - EPS_FREECAD * 10
            z_max = limits[:, 4].max() + EPS_FREECAD * 10
            boundary = make_polygon(
                [
                    [x_min, 0, z_min],
                    [x_min, 0, z_max],
                    [x_max, 0, z_max],
                    [x_max, 0, z_min],
                ],
                closed=True,
            )
            half_bb = extrude_shape(BluemiraFace(boundary), [0, y_min, 0])
            solids = [boolean_cut(sol, half_bb) for sol in solids]
        return show_cad(solids, *args, **kwargs)

    def __repr__(self) -> str:
        return super().__repr__().replace(" at ", f"of {len(self)} stacks at ")


class CellStack(ParentLinkable, Sequence):
    """A stack of cells (all sharing the same CW and CCW cell walls),
    arranged from plasma facing side to exterior side.
    """

    _allowed_parent_class = CSGReactor

    def __init__(
        self, cells: Sequence[Cell], *, validate: bool = True, link_children: bool = True
    ):
        """
        Parameters
        ----------
        validate:
            validate the geometry (i.e. check if neighbouring cell stacks are indeed
            neighbouring, oriented correctly).
        link_children:
            Set itself as the parent of each stack.

        Raises
        ------
        CSGGeometryValidationError
            If the ccw and cw cell walls for this cell stack aren't set (by setting the
            parent), then we cannot validate the geometry as requested.
        """
        self.cells = cells
        self._parent = None
        self._ccw_wall = None
        self._cw_wall = None

        if link_children:
            self.set_cells_properties(self.cells, self)
        if validate:
            self.validate_cells_contiguity(self.cells)
            if (not self.cw_wall) or (not self.ccw_wall):
                raise CSGGeometryValidationError(
                    f"No cell wall is set for {self}. "
                    "Cannot validate cell wall alignment!"
                )
            self.validate_cells_alignment(self.ccw_wall)

        self.vertices = Vertices(
            cells[0].vertices.ccw_in,
            cells[0].vertices.cw_in,
            cells[0].vertices.ccw_ex,
            cells[0].vertices.cw_ex,
        )
        # these are extra regions trackers, won't be directly used.
        self.subtracted_regions = []
        self.added_regions = []

    def get_edge_points(self):
        """Get a list of sample points on the edge of the volume of the cell stack, such
        that if these points are included in the CSG volume, the rest of the cell stack
        is guaranteed to also be included in the CSG volume.
        """
        return [*self.in_face.samples, *self.ex_face.samples]

    def add_region(self, region):
        for cell in self.cells:
            cell.add_region(region)
        self.added_regions.append(region)

    def subtract_region(self, region):
        for cell in self.cells:
            cell.subtract_region(region)
        self.subtracted_regions.append(region)

    @property
    def csg_regions(self):
        return [
            *chain(*[cell.csg_region[0] for cell in self.cells]),
            *self.added_regions,
        ], self.subtracted_regions

    @property
    def ccw_wall(self):
        return self._ccw_wall

    @ccw_wall.setter
    def ccw_wall(self, _ccw_wall: CellWall):
        """ccw_wall should only be set by the parent.

        Raises
        ------
        ReadOnlyAttributeError
            If the ccw_wall has already been written once, it cannot be written again.
        """
        # check_if_read_only_attr_has_been_set(self._ccw_wall, self)
        self._ccw_wall = _ccw_wall

    @property
    def cw_wall(self):
        return self._cw_wall

    @cw_wall.setter
    def cw_wall(self, _cw_wall: CellWall):
        """cw_wall should only be set by the parent.

        Raises
        ------
        ReadOnlyAttributeError
            If the cw_wall has already been written once, it cannot be written again.
        """
        # check_if_read_only_attr_has_been_set(self._cw_wall, self)
        self._cw_wall = _cw_wall

    @property
    def in_face(self) -> RadialInterface:
        return self.cells[0].in_face

    @property
    def ex_face(self) -> RadialInterface:
        return self.cells[-1].ex_face

    def __getitem__(self, index) -> Cell | CellStack:
        if isinstance(index, slice):
            return CellStack(self.cells[index], validate=False, link_children=False)
        return self.cells[index]

    def __len__(self) -> int:
        return len(self.cells)

    def get_interface(self, index: int) -> RadialInterface:
        """Get the n-th interface by index n.

        Parameters
        ----------
        index:
            The n-th interface that the user wants.

        Returns
        -------
        :
            The interface that separates the n-th and n+1-th cell (if n is positive); or
            the interface that separates the n-th and (n-1)-th cell (if n is negative).

        Raises
        ------
        IndexError
            Can only accept integer between the range [-l+1, l-2], where
            l = len(self)
        """
        l = len(self.cells)  # noqa: E741
        if index >= 0:
            if index > l - 2:
                raise IndexError(f"Can only go up to the {l - 2}-th interface.")
            return self.cell[index].ex_face
        if index < 1 - l:
            raise IndexError(f"Can only go down to the {1 - l}-th interface.")
        return self.cells[index].in_face

    @property
    def volumes(self) -> list[np.float64]:
        return [cell.volume for cell in self.cells]

    @property
    def total_volume(self):
        return sum(self.volumes)

    @property
    def outlines(self) -> list[BluemiraWire]:
        return [cell.outline for cell in self.cells]

    @property
    def solids(self) -> list[BluemiraSolid]:
        return [cell.solid for cell in self.cells]

    def plot_2d(self, *args, **kwargs) -> Axes:
        return plot_2d(self.outlines, *args, **kwargs)

    def show_cad(self, *args, show_half_only: bool = False, **kwargs) -> Axes:
        solids = self.solids
        if show_half_only:
            bounding_boxes = [sol.bounding_box for sol in solids]
            limits = np.array([
                (bb.x_min, bb.x_max, bb.y_min, bb.z_min, bb.z_max)
                for bb in bounding_boxes
            ])
            x_min = limits[:, 0].min() - EPS_FREECAD * 10
            x_max = limits[:, 1].max() + EPS_FREECAD * 10
            y_min = limits[:, 2].min() - EPS_FREECAD * 10
            z_min = limits[:, 3].min() - EPS_FREECAD * 10
            z_max = limits[:, 4].max() + EPS_FREECAD * 10
            boundary = make_polygon(
                [
                    [x_min, 0, z_min],
                    [x_min, 0, z_max],
                    [x_max, 0, z_max],
                    [x_max, 0, z_min],
                ],
                closed=True,
            )
            half_bb = extrude_shape(BluemiraFace(boundary), [0, y_min, 0])
            solids = [boolean_cut(sol, half_bb) for sol in solids]
        return show_cad(solids, *args, **kwargs)

    @staticmethod
    def set_cells_properties(cells: Sequence[Cell], parent: CellStack) -> None:
        """
        Raises
        ------
        ReadOnlyAttributeError
            Thrown when a cell's parent has already been set.
        """
        for cell in cells:
            cell.parent = parent

    @staticmethod
    def validate_cells_contiguity(cells: Sequence[Cell]) -> None:
        """
        Raises
        ------
        CSGGeometryValidationError
            Thrown if the cells aren't sequentially ordered (inward to outward) and
            neighbouring.
        """
        for in_cell, ex_cell in pairwise(cells):
            check_cells_are_neighbours_by_face(in_cell, ex_cell)

    @staticmethod
    def validate_cells_alignment(
        cells: Sequence[Cell], cell_wall_ccw: CellWall, cell_wall_cw: CellWall
    ) -> None:
        """Validate the alignment of cell with the cell wall

        Raises
        ------
        CSGGeometryValidationError
            Thrown if cells' corners do not start and end in the ccw and cw cell walls.
        """
        for cell in cells:
            check_cell_wall_alignment(cell, cell_wall_ccw, cell_wall_cw)

    def __repr__(self) -> str:
        if self._parent:
            return (
                super()
                .__repr__()
                .replace(
                    " at ",
                    f" of {len(self)} cells belonging "
                    f"to the {self.parent.index(self)}-th stack in {self.parent} at ",
                )
            )
        return super().__repr__().replace(" at ", f" of {len(self)} cells orphaned at ")


class Cell(ParentLinkable):
    """A  cell with four defined sides:
    Each side is made of at least one surface.
    """

    _allowed_parent_class = CellStack

    def __init__(self, in_face: RadialInterface, ex_face: RadialInterface):
        """
        Define cell using only its interior and exterior interface.

        Parameters
        ----------
        in_face:
            interior (pointed towards the plasma) side of the cell.
        ex_face:
            exterior (pointed towards the outside of the tokamak) side of the cell.

        Attributes
        ----------
        vertices:
            A collection of vertices defining the four corners of the cell.

        Raises
        ------
        CSGGeometryValidationError
            If the vertices are incorrectly ordered, it suggests that we may have an
            incorrectly ordered wire. Thus an error is raised.
        """
        self.in_face = in_face
        self.ex_face = ex_face
        # all RadialInterface should have wires that run clockwise
        vertices = Vertices.from_bluemira_coordinates(
            in_face.wire.start_point(),
            in_face.wire.end_point(),
            ex_face.wire.start_point(),
            ex_face.wire.end_point(),
        )
        check_vertices_ordering(vertices)
        self.vertices = vertices
        self._parent = None
        self.subtracted_regions = []  # each is an intersection list
        self.added_regions = []  # each is an intersection list

    def add_region(self, region):
        self.added_regions.append(region)

    def subtract_region(self, region):
        self.subtracted_regions.append(region)

    def get_edge_points(self) -> npt.NDArray:
        """Get a list of sample points on the edge of the volume of the cell,
        such that if these points are included in the CSG volume, the rest of the cell is
        guaranteed to also be included in the CSG volume.
        """
        return [*self.in_face.samples, *self.ex_face.samples]

    @property
    def csg_region(self):
        return (
            [  # union list
                [  # intersection list
                    self.choose_region(self.ccw_wall.csg_surf),
                    self.choose_region(self.in_face.csg_surf),
                    self.choose_region(self.cw_wall.csg_surf),
                    self.choose_region(self.ex_face.csg_surf),
                ],
                *self.added_regions,
            ],
            self.subtracted_regions,
        )

    def __repr__(self) -> str:
        centroid = self.vertices.centroid
        x, z = centroid[0], centroid[-1]
        if self._parent:
            return (
                super()
                .__repr__()
                .replace(
                    " at ",
                    f" located at ({x=},{z=}) belonging to the "
                    f"{self.parent.index(self)}-th cell in {self.parent} at ",
                )
            )
        return (
            super().__repr__().replace(" at ", f" located at ({x=},{z=}) orphaned at ")
        )

    @property
    def volume(self) -> np.float64:
        """Either calculate, or obtain from freecad, the volume of the cell, in [m^3]."""
        if all([
            # made of straight lines, and is axisymmetric:
            isinstance(self.in_face, LineSegments),
            isinstance(self.ex_face, LineSegments),
            not self.added_regions,
            not self.subtracted_regions,
        ]):
            return polygon_revolve_signed_volume(  # faster, more accurate implementation
                np.concatenate([self.in_face.xz.T[::-1], self.ex_face.xz.T])  # clockwise
            )
        return abs(self.solid.volume)

    @property
    def solid(self) -> BluemiraSolid:
        if not hasattr(self, "_solid"):
            self._solid = BluemiraSolid(revolve_shape(self.outline, 360))
        return self._solid

    @property
    def outline(self) -> BluemiraWire:
        """
        Raises
        ------
        CSGGeometryValidationError
            If, for any reason, the outline drawn isn't closed as the final wire is not
            joined onto the first wire, then this error is raised. This would typically
            be due to developer error.
        """
        if not hasattr(self, "_outline"):
            self._ccw_wire = make_polygon([
                Vertices.convert_to_coordinates(self.vertices.ccw_ex),
                Vertices.convert_to_coordinates(self.vertices.ccw_in),
            ])
            self._cw_wire = make_polygon([
                Vertices.convert_to_coordinates(self.vertices.cw_in),
                Vertices.convert_to_coordinates(self.vertices.cw_ex),
            ])
            self._outline = BluemiraWire([
                self.in_face.wire,
                self._cw_wire,
                self.ex_face.wire,
                self._ccw_wire,
            ])
            if not self._outline.is_closed():
                raise CSGGeometryValidationError("Wire not closed!")
        return self._outline

    def plot_2d(self, *args, **kwargs) -> Axes:
        """Plot the poloidal cross-section on a 2d plot"""
        return plot_2d(self.outline, *args, **kwargs)

    def show_cad(self, *args, show_half_only=True, **kwargs) -> Axes:
        """Plot 3D plots of the poloidal"""
        solid = self.solid
        if show_half_only:
            bb = self.solid.bounding_box
            r = abs(bb.x_max) + EPS_FREECAD * 10
            x_min = -r
            x_max = r
            z_min = bb.z_min - EPS_FREECAD * 10
            z_max = bb.z_max + EPS_FREECAD * 10
            boundary = make_polygon(
                [
                    [x_min, 0, z_min],
                    [x_min, 0, z_max],
                    [x_max, 0, z_max],
                    [x_max, 0, z_min],
                ],
                closed=True,
            )
            half_bb = extrude_shape(BluemiraFace(boundary), [0, -r, 0])
            solid = boolean_cut(solid, half_bb)
        return show_cad(solid, *args, **kwargs)

    def plot_and_fill(self, color, ax=None) -> Axes:
        """
        Plot the edges of the polygon, and then fill it with the specified color.
        """
        if not ax:
            ax = plt.axes()
        plot_2d(show=False, ax=ax)
        in_p = self.in_face.samples
        ex_p = self.ex_face.samples
        approx_outline = np.concatenate([in_p.xz.T[::-1], ex_p.xz.T])  # clockwise
        ax.add_patch(mpl_Polygon(approx_outline), color=color)
        return ax


class CSGSurfaces:
    """An object that can be translated into a unique surface/collection of surfaces
    in CSG representation.
    """

    def __init__(self):
        """Create an empty slot ._csg_surf to be filled in later."""
        self._csg_surf = None

    @property
    def csg_surf(self):  # type not defined yet since we don't want to import openmc here
        return self._csg_surf

    @csg_surf.setter
    def csg_surf(self, csg_surfaces):
        """
        Raises
        ------
        ReadOnlyAttributeError
            if the csg_surfaces has already been set once, it cannot be set again.
        """
        check_if_read_only_attr_has_been_set(self._csg_surf, self)
        self._csg_surf = csg_surfaces

    @property
    def wire(self):
        if not hasattr(self, "_wire"):
            self._make_wire()
        return self._wire

    def _make_wire(self):
        raise NotImplementedError(f"{self.__class__} does not have a _make_wire method.")


class RadialInterface(CSGSurfaces):
    """A wire with an associated csg surface representation."""

    def __init__(self, wire: BluemiraWire):
        """Wrap a bluemira wire along with its csg surface representation."""
        super().__init__()
        self._wire = wire

    def __eq__(self, other: RadialInterface):
        return self.wire.is_same(other.wire) and self.csg_surf == other.csg_surf

    def __hash__(self):
        return hash((self.wire, self.csg_surf))

    def make_samples(self, resolution: int = 5) -> None:
        """
        Choose how many points to sample.

        Parameters
        ----------
        resolution:
            The higher this integer is, the more sample points are chosen.
            self.wire is made of n wires. the number of points sampled = num_wires*n

        Notes
        -----
        This uses the BluemiraWire.discretise(byedges=True) option, which means that if
        one of the underlying wires is a splne, the 'resolution' parameter would have no
        effect on that wire.
        """
        self._samples = self.wire.discretise(
            resolution * len(self.wire.edges), byedges=True
        )

    @property
    def samples(self) -> Coordinates:
        """A numpy array of sample points, each point is a 3D numpy array (xyz)."""
        if not hasattr(self, "_samples"):
            self.make_samples()
        return self._samples


class LineSegments(CSGSurfaces):
    """An abstract class for n-1 several straight line segments joint together."""

    def __init__(self, points: Iterable[npt.NDArray[np.float64]]):
        """Store the points at the start and ends of the n-1 straight lines.

        Parameters
        ----------
        points:
            len(points)==n

        Raises
        ------
        CSGGeometryValidationError
            If only one point or no point is provided, then no line can be drawn.
        """
        if len(points) < 2:
            raise CSGGeometryValidationError(
                f"Must use at least 2 points to form {self.__class__}"
            )
        self.points = points

    def _make_wire(self):
        """Create a wire if it does not already exists"""
        self._wire = make_polygon(
            [Vertices.convert_to_coordinates(p) for p in self.points],
            closed=False,
        )

    def contains_all_points(
        self, *points: Iterable[npt.NDArray[np.float64]], atol=EPS_FREECAD
    ) -> bool:
        """Check if a list of points all lie on this line."""
        return all(self.contains(p) for p in points)

    def contains(self, point: npt.NDArray[np.float64], atol=EPS_FREECAD) -> bool:
        """Check if a single point lies on this line."""
        return self.wire.contains(point)


class StraightLine(LineSegments):
    """Straight line representable in CSG surface form."""

    def __init__(
        self,
        points: npt.NDArray[np.float64],
        *,
        snapped_to_horizontal: bool = False,
        snapped_to_vertical: bool = False,
    ):
        """A line, anchored at two points, and made into an associated infinite CSG
        plane/cone/cylinder.

        Raises
        ------
        CSGGeometryError
            Raised if the user is trying to simultaneously snap to vertical and horizonal
        """
        super().__init__(points)
        if snapped_to_horizontal and snapped_to_vertical:
            raise CSGGeometryError(
                f"{self} can only be snapped to be either horizontal or vertical, not"
                "both at the same time!"
            )
        self.snapped_to_horizontal = snapped_to_horizontal  # makes for a ZPlane
        self.snapped_to_vertical = snapped_to_vertical  # makes for a ZCylinder

    def _make_wire(self):
        """Create a wire if it does not already exists"""
        if self.snapped_to_horizontal:
            point_1 = self.points[0]
            point_2 = np.array([self.points[1][0], self.points[0][1]])
        elif self.snapped_to_vertical:
            point_1 = self.points[0]
            point_2 = np.array([self.points[0][0], self.points[1][1]])
        else:
            point_1, point_2 = self.points
        self._wire = make_polygon([point_1, point_2], closed=False)


class CellWall(StraightLine):
    """Interface between neighbouring cell stacks."""


class Vertices:
    """A collection of vertices denoting the corners of a cell/cell stack.
    Only record the x- and z-coordinates.
    """

    index_mapping = MappingProxyType({0: "ccw_in", 1: "cw_in", 2: "ccw_ex", 3: "cw_ex"})

    def __init__(
        self,
        ccw_in: npt.NDArray[np.float64],
        cw_in: npt.NDArray[np.float64],
        ccw_ex: npt.NDArray[np.float64],
        cw_ex: npt.NDArray[np.float64],
    ):
        """
        Taking only the x- and z-coordinates.

        Parameters
        ----------
        ccw_in:
            The vertex on the interior ccw side of the cell/cell stack, given as a numpy
            array of shape (2,).
        cw_in:
            The vertex on the interior cw side of the cell/cell stack, given as a numpy
            array of shape (2,).
        ccw_ex:
            The vertex on the exterior ccw side of the cell/cell stack, given as a numpy
            array of shape (2,).
        cw_ex:
            The vertex on the exterior cw side of the cell/cell stack, given as a numpy
            array of shape (2,).
        """
        self.ccw_in = ccw_in
        self.cw_in = cw_in
        self.ccw_ex = ccw_ex
        self.cw_ex = cw_ex

    @property
    def centroid(self) -> npt.NDArray[np.float64]:
        """Give the centroid of the four vertices"""
        return np.array(list(self)).mean(axis=0)

    @property
    def as_3d_array(self) -> npt.NDArray:
        return np.insert(list(self), 1, 0, axis=1)

    @staticmethod
    def convert_to_coordinates(coord_2d: npt.NDArray[np.float64]) -> Coordinates:
        """Convert a single point represented by a 2D numpy array into a 3D
        :class:`bluemira.geometry.coordinates.Coordinates` (on the x-z plane).
        """
        return Coordinates([coord_2d[0], 0, coord_2d[-1]])

    @staticmethod
    def convert_from_coordinates(coord_3d: Coordinates) -> npt.NDArray[np.float64]:
        """Convert a single point represented by a 3D
        :class:`bluemira.geometry.coordinates.Coordinates` (on the x-z plane) into a
        2D numpy array.
        """
        return np.squeeze(coord_3d.xz)

    @classmethod
    def from_coordinates(
        cls,
        ccw_in: Coordinates,
        cw_in: Coordinates,
        ccw_ex: Coordinates,
        cw_ex: Coordinates,
    ) -> Vertices:
        return cls(
            cls.convert_from_coordinates(ccw_in),
            cls.convert_from_coordinates(cw_in),
            cls.convert_from_coordinates(ccw_ex),
            cls.convert_from_coordinates(cw_ex),
        )

    def __getitem__(self, index: int) -> npt.NDArray[np.float64]:
        """Get vertices using integer indices, rather than their names names.

        Raises
        ------
        TypeError
            When slices or keys or other non-integer indices are passed in, this error
            is thrown.
        """
        if not isinstance(index, int):
            raise TypeError("Vertices only support integer indices.")
        return getattr(self, self.index_mapping[index])

    def __eq__(self, other: Vertices) -> bool:
        """Two sets of vertices are equal if they land on the same coordinates.

        Raises
        ------
        TypeError
            Can only compare against other Vertices object.
        """
        if not isinstance(other, Vertices):
            raise TypeError(f"Cannot compare {type(self)} with {type(other)}.")
        return all(
            (my_v == their_v).all() for my_v, their_v in zip(self, other, strict=True)
        )

    def __iter__(self):
        return iter(getattr(self, attr) for attr in self.index_mapping.values())

    def __hash__(self):
        return hash(tuple(tuple(v) for v in self))
