# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Create csg geometry by converting from bluemira geometry objects made of wires. All units
in this module are in SI (distrance:[m]) unless otherwise specified by the docstring.

Despite having a similar structure to
:mod:bluemira.radiation_transport.neutronics.make_pre_cell, we do not merge these two
modules together to ensure modularity, i.e. we can replace this file with a different csg
neutronics plugin than openmc later if the need arises.
"""

from __future__ import annotations

from itertools import chain, pairwise
from typing import TYPE_CHECKING

import numpy as np
import openmc
import openmc.region

from bluemira.codes.openmc.material import CellType
from bluemira.geometry.error import GeometryError
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.tools import (
    is_convex,
    make_polygon,
    polygon_revolve_signed_volume,
    revolve_shape,
)
from bluemira.geometry.wire import BluemiraWire
from bluemira.radiation_transport.neutronics.constants import (
    DTOL_CM,
    to_cm3,
)
from bluemira.radiation_transport.neutronics.radial_wall import Vert

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy.typing as npt

    from bluemira.codes.openmc.csg_tools import OpenMCEnvironment
    from bluemira.codes.openmc.material import MaterialsLibrary
    from bluemira.geometry.coordinates import Coordinates
    from bluemira.radiation_transport.neutronics.geometry import (
        DivertorThickness,
        TokamakDimensions,
    )
    from bluemira.radiation_transport.neutronics.make_pre_cell import (
        DivertorPreCell,
        DivertorPreCellArray,
        PreCell,
        PreCellArray,
    )
    from bluemira.radiation_transport.neutronics.wires import (
        WireInfoList,
    )


def is_monotonically_increasing(series):
    """Check if a series is monotonically increasing"""  # or decreasing
    return all(np.diff(series) >= 0)  # or all(diff<0)


def get_depth_values(
    pre_cell: PreCell, blanket_dimensions: TokamakDimensions
) -> npt.NDArray[np.float64]:
    """
    Choose the depth values that this pre-cell is suppose to use, according to where it
    is physically positioned (hence is classified as inboard or outboard).

    Parameters
    ----------
    pre_cell:
        :class:`~PreCell` to be classified as either inboard or outboard
    blanket_dimensions:
        :class:`bluemira.radiation_transport.neutronics.params.TokamakDimensions`
        recording the dimensions of the blanket in SI units (unit: [m]).

    Returns
    -------
    depth_series:
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
    # reference radius
    return (
        pre_cell.vertex[0].mean() < blanket_dimensions.inboard_outboard_transition_radius
    )


class BlanketCell(openmc.Cell):
    """
    A generic blanket cell that forms the base class for the five specialised types of
    blanket cells.

    It's a special case of openmc.Cell, in that it has 3 to 4 surfaces
    (mandatory surfaces: exterior_surface, ccw_surface, cw_surface;
    optional surface: interior_surface), and it is more wieldy because we don't have to
    specify the relevant half-space for each surface; instead the corners of the cell
    is provided by the user, such that the appropriate regions are chosen.
    """

    def __init__(
        self,
        exterior_surface: openmc.Surface,
        ccw_surface: openmc.Surface,
        cw_surface: openmc.Surface,
        interior_surface: openmc.Surface | None,
        vertices: Coordinates,
        csg: OpenMCEnvironment,
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

        Raises
        ------
        GeometryError
            Ordering of wires results in negative volume
        """
        self.exterior_surface = exterior_surface
        self.ccw_surface = ccw_surface
        self.cw_surface = cw_surface
        self.interior_surface = interior_surface
        self.vertex = vertices

        super().__init__(
            cell_id=cell_id,
            name=name,
            fill=fill,
            region=csg.region_from_surface_series(
                [exterior_surface, ccw_surface, cw_surface, interior_surface],
                self.vertex.T,  # We just assume it is convex
                control_id=bool(cell_id),
            ),
        )

        self.volume = to_cm3(polygon_revolve_signed_volume(vertices[::2].T))
        if self.volume <= 0:
            raise GeometryError("Wrong ordering of vertices!")


class BlanketCellStack:
    """
    A stack of openmc.Cells, first cell is closest to the interior and last cell is
    closest to the exterior. They should all be situated at the same poloidal angle.
    """

    def __init__(self, cell_stack: list[BlanketCell], csg: OpenMCEnvironment):
        """
        The shared surface between adjacent cells must be the SAME one, i.e. same id and
        hash, not just identical.
        They must share the SAME counter-clockwise surface and the SAME clockwise surface
        (left and right side surfaces of the stack, for the stack pointing straight up).

        Because the bounding_box function is INCORRECT, we can't perform a check on the
        bounding box to confirm that the stack is linearly increasing/decreasing in xyz
        bounds.

        Series of cells:

            SurfaceCell
            FirstWallCell
            BreedingZoneCell
            ManifoldCell
            VacuumVesselCell

        Raises
        ------
        ValueError
            Contiguous stack of cells expected
        """
        self.cell_stack = cell_stack
        self._csg = csg
        for int_cell, ext_cell in pairwise(cell_stack):
            if int_cell.exterior_surface is not ext_cell.interior_surface:
                raise ValueError("Expected a contiguous stack of cells!")

    def __len__(self) -> int:
        """Number of cells in stack"""
        return len(self.cell_stack)

    def __getitem__(self, index_or_slice) -> list[BlanketCell] | BlanketCell:
        """Get cell from stack"""
        return self.cell_stack[index_or_slice]

    def __iter__(self) -> Iterator[BlanketCell]:
        """Iterator for BlanketCellStack"""
        return iter(self.cell_stack)

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
        cut_point_series:
            array of shape (M+1, 2) where M = number of cells in the blanket cell stack
            (i.e. number of layers in the blanket). Each point has two dimensions
        direction_vector:
            direction that these points are all supposed to go towards.

        Raises
        ------
        GeometryError
            Crossing surfaces
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

    def get_overall_region(self, *, control_id: bool = False) -> openmc.Region:
        """
        Calculate the region covering the entire cell stack.

        Parameters
        ----------
        control_id
            Passed as argument onto
            :meth:`~bluemira.radiation_transport.neutronics.csg_env.OpenMCEnvironment.region_from_surface_series`

        Raises
        ------
        GeometryError
            Vertices must be convex
        """
        vertices = np.vstack((
            self.cell_stack[0].vertex.T[(1, 2),],
            self.cell_stack[-1].vertex.T[(3, 0),],
        ))
        if not is_convex(vertices):
            raise GeometryError(f"{self}'s vertices need to be convex!")
        return self._csg.region_from_surface_series(
            [
                self.exterior_surface,
                self.ccw_surface,
                self.cw_surface,
                self.interior_surface,
            ],
            vertices,
            control_id=control_id,
        )

    @classmethod
    def from_pre_cell(
        cls,
        pre_cell: PreCell,
        ccw_surface: openmc.Surface,
        cw_surface: openmc.Surface,
        depth_series: npt.NDArray,
        csg: OpenMCEnvironment,
        fill_lib: MaterialsLibrary,
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
            If None: we will not be controlling the cell and surfaces id.

        Raises
        ------
        TypeError
            Incorrect number of edges on external wire
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
        i = "(unspecified)" if blanket_stack_num is None else blanket_stack_num
        cls.check_cut_point_ordering(
            wall_cut_pts[:, 0],
            dirs[0],
            location_msg=f"\nOccuring in cell stack {i}'s CCW wall:"
            "\nCheck if the thickness specified can fit into the blanket?",
        )
        cls.check_cut_point_ordering(
            wall_cut_pts[:, 1],
            dirs[1],
            location_msg=f"\nOccuring in cell stack {i}'s CW wall:"
            "\nCheck if the thickness specified can fit into the blanket?",
        )
        # 2. Accumulate the corners of each cell.
        vertices = [
            np.array([
                [out[1, 0], inn[1, 0], inn[0, 0], out[0, 0]],
                np.full(4, 0),
                [out[1, 1], inn[1, 1], inn[0, 1], out[0, 1]],
            ])
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
        cell_types = np.array(
            (
                CellType.BlanketSurface,
                CellType.BlanketFirstWall,
                CellType.BlanketBreedingZone,
                CellType.BlanketManifold,
                CellType.VacuumVessel,
            ),
            dtype=object,
        )[layer_mask]
        for k, (points, cell_type) in enumerate(
            zip(wall_cut_pts[1:][layer_mask], cell_types, strict=False)
        ):
            j = k + 1  # = range(1, M+1)
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
                    fill=fill_lib.match_material(cell_type, inboard=inboard),
                )
            )
            int_surf = ext_surf
        int_surf.name = "vacuum-vessel-facing surface"

        return cls(cell_stack, csg)


class BlanketCellArray:
    """
    An array of BlanketCellStack. Interior and exterior curve both should be convex.

    Parameters
    ----------
    blanket_cell_array
        a list of BlanketCellStack

    """

    def __init__(
        self, blanket_cell_array: list[BlanketCellStack], csg: OpenMCEnvironment
    ):
        """
        Create array from a list of BlanketCellStack

        Raises
        ------
        GeometryError
            Neighbouring cell stack not aligned
        """
        self.blanket_cell_array = blanket_cell_array
        self.poloidal_surfaces = [self.blanket_cell_array[0].ccw_surface]
        self.radial_surfaces = []
        self._csg = csg
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
        if not is_convex(self.exterior_vertices()):
            raise GeometryError(
                "The exterior vertices of all of the cell-stack should"
                "form a convex curve!"
            )
        if not is_convex(self.interior_vertices()):
            raise GeometryError(
                "The interior vertices of all of the cell-stack should"
                "form a convex curve!"
            )

    def __len__(self) -> int:
        """Number of cell stacks"""
        return len(self.blanket_cell_array)

    def __getitem__(self, index_or_slice) -> list[BlanketCellStack] | BlanketCellStack:
        """Get cell stack"""
        return self.blanket_cell_array[index_or_slice]

    def __iter__(self) -> Iterator[BlanketCellStack]:
        """Iterator for BlanketCellArray"""
        return iter(self.blanket_cell_array)

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
        exterior_vertices:
            array of shape (N+1, 3) arranged clockwise (inboard to outboard).
        """
        return np.asarray([
            self.blanket_cell_array[0][-1].vertex.T[Vert.exterior_start],
            *(
                stack[-1].vertex.T[Vert.exterior_end]
                for stack in self.blanket_cell_array
            ),
        ])

    def interior_vertices(self) -> npt.NDArray:
        """
        Returns all of the tokamak's poloidal cross-section's inside corners'
        coordinates, in 3D.

        Parameters
        ----------
        interior_vertices:
            array of shape (N+1, 3) arranged clockwise (inboard to outboard).
        """
        return np.asarray([
            self.blanket_cell_array[0][0].vertex.T[Vert.interior_start],
            *(stack[0].vertex.T[Vert.interior_end] for stack in self.blanket_cell_array),
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
            :meth:`~bluemira.radiation_transport.neutronics.csg_env.OpenMCEnvironment.region_from_surface_series`.

        Raises
        ------
        GeometryError
            Vertices must be convex
        """
        exclusion_zone_by_stack = []
        for stack in self.blanket_cell_array:
            stack_vertices = np.vstack((
                stack[0].vertex.T[(1, 2),],
                stack[-1].vertex.T[(3, 0),],
            ))
            if not is_convex(stack_vertices):
                raise GeometryError(f"{self}'s vertices need to be convex!")

            exclusion_zone_by_stack.append(
                self._csg.region_from_surface_series(
                    [stack.cw_surface, stack.ccw_surface, stack.interior_surface],
                    stack_vertices,
                    control_id=control_id,
                )
            )
        return openmc.Union(exclusion_zone_by_stack)

    @classmethod
    def from_pre_cell_array(
        cls,
        pre_cell_array: PreCellArray,
        materials: MaterialsLibrary,
        blanket_dimensions: TokamakDimensions,
        csg: OpenMCEnvironment,
        *,
        control_id: bool = False,
    ) -> BlanketCellArray:
        """
        Create a BlanketCellArray from a
        :class:`~bluemira.radiation_transport.neutronics.make_pre_cell.PreCellArray`.
        This method assumes itself is the first method to be run to create cells in the
        :class:`~openmc.Universe.`

        Parameters
        ----------
        pre_cell_array
            PreCellArray
        materials
            :class:`~MaterialsLibrary` so that it separates into .inboard, .outboard,
            .divertor, .tf_coil_windings, etc.
        blanket_dimensions
            :class:`bluemira.radiation_transport.neutronics.params.TokamakDimensions`
            recording the dimensions of the blanket in SI units (unit: [m]).
        control_id
            Passed as argument onto
            :meth:`~bluemira.radiation_transport.neutronics.csg_env.OpenMCEnvironment.region_from_surface_series`.
        """
        cell_walls = pre_cell_array.cell_walls
        # TODO @je-cook: when contorl_id, we're forced to start at id=0
        # 3531

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
                fill_lib=materials,
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
        csg: OpenMCEnvironment,
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
        self._csg = csg

        region = self._csg.region_from_surface_series(
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

        Raises
        ------
        GeometryError
            Volume is negative
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
        self,
        *,
        away_from_plasma: bool = True,
        control_id: bool = False,
        additional_test_points: npt.NDArray | None = None,
    ) -> openmc.Region:
        """
        Get the exclusion zone of a semi-CONVEX cell.

        This can only be validly used:

            If away_from_plasma=True, then the interior side of the cell must be convex.
            If away_from_plasma=False, then the exterior_side of the cell must be convex.

        Usage:

            next_cell_region = flat_intersection(..., ~this_cell.exclusion_zone())

        Parameters
        ----------
        control_id
            Passed as argument onto
            :func:`~bluemira.radiation_transport.neutronics.csg_env.region_from_surface_series`

        Raises
        ------
        GeometryError
            Interior and exterior wire vertices must be convex
        """
        if away_from_plasma:
            vertices_array = self.interior_wire.get_3D_coordinates()
            if additional_test_points is not None:
                vertices_array = np.concatenate([additional_test_points, vertices_array])

            if not is_convex(vertices_array):
                raise GeometryError(
                    f"{self} (excluding the surface)'s vertices needs to be convex!"
                )
            return self._csg.region_from_surface_series(
                [self.cw_surface, self.ccw_surface, *self.interior_surfaces],
                vertices_array,
                control_id=control_id,
            )
        # exclusion zone facing towards the plasma
        vertices_array = self.exterior_wire.get_3D_coordinates()
        if additional_test_points is not None:
            vertices_array = np.concatenate([additional_test_points, vertices_array])

        if not is_convex(vertices_array):
            raise GeometryError(
                f"{self} (excluding the vacuum vessel)'s vertices needs to be convex!"
            )
        return self._csg.region_from_surface_series(
            [self.cw_surface, self.ccw_surface, *self.exterior_surfaces],
            vertices_array,
            control_id=control_id,
        )


class DivertorCellStack:
    """
    A CONVEX object! i.e. all its exterior points together should make a convex hull.
    A stack of DivertorCells (openmc.Cells), first cell is closest to the interior and
    last cell is closest to the exterior. They should all be situated on the same
    poloidal angle.
    """

    def __init__(self, divertor_cell_stack: list[DivertorCell], csg: OpenMCEnvironment):
        self.cell_stack = divertor_cell_stack
        self._csg = csg
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

    def __iter__(self) -> Iterator[DivertorCell]:
        """Iterator for DivertorCellStack"""
        return iter(self.cell_stack)

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
            :func:`~bluemira.radiation_transport.neutronics.csg_env.region_from_surface_series`

        Raises
        ------
        GeometryError
            All vertices myst be convex
        """
        all_vertices = self.get_all_vertices()
        if not is_convex(all_vertices):
            raise GeometryError(f"overall_region of {self} needs to be convex!")

        return self._csg.region_from_surface_series(
            [
                self.cw_surface,
                self.ccw_surface,
                *self.interior_surfaces,
                *self.exterior_surfaces,
            ],
            all_vertices,
            control_id=control_id,
        )

    @classmethod
    def from_divertor_pre_cell(
        cls,
        divertor_pre_cell: DivertorPreCell,
        cw_surface: openmc.Surface,
        ccw_surface: openmc.Surface,
        materials: MaterialsLibrary,
        csg: OpenMCEnvironment,
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
        # I apologise that this is still hard to read.
        # I'm trying to make cell_stack a 3-element list if armour_thickness>0,
        # but a 2-element list if armour_thickness==0.

        outermost_wire = divertor_pre_cell.exterior_wire
        outermost_surf = csg.surfaces_from_info_list(outermost_wire)
        outer_wire = divertor_pre_cell.vv_wire
        outer_surf = csg.surfaces_from_info_list(outer_wire)
        if armour_thickness > 0:
            inner_wire = divertor_pre_cell.offset_interior_wire(armour_thickness)
            inner_surf = csg.surfaces_from_info_list(inner_wire)
            innermost_wire = divertor_pre_cell.interior_wire
            innermost_surf = csg.surfaces_from_info_list(innermost_wire)
        else:
            inner_wire = divertor_pre_cell.interior_wire
            inner_surf = csg.surfaces_from_info_list(inner_wire)
        # make the middle cell
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
                fill=materials.match_material(CellType.DivertorBulk),
            ),
        ]
        # make the vacuum vessel cell
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
                subtractive_region=cell_stack[0].exclusion_zone(
                    away_from_plasma=False,
                    additional_test_points=innermost_wire.get_3D_coordinates()
                    if armour_thickness > 0
                    else inner_wire.get_3D_coordinates(),
                ),
                name="Vacuum Vessel behind the divertor in divertor cell stack"
                f"{stack_num}",
                fill=materials.match_material(CellType.DivertorFirstWall),
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
                        away_from_plasma=True,
                        additional_test_points=outermost_wire.get_3D_coordinates(),
                    ),
                    fill=materials.match_material(CellType.DivertorSurface),
                ),
            )
            # again, unfortunately, this does mean that the surface armour cell has the
            # largest ID.
        return cls(cell_stack, csg)


class DivertorCellArray:
    """Turn the divertor into a cell array"""

    def __init__(self, cell_array: list[DivertorCellStack]):
        """Create array from a list of DivertorCellStack.

        Raises
        ------
        GeometryError
            Neighbouring cell stack dont share the same poloidal wall
        """
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

    def __iter__(self) -> Iterator[DivertorCellStack]:
        """Iterator for DivertorCellArray"""
        return iter(self.cell_array)

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
        return list(
            chain.from_iterable([surf_stack[0] for surf_stack in self.radial_surfaces])
        )

    def exterior_surfaces(self) -> list[openmc.Surface]:
        """
        Get all of the outermost (vacuum-vessel-facing) surface.
        Runs clockwise.
        """
        return list(
            chain.from_iterable([
                surf_stack[-1][::-1] for surf_stack in self.radial_surfaces
            ])
        )

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
            :func:`~bluemira.radiation_transport.neutronics.csg_env.region_from_surface_series`
        """
        return openmc.Union([
            stack[0].exclusion_zone(
                away_from_plasma=True,
                control_id=control_id,
                additional_test_points=stack.exterior_wire.get_3D_coordinates(),
            )
            for stack in self.cell_array
        ])

    @classmethod
    def from_pre_cell_array(
        cls,
        pre_cell_array: DivertorPreCellArray,
        materials: MaterialsLibrary,
        divertor_thickness: DivertorThickness,
        csg: OpenMCEnvironment,
        override_start_end_surfaces: tuple[openmc.Surface, openmc.Surface] | None = None,
    ) -> DivertorCellArray:
        """
        Create the entire divertor from the pre-cell array.

        Parameters
        ----------
        pre_cell_array
            The array of divertor pre-cells.
        materials
            container of openmc.Material
        divertor_thickness
            A parameter
            :class:`bluemira.radiation_transport.neutronics.params.DivertorThickness`.
            For now it only has one scalar value stating how thick the
            divertor armour should be.
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
                    materials,
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
