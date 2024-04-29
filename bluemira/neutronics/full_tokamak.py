# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Only contains 1 class that controls the overall conversion from bluemira model to csg.
Separated from slicing.py to prevent import errors
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np
import openmc

from bluemira.geometry.constants import D_TOLERANCE
from bluemira.neutronics.constants import to_cm, to_cm3
from bluemira.neutronics.make_csg import (
    BlanketCellArray,
    DivertorCellArray,
    TFCoils,
    find_suitable_z_plane,
    flat_intersection,
    flat_union,
    region_from_surface_series,
    surface_from_2points,
)
from bluemira.neutronics.radial_wall import polygon_revolve_signed_volume
from bluemira.neutronics.slicing import (
    DivertorWireAndExteriorCurve,
    PanelsAndExteriorCurve,
)

if TYPE_CHECKING:
    from numpy import typing as npt

    from bluemira.geometry.wire import BluemiraWire
    from bluemira.neutronics.make_materials import MaterialsLibrary
    from bluemira.neutronics.make_pre_cell import DivertorPreCellArray, PreCellArray
    from bluemira.neutronics.params import TokamakDimensions


@dataclass
class ReactorGeometry:
    """
    Data storage stage

    Parameters
    ----------
    panel_break_points:
        The start and end points for each first-wall panel
        (for N panels, the shape is (N+1, 2)).
    divertor_wire:
        The plasma-facing side of the divertor.
    boundary:
        interface between the inside of the vacuum vessel and the outside of the blanket
    vacuum_vessel_wire:
        The outer-boundary of the vacuum vessel
    """

    panel_break_points: npt.NDArray
    divertor_wire: BluemiraWire
    boundary: BluemiraWire
    vacuum_vessel_wire: BluemiraWire


@dataclass
class CuttingStage:
    """Stage of making cuts to the exterior curve/ outer boundary."""

    blanket: PanelsAndExteriorCurve
    divertor: DivertorWireAndExteriorCurve


@dataclass
class PreCellStage:
    """Stage of making pre-cells"""

    blanket: PreCellArray
    divertor: DivertorPreCellArray

    def external_coordinates(self):
        """
        Get the outermost coordinates of the tokamak cross-section from pre-cell array
        and divertor pre-cell array.
        Runs clockwise, beginning at the inboard blanket-divertor joint.
        """
        return np.concatenate([
            self.blanket.exterior_vertices(),
            self.divertor.exterior_vertices()[::-1],
        ])

    def bounding_box(self):
        all_ext_vertices = self.external_coordinates()
        z_min = all_ext_vertices[:, -1].min()
        z_max = all_ext_vertices[:, -1].max()
        r_max = max(abs(all_ext_vertices[:, 0]))
        return z_max, z_min, r_max, -r_max


@dataclass
class CellStage:
    """Stage of making cells."""

    blanket: BlanketCellArray
    divertor: DivertorCellArray
    tf_coils: TFCoils
    central_solenoid: openmc.Cell
    plasma: openmc.Cell
    ext_void: openmc.Cell
    universe: openmc.Region

    @property
    def cells(self):
        """Get the list of all cells."""
        return (
            *chain.from_iterable((*self.blanket, *self.divertor)),
            *self.tf_coils,
            self.central_solenoid,
            self.plasma,
            self.ext_void,
        )

    def get_all_hollow_merged_cells(self):
        """Blanket and divertor cells"""
        return [
            *[openmc.Cell(region=stack.get_overall_region()) for stack in self.blanket],
            *[openmc.Cell(region=stack.get_overall_region()) for stack in self.divertor],
        ]


def round_up_next_openmc_ids(surface_step_size: int = 1000, cell_step_size: int = 100):
    """
    Make openmc's surfaces' and cells' next IDs to be incremented to the next
    pre-determined interval.
    """
    openmc.Surface.next_id = (
        int(max(openmc.Surface.used_ids) / surface_step_size + 1) * surface_step_size + 1
    )
    openmc.Cell.next_id = (
        int(max(openmc.Cell.used_ids) / cell_step_size + 1) * cell_step_size + 1
    )


class SingleNullTokamak:
    """
    Convert 3 things: panel_break_points, divertor_wire, and boundary_wire into
    pre-cell array, then cell-arrays.
    """

    def __init__(
        self,
        panel_break_points: npt.NDArray,
        divertor_wire: BluemiraWire,
        boundary: BluemiraWire,
        vacuum_vessel_wire: BluemiraWire,
    ):
        """
        Parameters
        ----------
        panel_break_points
            np.ndarray of shape (N+1, 2)
        divertor_wire
            BluemiraWire (3D object) outlining the top (plasma facing side) of the
            divertor.
        boundary
            BluemiraWire (3D object) outlining the outside boundary of the vacuum vessel.
        """
        self.geom = ReactorGeometry(
            panel_break_points, divertor_wire, boundary, vacuum_vessel_wire
        )

        self.cutting = CuttingStage(
            blanket=PanelsAndExteriorCurve(
                self.geom.panel_break_points, self.geom.boundary, vacuum_vessel_wire
            ),
            divertor=DivertorWireAndExteriorCurve(
                self.geom.divertor_wire, self.geom.boundary, vacuum_vessel_wire
            ),
        )

    def make_pre_cell_arrays(
        self,
        snap_to_horizontal_angle: float = 45,
        blanket_discretisation: int = 10,
        divertor_discretisation: int = 5,
    ):
        """
        Parameters
        ----------
        snap_to_horizontal_angle:
            see :meth:`~PanelsAndExteriorCurve.make_quadrilateral_pre_cell_array`
        """
        first_point = self.geom.divertor_wire.edges[0].start_point()
        last_point = self.geom.divertor_wire.edges[-1].end_point()

        blanket = self.cutting.blanket.make_quadrilateral_pre_cell_array(
            discretisation_level=blanket_discretisation,
            starting_cut=first_point.xz.flatten(),
            ending_cut=last_point.xz.flatten(),
            snap_to_horizontal_angle=snap_to_horizontal_angle,
        )

        self.pre_cell_arrays = PreCellStage(
            blanket=blanket.straighten_exterior(preserve_volume=True),
            divertor=self.cutting.divertor.make_divertor_pre_cell_array(
                discretisation_level=divertor_discretisation
            ),
        )

    @staticmethod
    def exterior_vertices(blanket, divertor) -> npt.NDArray:
        """
        Get the 3D coordinates of every point at the outer boundary of the tokamak's
        poloidal cross-section.

        Returns
        -------
        coordinates
            array of shape (N+1+n*M, 3), where N = number of blanket pre-cells,
            M = number of divertor pre-cells, n = discretization_level used when chopping
            up the divertor in
            :meth:`bluemira.neutronics.DivertorWireAndExteriorCurve.make_divertor_pre_cell_array`
        """
        return np.concatenate([
            blanket.exterior_vertices(),
            divertor.exterior_vertices()[::-1],
        ])

    @staticmethod
    def interior_vertices(blanket, divertor) -> npt.NDArray:
        """
        Get the 3D coordinates of every point at the interior boundary of the tokamak's
        poloidal cross-section

        Returns
        -------
        coordinates
            array of shape ((N+1)+sum(number of interior points of the divertor), 3),
            where N = number of blanket pre-cells, M = number of divertor pre-cells.
            Runs clockwise, beginning at the inboard blanket-divertor joining point.
        """
        return np.concatenate([
            blanket.interior_vertices(),
            divertor.interior_vertices()[::-1],
        ])

    @staticmethod
    def make_universe_box(
        z_min: float, z_max: float, r_max: float, *, control_id: bool = False
    ):
        """Box up the universe in a cylinder (including top and bottom)."""
        bottom = find_suitable_z_plane(
            z_min,
            boundary_type="vacuum",
            surface_id=999 if control_id else None,
            name="Universe bottom",
        )
        top = find_suitable_z_plane(
            z_max,
            boundary_type="vacuum",
            surface_id=1000 if control_id else None,
            name="Universe top",
        )
        universe_cylinder = openmc.ZCylinder(
            r=to_cm(r_max),
            surface_id=1001 if control_id else None,
            boundary_type="vacuum",
            name="Max radius of Universe",
        )
        return -top & +bottom & -universe_cylinder

    def make_cell_arrays(
        self,
        material_library: MaterialsLibrary,
        tokamak_dimensions: TokamakDimensions,
        *,
        control_id: bool = False,
    ) -> CellStage:
        """Make pre-cell arrays for the blanket and the divertor.

        Parameters
        ----------
        material_library:
            library containing information about the materials
        tokamak_dimensions:
            A parameter :class:`bluemira.neutronics.params.TokamakDimensions`,
            Specifying the dimensions of various layers in the blanket, divertor, and
            central solenoid.
        control_id: bool
            Whether to set the blanket Cells and surface IDs by force or not.
            With this set to True, it will be easier to understand where each cell came
            from. However, it will lead to warnings and errors if a cell/surface is
            generated to use a cell/surface ID that has already been used respectively.
            Keep this as False if you're running openmc simulations multiple times in one
            session.
        """
        # determine universe_box
        z_max, z_min, r_max, _r_min = self.pre_cell_arrays.bounding_box()
        universe = self.make_universe_box(
            z_min - D_TOLERANCE,
            z_max + D_TOLERANCE,
            r_max + D_TOLERANCE,
            control_id=control_id,
        )

        blanket = BlanketCellArray.from_pre_cell_array(
            self.pre_cell_arrays.blanket,
            material_library,
            tokamak_dimensions,
            control_id=control_id,
        )

        # change the cell and surface id register before making the divertor.
        # (ids will only count up from here.)
        if control_id:
            round_up_next_openmc_ids()

        divertor = DivertorCellArray.from_pre_cell_array(
            self.pre_cell_arrays.divertor,
            material_library,
            tokamak_dimensions.divertor,
            override_start_end_surfaces=(blanket[0].ccw_surface, blanket[-1].cw_surface),
            # ID cannot be controlled at this point.
        )

        # make the plasma cell and the exterior void.
        if control_id:
            round_up_next_openmc_ids()

        cs, tf = self.make_coils(
            tokamak_dimensions.central_solenoid.inner_diameter / 2,
            (
                (
                    tokamak_dimensions.central_solenoid.outer_diameter
                    - tokamak_dimensions.central_solenoid.inner_diameter
                )
                / 2
            ),
            z_min - D_TOLERANCE,
            z_max + D_TOLERANCE,
            material_library,
        )
        plasma, ext_void = self.make_void_cells(
            tf, cs, universe, blanket, divertor, control_id=control_id
        )

        self.cell_array = CellStage(
            blanket=blanket,
            divertor=divertor,
            tf_coils=tf,
            central_solenoid=cs,
            plasma=plasma,
            ext_void=ext_void,
            universe=universe,
        )
        self.set_volumes(
            self.cell_array.universe,
            self.cell_array.tf_coils,
            self.cell_array.central_solenoid,
            self.cell_array.ext_void,
            self.cell_array.blanket,
            self.cell_array.divertor,
            self.cell_array.plasma,
        )

        return self.cell_array

    @staticmethod
    def make_coils(
        solenoid_radius: float,
        tf_coil_thick: float,
        z_min: float,
        z_max: float,
        material_library,
    ) -> tuple[openmc.Cell, TFCoils]:
        """
        Make tf coil and the central solenoid. The former wraps around the latter.

        Parameters
        ----------
        solenoid_radius:
            Central solenoid radius [m]
        tf_coil_thick:
            Thickness of the tf-coil, wrapped around the central solenoid [m]
        z_max:
            z-coordinate of the the top z-plane shared by both cylinders
            (cs and tf coil)
        z_min
            z-coordinate of the the bottom z-plane shared by both cylinders
            (cs and tf coil)
        """
        solenoid = openmc.ZCylinder(r=to_cm(solenoid_radius))
        central_tf_coil = openmc.ZCylinder(r=to_cm(tf_coil_thick + solenoid_radius))
        top = find_suitable_z_plane(
            z_max,
            [z_max - D_TOLERANCE, z_max + D_TOLERANCE],
            name="Top of central solenoid",
        )
        bottom = find_suitable_z_plane(
            z_min,
            [z_min - D_TOLERANCE, z_min + D_TOLERANCE],
            name="Bottom of central solenoid",
        )
        central_solenoid = openmc.Cell(
            name="Central solenoid",
            fill=material_library.container_mat,
            region=+bottom & -top & -solenoid,
        )
        tf_coils = TFCoils([
            openmc.Cell(
                name="TF coil (sheath around central solenoid)",
                fill=material_library.tf_coil_mat,
                region=+bottom & -top & +solenoid & -central_tf_coil,
            )
        ])
        central_solenoid.volume = (top.z0 - bottom.z0) * np.pi * solenoid.r**2
        tf_coils[0].volume = (
            (top.z0 - bottom.z0) * np.pi * (central_tf_coil.r**2 - solenoid.r**2)
        )
        return central_solenoid, tf_coils

    def blanket_and_divertor_outer_region(
        self, blanket, divertor, *, control_id: bool = False
    ) -> openmc.Region:
        """
        Get the entire tokamak's poloidal cross-section (everything inside
        self.geom.boundary) as an openmc.Region.
        """
        surfaces = [
            *blanket.exterior_surfaces(),
            *chain.from_iterable(divertor.exterior_surfaces()),
        ]
        return region_from_surface_series(
            surfaces, self.exterior_vertices(blanket, divertor), control_id=control_id
        )

    @staticmethod
    def plasma_void(blanket, divertor, *, control_id: bool = False) -> openmc.Region:
        """Get the plasma chamber's poloidal cross-section"""
        blanket_interior_pts = blanket.interior_vertices()
        dividing_surface = surface_from_2points(
            blanket_interior_pts[0][::2], blanket_interior_pts[-1][::2]
        )
        blanket_surfaces = [*blanket.interior_surfaces(), dividing_surface]
        plasma = region_from_surface_series(
            blanket_surfaces, blanket_interior_pts, control_id=control_id
        )

        div_surfaces = [
            *chain.from_iterable(divertor.exterior_surfaces()),
            dividing_surface,
        ]
        exhaust_including_divertor = region_from_surface_series(
            div_surfaces,
            divertor.exterior_vertices(),
            control_id=control_id,
        )

        divertor_zone = divertor.exclusion_zone(control_id=control_id)
        return flat_union([plasma, exhaust_including_divertor]) & ~divertor_zone

    def make_void_cells(
        self,
        tf_coils,
        central_solenoid,
        universe,
        blanket,
        divertor,
        *,
        control_id: bool = False,
    ):
        """Make the plasma chamber and the outside ext_void. This should be called AFTER
        the blanket and divertor cells are created.
        """
        full_tokamak_region = self.blanket_and_divertor_outer_region(
            blanket, divertor, control_id=control_id
        )
        void_region = universe & ~full_tokamak_region
        if tf_coils:
            void_region &= ~tf_coils[0].region
        if central_solenoid:
            void_region &= ~central_solenoid.region

        return (
            openmc.Cell(
                region=self.plasma_void(blanket, divertor, control_id=control_id),
                fill=None,
                name="Plasma void",
            ),
            openmc.Cell(
                region=flat_intersection(void_region),
                fill=None,
                name="Exterior void",
            ),
        )

    def set_volumes(
        self, universe, tf_coils, central_solenoid, ext_void, blanket, divertor, plasma
    ):
        """
        Sets the volume of the voids. Not necessary/ used anywhere yet.
        """
        exterior_vertices = self.exterior_vertices(blanket, divertor)
        total_universe_volume = (
            #  top - bottom
            (universe[0].surface.z0 - universe[1].surface.z0)
            * np.pi
            * universe[2].surface.r ** 2  # cylinder
        )  # cm^3
        universe.volume = total_universe_volume

        outer_boundary_volume = to_cm3(
            polygon_revolve_signed_volume(exterior_vertices[:, ::2])
        )
        ext_void_volume = total_universe_volume - outer_boundary_volume
        if tf_coils:
            for coil in tf_coils:
                ext_void_volume -= coil.volume
        if central_solenoid:
            ext_void_volume -= central_solenoid.volume
        ext_void.volume = ext_void_volume
        blanket_volumes = sum(cell.volume for cell in chain.from_iterable(blanket))
        divertor_volumes = sum(cell.volume for cell in chain.from_iterable(divertor))
        plasma.volume = outer_boundary_volume - blanket_volumes - divertor_volumes

    def __repr__(self):
        """
        Make the name display what stage had been instantiated: pre-cell-array and
        cell-array.
        """
        has_pca = (
            "pre-cell-array generated"
            if hasattr(self, "pre_cell_array")
            else "no pre-cell-array"
        )
        has_ca = (
            "cell-array generated" if hasattr(self, "cell_array") else "no cell-array"
        )
        return super().__repr__().replace(" at ", f" with {has_pca}, {has_ca} at ")
