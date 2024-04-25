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
from bluemira.neutronics.slicing import (
    DivertorWireAndExteriorCurve,
    PanelsAndExteriorCurve,
)

if TYPE_CHECKING:
    from numpy import typing as npt

    from bluemira.geometry.wire import BluemiraWire
    from bluemira.neutronics.make_pre_cell import DivertorPreCellArray, PreCellArray
    from bluemira.neutronics.params import TokamakDimensions


class StageOfComputation:
    """Abstract base class of all of the stage of computations classes below."""


@dataclass
class RawData(StageOfComputation):
    """Data storage stage"""

    panel_break_points: npt.NDArray
    divertor_wire: BluemiraWire
    outer_boundary: BluemiraWire


@dataclass
class CuttingStage(StageOfComputation):
    """Stage of making cuts to the exterior curve/ outer boundary."""

    blanket: PanelsAndExteriorCurve = None
    divertor: DivertorWireAndExteriorCurve = None


@dataclass
class PreCellStage(StageOfComputation):
    """Stage of making pre-cells"""

    blanket: PreCellArray = None
    divertor: DivertorPreCellArray = None


@dataclass
class CellStage(StageOfComputation):
    """Stage of making cells."""

    blanket: BlanketCellArray = None
    divertor: DivertorCellArray = None
    tf_coils: TFCoils = None
    central_solenoid: openmc.Cell = None
    plasma: openmc.Cell = None
    ext_void: openmc.Cell = None
    universe_region: openmc.Region = None

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
    Convert 3 things: panel_break_points, divertor_wire, and outer_boundary_wire into
    pre-cell array, then cell-arrays.
    """

    def __init__(
        self,
        panel_break_points: npt.NDArray,
        divertor_wire: BluemiraWire,
        outer_boundary: BluemiraWire,
    ):
        """
        Parameters
        ----------
        panel_break_points
            np.ndarray of shape (N+1, 2)
        divertor_wire
            BluemiraWire (3D object) outlining the top (plasma facing side) of the
            divertor.
        outer_boundary
            BluemiraWire (3D object) outlining the outside boundary of the vacuum vessel.
        """
        self.data = RawData(panel_break_points, divertor_wire, outer_boundary)

        self.cutting = CuttingStage()
        self.cutting.blanket = PanelsAndExteriorCurve(
            self.data.panel_break_points, self.data.outer_boundary
        )
        self.cutting.divertor = DivertorWireAndExteriorCurve(
            self.data.divertor_wire, self.data.outer_boundary
        )

    def make_pre_cell_arrays(
        self,
        preserve_volume: bool,
        snap_to_horizontal_angle: float = 45,
        discretization_combo: tuple[float, float] = (20, 4),
    ) -> tuple[PreCellArray, DivertorPreCellArray]:
        """
        Parameters
        ----------
        preserve_volume:
            see :meth:`~PreCellArray.straighten_exterior`
        snap_to_horizontal_angle:
            see :meth:`~PanelsAndExteriorCurve.make_quadrilateral_pre_cell_array`
        """
        self.pre_cell_array = PreCellStage()

        # blanket
        first_point = self.data.divertor_wire.edges[
            0
        ].start_point()  # TODO: Shall I extend this further outwards?
        last_point = self.data.divertor_wire.edges[
            -1
        ].end_point()  # TODO: Shall I extend this further outwards?
        self.pre_cell_array.blanket = (
            self.cutting.blanket.make_quadrilateral_pre_cell_array(
                snap_to_horizontal_angle=snap_to_horizontal_angle,
                starting_cut=first_point.xz.flatten(),
                ending_cut=last_point.xz.flatten(),
                discretization_level=discretization_combo[0],
            )
        )
        self.pre_cell_array.blanket = self.pre_cell_array.blanket.straighten_exterior(
            preserve_volume
        )
        # divertor
        self.pre_cell_array.divertor = (
            self.cutting.divertor.make_divertor_pre_cell_array(
                discretization_level=discretization_combo[1]
            )
        )
        return self.pre_cell_array.blanket, self.pre_cell_array.divertor

    @staticmethod
    def get_coordinates_from_pre_cell_arrays(
        pre_cell_array: PreCellArray, divertor_pre_cell_array: DivertorPreCellArray
    ):
        """
        Get the outermost coordinates of the tokamak cross-section from pre-cell array
        and divertor pre-cell array.
        Runs clockwise, beginning at the inboard blanket-divertor joint.
        """
        return np.concatenate([
            pre_cell_array.get_exterior_vertices(),
            divertor_pre_cell_array.get_exterior_vertices()[::-1],
        ])

    def get_exterior_vertices(self) -> npt.NDArray:
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
            self.cell_array.blanket.get_exterior_vertices(),
            self.cell_array.divertor.get_exterior_vertices()[::-1],
        ])

    def get_interior_vertices(self) -> npt.NDArray:
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
            self.cell_array.blanket.get_interior_vertices(),
            self.cell_array.divertor.get_interior_vertices()[::-1],
        ])

    def set_universe_box(
        self, z_min: float, z_max: float, r_max: float, control_id: bool = False
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
        outer_cylinder = openmc.ZCylinder(
            r=to_cm(r_max),
            surface_id=1001 if control_id else None,
            boundary_type="vacuum",
            name="Max radius of Universe",
        )
        self.cell_array.universe_region = -top & +bottom & -outer_cylinder

    def make_cell_arrays(
        self,
        material_dict,
        tokamak_dimensions: TokamakDimensions,
        control_id: bool = False,
    ) -> tuple[BlanketCellArray, DivertorCellArray, openmc.Cell]:
        """Make pre-cell arrays for the blanket and the divertor.

        Parameters
        ----------
        material_dict:
            TODO: fill in later: Change this to MaterialsLibrary directly later.
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
        all_ext_vertices = self.get_coordinates_from_pre_cell_arrays(
            self.pre_cell_array.blanket, self.pre_cell_array.divertor
        )
        z_min = all_ext_vertices[:, -1].min()
        z_max = all_ext_vertices[:, -1].max()
        r_max = max(abs(all_ext_vertices[:, 0]))

        self.cell_array = CellStage()
        self.set_universe_box(
            z_min - D_TOLERANCE,
            z_max + D_TOLERANCE,
            r_max + D_TOLERANCE,
            control_id=control_id,
        )

        self.cell_array.blanket = BlanketCellArray.from_pre_cell_array(
            self.pre_cell_array.blanket,
            material_dict,
            tokamak_dimensions,
            control_id=control_id,
        )

        # change the cell and surface id register before making the divertor.
        # (ids will only count up from here.)
        if control_id:
            round_up_next_openmc_ids()

        self.cell_array.divertor = DivertorCellArray.from_divertor_pre_cell_array(
            self.pre_cell_array.divertor,
            material_dict,
            tokamak_dimensions.divertor,
            override_start_end_surfaces=(
                self.cell_array.blanket[0].ccw_surface,
                self.cell_array.blanket[-1].cw_surface,
            ),
            # ID cannot be controlled at this point.
        )

        # make the plasma cell and the exterior void.
        if control_id:
            round_up_next_openmc_ids()
        self.material_dict = material_dict
        self.make_cs_coils(
            2,
            1,
            z_min - D_TOLERANCE,
            z_max + D_TOLERANCE,
        )
        self.make_void_cells(control_id)
        # self.make_container()

        return (
            self.cell_array.blanket,
            self.cell_array.divertor,
            self.cell_array.tf_coils,
            self.cell_array.central_solenoid,
            self.cell_array.plasma,
            self.cell_array.ext_void,
        )

    def make_cs_coils(
        self, solenoid_radius: float, tf_coil_thick: float, z_min: float, z_max: float
    ):
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
        self.cell_array.central_solenoid = openmc.Cell(
            name="Central solenoid",
            fill=self.material_dict["CentralSolenoid"],
            region=+bottom & -top & -solenoid,
        )
        self.cell_array.tf_coils = TFCoils([
            openmc.Cell(
                name="TF coil (sheath around central solenoid)",
                fill=self.material_dict["TFCoil"],
                region=+bottom & -top & +solenoid & -central_tf_coil,
            )
        ])
        self.cell_array.central_solenoid.volume = (
            (top.z0 - bottom.z0) * np.pi * solenoid.r**2
        )
        self.cell_array.tf_coils.volume = (
            (top.z0 - bottom.z0) * np.pi * (central_tf_coil.r**2 - solenoid.r**2)
        )
        return self.cell_array.central_solenoid, self.cell_array.tf_coils

    def make_container(self):
        """Make container"""
        self.cell_array.container = ...
        raise NotImplementedError("Method incomplete.")

    def get_blanket_and_divertor_outer_region(
        self, control_id: bool = False
    ) -> openmc.Regoin:
        """
        Get the entire tokamak's poloidal cross-section (everything inside
        self.data.outer_boundary) as an openmc.Region.
        """
        exterior_vertices = self.get_exterior_vertices()
        _surfaces = list(self.cell_array.blanket.get_exterior_surfaces())
        for div_pre_cell_bottom in self.cell_array.divertor.get_exterior_surfaces():
            _surfaces.extend(div_pre_cell_bottom)
        return region_from_surface_series(_surfaces, exterior_vertices, control_id)

    def get_blanket_and_divertor_inner_region(
        self, control_id: bool = False
    ) -> openmc.Region:
        """Get the plasma chamber's poloidal cross-section"""
        _blanket_interior_pts = self.cell_array.blanket.get_interior_vertices()
        dividing_surface = surface_from_2points(
            _blanket_interior_pts[0][::2], _blanket_interior_pts[-1][::2]
        )
        _blanket_surfaces = self.cell_array.blanket.get_interior_surfaces()
        _blanket_surfaces.append(dividing_surface)
        plasma = region_from_surface_series(
            _blanket_surfaces, _blanket_interior_pts, control_id
        )

        _div_surfaces = []
        for surf_list in self.cell_array.divertor.get_exterior_surfaces():
            _div_surfaces.extend(surf_list)
        _div_surfaces.append(dividing_surface)
        exhaust_including_divertor = region_from_surface_series(
            _div_surfaces, self.cell_array.divertor.get_exterior_vertices(), control_id
        )

        divertor_zone = self.cell_array.divertor.get_exclusion_zone(control_id)
        return flat_union([plasma, exhaust_including_divertor]) & ~divertor_zone

    def make_void_cells(self, control_id: bool = False):
        """Make the plasma chamber and the outside ext_void. This should be called AFTER
        the blanket and divertor cells are created.
        """
        full_tokamak_region = self.get_blanket_and_divertor_outer_region(control_id)
        void_region = self.cell_array.universe_region & ~full_tokamak_region
        if self.cell_array.tf_coils:
            void_region = void_region & ~self.cell_array.tf_coils[0].region
        if self.cell_array.central_solenoid:
            void_region = void_region & ~self.cell_array.central_solenoid.region
        self.cell_array.ext_void = openmc.Cell(
            region=flat_intersection(void_region),
            fill=None,
            name="Exterior void",
        )

        self.cell_array.plasma = openmc.Cell(
            region=self.get_blanket_and_divertor_inner_region(control_id),
            fill=None,
            name="Plasma void",
        )
        self.set_void_volumes()

        return self.cell_array.plasma, self.cell_array.ext_void

    def set_void_volumes(self):
        """
        Sets the volume of the voids. Not necessary/ used anywhere yet.
        """
        exterior_vertices = self.get_exterior_vertices()
        total_universe_volume = (
            (
                self.cell_array.universe_region[0].surface.z0  # top
                - self.cell_array.universe_region[1].surface.z0
            )  # bottom
            * np.pi
            * self.cell_array.universe_region[2].surface.r ** 2  # cylinder
        )  # cm^3
        self.cell_array.universe_region.volume = total_universe_volume

        outer_boundary_volume = to_cm3(
            polygon_revolve_signed_volume(exterior_vertices[:, ::2])
        )
        ext_void_volume = total_universe_volume - outer_boundary_volume
        if self.cell_array.tf_coils:
            for coil in self.cell_array.tf_coils:
                ext_void_volume -= coil.volume
        if self.cell_array.central_solenoid:
            ext_void_volume -= self.cell_array.central_solenoid.volume
        self.cell_array.ext_void.volume = ext_void_volume
        blanket_volumes = sum(
            cell.volume for cell in chain.from_iterable(self.cell_array.blanket)
        )
        divertor_volumes = sum(
            cell.volume for cell in chain.from_iterable(self.cell_array.divertor)
        )
        self.cell_array.plasma.volume = (
            outer_boundary_volume - blanket_volumes - divertor_volumes
        )

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
