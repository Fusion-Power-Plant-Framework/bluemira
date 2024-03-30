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
from typing import TYPE_CHECKING, Tuple

import numpy as np
import openmc

from bluemira.geometry.constants import EPS_FREECAD
from bluemira.neutronics.make_csg import (
    BlanketCellArray,
    DivertorCellArray,
    find_suitable_z_plane,
    flat_intersection,
    region_from_surface_series,
)
from bluemira.neutronics.slicing import (
    DivertorWireAndExteriorCurve,
    PanelsAndExteriorCurve,
)

if TYPE_CHECKING:
    from numpy import typing as npt

    from bluemira.geometry.wire import BluemiraWire
    from bluemira.neutronics.make_pre_cell import DivertorPreCellArray, PreCellArray
    from bluemira.neutronics.params import TokamakThicknesses


class StageOfComputation:
    """Abstract base class of all of the stage of computations classes below."""

    pass


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
    plasma: openmc.Cell = None


@dataclass
class OpenMCModelGenerator:
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
        discretization_combo: Tuple[float, float] = (20, 4),
    ) -> Tuple[PreCellArray, DivertorPreCellArray]:
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
        ].start_point()  # TODO: extend this further.
        last_point = self.data.divertor_wire.edges[
            -1
        ].end_point()  # TODO: extend this further.
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
        """
        return np.concatenate([
            pre_cell_array.get_exterior_vertices(),
            divertor_pre_cell_array.get_exterior_vertices()[::-1],
        ])

    @staticmethod
    def get_coordinates_from_cell_arrays(
        blanket_cell_array: BlanketCellArray,
        divertor_cell_array: DivertorCellArray,
    ) -> npt.NDArray:
        """
        Get the 2D coordinates of every point at the outer boundary of the tokamak's
        poloidal cross-section.

        Parameters
        ----------
        blanket_cell_array:
            BlanketCellArray
        divertor_cell_array:
            DivertorCellArray

        Returns
        -------
        coordinates
            array of shape (N+1+n*M, 2), where N = number of blanket pre-cells,
            M = number of divertor pre-cells, n = discretization_level used when chopping
            up the divertor in
            :meth:`bluemira.neutronics.DivertorWireAndExteriorCurve.make_divertor_pre_cell_array`
        """
        return np.concatenate([
            blanket_cell_array.get_exterior_vertices(),
            divertor_cell_array.get_exterior_vertices(),
        ])

    def set_universe_box(self, z_min: float, z_max: float, r_max: float):
        """Box up the universe in a cylinder (including top and bottom)."""
        find_suitable_z_plane(
            z_min,
            boundary_type="vacuum",
            surface_id=999,
            name="Blanket bottom",
        )
        find_suitable_z_plane(
            z_max,
            boundary_type="vacuum",
            surface_id=1000,
            name="Blanket top",
        )
        self.universe_cylinder = openmc.ZCylinder(
            r_max,
            surface_id=1001,
            boundary_type="vacuum",
        )

    def make_cell_arrays(
        self,
        material_dict,
        thickness: TokamakThicknesses,
    ) -> Tuple[BlanketCellArray, DivertorCellArray]:
        """Make pre-cell arrays for the blanket and the divertor."""
        BODGED_THICKNESS = 0.10  # TODO: fix this
        # determine universe_box
        all_ext_vertices = self.get_coordinates_from_pre_cell_arrays(
            self.pre_cell_array.blanket, self.pre_cell_array.divertor
        )
        z_min = all_ext_vertices[:, -1].min()
        z_max = all_ext_vertices[:, -1].max()
        r_max = max(abs(all_ext_vertices[:, 0]))
        self.set_universe_box(z_min - EPS_FREECAD, z_max, r_max)

        self.cell_array = CellStage()

        self.cell_array.blanket = BlanketCellArray.from_pre_cell_array(
            self.pre_cell_array.blanket, material_dict, thickness, id_control=True
        )
        # change the id register. (It will only count up from here.)
        openmc.Surface.next_id = int(max(openmc.Surface.used_ids) / 100 + 1) * 100
        self.cell_array.divertor = DivertorCellArray.from_divertor_pre_cell_array(
            self.pre_cell_array.divertor,
            material_dict,
            BODGED_THICKNESS,
            override_start_end_surfaces=(
                self.cell_array.blanket[0].ccw_surface,
                self.cell_array.blanket[-1].cw_surface,
            ),
        )
        return self.cell_array.blanket, self.cell_array.divertor

    def get_full_tokamak_region(self) -> openmc.Regoin:
        """
        Get the entire tokamak's poloidal cross-section (everything inside
        self.data.outer_boundary) as an openmc.Region.
        """
        vertices_array = self.get_coordinates_from_cell_arrays(
            self.cell_array.blanket,
            self.cell_array.divertor,
        )
        _surfaces = list(self.cell_array.blanket.get_exterior_surfaces())
        for div_pre_cell_bottom in self.cell_array.divertor.get_exterior_surfaces():
            _surfaces.extend(div_pre_cell_bottom)
        return region_from_surface_series(_surfaces, vertices_array, True)

    def make_plasma_cell(self):
        """Make the plasma chamber."""
        region = flat_intersection([
            self.get_full_tokamak_region(),
            ~self.cell_array.blanket.get_exclusion_zone(True),
            ~self.cell_array.divertor.get_exclusion_zone(True),
        ])

        self.cell_array.plasma = openmc.Cell(
            region=region,
            fill=None,
            name="Plasma void",
        )
        return self.cell_array.plasma
