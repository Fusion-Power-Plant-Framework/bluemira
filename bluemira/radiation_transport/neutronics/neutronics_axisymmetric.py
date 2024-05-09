# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Axis-symmetric CSG CAD models for neutronics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from bluemira.base.look_and_feel import bluemira_print
from bluemira.base.parameter_frame import Parameter, ParameterFrame, make_parameter_frame
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.error import GeometryError
from bluemira.geometry.plane import calculate_plane_dir
from bluemira.geometry.tools import get_wire_plane_intersect, is_convex, make_polygon
from bluemira.neutronics.make_pre_cell import PreCell
from bluemira.neutronics.slicing import (
    DivertorWireAndExteriorCurve,
    PanelsAndExteriorCurve,
)

if TYPE_CHECKING:
    from numpy import typing as npt

    from bluemira.base.reactor import ComponentManager
    from bluemira.geometry.wire import BluemiraWire
    from bluemira.radiation_transport.neutronics.geometry import TokamakDimensions
    from bluemira.radiation_transport.neutronics.make_pre_cell import (
        DivertorPreCellArray,
        PreCellArray,
    )
    from bluemira.radiation_transport.neutronics.materials import NeutronicsMaterials


@dataclass
class ReactorGeometry:
    """
    Data storage stage

    Parameters
    ----------
    divertor_wire:
        The plasma-facing side of the divertor.
    panel_break_points:
        The start and end points for each first-wall panel
        (for N panels, the shape is (N+1, 2)).
    boundary:
        interface between the inside of the vacuum vessel and the outside of the blanket
    vacuum_vessel_wire:
        The outer-boundary of the vacuum vessel
    """

    divertor_wire: BluemiraWire
    panel_break_points: npt.NDArray
    boundary: BluemiraWire
    vacuum_vessel_wire: BluemiraWire


@dataclass
class CuttingStage:
    """Stage of making cuts to the exterior curve/ outer boundary."""

    blanket: PanelsAndExteriorCurve
    divertor: DivertorWireAndExteriorCurve


class PreCellStage:
    """Stage of making pre-cells"""

    def __init__(self, blanket: PreCellArray, divertor: DivertorPreCellArray):
        """Check convexity after initialization"""
        self.blanket = blanket.copy()
        self.divertor = divertor.copy()
        # 1. stretch first blanket cell in PreCellArray to reach div_start_wire
        div_start_wire = self.divertor[0].cw_wall.restore_to_wire()
        # pull everything down to: div_start_wire.
        # Alternatively, choose div_start_wire=self.divertor[0].outline
        old_vv_wire = self.blanket[0].vv_wire
        ext_pt, i_high, i_low = np.insert(self.blanket[0].vertex, 1, 0, axis=0).T[:3]
        i_end = get_wire_plane_intersect(
            div_start_wire, *calculate_plane_dir(i_high, i_low)
        )
        # v_end = get_wire_plane_intersect(
        #     div_start_wire,
        #     *calculate_plane_dir(old_vv_wire.end_point().xyz.flatten(),
        #     old_vv_wire.start_point().xyz.flatten())
        # )
        in_wire = make_polygon(np.array([i_high, i_end]).T, closed=False)
        vv_wire = make_polygon(
            np.array([
                old_vv_wire.end_point().xyz.flatten(),
                self.divertor[0].vv_wire.end_point,
            ]).T,
            closed=False,
        )
        ex_wire = make_polygon(
            np.array([self.divertor[0].vertex.T[0], ext_pt]).T, closed=False
        )
        new_start_cell = PreCell(in_wire, vv_wire, ex_wire)
        self.blanket[0] = new_start_cell

        # 2. stretch first blanket cell in PreCellArray to reach div_end_wire
        div_end_wire = self.divertor[-1].ccw_wall.restore_to_wire()
        old_vv_wire = self.blanket[-1].vv_wire
        i_low, i_high, ext_pt = np.insert(self.blanket[-1].vertex, 1, 0, axis=0).T[-3:]
        i_start = get_wire_plane_intersect(
            div_end_wire, *calculate_plane_dir(i_high, i_low)
        )
        # v_end = get_wire_plane_intersect(
        #     div_end_wire,
        #     *calculate_plane_dir(old_vv_wire.start_point().xyz.flatten(),
        #     old_vv_wire.end_point().xyz.flatten())
        # )
        in_wire = make_polygon(np.array([i_start, i_high]).T, closed=False)
        vv_wire = make_polygon(
            np.array([
                old_vv_wire.start_point().xyz.flatten(),
                self.divertor[-1].vv_wire.start_point,
            ]).T,
            closed=False,
        )
        ex_wire = make_polygon(
            np.array([ext_pt, self.divertor[-1].vertex.T[-1]]).T, closed=False
        )
        new_end_cell = PreCell(in_wire, vv_wire, ex_wire)
        self.blanket[-1] = new_end_cell

        # re-initialize so that the cell_walls are re-calculated
        self.blanket = self.blanket.copy()

        ext_coords = self.external_coordinates()
        if not is_convex(ext_coords):
            raise GeometryError(
                f"The vertices of {self.blanket} + {self.divertor} must form "
                "a convex outline!"
            )

    def external_coordinates(self) -> npt.NDArray:
        """
        Get the outermost coordinates of the tokamak cross-section from pre-cell array
        and divertor pre-cell array.
        Runs clockwise, beginning at the inboard blanket-divertor joint.
        """
        return np.concatenate([
            self.blanket.exterior_vertices(),
            self.divertor.exterior_vertices()[::-1],
        ])

    def bounding_box(self) -> tuple[float, ...]:
        """Get bounding box of pre cell stage"""
        all_ext_vertices = self.external_coordinates()
        z_min = all_ext_vertices[:, -1].min()
        z_max = all_ext_vertices[:, -1].max()
        r_max = max(abs(all_ext_vertices[:, 0]))
        return z_max, z_min, r_max, -r_max

    def half_bounding_box(self) -> tuple[float, ...]:
        """
        Get bounding box of the 2D poloidal cross-section of the right-hand half of the
        reactor.
        """
        all_ext_vertices = self.external_coordinates()
        z_min = all_ext_vertices[:, -1].min()
        z_max = all_ext_vertices[:, -1].max()
        r_max = max(abs(all_ext_vertices[:, 0]))
        r_min = min(abs(all_ext_vertices[:, 0]))
        return z_max, z_min, r_max, r_min


@dataclass
class NeutronicsReactorParameterFrame(ParameterFrame):
    """Neutronics reactor parameters"""

    inboard_fw_tk: Parameter[float]
    inboard_breeding_tk: Parameter[float]
    outboard_fw_tk: Parameter[float]
    outboard_breeding_tk: Parameter[float]
    r_tf_in: Parameter[float]
    tk_tf_inboard: Parameter[float]
    fw_divertor_surface_tk: Parameter[float]
    fw_blanket_surface_tk: Parameter[float]
    blk_ib_manifold: Parameter[float]
    blk_ob_manifold: Parameter[float]


class NeutronicsReactor(ABC):
    """Pre csg cell reactor"""

    param_cls = NeutronicsReactorParameterFrame

    def __init__(
        self,
        params: dict | ParameterFrame,
        divertor: ComponentManager,
        blanket: ComponentManager,
        vacuum_vessel: ComponentManager,
        materials_library: NeutronicsMaterials,
        *,
        snap_to_horizontal_angle: float = 45,
        blanket_discretisation: int = 10,
        divertor_discretisation: int = 5,
    ):
        bluemira_print("Creating axis-symmetric neutronics model")

        self.params = make_parameter_frame(params, self.param_cls)
        self.material_library = materials_library
        (
            self.tokamak_dimensions,
            divertor_wire,
            panel_points,
            blanket_wire,
            vacuum_vessel_wire,
        ) = self._get_wires_from_components(divertor, blanket, vacuum_vessel)

        self.geom = ReactorGeometry(
            divertor_wire, panel_points, blanket_wire, vacuum_vessel_wire
        )

        self._pre_cell_stage = self._create_pre_cell_stage(
            blanket_discretisation, divertor_discretisation, snap_to_horizontal_angle
        )

    def _create_pre_cell_stage(
        self, blanket_discretisation, divertor_discretisation, snap_to_horizontal_angle
    ):
        cutting = CuttingStage(
            blanket=PanelsAndExteriorCurve(
                self.geom.panel_break_points,
                self.geom.boundary,
                self.geom.vacuum_vessel_wire,
            ),
            divertor=DivertorWireAndExteriorCurve(
                self.geom.divertor_wire, self.geom.boundary, self.geom.vacuum_vessel_wire
            ),
        )
        divertor = cutting.divertor.make_divertor_pre_cell_array(
            discretisation_level=divertor_discretisation
        )
        first, last = divertor.exterior_vertices()[(0, -1),]

        blanket = cutting.blanket.make_quadrilateral_pre_cell_array(
            discretisation_level=blanket_discretisation,
            starting_cut=Coordinates(first).xz.flatten(),
            ending_cut=Coordinates(last).xz.flatten(),
            snap_to_horizontal_angle=snap_to_horizontal_angle,
        )

        return PreCellStage(
            blanket=blanket.straighten_exterior(preserve_volume=True), divertor=divertor
        )

    @property
    def bounding_box(self) -> tuple[float, ...]:
        """Bounding box of Neutronics reactor"""
        return self._pre_cell_stage.bounding_box()

    @property
    def half_bounding_box(self) -> tuple[float, ...]:
        """Bounding box of the right-hand half of the 2D poloidal cross-section"""
        return self._pre_cell_stage.half_bounding_box()

    @property
    def blanket(self):
        """Blanket pre cell"""
        return self._pre_cell_stage.blanket

    @property
    def divertor(self):
        """Divertor pre cell"""
        return self._pre_cell_stage.divertor

    def plot_2d(self, *args, **kwargs):
        """Plot neutronics reactor 2d profile"""
        show = kwargs.pop("show", True)
        ax = kwargs.pop("ax", None)
        ax = self.blanket.plot_2d(*args, ax=ax, show=False, **kwargs)
        return self.divertor.plot_2d(*args, ax=ax, show=show, **kwargs)

    @abstractmethod
    def _get_wires_from_components(
        self,
        divertor: ComponentManager,
        blanket: ComponentManager,
        vacuum_vessel: ComponentManager,
    ) -> tuple[
        TokamakDimensions, BluemiraWire, npt.NDArray, BluemiraWire, BluemiraWire
    ]: ...
