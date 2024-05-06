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

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from bluemira.base.parameter_frame import Parameter, ParameterFrame, make_parameter_frame
from bluemira.geometry.coordinates import vector_intersect
from bluemira.geometry.tools import deserialise_shape
from bluemira.neutronics.params import TokamakDimensions
from bluemira.neutronics.slicing import (
    DivertorWireAndExteriorCurve,
    PanelsAndExteriorCurve,
)

if TYPE_CHECKING:
    from numpy import typing as npt

    from bluemira.base.reactor import ComponentManager
    from bluemira.geometry.wire import BluemiraWire
    from bluemira.neutronics.make_pre_cell import DivertorPreCellArray, PreCellArray
    from bluemira.neutronics.materials import NeutronicsMaterials


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


@dataclass
class PreCellStage:
    """Stage of making pre-cells"""

    blanket: PreCellArray
    divertor: DivertorPreCellArray

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


def some_function_on_blanket_wire(*_args):
    """DELETE ME"""
    # Loading data
    with open("data/inner_boundary") as j:
        deserialise_shape(json.load(j))
    with open("data/outer_boundary") as j:
        outer_boundary = deserialise_shape(json.load(j))
        # TODO: need to add method of scaling BluemiraWire (issue #3038 /
        # TODO: raise new issue about needing method to scale BluemiraWire)
    with open("data/divertor_face.correct.json") as j:
        divertor_bmwire = deserialise_shape(json.load(j))
    with open("data/vv_bndry_outer.json") as j:
        vacuum_vessel_bmwire = deserialise_shape(json.load(j))

    fw_panel_bp_list = [
        np.load("data/fw_panels_10_0.1.npy"),
        np.load("data/fw_panels_25_0.1.npy"),
        np.load("data/fw_panels_25_0.3.npy"),
        np.load("data/fw_panels_50_0.3.npy"),
        np.load("data/fw_panels_50_0.5.npy"),
    ]
    panel_breakpoint_t = fw_panel_bp_list[0].T
    # MANUAL FIX of the coordinates, because the data we're given is not perfect.
    panel_breakpoint_t[0] = vector_intersect(
        panel_breakpoint_t[0],
        panel_breakpoint_t[1],
        divertor_bmwire.edges[0].start_point()[::2].flatten(),
        divertor_bmwire.edges[0].end_point()[::2].flatten(),
    )
    panel_breakpoint_t[-1] = vector_intersect(
        panel_breakpoint_t[-2],
        panel_breakpoint_t[-1],
        divertor_bmwire.edges[-1].start_point()[::2].flatten(),
        divertor_bmwire.edges[-1].end_point()[::2].flatten(),
    )
    return panel_breakpoint_t, outer_boundary, divertor_bmwire, vacuum_vessel_bmwire


@dataclass
class NeutronicsReactorParameterFrame(ParameterFrame):
    """Neutronics reactor parameters"""

    inboard_fw_tk: Parameter[float]
    inboard_breeding_tk: Parameter[float]
    outboard_fw_tk: Parameter[float]
    outboard_breeding_tk: Parameter[float]
    blanket_io_cut: Parameter[float]
    tf_inner_radius: Parameter[float]
    tf_outer_radius: Parameter[float]
    divertor_surface_tk: Parameter[float]
    blanket_surface_tk: Parameter[float]
    blk_ib_manifold: Parameter[float]
    blk_ob_manifold: Parameter[float]


class NeutronicsReactor:
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
        self.tokamak_dimensions = TokamakDimensions.from_parameterframe(
            make_parameter_frame(params, self.param_cls)
        )
        self.material_library = materials_library

        divertor_wire, panel_points, blanket_wire, vacuum_vessel_wire = (
            self._get_wires_from_components(divertor, blanket, vacuum_vessel)
        )

        self.geom = ReactorGeometry(
            divertor_wire, panel_points, blanket_wire, vacuum_vessel_wire
        )

        self._pre_cell_stage = self._create_pre_cell_stage(
            blanket_discretisation, divertor_discretisation, snap_to_horizontal_angle
        )

    def _create_pre_cell_stage(
        self, blanket_discretisation, divertor_discretisation, snap_to_horizontal_angle
    ):
        first_point = self.geom.divertor_wire.edges[0].start_point()
        last_point = self.geom.divertor_wire.edges[-1].end_point()

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
        blanket = cutting.blanket.make_quadrilateral_pre_cell_array(
            discretisation_level=blanket_discretisation,
            starting_cut=first_point.xz.flatten(),
            ending_cut=last_point.xz.flatten(),
            snap_to_horizontal_angle=snap_to_horizontal_angle,
        )

        return PreCellStage(
            blanket=blanket.straighten_exterior(preserve_volume=True),
            divertor=cutting.divertor.make_divertor_pre_cell_array(
                discretisation_level=divertor_discretisation
            ),
        )

    @property
    def bounding_box(self) -> tuple[float, ...]:
        """Bounding box of Neutronics reactor"""
        return self._pre_cell_stage.bounding_box()

    @property
    def blanket(self):
        """Blanket pre cell"""
        return self._pre_cell_stage.blanket

    @property
    def divertor(self):
        """Divertor pre cell"""
        return self._pre_cell_stage.divertor

    @staticmethod
    def _get_wires_from_components(
        divertor: ComponentManager,
        blanket: ComponentManager,
        vacuum_vessel: ComponentManager,
    ) -> tuple[BluemiraWire, npt.NDArray, BluemiraWire, BluemiraWire]:
        panel_points, outer_boundary, divertor_wire, vacuum_vessel_wire = (
            some_function_on_blanket_wire(divertor, blanket, vacuum_vessel)
        )
        return divertor_wire, panel_points, outer_boundary, vacuum_vessel_wire
