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
from itertools import chain
from typing import TYPE_CHECKING

import numpy as np
import openmc

from bluemira.base.parameter_frame import ParameterFrame, make_parameter_frame
from bluemira.geometry.constants import D_TOLERANCE
from bluemira.geometry.coordinates import vector_intersect
from bluemira.geometry.tools import deserialise_shape
from bluemira.neutronics.constants import to_cm, to_cm3
from bluemira.neutronics.make_csg import (
    BlanketCellArray,
    BluemiraNeutronicsCSG,
    DivertorCellArray,
    flat_intersection,
    flat_union,
)
from bluemira.neutronics.make_materials import create_materials
from bluemira.neutronics.params import (
    BlanketType,
    OpenMCNeutronicsSolverParams,
    TokamakDimensions,
    get_preset_physical_properties,
)
from bluemira.neutronics.radial_wall import polygon_revolve_signed_volume
from bluemira.neutronics.slicing import (
    DivertorWireAndExteriorCurve,
    PanelsAndExteriorCurve,
)

if TYPE_CHECKING:
    from numpy import typing as npt

    from bluemira.base.reactor import ComponentManager
    from bluemira.geometry.wire import BluemiraWire
    from bluemira.neutronics.make_pre_cell import DivertorPreCellArray, PreCellArray


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


@dataclass
class CellStage:
    """Stage of making cells."""

    blanket: BlanketCellArray
    divertor: DivertorCellArray
    tf_coils: list[openmc.Cell]
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


def exterior_vertices(blanket, divertor) -> npt.NDArray:
    """
    Get the 3D coordinates of every point at the outer boundary of the tokamak's
    poloidal cross-section.

    Returns
    -------
    coordinates
        array of shape (N+1+n*M, 3), where N = number of blanket pre-cells,
        M = number of divertor pre-cells, n = discretisation_level used when chopping
        up the divertor in
        :meth:`bluemira.neutronics.DivertorWireAndExteriorCurve.make_divertor_pre_cell_array`
    """
    return np.concatenate([
        blanket.exterior_vertices(),
        divertor.exterior_vertices()[::-1],
    ])


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


def make_universe_box(
    csg, z_min: float, z_max: float, r_max: float, *, control_id: bool = False
):
    """Box up the universe in a cylinder (including top and bottom)."""
    bottom = csg.find_suitable_z_plane(
        z_min,
        boundary_type="vacuum",
        surface_id=999 if control_id else None,
        name="Universe bottom",
    )
    top = csg.find_suitable_z_plane(
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


def make_coils(
    csg,
    solenoid_radius: float,
    tf_coil_thick: float,
    z_min: float,
    z_max: float,
    material_library,
) -> tuple[openmc.Cell, list[openmc.Cell]]:
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
    top = csg.find_suitable_z_plane(
        z_max,
        [z_max - D_TOLERANCE, z_max + D_TOLERANCE],
        name="Top of central solenoid",
    )
    bottom = csg.find_suitable_z_plane(
        z_min,
        [z_min - D_TOLERANCE, z_min + D_TOLERANCE],
        name="Bottom of central solenoid",
    )
    central_solenoid = openmc.Cell(
        name="Central solenoid",
        fill=material_library.container_mat,
        region=+bottom & -top & -solenoid,
    )
    tf_coils = [
        openmc.Cell(
            name="TF coil (sheath around central solenoid)",
            fill=material_library.tf_coil_mat,
            region=+bottom & -top & +solenoid & -central_tf_coil,
        )
    ]
    central_solenoid.volume = (top.z0 - bottom.z0) * np.pi * solenoid.r**2
    tf_coils[0].volume = (
        (top.z0 - bottom.z0) * np.pi * (central_tf_coil.r**2 - solenoid.r**2)
    )
    return central_solenoid, tf_coils


def blanket_and_divertor_outer_region(
    csg, blanket, divertor, *, control_id: bool = False
) -> openmc.Region:
    """
    Get the entire tokamak's poloidal cross-section (everything inside
    self.geom.boundary) as an openmc.Region.
    """
    surfaces = [
        *blanket.exterior_surfaces(),
        *chain.from_iterable(divertor.exterior_surfaces()),
    ]
    return csg.region_from_surface_series(
        surfaces, exterior_vertices(blanket, divertor), control_id=control_id
    )


def plasma_void(csg, blanket, divertor, *, control_id: bool = False) -> openmc.Region:
    """Get the plasma chamber's poloidal cross-section"""
    blanket_interior_pts = blanket.interior_vertices()
    dividing_surface = csg.surface_from_2points(
        blanket_interior_pts[0][::2], blanket_interior_pts[-1][::2]
    )
    blanket_surfaces = [*blanket.interior_surfaces(), dividing_surface]
    plasma = csg.region_from_surface_series(
        blanket_surfaces, blanket_interior_pts, control_id=control_id
    )

    div_surfaces = [
        *chain.from_iterable(divertor.exterior_surfaces()),
        dividing_surface,
    ]
    exhaust_including_divertor = csg.region_from_surface_series(
        div_surfaces,
        divertor.exterior_vertices(),
        control_id=control_id,
    )

    divertor_zone = divertor.exclusion_zone(control_id=control_id)
    return flat_union([plasma, exhaust_including_divertor]) & ~divertor_zone


def make_void_cells(
    csg,
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
    full_tokamak_region = blanket_and_divertor_outer_region(
        csg, blanket, divertor, control_id=control_id
    )
    void_region = universe & ~full_tokamak_region
    if tf_coils:
        void_region &= ~tf_coils[0].region
    if central_solenoid:
        void_region &= ~central_solenoid.region

    return (
        openmc.Cell(
            region=plasma_void(csg, blanket, divertor, control_id=control_id),
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
    universe: openmc.Universe,
    tf_coils: list[openmc.Cell],
    central_solenoid: openmc.Cell,
    ext_void: openmc.Cell,
    blanket: BlanketCellArray,
    divertor: DivertorCellArray,
    plasma: openmc.Cell,
):
    """
    Sets the volume of the voids. Not necessary/ used anywhere yet.
    """
    ext_vertices = exterior_vertices(blanket, divertor)
    total_universe_volume = (
        #  top - bottom
        (universe[0].surface.z0 - universe[1].surface.z0)
        * np.pi
        * universe[2].surface.r ** 2  # cylinder
    )  # cm^3
    universe.volume = total_universe_volume

    outer_boundary_volume = to_cm3(polygon_revolve_signed_volume(ext_vertices[:, ::2]))
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


def make_cell_arrays(
    pre_cell_reactor: NeutronicsReactor,
    csg: BluemiraNeutronicsCSG,
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

    z_max, z_min, r_max, _r_min = pre_cell_reactor.bounding_box
    universe = make_universe_box(
        csg,
        z_min - D_TOLERANCE,
        z_max + D_TOLERANCE,
        r_max + D_TOLERANCE,
        control_id=control_id,
    )

    blanket = BlanketCellArray.from_pre_cell_array(
        pre_cell_reactor.blanket,
        pre_cell_reactor.material_library,
        pre_cell_reactor.tokamak_dimensions,
        csg,
        control_id=control_id,
    )

    # change the cell and surface id register before making the divertor.
    # (ids will only count up from here.)
    if control_id:
        round_up_next_openmc_ids()

    divertor = DivertorCellArray.from_pre_cell_array(
        pre_cell_reactor.divertor,
        pre_cell_reactor.material_library,
        pre_cell_reactor.tokamak_dimensions.divertor,
        csg=csg,
        override_start_end_surfaces=(blanket[0].ccw_surface, blanket[-1].cw_surface),
        # ID cannot be controlled at this point.
    )

    # make the plasma cell and the exterior void.
    if control_id:
        round_up_next_openmc_ids()

    cs, tf = make_coils(
        csg,
        pre_cell_reactor.tokamak_dimensions.central_solenoid.inner_diameter / 2,
        (
            (
                pre_cell_reactor.tokamak_dimensions.central_solenoid.outer_diameter
                - pre_cell_reactor.tokamak_dimensions.central_solenoid.inner_diameter
            )
            / 2
        ),
        z_min - D_TOLERANCE,
        z_max + D_TOLERANCE,
        pre_cell_reactor.material_library,
    )
    plasma, ext_void = make_void_cells(
        csg, tf, cs, universe, blanket, divertor, control_id=control_id
    )

    cell_array = CellStage(
        blanket=blanket,
        divertor=divertor,
        tf_coils=tf,
        central_solenoid=cs,
        plasma=plasma,
        ext_void=ext_void,
        universe=universe,
    )
    set_volumes(
        cell_array.universe,
        cell_array.tf_coils,
        cell_array.central_solenoid,
        cell_array.ext_void,
        cell_array.blanket,
        cell_array.divertor,
        cell_array.plasma,
    )

    return cell_array


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


class NeutronicsReactor:
    """Pre csg cell reactor"""

    param_cls = OpenMCNeutronicsSolverParams

    def __init__(
        self,
        params: dict | ParameterFrame,
        divertor: ComponentManager,
        blanket: ComponentManager,
        vacuum_vessel: ComponentManager,
        *,
        snap_to_horizontal_angle: float = 45,
        blanket_discretisation: int = 10,
        divertor_discretisation: int = 5,
    ):
        self.params = make_parameter_frame(params, self.param_cls)
        _breeder_materials, _tokamak_geometry = get_preset_physical_properties(
            BlanketType.HCPB  # blanket.blanket_type
        )

        self.tokamak_dimensions = TokamakDimensions.from_tokamak_geometry(
            _tokamak_geometry,
            self.params.major_radius.value,
            tf_inner_radius=2,
            tf_outer_radius=4,
            divertor_surface_tk=0.1,
            blanket_surface_tk=0.01,
            blk_ib_manifold=0.02,
            blk_ob_manifold=0.2,
        )

        self._material_library = create_materials(_breeder_materials)

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
    def material_library(self):
        """Reactor material library"""
        return self._material_library

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
