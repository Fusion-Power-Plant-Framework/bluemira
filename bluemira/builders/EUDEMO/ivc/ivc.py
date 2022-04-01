# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.
"""
Builders for the first wall of the reactor, including divertor
"""

from copy import deepcopy
from typing import Any, Dict, Tuple

import numpy as np

from bluemira.base.builder import BuildConfig
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.error import BuilderError
from bluemira.builders.EUDEMO.ivc.blanket import BlanketThicknessBuilder
from bluemira.builders.EUDEMO.ivc.divertor import DivertorSilhouetteBuilder
from bluemira.builders.EUDEMO.ivc.wall import WallBuilder
from bluemira.builders.shapes import Builder
from bluemira.equilibria import Equilibrium
from bluemira.equilibria.find import find_OX_points, get_legs
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import (
    boolean_cut,
    convex_hull_wires_2d,
    make_polygon,
    offset_wire,
)
from bluemira.geometry.wire import BluemiraWire


def _cut_wall_below_x_point(shape: BluemiraWire, x_point_z: float) -> BluemiraWire:
    """
    Remove the parts of the wire below the given value in the z-axis.
    """
    # Create a box that surrounds the wall below the given z
    # coordinate, then perform a boolean cut to remove that portion
    # of the wall's shape.
    bounding_box = shape.bounding_box
    cut_box_points = np.array(
        [
            [bounding_box.x_min, 0, bounding_box.z_min],
            [bounding_box.x_min, 0, x_point_z],
            [bounding_box.x_max, 0, x_point_z],
            [bounding_box.x_max, 0, bounding_box.z_min],
            [bounding_box.x_min, 0, bounding_box.z_min],
        ]
    )
    cut_zone = make_polygon(cut_box_points, label="_shape_cut_exclusion")
    # For a single-null, we expect three 'pieces' from the cut: the
    # upper wall shape and the two separatrix legs
    pieces = boolean_cut(shape, [cut_zone])

    wall_piece = pieces[np.argmax([p.center_of_mass[2] for p in pieces])]
    if wall_piece.center_of_mass[2] < x_point_z:
        raise ValueError(
            "Could not cut wall shape below x-point. "
            "No parts of the wall found above x-point."
        )
    return wall_piece


class InVesselComponentBuilder(Builder):
    """
    Build a first wall with a divertor.

    This class runs the builders for the wall shape and the divertor,
    then combines the two.

    For a single-null plasma, the builder outputs a Component with the
    structure:

    .. code-block::

        in_vessel_components (Component)
        └── xz (Component)
            └── wall (Component)
                └── wall_boundary (PhysicalComponent)
            └── blanket (Component)
                └── blanket_boundary (PhysicalComponent)
            └── divertor (Component)
                ├── inner_target (PhysicalComponent)
                ├── outer_target (PhysicalComponent)
                ├── dome (PhysicalComponent)
                ├── inner_baffle (PhysicalComponent)
                └── outer_baffle (PhysicalComponent)
    """

    _required_params = [
        # Wall
        "plasma_type",
        "R_0",  # major radius
        "kappa_95",  # 95th percentile plasma elongation
        "r_fw_ib_in",  # inboard first wall inner radius
        "r_fw_ob_in",  # inboard first wall outer radius
        # Wall flux and geometric offsets
        "fw_psi_n",  # Normalised psi boundary to fit FW to
        "tk_sol_ib",  # Inboard SOL thickness (used as a geometric offset to LCFS)
        "A",  # aspect ratio
        # Divertor silhouette
        "div_L2D_ib",  # Inboard leg length
        "div_L2D_ob",  # Outboard leg length
        "div_Ltarg",  # Target length
        "div_open",  # Divertor open/closed configuration
        # Blanket thickness
        "tk_bb_ib",  # Inboard blanket thickness
        "tk_bb_ob",  # Outboard blanket thickness
    ]

    COMPONENT_DIVERTOR = "divertor"
    COMPONENT_WALL = "wall"
    COMPONENT_BLANKET = "blanket"

    def __init__(
        self,
        params: Dict[str, Any],
        build_config: BuildConfig,
        equilibrium: Equilibrium,
        **kwargs,
    ):
        super().__init__(params, build_config, **kwargs)

        self._build_config = build_config
        self.equilibrium = equilibrium
        self.o_points, self.x_points = find_OX_points(
            self.equilibrium.x, self.equilibrium.z, self.equilibrium.psi()
        )

    def reinitialise(self, params, **kwargs) -> None:
        """
        Initialise the state of this builder ready for a new run.
        """
        return super().reinitialise(params, **kwargs)

    def mock(self):
        """
        Create a basic shape for the wall's boundary.
        """
        pass

    def run(self):
        """Run the builder design problem."""
        pass

    def build(self) -> Component:
        """
        Build the component.
        """
        wall = self._build_wall()
        closed_wall_shape = wall.get_component(WallBuilder.COMPONENT_WALL_BOUNDARY).shape
        cut_wall, cut_wall_wire = self._cut_wall(wall)

        blanket = self._build_blanket_thickness(closed_wall_shape)
        divertor = self._build_divertor(cut_wall_wire)

        first_wall = Component(self.name)
        first_wall.add_child(self._build_xz(cut_wall, blanket, divertor))
        return first_wall

    def _build_wall(self) -> Component:
        """Build the first wall component."""
        build_config = deepcopy(self._build_config)
        build_config.update({"name": self.COMPONENT_WALL})
        keep_out_zone = self._make_wall_keep_out_zone()
        builder = WallBuilder(
            self.params, build_config=build_config, keep_out_zone=keep_out_zone
        )
        return builder()

    def _build_blanket_thickness(self, wall_shape) -> Component:
        """
        Build the blanket thickness component.

        The input wall shape must be closed, i.e., do not cut the wall
        below the x-point before passing it into this function.
        """
        build_config = deepcopy(self._build_config)
        build_config.update({"name": self.COMPONENT_BLANKET})
        build_config.pop("runmode", None)
        blanket_builder = BlanketThicknessBuilder(
            self.params,
            build_config,
            wall_shape,
            self.x_points[0].z,
        )
        return blanket_builder()

    def _build_divertor(self, wall: BluemiraWire) -> Component:
        """Build the divertor silhouette component tree."""
        build_config = deepcopy(self._build_config)
        build_config.update({"name": self.COMPONENT_DIVERTOR})
        build_config.pop("runmode", None)

        x_lims = (wall.start_point().x[0], wall.end_point().x[0])
        z_lims = (wall.start_point().z[0], wall.end_point().z[0])
        builder = DivertorSilhouetteBuilder(
            self.params,
            build_config,
            equilibrium=self.equilibrium,
            x_limits=x_lims,
            z_limits=z_lims,
        )
        return builder()

    def _build_xz(self, wall, blanket, divertor) -> Component:
        """
        Build the component tree in the xz-plane.
        """
        parent_component = Component("xz")
        wall_xz = wall.get_component("xz")
        Component(
            self.COMPONENT_WALL,
            parent=parent_component,
            children=list(wall_xz.children),
        )
        blanket_xz_component = blanket.get_component("xz")
        Component(
            self.COMPONENT_BLANKET,
            parent=parent_component,
            children=list(blanket_xz_component.children),
        )
        divertor_xz_component = divertor.get_component("xz")
        Component(
            self.COMPONENT_DIVERTOR,
            parent=parent_component,
            children=list(divertor_xz_component.children),
        )
        return parent_component

    def _cut_wall(self, wall: Component) -> Tuple[Component, BluemiraWire]:
        """Cut the given wall component below the equilibrium's x-point."""
        # Cut wall below x-point in xz, a divertor will be put in the
        # space
        wall_xz = wall.get_component("xz")
        wall_boundary = wall_xz.get_component(WallBuilder.COMPONENT_WALL_BOUNDARY)
        x_point_z = self.x_points[0].z
        if wall_boundary.shape.bounding_box.z_min >= x_point_z:
            raise BuilderError(
                "First wall boundary does not inclose separatrix x-point."
            )
        cut_shape = _cut_wall_below_x_point(wall_boundary.shape, x_point_z)

        # Replace the "uncut" wall boundary with the new shape
        wall_xz.prune_child(WallBuilder.COMPONENT_WALL_BOUNDARY)
        wall_xz.add_child(
            PhysicalComponent(WallBuilder.COMPONENT_WALL_BOUNDARY, cut_shape)
        )
        return wall, cut_shape

    def _make_wall_keep_out_zone(self) -> BluemiraWire:
        """
        Create a "keep-out zone" to be used as a constraint in the
        wall shape optimiser.
        """
        geom_offset = self._params.tk_sol_ib.value
        psi_n = self._params.fw_psi_n.value
        geom_offset = 0.2  # TODO: Unpin
        psi_n = 1.05  # TODO: Unpin
        geom_offset_zone = self._make_geometric_keep_out_zone(geom_offset)
        flux_surface_zone = self._make_flux_surface_keep_out_zone(psi_n)
        leg_zone = self._make_divertor_leg_keep_out_zone(
            self._params.div_L2D_ib.value, self._params.div_L2D_ob.value
        )
        return convex_hull_wires_2d(
            [geom_offset_zone, flux_surface_zone, leg_zone], ndiscr=200, plane="xz"
        )

    def _make_geometric_keep_out_zone(self, offset: float) -> BluemiraWire:
        """
        Make a "keep-out zone" from a geometric offset of the LCFS.
        """
        lcfs = make_polygon(self.equilibrium.get_LCFS().xyz, closed=True)
        return offset_wire(lcfs, offset, join="arc")

    def _make_flux_surface_keep_out_zone(self, psi_n: float) -> BluemiraWire:
        """
        Make a "keep-out zone" from an equilibrium's flux surface.
        """
        flux_surface_zone = self.equilibrium.get_flux_surface(psi_n)
        # Chop the flux surface to only take the upper half
        indices = flux_surface_zone.z >= self.o_points[0][1]
        flux_surface_zone = make_polygon(flux_surface_zone.xyz[:, indices], closed=True)
        return flux_surface_zone

    def _make_divertor_leg_keep_out_zone(
        self, leg_length_ib_2D, leg_length_ob_2D
    ) -> BluemiraWire:
        """
        Make a "keep-out zone" from an equilibrium's divertor legs
        """
        legs = get_legs(self.equilibrium, n_layers=1, dx_off=0.0)
        ib_leg = make_polygon(legs["lower_inner"][0].xyz)
        ob_leg = make_polygon(legs["lower_outer"][0].xyz)

        p_ib = ib_leg.value_at(distance=leg_length_ib_2D)

        p_ob = ob_leg.value_at(distance=leg_length_ob_2D)

        return make_polygon([p_ib, p_ob], closed=False)


def build_ivc_xz_shapes(
    components: Component, rm_clearance: float
) -> Tuple[BluemiraFace, BluemiraFace, BluemiraWire]:
    """
    Build in-vessel component shapes in the xz-plane.

    This function is intended to be used to create shapes from the
    component tree outputted by InVesselComponentBuilder.

    Takes the blanket boundary, creates a face from it, then cuts away
    the plasma-facing (first wall) shape. It then cuts a remote
    maintainance clearance between the divertor and wall and returns
    the two shapes, along with the wire defining the boundary of the
    blanket.

    Parameters
    ----------
    components: Component
        A component tree outputted by InVesselComponentBuilder.build().
    rm_clearance: float
        The thickness of the clearance between the blanket and the
        divertor.

    Returns
    -------
    blanket_face: BluemiraFace
        Face representing the space for the blankets.
    divertor_face: BluemiraFace
        Face representing the space for the divertor.
    blanket_boundary: BlurmiraWire
        The shape of the outer boundary of the blanket.
    """
    # Make the in-vessel "shell"
    blanket_boundary = _extract_wire(
        components, BlanketThicknessBuilder.COMPONENT_BOUNDARY
    )
    plasma_facing_wire = _build_plasma_facing_wire(components)
    in_vessel_face = BluemiraFace([blanket_boundary, plasma_facing_wire])

    # Cut a clearance between the blankets and divertor - getting two
    # new faces
    vessel_bbox = in_vessel_face.bounding_box
    rm_clearance_face = _make_clearance_face(
        vessel_bbox.x_min,
        vessel_bbox.x_max,
        _extract_wire(
            components, WallBuilder.COMPONENT_WALL_BOUNDARY
        ).bounding_box.z_min,
        rm_clearance,
    )
    blanket_face, divertor_face = _cut_vessel_shape(in_vessel_face, rm_clearance_face)
    return blanket_face, divertor_face, blanket_boundary


def _build_plasma_facing_wire(components: Component) -> BluemiraWire:
    """
    Build a wire of the plasma facing shape of the first wall
    """
    # Note, the order of these lables matters. We need to go
    # anti-clockwise around the shape in order to get our closed wire
    labels = [
        WallBuilder.COMPONENT_WALL_BOUNDARY,
        DivertorSilhouetteBuilder.COMPONENT_INNER_BAFFLE,
        DivertorSilhouetteBuilder.COMPONENT_INNER_TARGET,
        DivertorSilhouetteBuilder.COMPONENT_DOME,
        DivertorSilhouetteBuilder.COMPONENT_OUTER_TARGET,
        DivertorSilhouetteBuilder.COMPONENT_OUTER_BAFFLE,
    ]
    wires = [_extract_wire(components, label) for label in labels]
    return BluemiraWire(wires)


def _cut_vessel_shape(
    in_vessel_face: BluemiraFace, rm_clearance_face: BluemiraFace
) -> Tuple[BluemiraFace, BluemiraFace]:
    """
    Cut a remote maintainance clearance into the given vessel shape.
    """
    pieces = boolean_cut(in_vessel_face, [rm_clearance_face])
    blanket_face = pieces[np.argmax([p.center_of_mass[2] for p in pieces])]
    divertor_face = pieces[np.argmin([p.center_of_mass[2] for p in pieces])]
    return blanket_face, divertor_face


def _make_clearance_face(x_min: float, x_max: float, z: float, thickness: float):
    """
    Makes a rectangular face in xz with the given thickness in z.

    The face is intended to be used to cut a remote maintainance
    clearance between blankets and divertor.
    """
    x_coords = [x_min, x_min, x_max, x_max]
    y_coords = [0, 0, 0, 0]
    z_coords = [
        z + thickness / 2,
        z - thickness / 2,
        z - thickness / 2,
        z + thickness / 2,
    ]
    return BluemiraFace(make_polygon([x_coords, y_coords, z_coords], closed=True))


def _extract_wire(component_tree: Component, label: str) -> BluemiraWire:
    """
    Extract the wire from the component with the given label.

    Throw if there is not exactly one component with the given label.
    """
    component = component_tree.get_component(label)
    if not component:
        raise ValueError(f"No component '{label}' in component tree.")
    if isinstance(component, list):
        raise ValueError(f"More than one component with label '{label}'.")
    return component.shape
