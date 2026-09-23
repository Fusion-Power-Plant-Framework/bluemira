# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Geometry for generalised neutronics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.look_and_feel import bluemira_print, bluemira_warn
from bluemira.base.reactor import ComponentManager
from bluemira.geometry.despliner import (
    despline_xz_component,
)
from bluemira.geometry.tools import (
    check_touching_geos,
    repair_gaps_between_faces,
    repair_overlapping_geos,
    revolve_shape,
)

if TYPE_CHECKING:
    from matproplib.material import Material


@dataclass
class NeutronicsComponent:
    """
    Store component data used during neutronics pre-processing.
    """

    name: str
    original_volume: float
    material: Material | None = None
    touching_components: list[str] | None = None
    desplined_xz_component: Component | None = None
    desplined_xyz_component: Component | None = None

    def revolve_and_create_xyz(self) -> Component:
        """
        Revolve the desplined xz geometry and create the corresponding xyz
        component.

        Returns
        -------
        Component
        """
        xyz_shape = revolve_shape(
            self.desplined_xz_component.children[0].shape,
            base=(0, 0, 0),
            direction=(0, 0, 1),
            degree=360.0,
        )

        return Component(
            "xyz",
            children=[
                PhysicalComponent(
                    name=self.name,
                    shape=xyz_shape,
                    material=self.material,
                )
            ],
        )


class NeutronicsGeometryManager(ComponentManager):
    """Manage components used to construct the neutronics geometry."""

    def __init__(
        self,
        component: Component,
        volume_differences: dict[str, float],
    ):
        super().__init__(component)
        self.volume_differences = volume_differences

    @staticmethod
    def check_touching(
        comp: Component,
        components: list[Component],
        overlap_tolerance: float,
    ) -> list[str]:
        """Return the names of components whose XZ faces touch ``comp``.

        ``comp`` is compared against the XZ face of each
        component in ``components``. Components with the same name as
        ``comp`` are excluded from the comparison.

        Parameters
        ----------
        comp
            Component whose XZ face is checked for touching.
        components
            Components whose XZ faces are checked against ``comp``.
        overlap_tolerance
            Tolerance used to determine whether two XZ face geometries
            are touching.

        Returns
        -------
        list[str]
            Names of components whose XZ faces touch the XZ face of ``comp``.
        """
        comp_xz_face = comp.get_component("xz").children[0].shape

        touching_components = []

        for other_comp in components:
            if other_comp.name == comp.name:
                continue

            other_xz_face = other_comp.get_component("xz").children[0].shape

            if check_touching_geos(
                comp_xz_face,
                other_xz_face,
                overlap_tolerance,
            ):
                touching_components.append(other_comp.name)

        return touching_components

    @classmethod
    def from_list_of_components(
        cls,
        components: list[Component],
        discretisations: list[int],
        overlap_tolerance: float = 1e-10,
        gap_tolerance: float = 1e-2,
    ) -> NeutronicsGeometryManager:
        """Create a neutronics geometry manager from components.

        Parameters
        ----------
        components
            Components to include in the neutronics geometry.
        discretisations
            Discretisation to use for each component.
        overlap_tolerance
            Tolerance used when checking whether XZ faces are touching.
        gap_tolerance
            Tolerance used when checking for gaps between components.

        Returns
        -------
        NeutronicsGeometryManager
            Manager containing the processed neutronics components.

        Raises
        ------
        ValueError
            If the number of discretisations does not match the number
            of components.
        """
        if len(discretisations) != len(components):
            raise ValueError(
                f"number of components {len(components)} "
                f"differs from provided number of discretisations "
                f"{len(discretisations)}"
            )

        # ---------------------------------------------------------
        # Desplining xz boundaries
        # * Assuming one child per component
        # If you have multiple children, unpack them before
        # running this
        # ---------------------------------------------------------
        all_neutronics_comps = []
        bluemira_print("Desplining xz components.")
        for comp, dscrt in zip(components, discretisations, strict=True):
            xyz_component = comp.get_component("xyz")
            xyz_shape = xyz_component.children[0].shape

            all_neutronics_comps.append(
                NeutronicsComponent(
                    name=comp.name,
                    original_volume=xyz_shape.volume,
                    touching_components=cls.check_touching(
                        comp,
                        components,
                        overlap_tolerance,
                    ),
                    desplined_xz_component=despline_xz_component(
                        comp,
                        dscrt,
                        fallback_to_existing_discretisation=True,
                    ).get_component("xz"),
                    material=xyz_component.children[0].material,
                )
            )

        # ---------------------------------------------------------
        # Check and fix overlaps.
        # * Assuming one child per component
        # If you have multiple children, unpack them before
        # running this
        # ---------------------------------------------------------
        bluemira_print("Checking for possible overlaps and fixing them.")
        cls.fix_xz_overlaps_and_gaps(
            all_neutronics_comps,
            overlap_tolerance=overlap_tolerance,
            gap_tolerance=gap_tolerance,
        )

        # ---------------------------------------------------------
        # Create XYZ components by revolution and assemble the
        # final component tree.
        # ---------------------------------------------------------
        component_tree = Component("Neutronics Geometry")
        volume_differences = {}

        for neutronics_comp in all_neutronics_comps:
            xyz_component = neutronics_comp.revolve_and_create_xyz()

            component_tree.add_child(
                Component(
                    neutronics_comp.name,
                    children=[
                        neutronics_comp.desplined_xz_component,
                        xyz_component,
                    ],
                )
            )

            # Calculate and store the relative volume difference.
            relative_volume_difference = (
                abs(
                    xyz_component.children[0].shape.volume
                    - neutronics_comp.original_volume
                )
                / neutronics_comp.original_volume
            )

            volume_differences[neutronics_comp.name] = relative_volume_difference

            bluemira_print(
                f"{neutronics_comp.name}: volume difference after pre-processing = "
                f"{relative_volume_difference:.4%}"
            )

        return cls(
            component_tree,
            volume_differences,
        )

    def get_all_components(self) -> list[Component]:
        """
        Return all component managers.

        Returns
        -------
        list[Component]
        """
        return self.component().children

    def inspect_xyz_overlaps(
        self,
        tolerance: float = 1e-10,
    ):
        """
        Inspect all xyz solids to ensure that there is no overlap
        Touching is allowed.

        Running on solids takes longer. Just use it for sanity
        checks.

        Parameters
        ----------
        tolerance
            Minimum intersection area considered to be an overlap.

        """
        all_comps = self.get_all_components()

        for i, comp_a in enumerate(all_comps):
            xyz_a = comp_a.get_component("xyz")
            name_a = comp_a.name
            xyz_a_solid = xyz_a.children[0].shape

            for _, comp_b in enumerate(
                all_comps[i + 1 :],
                start=i + 1,
            ):
                # assumes only one xz child
                name_b = comp_b.name
                xyz_b = comp_b.get_component("xyz")
                xyz_b_solid = xyz_b.children[0].shape
                intersection = xyz_a_solid.shape.intersect(xyz_b_solid.shape)
                intersection_vol = intersection.Volume()

                if intersection_vol > tolerance:
                    bluemira_warn(
                        f"Desplining created overlapping solids. "
                        f"{name_a} overlaps {name_b} by an volume of "
                        f"{intersection_vol} m^3"
                    )

    @staticmethod
    def fix_xz_overlaps_and_gaps(
        neutronics_comps: list[NeutronicsComponent],
        overlap_tolerance: float = 1e-10,
        gap_tolerance: float = 1e-2,
    ) -> None:
        """Fix overlaps and gaps between desplined XZ faces.

        Components that should touch are identified from the
        ``touching_components`` stored on each ``NeutronicsComponent``.

        Overlaps introduced during desplining are repaired first. For each
        component, all overlapping neighbours that should touch it are repaired
        in a single operation. Finally, gaps between touching components are
        repaired by sewing their XZ faces.

        Parameters
        ----------
        neutronics_comps
            Neutronics components containing the desplined XZ geometries and
            names of components they should touch.
        overlap_tolerance
            Minimum intersection area considered to be an overlap.
        gap_tolerance
            Maximum sewing tolerance for neighbouring XZ faces that should touch.

        Notes
        -----
        Assumes one child per XZ component.

        Overlaps are corrected with respect to the order of the components in
        ``neutronics_comps``. Each touching pair is processed once when repairing
        gaps.

        The ``desplined_xz_component`` geometries are updated in place.
        """
        comps_by_name = {
            neutronics_comp.name: neutronics_comp for neutronics_comp in neutronics_comps
        }

        # ---------------------------------------------------------
        # Fix overlaps.
        #
        # For each component, collect all overlapping neighbours
        # that should touch it and fix them in one operation.
        # ---------------------------------------------------------
        for neutronics_comp in neutronics_comps:
            comp_a = neutronics_comp.desplined_xz_component
            xz_a = comp_a.get_component("xz")
            xz_a_face = xz_a.children[0].shape

            overlapping = []

            for touching_name in neutronics_comp.touching_components or []:
                comp_b = comps_by_name[touching_name].desplined_xz_component
                xz_b_face = comp_b.get_component("xz").children[0].shape

                intersection = xz_a_face.shape.intersect(xz_b_face.shape)
                intersection_area = intersection.Area()

                if intersection_area > overlap_tolerance:
                    overlapping.append((touching_name, xz_b_face, intersection_area))

            if overlapping:
                bluemira_warn(
                    f"{neutronics_comp.name} overlaps "
                    f"{', '.join(name for name, _, _ in overlapping)} "
                    f"by areas "
                    f"{', '.join(f'{area} m^2' for _, _, area in overlapping)}. "
                    f"In the original geometry, they should touch instead. "
                    f"Running overlap fixing."
                )

                xz_a.children[0].shape = repair_overlapping_geos(
                    xz_a_face,
                    [face for _, face, _ in overlapping],
                )

        # ---------------------------------------------------------
        # Repair gaps.
        #
        # Sew every pair that should touch according to the
        # original geometry. Each pair is processed only once.
        # ---------------------------------------------------------

        # Track processed pairs to prevent repairing the same touching pair twice.
        repaired_pairs: set[frozenset[str]] = set()

        for neutronics_comp in neutronics_comps:
            # Note: The list iteration order determines which
            # comp gets processed first
            comp_a = neutronics_comp.desplined_xz_component

            for touching_name in neutronics_comp.touching_components:
                pair = frozenset((neutronics_comp.name, touching_name))

                if pair in repaired_pairs:
                    continue

                comp_b = comps_by_name[touching_name].desplined_xz_component

                xz_a = comp_a.get_component("xz")
                xz_b = comp_b.get_component("xz")

                repaired_a, repaired_b = repair_gaps_between_faces(
                    [
                        xz_a.children[0].shape,
                        xz_b.children[0].shape,
                    ],
                    tolerance=gap_tolerance,
                )

                xz_a.children[0].shape = repaired_a
                xz_b.children[0].shape = repaired_b

                repaired_pairs.add(pair)
