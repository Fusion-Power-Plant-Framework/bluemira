# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Geometry for generalised neutronics
"""

from __future__ import annotations

from enum import Enum, auto

from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.look_and_feel import bluemira_print, bluemira_warn
from bluemira.base.reactor import ComponentManager
from bluemira.geometry.despliner import despline_xz_component
from bluemira.geometry.tools import (
    check_touching_geos,
    repair_gaps_between_faces,
    repair_overlapping_geos,
    revolve_shape,
)


class GeometryModel(Enum):
    """
    Enumeration of different geometry model to be
    considered for neutronics simulations
    """

    # In-house axisymmetric neutronics CSG maker
    BLUEMRIA_CSG = auto()

    # USER Specified / NOT IMPLEMENTED YET
    CUSTOM = auto()

    @classmethod
    def _missing_(cls, value: str):
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(
                f"{cls.__name__} has no type {value}. "
                f"Please select from {(*cls._member_names_,)}"
            ) from None


class NeutronicsGeometryManagers(ComponentManager):
    """
    Class containing all the Component Managers for neutronics.

    Parameters
    ----------
    All managers to be considered in the neutronics model.
    """

    @classmethod
    def from_list_of_components(
        cls,
        components: list[Component],
        discretisations: list[int],
        overlap_tolerance: float = 1e-10,
        gap_tolerance: float = 1e-2,
    ) -> NeutronicsGeometryManagers:
        """Create a NeutronicsGeometryManagers instance from component managers.

        Parameters
        ----------
        components
            All components to include in the neutronics geometry.
        discretisations
            Discretisation for each component manager.

        Returns
        -------
        NeutronicsGeometryManagers
            A NeutronicsGeometryManagers instance containing the
            desplined component managers.

        Raises
        ------
        ValueError
            If the number of discretisations does not match the number
            of component managers.
        """
        if len(discretisations) != len(components):
            raise ValueError(
                f"number of components {len(components)} "
                f"differs from provided number of discretisations "
                f"{len(discretisations)}"
            )
        all_orig_xyzs = [comp.get_component("xyz") for comp in components]
        # ---------------------------------------------------------
        # Desplining xz boundaries
        # * Assuming one child per component
        # If you have multiple children, unpack them before
        # running this
        # ---------------------------------------------------------
        bluemira_print("Desplining xz components.")
        all_desplined_comps = [
            despline_xz_component(
                comp,
                dscrt,
                fallback_to_existing_discretisation=True,
            )
            for comp, dscrt in zip(
                components,
                discretisations,
                strict=True,
            )
        ]
        # ---------------------------------------------------------
        # Check and fix overlaps.
        # * Assuming one child per component
        # If you have multiple children, unpack them before
        # running this
        # ---------------------------------------------------------
        bluemira_print("Checking for possible overlaps and fixing them.")
        updated_desplined_xzs = cls.fix_xz_overlaps_and_gaps(
            components,
            all_desplined_comps,
            overlap_tolerance=overlap_tolerance,
            gap_tolerance=gap_tolerance,
        )

        # Create XYZ components by revolution and add them as children.
        component_tree = Component("Neutronics Geometry")

        for i, desp_comp in enumerate(updated_desplined_xzs):
            xyz_shape = revolve_shape(
                desp_comp.get_component("xz").children[0].shape,
                base=(0, 0, 0),
                direction=(0, 0, 1),
                degree=360.0,
            )

            name = all_orig_xyzs[i].children[0].name
            material = all_orig_xyzs[i].get_component_properties("material")

            desp_comp.add_child(
                Component(
                    "xyz",
                    children=[
                        PhysicalComponent(
                            name=name,
                            shape=xyz_shape,
                            material=material,
                        )
                    ],
                )
            )
            component_tree.add_child(desp_comp)

        # Initiate and return a new instance of this class.
        # Each component should have one XZ and
        # one XYZ component for simplicity. No need to keep the extensive
        # component hierarchy from the original reactor.
        return cls(component_tree)

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
        original_comps: list[Component],
        desplined_comps: list[Component],
        overlap_tolerance: float = 1e-10,
        gap_tolerance: float = 1e-2,
    ) -> list[Component]:
        """
        Fix overlaps and gaps between XZ faces introduced by desplining,
        if any.

        Pairs that should touch are identified from the original geometry.
        Overlaps are then fixed one component at a time, with all overlapping
        neighbours repaired in a single operation. Finally, all pairs that
        should touch are sewn to repair any remaining gaps.

        Parameters
        ----------
        original_comps:
            Components containing the original geometries.
        desplined_comps:
            Components containing the desplined geometries.
        overlap_tolerance:
            Minimum intersection area considered to be an overlap.
        gap_tolerance:
            Maximum sewing tolerance for neighbouring faces that should touch.

        Returnslist[Component]
            Components with repaired XZ faces.

        Returns
        -------
        list[Component]

        Notes
        -----
        Assumes one child per XZ component.

        Overlaps are corrected with respect to the order of the components in
        the list. For each component, all overlapping faces are used in a
        single repair operation.
        """
        # ---------------------------------------------------------
        # Find pairs that should touch from the original geometry.
        # ---------------------------------------------------------
        touching_pairs = []

        for i, comp_a in enumerate(original_comps):
            xz_a_face = comp_a.get_component("xz").children[0].shape

            for j, comp_b in enumerate(
                original_comps[i + 1 :],
                start=i + 1,
            ):
                xz_b_face = comp_b.get_component("xz").children[0].shape

                if check_touching_geos(
                    xz_a_face,
                    xz_b_face,
                    overlap_tolerance,
                ):
                    touching_pairs.append((i, j))

        # ---------------------------------------------------------
        # Fix overlaps.
        #
        # For each component, collect all overlapping neighbours
        # that should touch it and fix them in one operation.
        # ---------------------------------------------------------
        for i, comp_a in enumerate(desplined_comps):
            xz_a = comp_a.get_component("xz")
            xz_a_face = xz_a.children[0].shape

            overlapping = []

            touching_indices = [j for pair_i, j in touching_pairs if pair_i == i]

            for j in touching_indices:
                comp_b = desplined_comps[j]
                xz_b_face = comp_b.get_component("xz").children[0].shape

                intersection = xz_a_face.shape.intersect(xz_b_face.shape)
                intersection_area = intersection.Area()

                if intersection_area > overlap_tolerance:
                    overlapping.append((j, comp_b.name, intersection_area))

            if overlapping:
                bluemira_warn(
                    f"{comp_a.name} overlaps "
                    f"{', '.join(name for _, name, _ in overlapping)} "
                    f"by areas "
                    f"{', '.join(f'{area} m^2' for _, _, area in overlapping)}. "
                    f"In the original geometry, they should touch instead. "
                    f"Running overlap fixing."
                )

                overlapping_faces = [
                    desplined_comps[j].get_component("xz").children[0].shape
                    for j, _, _ in overlapping
                ]

                xz_a.children[0].shape = repair_overlapping_geos(
                    xz_a_face,
                    overlapping_faces,
                )

        # ---------------------------------------------------------
        # Repair gaps.
        #
        # Sew every pair that should touch according to the
        # original geometry. Sewing is also safe when the pair
        # already has a matching boundary.
        # ---------------------------------------------------------
        for i, j in touching_pairs:
            xz_a = desplined_comps[i].get_component("xz")
            xz_b = desplined_comps[j].get_component("xz")

            repaired_a, repaired_b = repair_gaps_between_faces(
                [
                    xz_a.children[0].shape,
                    xz_b.children[0].shape,
                ],
                tolerance=gap_tolerance,
            )

            xz_a.children[0].shape = repaired_a
            xz_b.children[0].shape = repaired_b

        return desplined_comps
