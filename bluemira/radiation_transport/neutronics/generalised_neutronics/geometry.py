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
from bluemira.geometry.error import GeometryError
from bluemira.geometry.tools import (
    check_touching_geos,
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
        updated_desplined_xzs = cls.inspect_fix_xz_overlaps(
            components,
            all_desplined_comps,
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

    @staticmethod
    def inspect_fix_xz_overlaps(
        original_comps: list[Component],
        desplined_comps: list[Component],
        tolerance: float = 1e-10,
    ) -> list[Component]:
        """
        Inspect all XZ faces to ensure that there is no overlap
        between any two faces. Touching is allowed.

        Not running on solids as that takes longer.

        Parameters
        ----------
        tolerance
            Minimum intersection area considered to be an overlap.

        Returns
        -------
        list[Component]
            Fixed XZ components.

        Raises
        ------
        ValueError
            If desplining creates overlapping faces which were not
            supposed to even touch in the original geometry.
        """
        for i, comp_a in enumerate(desplined_comps):
            xz_a = comp_a.get_component("xz")
            name_a = comp_a.name
            xz_a_face = xz_a.children[0].shape
            overlapping = []

            for j, comp_b in enumerate(
                desplined_comps[i + 1 :],
                start=i + 1,
            ):
                # assumes only one xz child
                name_b = comp_b.name
                xz_b = comp_b.get_component("xz")
                xz_b_face = xz_b.children[0].shape
                intersection = xz_a_face.shape.intersect(xz_b_face.shape)
                intersection_area = intersection.Area()

                if intersection_area > tolerance:
                    # Check if the original XZ faces were supposed to touch.
                    orig_a = next(
                        geo for geo in original_comps if geo.name == name_a
                    ).children[0]
                    orig_b = next(
                        geo for geo in original_comps if geo.name == name_b
                    ).children[0]

                    should_touch = check_touching_geos(
                        orig_a.children[0].shape,
                        orig_b.children[0].shape,
                        tolerance,
                    )

                    if not should_touch:
                        raise ValueError(
                            f"Desplining created overlapping faces. "
                            f"{name_a} overlaps {name_b} by an area of "
                            f"{intersection_area} m^2. In original geometry, "
                            f"they should not even touch. Please increase "
                            f"the discretisations to avoid overlaps in "
                            f"rebuilt geometry."
                        )

                    overlapping.append((j, name_b, intersection_area))

            if overlapping:
                bluemira_warn(
                    f"{name_a} overlaps "
                    f"{', '.join(name for _, name, _ in overlapping)} "
                    f"by areas "
                    f"{', '.join(f'{area} m^2' for _, _, area in overlapping)}. "
                    f"In original geometry, they should touch instead. "
                    f"Running overlap fixing."
                )

                for j, name_b, _ in overlapping:
                    xz_b = desplined_comps[j].get_component("xz")
                    xz_b_face = xz_b.children[0].shape
                    try:
                        xz_a, xz_b = repair_overlapping_geos(
                            xz_a_face,
                            xz_b_face,
                        )

                        desplined_comps[j].get_component("xz").children[
                            0
                        ].shape = xz_b_face

                    except GeometryError as e:
                        bluemira_warn(
                            f"Could not repair overlap between "
                            f"{name_a} and {name_b}: {e}"
                        )

                desplined_comps[i].get_component("xz").children[0].shape = xz_a

        return desplined_comps
