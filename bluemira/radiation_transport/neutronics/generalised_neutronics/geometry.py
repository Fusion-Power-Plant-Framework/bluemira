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
from bluemira.geometry.despliner import create_desplined_xz_component
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


def get_all_geo(
    components: list[Component],
    dims: str = "xyz",
) -> list[tuple[Component, str]]:
    """
    Get all XYZ solids / XZ faces and their parent component names.

    Parameters
    ----------
    components
        Components to inspect.
    dims
        Geometry dimension to retrieve, e.g. ``"xyz"`` or ``"xz"``.

    Returns
    -------
    list[tuple[Component, str]]
        Geometry component and the name of its parent component.
    """
    all_geo = []
    for component in components:
        comps = component.get_component(dims, first=False)
        all_geo.extend((comp, comp.parent.name) for comp in comps)
    return all_geo


class NeutronicsGeometryManagers(ComponentManager):
    """
    Class containing all the Component Managers for neutronics.

    Parameters
    ----------
    All managers to be considered in the neutronics model.
    """

    @classmethod
    def from_component_managers(
        cls,
        component_managers: list[ComponentManager],
        discretisations: list[int],
    ) -> NeutronicsGeometryManagers:
        """Create a NeutronicsGeometryManagers instance from component managers.

        Parameters
        ----------
        component_managers
            Component managers to include in the neutronics geometry.
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
        if len(discretisations) != len(component_managers):
            raise ValueError(
                f"number of components {len(component_managers)} "
                f"differs from provided number of discretisations "
                f"{len(discretisations)}"
            )

        # Retrieve all original XZ and XYZ components.
        all_orig_components = [manager.component() for manager in component_managers]
        all_orig_xzs = get_all_geo(all_orig_components, "xz")
        all_orig_xyzs = get_all_geo(all_orig_components, "xyz")

        # Assign the discretisation of each parent component to each XZ.
        all_dscrt = []
        for component, dscrt in zip(
            all_orig_components,
            discretisations,
            strict=True,
        ):
            all_dscrt.extend(dscrt for _ in component.get_component("xz", first=False))

        # First perform desplining.
        all_desplined_comps = [
            (
                create_desplined_xz_component(
                    xz,
                    dscrt,
                    fallback_to_existing_discretisation=True,
                ),
                parent_name,
            )
            for (xz, parent_name), dscrt in zip(
                all_orig_xzs,
                all_dscrt,
                strict=True,
            )
        ]

        # Check and fix overlaps.
        bluemira_print("Checking for possible overlaps and fixing them.")
        updated_desplined_xzs = cls.inspect_fix_xz_overlaps(
            all_orig_xzs,
            all_desplined_comps,
        )

        # Create XYZ components by revolution and add them as children.
        component_tree = Component("Neutronics Geometry")

        for i, (desp_comp, parent_name) in enumerate(updated_desplined_xzs):
            xyz_shape = revolve_shape(
                desp_comp.get_component("xz").children[0].shape,
                base=(0, 0, 0),
                direction=(0, 0, 1),
                degree=360.0,
            )

            name = all_orig_xyzs[i][0].children[0].name
            material = all_orig_xyzs[i][0].get_component_properties("material")

            desp_parent_component = Component(parent_name)
            desp_parent_component.add_child(desp_comp)
            desp_parent_component.add_child(
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

            component_tree.add_child(desp_parent_component)

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
        original_xz_comps: list[tuple[Component, str]],
        desplined_xz_comps: list[tuple[Component, str]],
        tolerance: float = 1e-10,
    ) -> list[tuple[Component, str]]:
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
        list[tuple[Component, str]]
            Fixed XZ components.

        Raises
        ------
        ValueError
            If desplining creates overlapping faces which were not
            supposed to even touch in the original geometry.
        """
        for i in range(len(desplined_xz_comps)):
            xz_a, name_a = desplined_xz_comps[i]
            xz_a_face = xz_a.children[0].shape
            overlapping = []

            for j, (xz_b, name_b) in enumerate(
                desplined_xz_comps[i + 1 :],
                start=i + 1,
            ):
                # assumes only one xz child
                xz_b_face = xz_b.children[0].shape
                intersection = xz_a_face.shape.intersect(xz_b_face.shape)
                intersection_area = intersection.Area()

                if intersection_area > tolerance:
                    # Check if the original XZ faces were supposed to touch.
                    orig_a = next(
                        geo for geo, name in original_xz_comps if name == name_a
                    )
                    orig_b = next(
                        geo for geo, name in original_xz_comps if name == name_b
                    )

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
                    xz_b, _ = desplined_xz_comps[j]
                    xz_b_face = xz_b.children[0].shape
                    try:
                        xz_a, xz_b = repair_overlapping_geos(
                            xz_a_face,
                            xz_b_face,
                        )

                        desplined_xz_comps[j][0].children[0].shape = xz_b_face

                    except GeometryError as e:
                        bluemira_warn(
                            f"Could not repair overlap between "
                            f"{name_a} and {name_b}: {e}"
                        )

                desplined_xz_comps[i][0].children[0].shape = xz_a

        return desplined_xz_comps
