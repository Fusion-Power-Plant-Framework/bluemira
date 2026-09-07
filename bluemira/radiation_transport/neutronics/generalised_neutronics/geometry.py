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

import cadquery as cq

from bluemira.base.components import Component
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.reactor import ComponentManager
from bluemira.materials.error import MaterialsError


class GeometryModel(Enum):
    """
    Enumeration of different geometry model to be
    considered for neutronics simulations
    """

    # In-house axissymmetric neutronics CSG maker
    BLUEMRIA_CSG = auto()
    # USER Specificd / NOT IMPLEMENTED YET
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
    Class containing all the Component Managers for neutronics

    Parameters
    ----------
    All managers to be considered in the neutronics model

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

        component_tree = Component("Neutronics Geometry")

        for manager, discretisation in zip(
            component_managers,
            discretisations,
            strict=True,
        ):
            component_tree.add_child(
                manager.get_desplined_component_tree(
                    discretisation=discretisation,
                )
            )

        geom_managers = cls(component_tree)
        geom_managers.inspect_overlaps()
        geom_managers.inspect_materials()

        return geom_managers

    def get_all_components(self) -> list[ComponentManager]:
        """
        Return all component managers.

        Returns
        -------
        list[ComponentManager]
        """
        return self.component().children

    def inspect_overlaps(self, tolerance: float = 1e-10) -> dict:
        """
        Inspect all managers to ensure that there is no overlap
        between any two CadQuery solids. Touching is allowed.

        Parameters
        ----------
        tolerance
            Minimum intersection volume considered to be an overlap.

        Returns
        -------
        dict
            Mapping from ``(component_name, solid_name)`` to a list of
            ``(component_name, solid_name, solid)`` tuples for all solids
            that overlap it.

        Raises
        ------
        TypeError
            If a component does not contain a CadQuery Solid.
        """
        all_solids = []

        for comp in self.get_all_components():
            component_name = comp.name
            xyzs = comp.get_component("xyz", first=False)

            for xyz in xyzs:
                for child in xyz.children:
                    if not isinstance(child.shape.shape, cq.Solid):
                        raise TypeError(
                            "inspect_overlaps is only available for "
                            f"CadQuery Solid objects. "
                            f"{component_name} contains "
                            f"{type(child.shape).__name__}."
                        )

                    all_solids.append((component_name, child.name, child.shape))

        overlaps = {}

        for i, (component_a, name_a, solid_a) in enumerate(all_solids):
            overlapping_solids = []

            for component_b, name_b, solid_b in all_solids[i + 1 :]:
                intersection = solid_a.shape.intersect(solid_b.shape)

                if intersection.Volume() > tolerance:
                    overlapping_solids.append((component_b, name_b))

                    bluemira_warn(
                        f"({component_a}) {name_a} overlaps ({component_b}) {name_b} "
                        f"by a volume {intersection.Volume()} m^3. Please increase "
                        f"the discretisations to avoid overlaps in rebuilt solids."
                    )

            if overlapping_solids:
                overlaps[component_a, name_a] = overlapping_solids

        return overlaps

    def inspect_materials(self):
        """
        Inspect all managers to ensure that they are assigned
        a material

        Raises
        ------
        MaterialsError
            If a component does not have a material assigned
        """
        for comp in self.get_all_components():
            for xyz in comp.get_component("xyz", first=False):
                if xyz.get_component_properties("material") is None:
                    raise MaterialsError(
                        f"Component manager '{comp.name}' does not have"
                        " a material assigned."
                    )

    # def imprint_and_replace(self, tolerance: float = 1e-10):
    #     """

    #     """
    #     overlaps = self.find_overlaps(tolerance)

    #     def _find_physical_component(component, name):
    #         """Find the PhysicalComponent with ``name`` under an ``xyz`` component."""
    #         xyzs = component.get_component("xyz", first=False)

    #         for xyz in xyzs:
    #             for child in xyz.children:
    #                 if isinstance(child, PhysicalComponent) and child.name == name:
    #                     return child

    #     for (component_name_1, name_1), overlapping_components in overlaps.items():

    #         parent_1 = self.component().get_component(component_name_1)
    #         component_1 = _find_physical_component(parent_1, name_1)

    #         for component_name_2, name_2 in overlapping_components:
    #             parent_2 = self.component().get_component(component_name_2)
    #             component_2 = _find_physical_component(parent_2, name_2)

    #             print(component_1)
    #             print(component_2)
    #             print("")
