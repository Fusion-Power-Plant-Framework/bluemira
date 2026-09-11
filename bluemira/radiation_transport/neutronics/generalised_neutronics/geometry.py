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
from typing import TYPE_CHECKING

from bluemira.base.components import Component
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.reactor import ComponentManager
from bluemira.geometry.tools import check_touching_solids
from bluemira.materials.error import MaterialsError

if TYPE_CHECKING:
    from bluemira.geometry.solid import BluemiraSolid


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
        # geom_managers.inspect_overlaps()
        geom_managers.inspect_materials()

        return geom_managers

    def get_all_components(self) -> list[Component]:
        """
        Return all component managers.

        Returns
        -------
        list[ComponentManager]
        """
        return self.component().children

    @staticmethod
    def _get_all_solids(
        components: list[Component],
    ) -> list[tuple[BluemiraSolid, list[str]]]:
        """
        Get all XYZ solids and their full component hierarchies.

        Returns
        -------
        list[tuple[BluemiraSolid, list[str]]]
        """
        all_solids = []

        for component in components:
            xyz_comps = component.get_component("xyz", first=False)

            for xyz in xyz_comps:
                hierarchy = []
                current = xyz

                while current is not None:
                    hierarchy.append(current.name)
                    current = current.parent

                all_solids.append((xyz.children[0].shape, hierarchy))

        return all_solids

    def inspect_overlaps(
        self, original_comp_managers: list[ComponentManager], tolerance: float = 1e-10
    ):
        """
        Inspect all managers to ensure that there is no overlap
        between any two CadQuery solids. Touching is allowed.

        Parameters
        ----------
        tolerance
            Minimum intersection volume considered to be an overlap.

        Raises
        ------
        ValueError
            If desplining Created Overlapping solids which
            are not supposed to even touch in the original
            geometry
        """
        all_solids = self._get_all_solids(self.get_all_components())
        all_orig_solids = self._get_all_solids([
            manager.component() for manager in original_comp_managers
        ])

        for i, (solid_a, hierarchy_a) in enumerate(all_solids):
            for solid_b, hierarchy_b in all_solids[i + 1 :]:
                intersection = solid_a.shape.intersect(solid_b.shape)

                if intersection.Volume() > tolerance:
                    # check if the original solids were supposed to touch
                    # Note: rebuilt geometry has "Neutronics Geometry" appended
                    # to the hierarchy
                    orig_a = next(
                        solid
                        for solid, hierarchy in all_orig_solids
                        if hierarchy == hierarchy_a[:-1]
                    )
                    orig_b = next(
                        solid
                        for solid, hierarchy in all_orig_solids
                        if hierarchy == hierarchy_b[:-1]
                    )
                    should_touch = check_touching_solids(orig_a, orig_b, tolerance)

                    if not should_touch:
                        raise ValueError(
                            f"Desplining Created Overlapping solids. "
                            f"{hierarchy_a[1]} overlaps {hierarchy_b[1]} by a volume "
                            f"{intersection.Volume()} m^3. In original Geometry, "
                            f" they should not even touch. Please increase "
                            "the discretisations to avoid overlaps in rebuilt solids."
                        )
                    bluemira_warn(
                        f"{hierarchy_a[1]} overlaps {hierarchy_b[1]} by a volume "
                        f"{intersection.Volume()} m^3. In original Geometry, "
                        f" they should touch instead. Running Overlap fixing. "
                    )
                    # Running Overlap

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
