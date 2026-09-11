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
from bluemira.geometry.error import GeometryError
from bluemira.geometry.tools import check_touching_solids, repair_overlapping_solids
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
        geom_managers.inspect_fix_overlaps(component_managers)
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
    ) -> list[tuple[BluemiraSolid, str]]:
        """
        Get all XYZ solids and their parent component names.

        Returns
        -------
        list[tuple[BluemiraSolid, str]]
        """
        all_solids = []
        for component in components:
            xyz_comps = component.get_component("xyz", first=False)
            all_solids.extend(
                (xyz.children[0].shape, xyz.parent.name) for xyz in xyz_comps
            )
        return all_solids

    def inspect_fix_overlaps(
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

        for i in range(len(all_solids)):
            solid_a, name_a = all_solids[i]
            overlapping = []

            for j, (solid_b, name_b) in enumerate(all_solids[i + 1 :], start=i + 1):
                intersection = solid_a.shape.intersect(solid_b.shape)

                if intersection.Volume() > tolerance:
                    # check if the original solids were supposed to touch
                    orig_a = next(
                        solid for solid, name in all_orig_solids if name == name_a
                    )
                    orig_b = next(
                        solid for solid, name in all_orig_solids if name == name_b
                    )
                    should_touch = check_touching_solids(orig_a, orig_b, tolerance)

                    if not should_touch:
                        raise ValueError(
                            f"Desplining Created Overlapping solids. "
                            f"{name_a} overlaps {name_b} by a volume "
                            f"{intersection.Volume()} m^3. In original Geometry, "
                            f" they should not even touch. Please increase "
                            "the discretisations to avoid overlaps in rebuilt solids."
                        )

                    overlapping.append((j, name_b, intersection.Volume()))

            if overlapping:
                bluemira_warn(
                    f"{name_a} overlaps "
                    f"{', '.join(name for _, name, _ in overlapping)} "
                    f"by volumes "
                    f"{', '.join(f'{volume} m^3' for _, _, volume in overlapping)}. "
                    f"In original Geometry, they should touch instead. "
                    f"Running Overlap fixing."
                )

                for j, name_b, _ in overlapping:
                    solid_b, _ = all_solids[j]

                    try:
                        solid_a, solid_b = repair_overlapping_solids(solid_a, solid_b)
                        all_solids[j] = (solid_b, name_b)

                        # Assuming only one xyz, works because of our earlier
                        #  _get_all_solid() logic
                        self.component().get_component(name_b).get_component(
                            "xyz"
                        ).children[0].shape = solid_b

                    except GeometryError as e:
                        bluemira_warn(
                            f"Could not repair overlap between "
                            f"{name_a} and {name_b}: {e}"
                        )

                all_solids[i] = (solid_a, name_a)

                # Immediately replace the repaired A
                self.component().get_component(name_a).get_component("xyz").children[
                    0
                ].shape = solid_a

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
