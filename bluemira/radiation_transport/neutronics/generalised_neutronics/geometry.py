# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Geometry for generalised neutronics
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, auto
from itertools import combinations
from typing import TypeAlias

import cadquery as cq

from bluemira.base.reactor import ComponentManager
from bluemira.geometry.error import GeometryError

ComponentManagerConfig: TypeAlias = tuple[ComponentManager, int]
"""Type alias for a tuple containing a ComponentManager and
its desplining discretisation."""


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


@dataclass
class ReactorGeometry:
    """
    Data storage stage

    Parameters
    ----------
    All managers to be considered in the neutronics model

    """

    # Instead of individual compoennt names, keep the below format
    # to make it usable for any number of components with any name
    comp_managers: dict[str, ComponentManagerConfig] = field(default_factory=dict)

    def __init__(self, _objects: dict[str, ComponentManagerConfig]) -> None:
        raise TypeError(
            "ReactorGeometry must be initialised using ReactorGeometry.from_dict()"
        )

    @classmethod
    def from_dict(
        cls,
        objects: dict[str, ComponentManagerConfig],
    ) -> ReactorGeometry:
        """Create a ReactorGeometry instance from a dictionary.

        Parameters
        ----------
        objects
            Dictionary mapping component names to ComponentManager and
            discretisation pairs.

        Returns
        -------
        ReactorGeometry
            A ReactorGeometry instance containing the provided component
            managers and discretisations.

        Raises
        ------
        TypeError
            If any value in ``objects`` is not a valid
            ComponentManagerConfig.
        """
        tuple_length = 2
        for name, config in objects.items():
            if (
                not isinstance(config, tuple)
                or len(config) != tuple_length
                or not isinstance(config[0], ComponentManager)
                or not isinstance(config[1], int)
            ):
                raise TypeError(
                    f"{name!r} must be a tuple of "
                    f"(ComponentManager, int), got {type(config).__name__}"
                )

        instance = object.__new__(cls)
        instance.comp_managers = dict(objects)
        return instance


def inspect_overlaps(
    geometry: ReactorGeometry, tolerance: float = 1e-10
) -> tuple[bool, list[str] | None]:
    """
    Inspect ReactorGeometry to ensure that there is no overlap
    between any two CadQuery solids. Touching is allowed.

    Returns
    -------
    tuple[bool, list[str] | None]

    Raises
    ------
    TypeError
        If a component does not contain a CadQuery Solid.
    """
    all_xyzs = {
        name: manager.component().get_component("xyz", first=False)
        for name, (manager, _) in geometry.comp_managers.items()
    }

    all_solids = {}

    for component_name, xyzs in all_xyzs.items():
        solids = []

        for xyz in xyzs:
            for child in xyz.children:
                if not isinstance(child.shape.shape, cq.Solid):
                    raise TypeError(
                        "inspect_overlaps is only available for "
                        f"CadQuery Solid objects. "
                        f"{component_name} contains "
                        f"{type(child.shape).__name__}."
                    )

                solids.append(child.shape)

        all_solids[component_name] = solids

    solids = [
        (component, index, solid)
        for component, component_solids in all_solids.items()
        for index, solid in enumerate(component_solids)
    ]

    overlaps = []

    for (
        (component_a, index_a, solid_a),
        (component_b, index_b, solid_b),
    ) in combinations(solids, 2):
        intersection = solid_a.shape.intersect(solid_b.shape)

        if intersection.Volume() > tolerance:
            overlaps.append(
                f"{component_a}[{index_a}] overlaps "
                f"{component_b}[{index_b}] by {intersection.Volume()}"
            )

    if overlaps:
        return True, overlaps
    return False, None


def despline_reactor_geometry(
    geometry: ReactorGeometry, overlap_tolerance: float = 1e-10
) -> ReactorGeometry:
    """Despline all relevant components in the reactor geometry.

    Parameters
    ----------
    geometry
        Reactor geometry containing the components to despline.

    Returns
    -------
    ReactorGeometry
        Desplined ReactorGeometry.

    Raises
    ------
    GeometryError
        if independent desplining of components causes clashes
    """
    desplined_components: dict[str, ComponentManagerConfig] = {}

    for comp_name, (manager, discretisation) in geometry.comp_managers.items():
        desplined_comp = manager.get_desplined_component_tree(
            discretisation=discretisation,
        )

        desplined_manager = deepcopy(manager)
        desplined_manager._component = desplined_comp

        desplined_components[comp_name] = (
            desplined_manager,
            discretisation,
        )

    geometry = ReactorGeometry.from_dict(desplined_components)
    overlap, error = inspect_overlaps(geometry, overlap_tolerance)

    if overlap:
        raise GeometryError(
            "Overlapping solids found:\n"
            + "\n".join(error)
            + "\n"
            + "We advise increasing the desplining discretisations."
        )
    return geometry
