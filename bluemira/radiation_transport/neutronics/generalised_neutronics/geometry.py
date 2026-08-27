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
from typing import TypeAlias

from bluemira.base.reactor import ComponentManager

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


def despline_reactor_geometry(geometry: ReactorGeometry) -> ReactorGeometry:
    """Despline all relevant components in the reactor geometry.

    Parameters
    ----------
    geometry
        Reactor geometry containing the components to despline.

    Returns
    -------
    ReactorGeometry
        Desplined ReactorGeometry.
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

    return ReactorGeometry.from_dict(desplined_components)
