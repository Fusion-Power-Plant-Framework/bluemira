# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Geometry for generalised neutronics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

from bluemira.base.reactor import ComponentManager


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

    comp_managers: dict[str, ComponentManager] = field(default_factory=dict)

    def __init__(self, _objects: dict[str, ComponentManager]) -> None:
        raise TypeError("Components must be initialised using Components.from_dict()")

    @classmethod
    def from_dict(cls, objects: dict[str, ComponentManager]) -> ReactorGeometry:
        """
        Create a ReactorGeometry instance from a dictionary.

        Returns
        -------
        ReactorGeometry

        Raises
        ------
        TypeError
            If any value in ``objects`` is not a ``ComponentManager`` instance.
        """
        for name, obj in objects.items():
            if not isinstance(obj, ComponentManager):
                raise TypeError(
                    f"{name!r} must be a ComponentManager, got {type(obj).__name__}"
                )

        instance = object.__new__(cls)
        instance.objects = dict(objects)
        return instance
