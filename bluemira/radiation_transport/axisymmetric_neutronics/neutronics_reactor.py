# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Axissymetric Neutronics Reactor
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from bluemira.base.look_and_feel import bluemira_print
from bluemira.base.parameter_frame import ParameterFrame
from bluemira.radiation_transport.axisymmetric_neutronics.geometry import (
    AxisymmetricNeutronicsGeomManager,
)

if TYPE_CHECKING:
    from bluemira.base.components import Component
    from bluemira.base.reactor import Reactor


class NeutronincsModel(Enum):
    """
    Enumeration of different model to be
    considered for neutronics simulations.

    The interpretation of the enums would
    be user-determined through the abstract
    methods defined in the NeutronicsReactor
    class
    """

    DEBUG = auto()
    SIMPLE = auto()
    ADVANCED = auto()

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
class AxisymmetricNeutronicsParameterFrame(ParameterFrame):
    """Neutronics reactor parameters"""


class AxisymmetricNeutronicsReactor(ABC):
    """
    Axisymmetric Neutronics Reactor
    """

    param_cls = AxisymmetricNeutronicsParameterFrame

    def __init__(
        self,
        neutronics_config: dict,
        params: dict | ParameterFrame,
        model: NeutronincsModel,
        reactor: Reactor,
    ):
        bluemira_print("Creating axisymmetric CSG neutronics model")

        self.config = neutronics_config
        self.params = params
        self.model = model
        self.geometry = self.get_desplined_geometry(reactor)

    def get_desplined_geometry(
        self, reactor: Reactor
    ) -> AxisymmetricNeutronicsGeomManager:
        """
        Creates a neutronics geometry manager from components,
        subject to the neutronics model

        Returns
        -------
        AxisymmetricNeutronicsGeomManager
        """
        all_comps = self.get_all_components(reactor, self.model)
        discretisations = [
            self.default_discretisations(comp.name)
            or self.default_discretisations(comp.parent().name)
            for comp in all_comps
        ]
        return AxisymmetricNeutronicsGeomManager.from_list_of_components(
            components=all_comps,
            discretisations=discretisations,
            overlap_tolerance=self.config.get("overlap_tolerance", 1e-10),
            gap_tolerance=self.config.get("gap_tolerance", 1e-6),
        )

    def plot_2d(self, *args, **kwargs):
        """
        Plot neutronics reactor 2d profile

        Returns
        -------
        :
            Axes on which the reactor is plotted.
        """
        show = kwargs.pop("show", True)
        ax = kwargs.pop("ax", None)
        return self.geometry.plot_2d(*args, ax=ax, show=show, **kwargs)

    @abstractmethod
    def get_all_components(
        self, reactor: Reactor, model: NeutronincsModel = NeutronincsModel.SIMPLE
    ) -> list[Component]:
        """
        Get all relevant components for the reactor

        Returns
        -------
        list
            list of components.
        """
        ...

    @abstractmethod
    def default_discretisations(self, comp_name: str) -> int:
        """
        Return the default spline discretisation for a component.

        Returns
        -------
        int
        """
        ...
