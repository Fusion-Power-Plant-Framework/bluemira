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
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from bluemira.base.look_and_feel import bluemira_print
from bluemira.base.parameter_frame import ParameterFrame
from bluemira.display.plotter import plot_2d
from bluemira.radiation_transport.axisymmetric_neutronics.geometry import (
    AxisymmetricNeutronicsGeomManager,
)

if TYPE_CHECKING:
    from bluemira.base.components import Component
    from bluemira.base.reactor import Reactor


class NeutronicsModel(Enum):
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
        model: NeutronicsModel,
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
        desplined_geometry = AxisymmetricNeutronicsGeomManager.from_list_of_components(
            components=all_comps,
            discretisations=discretisations,
            overlap_tolerance=self.config.get("overlap_tolerance", 1e-10),
            gap_tolerance=self.config.get("gap_tolerance", 1e-6),
        )

        plot_comparisons = self.config.get("plot_comparisons", False)
        if plot_comparisons:
            neutronics_output_path = Path(
                self.config.get("neutronics_output_path", "./outputs/")
            )
            self.plot_desplining_comparisons(
                orig_comps=[comp.get_component("xz") for comp in all_comps],
                desplined_comps=desplined_geometry.component().get_component(
                    "xz", first=False
                ),
                output_path=neutronics_output_path,
            )
        return desplined_geometry

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

    @staticmethod
    def plot_desplining_comparisons(
        orig_comps: list[Component],
        desplined_comps: list[Component],
        output_path: Path | None = None,
    ):
        """
        Create comparison plots in the xz plane before and after
        desplining components.

        Raises
        ------
        ValueError
            if original and desplined components do not have the
            same length
        """
        output_path.mkdir(parents=True, exist_ok=True)

        if len(orig_comps) != len(desplined_comps):
            raise ValueError(
                "original and desplined components must have the same length."
            )

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(12, 6),
            sharex=True,
            sharey=True,
        )

        for orig_comp, desplined_comp in zip(
            orig_comps,
            desplined_comps,
            strict=True,
        ):
            orig_comp.plot_2d(ax=axes[0], show=False)
            for wire in orig_comp.leaves[0].shape.boundary:
                plot_2d(wire.vertexes.T, ax=axes[0], show=False)

            desplined_comp.plot_2d(ax=axes[1], show=False)
            for wire in desplined_comp.leaves[0].shape.boundary:
                plot_2d(wire.vertexes.T, ax=axes[1], show=False)

        axes[0].set_title("Original")
        axes[1].set_title("Desplined")

        fig.tight_layout()

        plot_path = output_path / "desplining_comparisons.png"
        fig.savefig(plot_path, bbox_inches="tight")

        bluemira_print(f"Plotted desplining comparisons as {plot_path}")

    @abstractmethod
    def get_all_components(
        self, reactor: Reactor, model: NeutronicsModel = NeutronicsModel.SIMPLE
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
