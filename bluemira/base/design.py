# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
Module containing the bluemira Design class.
"""

from typing import Dict, List, Type

from bluemira.base.builder import Builder, BuildConfig
from bluemira.base.components import Component
from bluemira.base.config import Configuration
from bluemira.base.error import BuilderError
from bluemira.base.look_and_feel import bluemira_print, print_banner
from bluemira.utilities.tools import get_class_from_module


class Design:
    """
    The Design class is the main bluemira class that controls the design study that is
    being performed. It allows a series of Builder objects to be created, which are
    configured using the input parameters and configuration and walked though to produce
    the analysis results and reactor components.
    """

    _required_params: List[str] = ["Name"]
    _params: Configuration
    _build_config: BuildConfig
    _builders: Dict[str, Builder]

    def __init__(self, params, build_config):
        print_banner()
        self._build_config = build_config
        self._extract_builders(params)
        self._params = Configuration.from_template(self._required_params)
        self._params.update_kw_parameters(params)

    @property
    def params(self) -> Configuration:
        """
        The ParameterFrame associated with this Design.
        """
        return self._params

    def run(self) -> Component:
        """
        Runs through the Builders associated with this Design. Components and
        Parameters are transferred onto the Design after each step.

        Returns
        -------
        component: Component
            The Component tree resulting from the various build stages in the Design.
        """
        bluemira_print(f"Running Design: {self._params.Name.value}")
        component = Component(self._params.Name)
        for builder in self._builders.values():
            component.add_child(builder(self._params))
            self._params.update_kw_parameters(
                builder._params.to_dict(), source=builder.name
            )
        return component

    def get_builder(self, builder_name: str) -> Builder:
        """
        Get the builder with the corresponding builder_name.

        Parameters
        ----------
        builder_name: str
            The name of the builder to get.

        Returns
        -------
        builder: Builder
            The builder corresponding to the provided name.
        """
        return self._builders[builder_name]

    def _extract_builders(self, params):
        """
        Extracts the builders from the config, which must be an ordered dictionary
        mapping the name of the builder to the corresponding options.
        """
        self._builders = {}
        for key, val in self._build_config.items():
            class_name = val.pop("class")
            val["name"] = key
            builder_class: Type[Builder] = get_class_from_module(
                class_name, default_module="bluemira.builders"
            )
            if key not in self._builders:
                self._builders[key] = builder_class(params, val)
            else:
                raise BuilderError(f"Builder {key} already exists in {self}.")
            self._required_params += self._builders[key]._required_params
