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

import abc
from typing import Dict, List, Optional, Type, Union

from bluemira.base.builder import Builder, BuildConfig
from bluemira.base.components import Component
from bluemira.base.config import Configuration
from bluemira.base.error import BuilderError
from bluemira.base.file import BM_ROOT, FileManager
from bluemira.base.look_and_feel import bluemira_print, print_banner
from bluemira.utilities.tools import get_class_from_module


CallbackConfig = Dict[str, Union[str, Dict[str, Union[str, Dict[str, str]]]]]


class DesignABC(abc.ABC):
    """
    The abstract Design class provides the framework for performing bluemira design
    studies. It allows a series of Builder objects to be created, which are
    configured using the input parameters and configuration and walked though to produce
    the analysis results and reactor components.
    """

    _required_params: List[str] = ["Name"]
    _params: Configuration
    _build_config: Dict[str, BuildConfig]
    _builders: Dict[str, Builder]
    _callbacks: CallbackConfig

    def __init__(
        self,
        params: Dict[str, Union[int, float, str]],
        build_config: Dict[str, BuildConfig],
    ):
        print_banner()
        self._builders = {}
        self._build_config = build_config
        self._extract_build_config(params)
        self._params = Configuration.from_template(self._required_params)
        self._params.update_kw_parameters(params)

    @property
    def params(self) -> Configuration:
        """
        The ParameterFrame associated with this Design.
        """
        return self._params

    @property
    def build_config(self):
        """
        The build configuration associated with this Design.
        """
        return self._build_config

    @abc.abstractmethod
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
        return component

    def _build_stage(self, builder: Builder, component_tree: Component) -> Component:
        callback = self._callbacks.get(builder.name, None)
        callback_args = {}
        if isinstance(callback, dict):
            if "func" not in callback:
                raise BuilderError(
                    "When defining a callback as a dictionary, the callback function "
                    "must be specified in the func key, and any args in the optional "
                    f"args key, got {callback}"
                )
            callback_args = callback.get("args", {})
            callback = callback["func"]

        component = self._builders[builder.name](
            self._params.to_dict(),
            component_tree,
            callback,
            callback_args,
        )

        self._params.update_kw_parameters(self._builders[builder.name]._params.to_dict())
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

    def register_builder(self, builder: Builder, builder_name: str):
        """
        Add the builder to the builder registry with key builder_name.

        Parameters
        ----------
        builder: Builder
            The Builder to be registered.
        builder_name: str
            The name to register the Builder under.
        """
        if builder_name in self._builders:
            raise BuilderError(f"Builder already registered with name {builder_name}.")

        self._builders[builder_name] = builder

    @abc.abstractmethod
    def _extract_build_config(self, params: Dict[str, Union[int, float, str]]):
        """
        Extracts the builders from the config, which must be an ordered dictionary
        mapping the name of the builder to the corresponding options.
        """
        self._callbacks = self._build_config.get("callbacks", {})


class Design(DesignABC):
    """
    The Design class is the main bluemira class that controls configurable design
    studies. It allows a series of Builder objects to be created, which are configured
    using the input parameters and configuration and walked though to produce the
    analysis results and reactor components.
    """

    _required_params: List[str] = ["Name"]
    _params: Configuration
    _build_config: Dict[str, BuildConfig]
    _builders: Dict[str, Builder]

    def run(self) -> Component:
        """
        Runs through the Builders associated with this Design. Components and
        Parameters are transferred onto the Design after each step.

        Returns
        -------
        component: Component
            The Component tree resulting from the various build stages in the Design.
        """
        component = super().run()

        for builder in self._builders.values():
            component.add_child(self._build_stage(builder, component))

        return component

    def _extract_build_config(self, params: Dict[str, Union[int, float, str]]):
        """
        Extracts the builders from the config, which must be an ordered dictionary
        mapping the name of the builder to the corresponding options.
        """
        super()._extract_build_config(params)

        for key, val in self._build_config.items():
            class_name = val.pop("class")
            builder_class: Type[Builder] = get_class_from_module(
                class_name, default_module="bluemira.builders"
            )

            val["name"] = key
            self.register_builder(builder_class(params, val), key)

            self._required_params += self.get_builder(key).required_params

            if "callback" in val:
                self._callbacks[key] = val["callback"]


class Reactor(DesignABC):
    """
    The Reactor class allows a Design to be implemented directly in the code. This can
    simplify some of logic when compared with the configurable Design class, in
    particularwhen passing around Component information. As such, individual Reactor
    instances must implement their own `run` method. The Reactor class also provides
    managed output via a FileManager to aid the persistence of input and output data.
    """

    _required_params: List[str] = list(Configuration().keys())
    _params: Configuration
    _build_config: BuildConfig
    _builders: Dict[str, Builder]
    _file_manager: FileManager

    def __init__(
        self,
        params: Dict[str, Union[int, float, str]],
        build_config: Dict[str, BuildConfig],
    ):
        super().__init__(params, build_config)

        self._create_file_manager()

    def _create_file_manager(self):
        """
        Create the FileManager for this Reactor.
        """
        self._file_manager = FileManager(
            reactor_name=self._params.Name.value,
            reference_data_root=self._reference_data_root,
            generated_data_root=self._generated_data_root,
        )
        self._file_manager.build_dirs()

    def _extract_build_config(self, params: Dict[str, Union[int, float, str]]):
        super()._extract_build_config(params)

        self._reference_data_root: str = self._build_config.get(
            "reference_data_root", f"{BM_ROOT}/data"
        )
        self._generated_data_root: str = self._build_config.get(
            "generated_data_root", f"{BM_ROOT}/generated_data"
        )
        self._plot_flag: bool = self._build_config.get("plot_flag", False)

    @property
    def file_manager(self):
        """
        The FileManager instance associated with this Reactor.
        """
        return self._file_manager

    def add_parameters(
        self, params: Dict[str, Union[int, float, str]], source: Optional[str] = None
    ):
        """
        Perform a bulk update of the parameters from the given source.
        """
        self._params.update_kw_parameters(params, source=source)
