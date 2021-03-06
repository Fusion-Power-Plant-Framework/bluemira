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

from __future__ import annotations

import abc
import copy
import functools
import typing
from typing import Callable, Dict, List, Optional, Set, Type, Union

from bluemira.base.builder import BuildConfig, Builder
from bluemira.base.components import Component
from bluemira.base.config import Configuration
from bluemira.base.error import DesignError
from bluemira.base.file import BM_ROOT, FileManager
from bluemira.base.look_and_feel import bluemira_print, print_banner
from bluemira.utilities.tools import get_class_from_module

if typing.TYPE_CHECKING:
    from bluemira.codes.interface import CodesSolver


class DesignABC(abc.ABC):
    """
    The abstract Design class provides the framework for performing bluemira design
    studies. It allows a series of Builder objects to be created, which are
    configured using the input parameters and configuration and walked though to produce
    the analysis results and reactor components.
    """

    _required_params: Set[str]
    _params: Configuration
    _build_config: Dict[str, BuildConfig]
    _builders: Dict[str, Builder]
    _solvers: Dict[str, CodesSolver]
    _stage: List[str] = []

    def __init__(
        self,
        params: Dict[str, Union[int, float, str]],
        build_config: Dict[str, BuildConfig],
    ):
        print_banner()
        self._build_config = copy.deepcopy(build_config)
        self._extract_build_config(params)
        self._validate_params(params)
        self._params = Configuration.from_template(self._required_params)
        self._params.update_kw_parameters(params)

    @property
    def required_params(self) -> Set[str]:
        """
        The names of the parameters that are required to run this design.
        """
        return self._required_params

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

    @property
    def stage(self):
        """
        Stage label

        Raises
        ------
        DesignError when stage not registered
        """
        try:
            return self._stage[-1]
        except IndexError:
            raise DesignError(
                "Design stage not defined please use the design_stage decorator on your function"
            ) from None

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

    def get_solver(self, solver_name: str) -> CodesSolver:
        """
        Get the solver with the corresponding solver_name.

        Parameters
        ----------
        solver_name: str
            The name of the solver to get.

        Returns
        -------
        solver: CodesSolver
            The solver corresponding to the provided name.
        """
        return self._solvers[solver_name]

    def register_solver(self, solver: CodesSolver):
        """
        Add this solver to the internal solver registry.

        Parameters
        ----------
        solver: CodesSolver
            The solver to be registered.
        name: str
            The name to register this solver with.

        Raises
        ------
        DesignError
            If name already exists in the registry.
        """
        if self.stage not in self._solvers:
            self._solvers[self.stage] = solver
        else:
            raise DesignError(f"Solver {self.stage} already exists in {self}.")

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

    def register_builder(self, builder: Builder):
        """
        Add this builder to the internal builder registry.

        Parameters
        ----------
        builder: Builder
            The builder to be registered.
        name: str
            The name to register this builder with.

        Raises
        ------
        DesignError
            If name already exists in the registry.
        """
        if self.stage not in self._builders:
            self._builders[self.stage] = builder
        else:
            raise DesignError(f"Builder {self.stage} already exists in {self}.")

    def _build_stage(self) -> Component:
        """
        Build the requested stage and update the design's parameters.

        Parameters
        ----------
        name: str
            The name of the stage to build.

        Returns
        -------
        component: Component
            The resulting component from the build.
        """
        component = self._builders[self.stage]()
        self._params.update_kw_parameters(self._builders[self.stage].params.to_dict())

        return component

    @staticmethod
    def design_stage(name: str) -> Callable:
        """
        Design stage decorator.
        Adds some printing to each stage and sets the 'stage' variable

        Parameters
        ----------
        name: str
            Name of design stage

        Returns
        -------
        decorated function

        """

        def decorator(func):
            @functools.wraps(func)
            def wrapper(self, *args, **kwargs):
                bluemira_print(f"Starting design stage: {name}")
                self._stage.append(name)
                ret = func(self, *args, **kwargs)
                self._stage.pop()
                bluemira_print(f"Completed design stage: {name}")
                return ret

            return wrapper

        return decorator

    @abc.abstractmethod
    def _extract_build_config(self, params: Dict[str, Union[int, float, str]]):
        """
        Extract the builders and associated required parameter names from the config,
        which must be an ordered dictionary mapping the name of the builder to the
        corresponding options.
        """
        self._builders = {}
        self._solvers = {}
        self._required_params = {"Name"}

    def _validate_params(self, params: Dict[str, Union[int, float, str]]):
        """
        Validate that the provided parameters are as expected.
        """
        missing_params = {
            param_name
            for param_name in self._required_params
            if param_name not in params
        }

        if len(missing_params) > 0:
            raise DesignError(
                f"Required parameters {', '.join(sorted(missing_params))} not provided to Design"
            )


class Design(DesignABC):
    """
    The Design class is the main bluemira class that controls configurable design
    studies. It allows a series of Builder objects to be created, which are configured
    using the input parameters and configuration and walked though to produce the
    analysis results and reactor components.
    """

    _required_params: Set[str]
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
            component.add_child(
                self.design_stage(builder.name)(Design._build_stage)(self)
            )

        bluemira_print("Design Complete!")

        return component

    def _extract_build_config(self, params: Dict[str, Union[int, float, str]]):
        """
        Extracts the builders from the config, which must be an ordered dictionary
        mapping the name of the builder to the corresponding options.
        """
        super()._extract_build_config(params)
        for key, val in self._build_config.items():
            class_name = val.pop("class")
            val["name"] = key
            builder_class: Type[Builder] = get_class_from_module(
                class_name, default_module="bluemira.builders"
            )
            self._stage.append(key)
            self.register_builder(builder_class(params, val))
            self._required_params |= set(self._builders[key]._required_params)
            self._stage.pop()


class ReactorDesign(DesignABC):
    """
    The ReactorDesign class allows a Design to be implemented directly in the code. This
    can simplify some of logic when compared with the configurable Design class, in
    particular when passing around Component information. As such, individual
    ReactorDesign instances must implement their own `run` method. The ReactorDesign
    class also provides managed output via a FileManager to aid the persistence of input
    and output data.
    """

    _required_params: Set[str]
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

        # For now the params can come from any parameters defined in Configuration.
        # In the future we probably want to register the Builders early and get the
        # parameters that the Builders need.
        self._required_params |= set(Configuration().keys())

        self._reference_data_root: str = self._build_config.get(
            "reference_data_root", f"{BM_ROOT}/data"
        )
        self._generated_data_root: str = self._build_config.get(
            "generated_data_root", f"{BM_ROOT}/generated_data"
        )
        self._plot_flag: bool = self._build_config.get("plot_flag", False)

    def _process_design_stage_config(
        self, default_config: BuildConfig = None
    ) -> Dict[str, BuildConfig]:
        config = {"name": self.stage}

        # Copy in top-level configuration
        for key, val in self._build_config.items():
            if not isinstance(val, dict):
                config[key] = val

        # Set the default configuration values
        config.update(default_config)

        # Set the specified configuration values
        config.update(self._build_config.get(self.stage, {}))

        return config

    @property
    def file_manager(self) -> FileManager:
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

    def _validate_params(self, params: Dict[str, Union[int, float, str]]):
        """
        Validation of Reactor parameters is currently not supported.
        """
        pass
