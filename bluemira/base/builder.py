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
Interfaces for builder and build steps classes
"""

from __future__ import annotations

import abc
import enum
import string
from typing import Any, Dict, List, Literal, Optional, Union

from bluemira.base.components import Component
from bluemira.base.config import Configuration
from bluemira.base.error import BuilderError
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print
from bluemira.base.parameter import ParameterFrame

__all__ = ["Builder", "BuildConfig"]


BuildConfig = Dict[str, Union[int, float, str, "BuildConfig"]]
"""
Type alias for representing nested build configuration information.
"""


# TODO: Consolidate with RunMode in codes.
class RunMode(enum.Enum):
    """
    Enum class to pass args and kwargs to the function corresponding to the chosen
    PROCESS runmode (Run, Read, or Mock).
    """

    RUN = enum.auto()
    READ = enum.auto()
    MOCK = enum.auto()

    def __call__(self, obj, *args, **kwargs):
        """
        Call function of object with lowercase name of enum

        Parameters
        ----------
        obj: instance
            instance of class the function will come from
        *args
           args of function
        **kwargs
           kwargs of function

        Returns
        -------
        function result
        """
        func = getattr(obj, self.name.lower())
        return func(*args, **kwargs)


class Builder(abc.ABC):
    """
    The Builder classes in bluemira define the various steps that will take place to
    build components when a Design is run.
    """

    _default_runmode: Optional[str] = None
    _required_params: List[str] = []
    _required_config: List[str] = []
    _params: Configuration
    _design_problem = None

    def __init__(self, params: Dict[str, Any], build_config: BuildConfig, **kwargs):
        self._name = build_config["name"]

        self._validate_config(build_config)
        self._extract_config(build_config)
        self._params = Configuration.from_template(self._required_params)
        self.reinitialise(params, **kwargs)

    def __call__(self) -> Component:
        """
        Perform the full build process using the current parameterisation of the builder.

        Returns
        -------
        component: Component
            The Component build by this builder.
        """
        bluemira_print(f"Running full build chain for {self.name}")
        if hasattr(self, "_runmode"):
            bluemira_print(f"Designing {self.name} using runmode: {self.runmode}")
            self._runmode(self)
        return self.build()

    @abc.abstractmethod
    def reinitialise(self, params, **kwargs) -> None:
        """
        Initialise the state of this builder ready for a new run.

        Parameters
        ----------
        params: Dict[str, Any]
            The parameterisation containing at least the required params for this
            Builder.
        """
        bluemira_debug(f"Reinitialising {self.name}")
        self._reset_params(params)

    @abc.abstractmethod
    def build(self) -> Component:
        """
        Runs this Builder's build process to populate the required Components.

        The result of the build is stored in the Builder's component property.
        """
        bluemira_print(f"Building {self.name}")

        return Component(self._name)

    @property
    def name(self) -> str:
        """
        The name of the builder.
        """
        return self._name

    @property
    def params(self) -> ParameterFrame:
        """
        The parameterisation of this builder.
        """
        return self._params

    @property
    def required_params(self) -> List[str]:
        """
        The variable names of the parameters that are needed to run this builder.
        """
        return self._required_params

    @property
    def required_config(self) -> List[str]:
        """
        The names of the build configuration values that are needed to run this builder.
        """
        return self._required_config

    @property
    def runmode(self):
        """
        The name of the method that will be executed when calling this builder.
        """
        return self._runmode.name.lower()

    @property
    def design_problem(self):
        """
        The design problem solved by this builder, if any.
        """
        return self._design_problem

    def run(self):
        """
        Optional abstract method to run a design problem while building.
        """
        raise NotImplementedError

    def read(self):
        """
        Optional abstract method to read the result of a design problem while building.
        """
        raise NotImplementedError

    def mock(self):
        """
        Optional abstract method to mock a design problem while building.
        """
        raise NotImplementedError

    def _validate_requirement(
        self, input, source: Literal["params", "config"]
    ) -> List[str]:
        missing = []
        for req in getattr(self, f"_required_{source}"):
            if req not in input.keys():
                missing += [req]
        return missing

    def _validate_config(self, build_config: BuildConfig):
        missing_config = self._validate_requirement(build_config, "config")

        if missing_config != []:
            raise BuilderError(
                f"Required config keys {', '.join(missing_config)} not provided to "
                f"Builder: {self._name}"
            )

    def _validate_params(self, params):
        missing_params = self._validate_requirement(params, "params")

        if missing_params != []:
            raise BuilderError(
                f"Required parameters {', '.join(missing_params)} not provided to "
                f"Builder {self._name}"
            )

    def _reset_params(self, params):
        self._validate_params(params)
        self._params.update_kw_parameters({k: params[k] for k in self._params.keys()})

    def _extract_config(self, build_config: BuildConfig):
        has_runmode = (
            "runmode" in build_config
            or getattr(self, "_default_runmode", None) is not None
        )
        if has_runmode:
            self._set_runmode(build_config)

    def _set_runmode(self, build_config: BuildConfig):
        """
        Set runmode according to the "runmode" parameter in build_config or the default
        run mode if not provided via build_config.
        """
        runmode = build_config.get("runmode", self._default_runmode)

        if not hasattr(self, runmode.lower()):
            raise NotImplementedError(
                f"Builder {self.__class__.__name__} has no {runmode.lower()} mode."
            )

        mode = (
            build_config.get("runmode", self._default_runmode)
            .upper()
            .translate(str.maketrans("", "", string.whitespace))
        )
        self._runmode = RunMode[mode]
