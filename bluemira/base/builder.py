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
from typing import Dict, List, Literal, Union

from bluemira.base.components import Component
from bluemira.base.error import BuilderError
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print
from bluemira.base.parameter import ParameterFrame


__all__ = ["Builder", "BuildConfig"]


BuildConfig = Dict[str, Union[int, float, str, "BuildConfig"]]
"""
Type alias for representing nested build configuration information.
"""


class Builder(abc.ABC):
    """
    The Builder classes in bluemira define the various steps that will take place to
    build components when a Design is run.
    """

    _required_params: List[str] = []
    _required_config: List[str] = []
    _params: ParameterFrame

    def __init__(self, params, build_config: BuildConfig, **kwargs):
        self._name = build_config["name"]

        self._validate_config(build_config)
        self._extract_config(build_config)
        self._params = ParameterFrame.from_template(self._required_params)
        self.reinitialise(params)

    def __call__(self, params, **kwargs) -> Component:
        """
        Perform the full build process, including reinitialisation, using the provided
        parameters.

        Parameters
        ----------
        params: Dict[str, Any]
            The parameterisation containing at least the required params for this
            Builder.

        Returns
        -------
        component: Component
            The Component build by this builder.
        """
        self.reinitialise(params)
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
    def build(self, **kwargs) -> Component:
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
    def required_parameters(self) -> List[str]:
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
        self._params.update_kw_parameters(params)

    def _extract_config(self, build_config: BuildConfig):
        pass
