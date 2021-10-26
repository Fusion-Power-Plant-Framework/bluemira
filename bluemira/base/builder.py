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

import abc
from typing import Any, Dict, List, Literal, Tuple

from bluemira.base.components import Component
from bluemira.base.error import BuilderError
from bluemira.base.look_and_feel import bluemira_print
from bluemira.base.parameter import ParameterFrame


class Builder(abc.ABC):
    """
    The Builder classes in bluemira define the various steps that will take place to
    build components when an Analysis is run.
    """

    _required_params: List[str] = []
    _required_config: List[str] = []
    _params: ParameterFrame

    def __init__(self, params, build_config: Dict[str, Any], **kwargs):
        self._name = build_config["name"]

        self._validate_config(build_config)
        self._extract_config(build_config)
        self._validate_params(params)
        self._params = ParameterFrame.from_template(self._required_params)
        self._params.update_kw_parameters(params)

    @abc.abstractmethod
    def build(self, params, **kwargs) -> List[Tuple[str, Component]]:
        """
        Runs this Builder's build process to generate the required Components.

        Parameters
        ----------
        params: Dict[str, Any]
            The parameterisation containing at least the required params for this
            Builder.

        Returns
        -------
        build_results: List[Tuple[str, Component]]
            The Components build by this builder, including the target paths.
        """
        self._validate_params(params)
        self._params.update_kw_parameters(params)
        bluemira_print(f"Building {self.name}")
        return [()]

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
    def required_config(self):
        """
        The names of the build configuration values that are needed to run this builder.
        """
        return self._required_config

    def _validate_requirement(self, input, source: Literal["params", "config"]):
        missing = []
        for req in getattr(self, f"_required_{source}"):
            if req not in input.keys():
                missing += [req]
        return missing

    def _validate_config(self, build_config):
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

    def _extract_config(self, build_config: Dict[str, Any]):
        pass
