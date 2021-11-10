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
Built-in build steps for making shapes
"""

from typing import Any, Dict, Type

from ..base.builder import Builder
from ..base.components import Component, PhysicalComponent
from ..geometry.parameterisations import GeometryParameterisation
from ..utilities.tools import get_module


class ParameterisedShapeBuilder(Builder):
    """
    Abstract builder class for building parameterised shapes.
    """

    _required_config = ["param_class", "variables_map"]

    def _extract_config(self, build_config: Dict[str, Any]):
        def _get_param_class(param_class: str) -> Type[GeometryParameterisation]:
            module = "bluemira.geometry.parameterisations"
            class_name = param_class
            if "::" in class_name:
                module, class_name = class_name.split("::")
            return getattr(get_module(module), class_name)

        self._param_class = _get_param_class(build_config["param_class"])
        self._variables_map: Dict[str, str] = build_config["variables_map"]
        self._extract_required_params()

    def _extract_required_params(self):
        self._required_params = []
        for var in self._variables_map.values():
            if isinstance(var, dict) and isinstance(var["value"], str):
                self._required_params += [var["value"]]
            elif isinstance(var, str):
                self._required_params += [var]

    def _derive_shape_params(self):
        shape_params = {}
        for key, val in self._variables_map.items():
            if isinstance(val, str):
                val = self._params.get(val)
            elif isinstance(val, dict):
                if isinstance(val["value"], str):
                    val["value"] = self._params.get(val["value"])
            shape_params[key] = val
        return shape_params

    def create_parameterisation(self):
        """
        Create the GeometryParameterisation from the provided param_class and
        variables_map.

        Returns
        -------
        shape: GeometryParameterisation
            The shape resulting from the provided param_class and variables_map.
        """
        shape_params = self._derive_shape_params()
        shape = self._param_class()
        for key, val in shape_params.items():
            if isinstance(val, dict):
                shape.adjust_variable(key, **val)
            else:
                shape.adjust_variable(key, val)
        return shape


class MakeParameterisedShape(ParameterisedShapeBuilder):
    """
    A builder that constructs a Component using a parameterised shape.
    """

    _required_config = ParameterisedShapeBuilder._required_params + ["target"]

    _param_class: Type[GeometryParameterisation]
    _variables_map: Dict[str, str]
    _target: str

    def _extract_config(self, build_config: Dict[str, Any]):
        super()._extract_config(build_config)

        self._target: str = build_config["target"]

    def build(self, params, **kwargs) -> Dict[str, Component]:
        """
        Build the components from parameterised shapes using the provided configuration
        and parameterisation.

        Parameters
        ----------
        params: Dict[str, Any]
            The parameters to use for building.

        Returns
        -------
        build_results: List[Tuple[str, Component]]
            The Components build by this builder, including the target paths. For this
            Builder the results will contain one item.
        """
        super().build(params, **kwargs)

        parameterisation = self.create_parameterisation()

        target = self._target.split("/")
        return [
            (
                "/".join(target[:-1]),
                PhysicalComponent(target[-1], parameterisation.create_shape()),
            )
        ]
