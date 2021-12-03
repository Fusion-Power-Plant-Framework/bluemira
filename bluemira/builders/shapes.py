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

from typing import Dict, Type

from bluemira.base.builder import BuildConfig, Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.utilities.tools import get_class_from_module


class ParameterisedShapeBuilder(Builder):
    """
    Abstract builder class for building parameterised shapes.
    """

    _required_config = Builder._required_config + ["param_class", "variables_map"]

    _param_class: Type[GeometryParameterisation]
    _variables_map: Dict[str, str]

    def _extract_config(self, build_config: BuildConfig):
        super()._extract_config(build_config)

        self._param_class: Type[GeometryParameterisation] = get_class_from_module(
            build_config["param_class"],
            default_module="bluemira.geometry.parameterisations",
        )
        self._variables_map: Dict[str, str] = build_config["variables_map"]
        self._required_params = build_config.get("additional_params", [])
        self._extract_required_params()

    def _extract_required_params(self):
        for var in self._variables_map.values():
            if isinstance(var, dict) and isinstance(var["value"], str):
                self._required_params += [var["value"]]
            elif isinstance(var, str):
                self._required_params += [var]

    def _derive_shape_params(self):
        shape_params = {}
        for key, val in self._variables_map.items():
            if isinstance(val, str):
                val = {"value": self._params.get(val)}
            elif isinstance(val, dict):
                if isinstance(val["value"], str):
                    val["value"] = self._params.get(val["value"])
            else:
                val = {"value": val}
            shape_params[key] = val
        return shape_params

    def reinitialise(self, params, **kwargs):
        """
        Create the GeometryParameterisation from the provided param_class and
        variables_map.

        Parameters
        ----------
        params: Dict[str, Any]
            The parameterisation containing at least the required params for this
            Builder.
        """
        super().reinitialise(params, **kwargs)

        shape_params = self._derive_shape_params()
        shape = self._param_class(shape_params)
        self._shape = shape


class DesignParameterisedShape(ParameterisedShapeBuilder):
    """
    A builder that designs a Component using a parameterised shape and an optional
    design optimisation callback.
    """

    _required_config = ParameterisedShapeBuilder._required_config + ["label"]

    _label: str

    def _extract_config(self, build_config: BuildConfig):
        super()._extract_config(build_config)

        self._label: str = build_config["label"]

    def build(self, component_tree=None, **kwargs) -> Component:
        """
        Build the components from parameterised shapes using the provided configuration
        and parameterisation.

        Returns
        -------
        component: Component
            The Component built by this builder.
        """
        component = super().build(**kwargs)

        component.add_child(
            PhysicalComponent(self._label, self._shape.create_shape(label=self._label))
        )

        return component
