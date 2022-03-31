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

import copy
from typing import Dict, Type

from bluemira.base.builder import BuildConfig, Builder, BuilderError
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print
from bluemira.geometry.optimisation import GeometryOptimisationProblem
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.utilities.optimiser import Optimiser
from bluemira.utilities.tools import get_class_from_module


class ParameterisedShapeBuilder(Builder):
    """
    Abstract builder class for building parameterised shapes.
    """

    _required_config = ["param_class", "variables_map"]

    _param_class: Type[GeometryParameterisation]
    _variables_map: Dict[str, str]

    def _extract_config(self, build_config: BuildConfig):
        super()._extract_config(build_config=build_config)

        self._param_class: Type[GeometryParameterisation] = get_class_from_module(
            build_config["param_class"],
            default_module="bluemira.geometry.parameterisations",
        )
        self._variables_map: Dict[str, str] = build_config["variables_map"]
        self._extract_required_params()

    def _extract_required_params(self):
        self._required_params = copy.deepcopy(self._required_params)
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

            if isinstance(val, dict):
                if isinstance(val["value"], str):
                    val["value"] = self._params.get(val["value"])
            else:
                val = {"value": val}

            shape_params[key] = val
        return shape_params

    def reinitialise(self, params):
        """
        Create the GeometryParameterisation from the provided param_class and
        variables_map.

        Parameters
        ----------
        params: Dict[str, Any]
            The parameterisation containing at least the required params for this
            Builder.
        """
        super().reinitialise(params)

        shape_params = self._derive_shape_params()
        shape = self._param_class(shape_params)
        self._shape = shape

    def save_shape(self, filename: str, **kwargs):
        """
        Save the shape to a json file.

        Parameters
        ----------
        filename: str
            The path to the file that the shape should be written to.
        """
        self._shape.to_json(file=filename, **kwargs)
        bluemira_print(f"{self._name} shape saved to {filename}")


class OptimisedShapeBuilder(ParameterisedShapeBuilder):
    """
    An abstract builder that optimises a parameterised shaped based on a design problem.
    """

    _required_config = ParameterisedShapeBuilder._required_config + ["problem_class"]
    _problem_class: Type[GeometryOptimisationProblem]
    _default_runmode: str = "run"

    def _extract_config(self, build_config: BuildConfig):
        super()._extract_config(build_config)

        problem_class = build_config["problem_class"]
        if isinstance(problem_class, str):
            self._problem_class = get_class_from_module(problem_class)
        elif isinstance(problem_class, type):
            self._problem_class = problem_class
        else:
            raise BuilderError(
                "problem_class must either be a str pointing to the class to be loaded "
                f"or the class itself - got {problem_class}."
            )
        self._problem_settings = build_config.get("problem_settings", {})
        self._algorithm_name = build_config.get("algorithm_name", "SLSQP")
        self._opt_conditions = build_config.get("opt_conditions", {"max_eval": 100})
        self._opt_parameters = build_config.get("opt_parameters", {})

    def run(self, *args, **kwargs):
        """
        Optimise the shape using the provided parameterisation and optimiser.
        """
        bluemira_debug(
            f"""Setting up design problem with:
algorithm_name: {self._algorithm_name}
n_variables: {self._shape.variables.n_free_variables}
opt_conditions: {self._opt_conditions}
opt_parameters: {self._opt_parameters}"""
        )
        optimiser = Optimiser(
            self._algorithm_name,
            self._shape.variables.n_free_variables,
            self._opt_conditions,
            self._opt_parameters,
        )

        if self._problem_settings != {}:
            bluemira_debug(
                f"Applying non-default settings to problem: {self._problem_settings}"
            )
        self._design_problem = self._problem_class(
            self._shape,
            optimiser,
            *args,
            **kwargs,
            **self._problem_settings,
        )

        bluemira_print(
            f"Solving design problem: {self._design_problem.__class__.__name__}"
        )
        if self._shape.n_ineq_constraints > 0:
            bluemira_debug("Applying shape constraints")
            self._design_problem.apply_shape_constraints()

        bluemira_debug("Solving...")
        self._shape = self._design_problem.optimise()


class SimpleBuilderMixin:
    """
    A mixin class for building a single labelled component from an abstract Builder.
    """

    _label: str

    def __init__(self, *args, **kwargs):
        self._required_config = copy.deepcopy(super()._required_config) + ["label"]

        super().__init__(*args, **kwargs)

    def _extract_config(self, build_config: BuildConfig):
        super()._extract_config(build_config)

        self._label: str = build_config["label"]

    def build(self) -> Component:
        """
        Build the components from parameterised shapes using the provided configuration
        and parameterisation.

        Returns
        -------
        component: Component
            The Component built by this builder.
        """
        component = super().build()

        component.add_child(
            PhysicalComponent(self._label, self._shape.create_shape(label=self._label))
        )

        return component


class MakeParameterisedShape(SimpleBuilderMixin, ParameterisedShapeBuilder):
    """
    A builder that constructs a Component using a parameterised shape.
    """

    pass


class MakeOptimisedShape(SimpleBuilderMixin, OptimisedShapeBuilder):
    """
    A builder that constructs an optimised Component using a parameterised shape and
    design problem.
    """

    pass
