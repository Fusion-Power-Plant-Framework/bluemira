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

from typing import Dict, List, Type

from ..base.builder import BuildConfig, Builder, BuildResult
from ..base.components import PhysicalComponent
from ..geometry.optimisation import GeometryOptimisationProblem
from ..geometry.parameterisations import GeometryParameterisation
from ..utilities.optimiser import Optimiser
from ..utilities.tools import get_class_from_module


class ParameterisedShapeBuilder(Builder):
    """
    Abstract builder class for building parameterised shapes.
    """

    _required_config = ["param_class", "variables_map"]

    _param_class: Type[GeometryParameterisation]
    _variables_map: Dict[str, str]

    def _extract_config(self, build_config: BuildConfig):
        self._param_class: Type[GeometryParameterisation] = get_class_from_module(
            build_config["param_class"],
            default_module="bluemira.geometry.parameterisations",
        )
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

    def reinitialise(self, params, **kwargs) -> None:
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
        shape = self._param_class()
        for key, val in shape_params.items():
            if isinstance(val, dict):
                shape.adjust_variable(key, **val)
            else:
                shape.adjust_variable(key, val)
        self._shape = shape


class MakeParameterisedShape(ParameterisedShapeBuilder):
    """
    A builder that constructs a Component using a parameterised shape.
    """

    _required_config = ParameterisedShapeBuilder._required_config + ["target"]

    _target: str

    def _extract_config(self, build_config: BuildConfig):
        super()._extract_config(build_config)

        self._target: str = build_config["target"]

    def build(self, **kwargs) -> List[BuildResult]:
        """
        Build the components from parameterised shapes using the provided configuration
        and parameterisation.

        Returns
        -------
        build_results: List[BuildResult]
            The Components built by this builder, including the target paths. For this
            Builder the results will contain one item.
        """
        super().build(**kwargs)

        target = self._target.split("/")
        return [
            BuildResult(
                target="/".join(target),
                component=PhysicalComponent(target[-1], self._shape.create_shape()),
            )
        ]


class MakeOptimisedShape(MakeParameterisedShape):
    """
    A builder that constructs a Component using a parameterised shape.
    """

    _required_config = MakeParameterisedShape._required_config + ["problem_class"]

    _problem_class: Type[GeometryOptimisationProblem]

    def __call__(self, params, optimise=True, **kwargs):
        """
        Perform the full build process, including reinitialisation and optimisation,
        using the provided parameters.

        Parameters
        ----------
        params: Dict[str, Any]
            The parameterisation containing at least the required params for this
            Builder.
        optimise: bool
            If True then the build will include optimisation, by default True.

        Returns
        -------
        build_results: List[BuildResult]
            The Components build by this builder, including the target paths.
        """
        self.reinitialise(params)
        if optimise:
            self.optimise()
        return self.build()

    def _extract_config(self, build_config: BuildConfig):
        super()._extract_config(build_config)

        problem_class = build_config["problem_class"]
        self._problem_class: Type[GeometryOptimisationProblem] = get_class_from_module(
            problem_class
        )
        self._algorithm_name = build_config.get("algorithm_name", "SLSQP")
        self._opt_conditions = build_config.get("opt_conditions", {"max_eval": 100})
        self._opt_parameters = build_config.get("opt_parameters", {})

    def optimise(self):
        """
        Optimise the shape using the provided parameterisation and optimiser.
        """
        optimiser = Optimiser(
            self._algorithm_name,
            self._shape.variables.n_free_variables,
            self._opt_conditions,
            self._opt_parameters,
        )
        problem = self._problem_class(self._shape, optimiser)
        problem.solve()
