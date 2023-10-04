# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
PROCESS IN.DAT template builder
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from enum import EnumType

    from bluemira.codes.process._equation_variable_mapping import Constraint, Objective
    from bluemira.codes.process._model_mapping import PROCESSOptimisationAlgorithm
    from bluemira.codes.process.api import _INVariable

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.codes.process._equation_variable_mapping import (
    FV_CONSTRAINT_ITVAR_MAPPING,
    ITERATION_VAR_MAPPING,
    OBJECTIVE_MIN_ONLY,
    VAR_ITERATION_MAPPING,
)
from bluemira.codes.process._inputs import ProcessInputs
from bluemira.codes.process.api import Impurities


class PROCESSTemplateBuilder:
    """
    An API patch to make PROCESS a little easier to work with before
    the PROCESS team write a Python API.
    """

    def __init__(self):
        self.models: Dict[str, int] = {}
        self.values: Dict[str, Any] = {}
        self.variables: Dict[str, float] = {}
        self.bounds: Dict[str, Dict[str, str]] = {}
        self.icc: List[int] = []
        self.ixc: List[int] = []
        self.fimp: List[float] = 14 * [0.0]

        self.minmax: int = 0
        self.ioptimiz: int = 0
        self.maxcal: int = 1000
        self.epsvmc: float = 1.0e-8

    def set_optimisation_algorithm(self, algorithm_choice: PROCESSOptimisationAlgorithm):
        """
        Set the optimisation algorithm to use
        """
        self.ioptimiz = algorithm_choice.value

    def set_optimisation_numerics(
        self, max_iterations: int = 1000, tolerance: float = 1e-8
    ):
        """
        Set optimisation numerics
        """
        self.maxcal = max_iterations
        self.epsvmc = tolerance

    def set_minimisation_objective(self, objective: Objective):
        """
        Set the minimisation objective equation to use when running PROCESS
        """
        self.minmax = objective.value

    def set_maximisation_objective(self, objective: Objective):
        """
        Set the maximisation objective equation to use when running PROCESS
        """
        minmax = objective.value
        if minmax in OBJECTIVE_MIN_ONLY:
            raise ValueError(
                f"Equation {objective} can only be used as a minimisation objective."
            )
        self.minmax = -minmax

    def set_model(self, model_choice: EnumType):
        """
        Set a model switch to the PROCESS run
        """
        self.models[model_choice.switch_name] = model_choice.value

    def add_constraint(self, constraint: Constraint):
        """
        Add a constraint to the PROCESS run
        """
        if constraint.value in self.icc:
            bluemira_warn(
                f"Constraint {constraint.name} is already in the constraint list."
            )

        if constraint.value in FV_CONSTRAINT_ITVAR_MAPPING:
            # Sensible (?) defaults. bounds are standard PROCESS for f-values for _most_
            # f-value constraints.
            self.add_fvalue_constraint(constraint, 0.5, 1e-3, 1.0)
        else:
            self.icc.append(constraint.value)

    def add_fvalue_constraint(
        self,
        constraint: Constraint,
        value: float,
        lower_bound: float = 1e-3,
        upper_bound: float = 1.0,
    ):
        """
        Add an f-value constraint to the PROCESS run
        """
        if constraint.value not in FV_CONSTRAINT_ITVAR_MAPPING:
            raise ValueError(
                f"Constraint '{constraint.name}' is not an f-value constraint."
            )
        self.icc.append(constraint.value)
        itvar = FV_CONSTRAINT_ITVAR_MAPPING[constraint.value]
        if itvar not in self.ixc:
            self.add_variable(
                VAR_ITERATION_MAPPING[itvar], value, lower_bound, upper_bound
            )

    def add_variable(
        self,
        name: str,
        value: float,
        lower_bound: Optional[float] = None,
        upper_bound: Optional[float] = None,
    ):
        """
        Add an iteration variable to the PROCESS run
        """
        itvar = ITERATION_VAR_MAPPING.get(name, None)
        if not itvar:
            raise ValueError(f"There is no iteration variable: '{name}'")

        if itvar in self.ixc:
            bluemira_warn(
                f"Iterable variable {name} is already in the variable list. Updating value and bounds."
            )
            self._add_to_dict(self.variables, name, value)

            if lower_bound:
                self.bounds[str(itvar)]["l"] = str(lower_bound)
            if upper_bound:
                self.bounds[str(itvar)]["u"] = str(upper_bound)

        else:
            self.ixc.append(itvar)
            self._add_to_dict(self.variables, name, value)

        if lower_bound or upper_bound:
            var_bounds = {}
            if lower_bound:
                var_bounds["l"] = str(lower_bound)
            if upper_bound:
                var_bounds["u"] = str(upper_bound)
            self.bounds[str(itvar)] = var_bounds

    def add_input_value(self, name: str, value: str):
        """
        Add a fixed input value to the PROCESS run
        """
        if name in self.values:
            bluemira_warn(f"Over-writing {name} from {self.values[name]} to {value}")
        self._add_to_dict(self.values, name, value)

    def add_input_values(self, mapping: Dict[str, Any]):
        """
        Add a dictionary of fixed input values to the PROCESS run
        """
        for name, value in mapping.items():
            self.add_input_value(name, value)

    def add_impurity(self, impurity: EnumType, value: float):
        """
        Add an impurity concentration
        """
        idx = impurity.value - 1
        self.fimp[idx] = value

    def _add_to_dict(self, mapping: Dict[str, Any], name: str, value: Any):
        if "fimp(" in name:
            num = int(name.strip("fimp(")[:2])
            impurity = Impurities(num)
            self.add_impurity(impurity, value)
        else:
            mapping[name] = value

    def make_inputs(self) -> Dict[str, _INVariable]:
        """
        Make the ProcessInputs InVariable for the specified template
        """
        if self.ioptimiz != 0 and self.minmax == 0:
            bluemira_warn(
                "You are running in optimisation mode, but have not set an objective function."
            )

        return ProcessInputs(
            bounds=self.bounds,
            icc=self.icc,
            ixc=self.ixc,
            minmax=self.minmax,
            ioptimz=self.ioptimiz,
            epsvmc=self.epsvmc,
            maxcal=self.maxcal,
            fimp=self.fimp,
            **self.values,
            **self.models,
            **self.variables,
        ).to_invariable()
