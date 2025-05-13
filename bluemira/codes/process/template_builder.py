# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
PROCESS IN.DAT template builder
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.codes.process._inputs import ProcessInputs
from bluemira.codes.process.api import Impurities
from bluemira.codes.process.equation_variable_mapping import (
    FV_CONSTRAINT_ITVAR_MAPPING,
    ITERATION_VAR_MAPPING,
    OBJECTIVE_MIN_ONLY,
    VAR_ITERATION_MAPPING,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from bluemira.codes.process.equation_variable_mapping import (
        Constraint,
        ConstraintSelection,
        Objective,
    )
    from bluemira.codes.process.model_mapping import (
        PROCESSModel,
        PROCESSOptimisationAlgorithm,
    )


class PROCESSTemplateBuilder:
    """
    An API patch to make PROCESS a little easier to work with before
    the PROCESS team write a Python API.
    """

    def __init__(self):
        self._models: dict[str, PROCESSModel] = {}
        self._constraints: list[Constraint] = []
        self.values: dict[str, Any] = {}
        self.variables: dict[str, float] = {}
        self.bounds: dict[str, dict[str, str]] = {}
        self.ixc: list[int] = []
        self.fimp: list[float] = 14 * [0.0]

        self.minmax: int = 0
        self.ioptimiz: int = 0
        self.maxcal: int = 1000
        self.epsvmc: float = 1.0e-8

    def set_run_title(self, run_title: str):
        """
        Set the run title
        """
        self.values["runtitle"] = run_title

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

        Parameters
        ----------
        max_iterations:
            maximum number of iteration/calculations in process.
        toleratnce:
            VMCON tolerance epsvmc
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

        Raises
        ------
        ValueError
            Objective can only be a minimisation
        """
        minmax = objective.value
        if minmax in OBJECTIVE_MIN_ONLY:
            raise ValueError(
                f"Equation {objective} can only be used as a minimisation objective."
            )
        self.minmax = -minmax

    def set_model(self, model_choice: PROCESSModel):
        """
        Set a model switch to the PROCESS run
        """
        if model_choice.switch_name in self._models:
            bluemira_warn(f"Over-writing model choice {model_choice}.")
        self._models[model_choice.switch_name] = model_choice

    def add_constraint(self, constraint: Constraint):
        """
        Add a constraint to the PROCESS run
        """
        if constraint in self._constraints:
            bluemira_warn(
                f"Constraint {constraint.name} is already in the constraint list."
            )

        if constraint.value in FV_CONSTRAINT_ITVAR_MAPPING:
            # Sensible (?) defaults. bounds are standard PROCESS for f-values for _most_
            # f-value constraints.
            self.add_fvalue_constraint(constraint, None, None, None)
        else:
            self._constraints.append(constraint)

    def add_fvalue_constraint(
        self,
        constraint: Constraint,
        value: float | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ):
        """
        Add an f-value constraint to the PROCESS run

        Raises
        ------
        ValueError
            Constraint not an f-value constraint
        """
        if constraint.value not in FV_CONSTRAINT_ITVAR_MAPPING:
            raise ValueError(
                f"Constraint '{constraint.name}' is not an f-value constraint."
            )
        self._constraints.append(constraint)

        itvar = FV_CONSTRAINT_ITVAR_MAPPING[constraint.value]
        if itvar not in self.ixc:
            self.add_variable(
                VAR_ITERATION_MAPPING[itvar], value, lower_bound, upper_bound
            )

    def add_variable(
        self,
        name: str,
        value: float | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ):
        """
        Add an iteration variable to the PROCESS run

        Raises
        ------
        ValueError
            Iteration variable not found
        """
        itvar = ITERATION_VAR_MAPPING.get(name, None)
        if not itvar:
            raise ValueError(f"There is no iteration variable: '{name}'")

        if itvar in self.ixc:
            bluemira_warn(
                f"Iteration variable '{name}' is already in the variable list."
                " Updating value and bounds."
            )
            self.adjust_variable(name, value, lower_bound, upper_bound)

        else:
            self.ixc.append(itvar)
            self._add_to_dict(self.variables, name, value)

        if lower_bound or upper_bound:
            var_bounds = {}
            if lower_bound:
                var_bounds["l"] = str(lower_bound)
            if upper_bound:
                var_bounds["u"] = str(upper_bound)
            if var_bounds:
                self.bounds[str(itvar)] = var_bounds

    def adjust_variable(
        self,
        name: str,
        value: float | None = None,
        lower_bound: float | None = None,
        upper_bound: float | None = None,
    ):
        """
        Adjust an iteration variable in the PROCESS run

        Raises
        ------
        ValueError
            No iteration variable found
        """
        itvar = ITERATION_VAR_MAPPING.get(name, None)
        if not itvar:
            raise ValueError(f"There is no iteration variable: '{name}'")
        if itvar not in self.ixc:
            bluemira_warn(
                f"Iteration variable '{name}' is not in the variable list. Adding it."
            )
            self.add_variable(name, value, lower_bound, upper_bound)
        else:
            self._add_to_dict(self.variables, name, value)
            if (lower_bound or upper_bound) and str(itvar) not in self.bounds:
                self.bounds[str(itvar)] = {}

            if lower_bound:
                self.bounds[str(itvar)]["l"] = str(lower_bound)
            if upper_bound:
                self.bounds[str(itvar)]["u"] = str(upper_bound)

    def add_input_value(self, name: str, value: float | Iterable[float]):
        """
        Add a fixed input value to the PROCESS run
        """
        if name in self.values:
            bluemira_warn(f"Over-writing {name} from {self.values[name]} to {value}")
        self._add_to_dict(self.values, name, value)

    def add_input_values(self, mapping: dict[str, Any]):
        """
        Add a dictionary of fixed input values to the PROCESS run
        """
        for name, value in mapping.items():
            self.add_input_value(name, value)

    def add_impurity(self, impurity: Impurities, value: float):
        """
        Add an impurity concentration
        """
        idx = impurity.value - 1
        self.fimp[idx] = value

    def _add_to_dict(self, mapping: dict[str, Any], name: str, value: Any):
        if "fimp(" in name:
            num = int(name.strip("fimp(")[:2])
            impurity = Impurities(num)
            self.add_impurity(impurity, value)
        else:
            mapping[name] = value

    def _check_model_inputs(self):
        """
        Check the required inputs for models have been provided.
        """
        for model in self._models.values():
            self._check_missing_inputs(model)

    def _check_constraint_inputs(self):
        """
        Check the required inputs for the constraints have been provided
        """
        for constraint in self._constraints:
            self._check_missing_iteration_variables(constraint)
            self._check_missing_inputs(constraint)

    def _check_missing_inputs(self, model: PROCESSModel | ConstraintSelection):
        missing_inputs = [
            input_name
            for input_name in model.requires_values
            if (input_name not in self.values and input_name not in self.variables)
        ]

        if missing_inputs:
            model_name = f"{model.__class__.__name__}.{model.name}"
            inputs = ", ".join([f"'{inp}'" for inp in missing_inputs])
            bluemira_warn(
                f"{model_name} requires inputs {inputs} which have not been specified."
                " Default values will be used."
            )

    def _check_missing_iteration_variables(self, constraint: ConstraintSelection):
        missing_itv = [
            VAR_ITERATION_MAPPING[itv_num]
            for itv_num in constraint.requires_variables
            if (
                VAR_ITERATION_MAPPING[itv_num] not in self.variables
                and VAR_ITERATION_MAPPING[itv_num] not in self.values
            )
        ]
        if missing_itv:
            con_name = f"{constraint.__class__.__name__}.{constraint.name}"
            inputs = ", ".join([f"'{inp}'" for inp in missing_itv])
            bluemira_warn(
                f"{con_name} requires iteration variable {inputs} "
                "which have not been specified. Default values will be used."
            )

    def make_inputs(self) -> ProcessInputs:
        """
        Make the ProcessInputs InVariable for the specified template
        """  # noqa: DOC201
        if self.ioptimiz != 0 and self.minmax == 0:
            bluemira_warn(
                "You are running in optimisation mode,"
                " but have not set an objective function."
            )

        self._check_constraint_inputs()
        icc = [con.value for con in self._constraints]
        self._check_model_inputs()
        models = {k: v.value for k, v in self._models.items()}

        return ProcessInputs(
            bounds=self.bounds,
            icc=icc,
            ixc=self.ixc,
            minmax=self.minmax,
            ioptimz=self.ioptimiz,
            epsvmc=self.epsvmc,
            maxcal=self.maxcal,
            fimp=self.fimp,
            **self.values,
            **models,
            **self.variables,
        )
