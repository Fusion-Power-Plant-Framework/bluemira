# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
OptimisationProblems for coilset design.

New optimisation schemes for the coilset can be provided by subclassing
from CoilsetOP, which is an abstract base class for OptimisationProblems
that use a coilset as their parameterisation object.

Subclasses must provide an optimise() method that returns an optimised
coilset according to a given optimisation objective function.
As the exact form of the state vector that is optimised is often
specific to each objective function, each subclass of CoilsetOP is
generally also specific to a given objective function, since
the method used to map the coilset object to the state vector
(and additional required arguments) will generally differ in each case.

"""

import numpy as np
import numpy.typing as npt

from bluemira.base.look_and_feel import bluemira_print_flush
from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraints import (
    UpdateableConstraint,
)
from bluemira.equilibria.optimisation.problem.base import (
    CoilsetOptimisationProblem,
    CoilsetOptimiserResult,
)
from bluemira.optimisation import Algorithm, AlgorithmType, optimise
from bluemira.utilities.positioning import PositionMapper


class NestedCoilsetPositionCOP(CoilsetOptimisationProblem):
    """
    Coilset OptimisationProblem for coil currents and positions
    subject to maximum current bounds and positions bounded within
    a provided region. Performs a nested optimisation for coil
    currents within each position optimisation function call.

    Parameters
    ----------
    sub_opt:
        Coilset OptimisationProblem to use for the optimisation of
        coil currents at each trial set of coil positions.
        sub_opt.coilset must exist, and will be modified
        during the optimisation.
    eq:
        Equilibrium object used to update magnetic field targets.
    position_mapper:
        position mapper object of the regions to optimise the coil positions within
    opt_algorithm:
        The optimisation algorithm to use (e.g. SLSQP)
    opt_conditions:
        The stopping conditions for the optimiser.
        for defaults see
        :class:`~bluemira.optimisation._algorithm.AlgorithDefaultTolerances`
        along with `max_eval=100`
    opt_parameters:
        Optimiser specific parameters,
        see https://nlopt.readthedocs.io/en/latest/NLopt_Reference/#algorithm-specific-parameters
        Otherwise, the parameters can be founded by digging through the source code.
    constraints:
        List of optimisation constraints to apply to the optimisation problem

    Notes
    -----
    Setting stopval and maxeval is the most reliable way to stop optimisation
    at the desired figure of merit and number of iterations respectively.
    Some NLOpt optimisers display unexpected behaviour when setting xtol and
    ftol, and may not terminate as expected when those criteria are reached.
    """

    def __init__(
        self,
        sub_opt: CoilsetOptimisationProblem,
        eq: Equilibrium,
        position_mapper: PositionMapper,
        opt_algorithm: AlgorithmType = Algorithm.SBPLX,
        opt_conditions: dict[str, float] | None = None,
        opt_parameters: dict[str, float] | None = None,
        constraints: list[UpdateableConstraint] | None = None,
    ):
        self.eq = eq
        self.position_mapper = position_mapper

        opt_dimension = self.position_mapper.dimension
        self.bounds = (np.zeros(opt_dimension), np.ones(opt_dimension))
        self.coilset = sub_opt.coilset
        self.sub_opt = sub_opt
        self.opt_algorithm = opt_algorithm
        self.opt_conditions = opt_conditions or self._opt_condition_defaults({
            "max_eval": 100
        })
        self.opt_parameters = opt_parameters
        self._constraints = [] if constraints is None else constraints

    def _get_initial_vector(self) -> npt.NDArray:
        """
        Get a vector representation of the initial coilset state from the PositionMapper.
        """
        cs_opt_state = self.coilset.get_optimisation_state(
            self.position_mapper.interpolator_names, current_scale=self.scale
        )
        return self.position_mapper.to_L(cs_opt_state.xs, cs_opt_state.zs)

    def optimise(self, x0: npt.NDArray | None = None):
        """
        Run the optimisation.

        Returns
        -------
        Optimised CoilSet object.
        """
        if x0 is None:
            x0 = self._get_initial_vector()

        eq_constraints, ineq_constraints = self._make_numerical_constraints(self.coilset)
        opt_result = optimise(
            f_objective=self.objective,
            x0=x0,
            algorithm=self.opt_algorithm,
            bounds=self.bounds,
            opt_conditions=self.opt_conditions,
            opt_parameters=self.opt_parameters,
            eq_constraints=eq_constraints,
            ineq_constraints=ineq_constraints,
        )

        optimal_positions = opt_result.x
        # Call the objective one last time, makes sure the coilset state
        # is set to the optimum
        self.objective(optimal_positions)

        return CoilsetOptimiserResult.from_opt_result(self.coilset, opt_result)

    def objective(self, vector: npt.NDArray[np.float64]) -> float:
        """Objective function to minimise."""
        pos_map = self.position_mapper.to_xz_dict(vector)
        self.coilset.set_optimisation_state(coil_position_map=pos_map)

        self.eq._remap_greens()
        self.eq._clear_OX_points()

        # Run the sub-optimisation
        sub_opt_result = self.sub_opt.optimise(fixed_coils=False)

        return sub_opt_result.f_x


class PulsedNestedPositionCOP(CoilsetOptimisationProblem):
    """
    Coilset position optimisation problem for multiple sub-optimisation problems.

    Parameters
    ----------
    coilset:
        Coilset for which to optimise positions
    position_mapper:
        Position mapper tool to parameterise coil positions
    sub_opt_problems:
        The list of sub-optimisation problems to solve
    opt_algorithm:
        The optimisation algorithm to use (e.g. SLSQP)
    opt_conditions:
        The stopping conditions for the optimiser.
        for defaults see
        :class:`~bluemira.optimisation._algorithm.AlgorithDefaultTolerances`
        along with `max_eval=100`
    constraints:
        Constraints to use. Note these should be applicable to the parametric position
        vector
    initial_currents:
        Initial currents to use when solving the current sub-optimisation problems
    debug:
        Whether or not to run in debug mode (will affect run-time noticeably)
    """

    def __init__(
        self,
        coilset: CoilSet,
        position_mapper: PositionMapper,
        sub_opt_problems: list[CoilsetOptimisationProblem],
        opt_algorithm: AlgorithmType = Algorithm.COBYLA,
        opt_conditions: dict[str, float] | None = None,
        opt_parameters: dict[str, float] | None = None,
        constraints=None,
        initial_currents=None,
        *,
        debug: bool = False,
    ):
        self.coilset = coilset
        self.position_mapper = position_mapper
        self.sub_opt_problems = sub_opt_problems
        self.opt_algorithm = opt_algorithm
        self.opt_conditions = opt_conditions or self._opt_condition_defaults({
            "max_eval": 100
        })
        self.opt_parameters = opt_parameters
        self._constraints = constraints

        if initial_currents:
            self.initial_currents = initial_currents / self.sub_opt_problems[0].scale
        else:
            self.initial_currents = np.zeros(coilset.get_control_coils().n_coils())
        self.debug = {0: debug}
        self.iter = {0: 0.0}
        opt_dimension = self.position_mapper.dimension
        self.bounds = (np.zeros(opt_dimension), np.ones(opt_dimension))

    @staticmethod
    def _run_reporting(itern, max_fom, verbose):
        """
        Keep track of objective function value over iterations.
        """
        i = max(list(itern.keys())) + 1
        itern[i] = max_fom

        if verbose:
            bluemira_print_flush(f"Coil position iteration {i} FOM value: {max_fom:.6e}")

    @staticmethod
    def _run_diagnostics(
        debug,
        sub_opt_prob: CoilsetOptimisationProblem,
        opt_result: CoilsetOptimiserResult,
    ):
        """
        In debug mode, store the LCFS at each iteration for each of the sub-optimisation
        problems.

        Notes
        -----
        This can significantly impact run-time.
        """
        if debug[0]:
            entry = max(list(debug.keys()))
            value = opt_result.f_x
            sub_opt_prob.eq._remap_greens()
            sub_opt_prob.eq._clear_OX_points()
            lcfs = sub_opt_prob.eq.get_LCFS()
            debug[entry].append([lcfs, value])

    def sub_opt_objective(
        self, vector: npt.NDArray[np.float64], *, verbose: bool = False
    ) -> float:
        """Run the sub-optimisations and return the largest figure of merit."""
        pos_map = self.position_mapper.to_xz_dict(vector)

        if self.debug[0]:
            # Increment debug dictionary
            i = max(list(self.debug.keys())) + 1
            self.debug[i] = []

        fom_values = []
        for sub_opt_prob in self.sub_opt_problems:
            sub_opt_prob.coilset.set_optimisation_state(coil_position_map=pos_map)

            result = sub_opt_prob.optimise(x0=self.initial_currents, fixed_coils=False)

            self._run_diagnostics(self.debug, sub_opt_prob, result)
            fom_values.append(result.f_x)
        max_fom = max(fom_values)

        self._run_reporting(self.iter, max_fom, verbose)
        return max_fom

    def objective(
        self, vector: npt.NDArray[np.float64], *, verbose: bool = False
    ) -> float:
        """The objective function of the parent optimisation."""
        return self.sub_opt_objective(vector, verbose=verbose)

    def _get_initial_vector(self) -> npt.NDArray[np.float64]:
        """
        Get a vector representation of the initial coilset state from the PositionMapper.
        """
        cs_opt_state = self.coilset.get_optimisation_state(
            self.position_mapper.interpolator_names, current_scale=self.scale
        )
        return self.position_mapper.to_L(cs_opt_state.xs, cs_opt_state.zs)

    def optimise(
        self, x0: npt.NDArray | None = None, *, verbose: bool = False
    ) -> CoilsetOptimiserResult:
        """
        Run the PulsedNestedPositionCOP

        Parameters
        ----------
        x0:
            Initial solution vector (parameterised positions)
        verbose:
            Whether or not to print progress information

        Returns
        -------
        coilset:
            Optimised CoilSet
        """
        if x0 is None:
            x0 = self._get_initial_vector()

        eq_constraints, ineq_constraints = self._make_numerical_constraints(self.coilset)
        opt_result = optimise(
            f_objective=lambda vector: self.objective(vector, verbose=verbose),
            x0=x0,
            df_objective=None,  # use a numerical approximation if needed
            algorithm=self.opt_algorithm,
            opt_conditions=self.opt_conditions,
            opt_parameters=self.opt_parameters,
            bounds=self.bounds,
            eq_constraints=eq_constraints,
            ineq_constraints=ineq_constraints,
        )

        optimal_positions = opt_result.x
        # Call the objective one last time, makes sure the coilset state
        # is set to the optimum
        self.sub_opt_objective(optimal_positions)

        # Clean up state of Equilibrium objects
        for sub_opt in self.sub_opt_problems:
            sub_opt.eq._remap_greens()
            sub_opt.eq._clear_OX_points()

        return CoilsetOptimiserResult.from_opt_result(self.coilset, opt_result)
