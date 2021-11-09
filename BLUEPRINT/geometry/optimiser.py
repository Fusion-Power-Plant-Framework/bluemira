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
Shape optimiser object and interfaces to scipy.
"""

import time
from scipy.optimize import minimize, differential_evolution, shgo
from bluemira.base.look_and_feel import bluemira_print
from bluemira.utilities.opt_tools import process_scipy_result
from BLUEPRINT.utilities.optimisation import convert_scipy_constraints

# Mapping of the best algorithm for each parameterisation
SHAPE_ALGO_MAP = {
    "PolySpline": "SLSQP",
    "PictureFrame": "SLSQP",
    "TripleArc": "SLSQP",
    "PrincetonD": "SLSQP",
    "BackwardPolySpline": "SLSQP",
    "TaperedPictureFrame": "SLSQP",
    "CurvedPictureFrame": "SLSQP",
}
# TODO: Confirm best algorithms on real problems...

# fmt:off
ALGO_KWARGS_MAP = {
    "SLSQP": {
        "ftol": 2e-3,                  # Precision goal for the value of f in the stopping criterion
        "eps": 1.4901161193847656e-8,  # Step size used for numerical approximation of the Jacobian
        "maxiter": 50,                 # Maximum number of iterations
    },
    "COBYLA": {
        "rhobeg": 1.0,                 # Reasonable initial changes to the variables
        "tol": 0.002,                  # Final accuracy in the optimization
        "catol": 0.0002,               # Tolerance (absolute) for constraint violations
        "maxiter": 100                 # Maximum number of function evaluations
    },
    "trust-constr": {
        "xtol": 2e-8,                  # Tolerance for termination by the change of the independent variable
        "gtol": 2e-3,                  # Tolerance for termination by the norm of the Lagrangian gradient
        "barrier_tol": 2e-3,           # Threshold on the barrier parameter for the algorithm termination
        "maxiter": 200                 # Maximum number of algorithm iterations
    },
    "DE": {
        "tol": 0.1,                    # Relative tolerance for convergence
        "popsize": 25,                 # A multiplier for setting the total population size
        "recombination": 0.7,          # The recombination a.k.a. crossover constant
        "maxiter": 15,                 # This is the number of generations, not func evals
    },
    "SHGO": {},
    "Nelder-Mead": {},
    "BFGS": {},
}
# fmt: on

SHAPE_KWARGS_MAP = {
    "PolySpline": {},
    "BackwardPolySpline": {},
    "PrincetonD": {"ftol": 2e-3, "eps": 1e-1},
    "PictureFrame": {"ftol": 2e-3, "eps": 2e-1},
    "TripleArc": {},
    "TaperedPictureFrame": {"ftol": 2e-3, "eps": 1e-3},
    "CurvedPictureFrame": {"ftol": 2e-3, "eps": 1e-1},
}

GRADIENT_BASED = ["SLSQP", "COBYLA", "trust-constr", "Nelder-Mead", "BFGS"]

GLOBAL_ALGO_MAP = {
    "DE": differential_evolution,
    "SHGO": shgo,
}


class ShapeOptimiser:
    """
    An optimiser object for Shapes

    Parameters
    ----------
    parameterisation: Type[Parameterisation]
        The Shape Parameterisation to optimise
    f_objective: callable
        The objective function of the optimisation problem
    f_ieq_constraints: callable
        The inequality constraints function of the optimisation problem
    f_eq_constraints: callable
        The equality constraints function of the optimisation problem
    args: tuple
        The additional arguments to pass into the optimisation functions (aside
        from the variable vector)
    algorithm: str
        The name of the optimisation algorithm to use

    Other Parameters
    ----------------
    opt_kwargs: dict
        The various optimisation tweaking parameters to use in the algorithm.
        Depends on the algorithm being used.
    """

    def __init__(
        self,
        parameterisation,
        f_objective,
        f_ieq_constraints=None,
        f_eq_constraints=None,
        args=None,
        algorithm=None,
        **opt_kwargs,
    ):
        self.algorithm = algorithm

        self.parameterisation = parameterisation

        if algorithm is None:
            # Get the most suitable optimisation algorithm if not specified
            algorithm = SHAPE_ALGO_MAP[parameterisation.name]

        self.algorithm = algorithm

        if args is None:
            # No additional arguments to feed into f_objective and f_constraints
            args = []
        self.args = args

        # Handle optimiser keyword arguments
        defaults = ALGO_KWARGS_MAP[algorithm]

        # Handle shape-specific kwarg-tweaks

        shape_kwargs = SHAPE_KWARGS_MAP[parameterisation.name]

        # Overwrite defaults with shape kwargs and specified minimiser kwargs
        # If no kwargs specified, will default safely.
        self.opt_kwargs = {**defaults, **shape_kwargs, **opt_kwargs}

        self.f_objective = f_objective
        self.constraints = []

        if f_eq_constraints:
            self.constraints.append(
                {"type": "eq", "fun": f_eq_constraints, "args": args}
            )
        if f_ieq_constraints:
            self.constraints.append(
                {"type": "ineq", "fun": f_ieq_constraints, "args": args}
            )

        self.constraints = tuple(self.constraints)

        # Constructors
        self.rms = None
        self.result = None

        # Assign the call method depending on the type of optimiser selected
        if algorithm in GRADIENT_BASED:
            self.__call__ = self.optimise_grad_based
        else:
            self.__call__ = self.optimise_global

    def __call__(self, verbose=False):
        """
        Perform the optimisation.

        Parameters
        ----------
        verbose: bool
            Whether or not to display the optimiser progress and information.

        Returns
        -------
        xnorm: np.array
            The normalised vector of optimal values
        """
        x_0, b_norm = self.parameterisation.set_oppvar()

        # Set the optimising verbosity. Will print convergence info
        self.opt_kwargs["disp"] = verbose

        tic = time.time()
        if self.algorithm in GRADIENT_BASED:
            result = self.optimise_grad_based(x_0, b_norm, self.opt_kwargs)
        else:
            result = self.optimise_global(b_norm)

        # We always want to see this (for now)
        bluemira_print(
            f"{self.parameterisation.name} optimisation time: {time.time()-tic:1.1f} s"
        )

        # Store result
        self.result = result

        # Check termination of optimisation algorithm
        x_norm = process_scipy_result(result)

        return x_norm

    def optimise_grad_based(self, x_0, b_norm, opt_kwargs):
        """
        Runs gradient-based optimisations algorithm on the Shape optimistion
        problem
        """
        result = minimize(
            self.f_objective,
            x_0,
            args=self.args,
            bounds=b_norm,
            constraints=self.constraints,
            method=self.algorithm,
            options=self.opt_kwargs,
        )
        return result

    def optimise_global(self, b_norm):
        """
        Runs non-gradient-based algorithms on the Shape optimisation
        """
        # Get the global optimiser handle
        minimise = GLOBAL_ALGO_MAP[self.algorithm]

        if self.algorithm == "DE":
            # Convert constraints because scipy has multiple personalities
            self.constraints = convert_scipy_constraints(self.constraints)

        if self.algorithm == "SHGO":
            # Drop kwargs because scipy has multiple personalities
            self.opt_kwargs = {}

        result = minimise(
            self.f_objective,
            bounds=b_norm,
            args=self.args,
            constraints=self.constraints,
            **self.opt_kwargs,
        )
        return result
