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
An example/tutorial of how to write a Solver and Task classes.
"""

# %%[markdown]
# # Writing a Solver

# %%[markdown]
# Bluemira provides an interface to write a general solver that performs
# a setup, run, and teardown stage.
# Bluemira can run these solvers during design stages.
#
# To define a solver, inherit from `SolverABC` and implement the interface.
# As an example, let's implement a solver to remove noise from a gaussian curve.
# We use `scipy` to fit a gaussian to some noisy data,
# then evaluate the fitting parameters to return a smooth curve.

# %%
import enum
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from bluemira.base.parameter import ParameterFrame
from bluemira.base.solver import RunMode, SolverABC, Task

# %%[markdown]
# Let's generate some noisy (Gaussian) data to use in the example.


# %%
def gaussian(x: np.ndarray, sigma: float, mu: float, vertical_offset: float):
    """Apply the Gaussian function to x."""
    exponent = -1 / 2 * ((x - mu) / sigma) ** 2
    return 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(exponent) + vertical_offset


def generate_noisy_gaussian(
    x: np.ndarray, sigma: float, mu: float, vertical_offset: float, rng_seed: int = 0
):
    """Generate some gaussian data with random noise."""
    gauss_y = gaussian(x, sigma, mu, vertical_offset)
    rng = np.random.default_rng(rng_seed)
    return gauss_y + rng.uniform(-0.05, 0.05, len(x))


# %%
x = np.linspace(-10, 10, 200)
gauss_data = generate_noisy_gaussian(x, 2, 1.8, 1.5)
gauss_params = ParameterFrame(
    [
        ["sigma", "sigma", 1.0, "dimensionless", "Standard Deviation", "Input"],
        ["mu", "mu", 1.0, "dimensionless", "Expectation value", "Input"],
        [
            "vertical_offset",
            "vertical_offset",
            0.0,
            "dimensionless",
            "The vertical offset",
            "Input",
        ],
        ["x", "x", x, "dimensionless", "Problem space", "Input"],
        [
            "gauss_data",
            "gauss_data",
            gauss_data,
            "dimensionless",
            "Gauss signal",
            "Input",
        ],
    ]
)

plt.plot(x, gauss_data)

# %%[markdown]
# First we need to enumerate the ways to run our solver.
# Your new `Enum` should inherit from `solver.RunMode`
# and define the possible run modes for your solver.
# Generally, your solver will always have a `RUN` mode.
# Common examples of other run modes are `READ` and `MOCK`.


# %%
class GaussFitRunMode(RunMode):
    """Enumeration of the run modes for the GaussFit solver"""

    RUN = enum.auto()
    MOCK = enum.auto()


# %%[markdown]
# Next you must define a `Task` class for each of the setup, run,
# and teardown stages of the problem.
# Each task must can define a method corresponding to each name in the
# enum (written in lowercase) we just defined.
# In this case, we've defined `RUN` and `MOCK` run modes,
# so we should define methods `run` and `mock` in at least one of our tasks.
# If a run mode is not defined for a given task, that stage is skipped.

# Having separate setup, run, and teardown classes can be useful.
# For example, writing a solver for an external program like PROCESS,
# you should ideally be able to re-use the setup/teardown tasks
# as the parameter mappings performed should be similar.

# If a solver stage is not required,
# for example if it needs no teardown stage,
# you can assign the special `NoOpTask` class to the property.
# This will skip the stage.

# %%
class GaussFitSetup(Task):
    """
    Task to set up the gaussian fitting problem.

    This makes some estimates for the fitting parameters.
    """

    def __init__(self, params: ParameterFrame):
        super().__init__(params)
        self._x = self.params["x"]
        self._y = self.params["y"]

    def run(self) -> Dict[str, float]:
        """
        Set up the fitting problem; estimate some fitting parameters.
        """
        return {
            "sigma": self._estimate_sigma(),
            "mu": self._estimate_mu(),
            "vertical_offset": self._estimate_vertical_offset(),
        }

    def mock(self) -> Dict[str, float]:
        """
        Return a calculation-free estimate of fitting parameters.
        """
        return {
            "sigma": 1.0,
            "mu": 1.0,
            "vertical_offset": 0.0,
        }

    def _estimate_mu(self) -> float:
        """The x-value where y is at its maximum estimates the mean."""
        return self._x[np.argmax(self._y)]

    def _estimate_sigma(self) -> float:
        """Estimate x distance between the half heights and halve."""
        half_height = (np.max(self._y) + np.min(self._y)) / 2
        x_above_hh = self._x[self._y > half_height]
        return abs(x_above_hh[0] - x_above_hh[-1]) / 2

    def _estimate_vertical_offset(self) -> float:
        """The minimum value of y; only noise makes this not exact."""
        return np.min(self._y)


class GaussFitRun(Task):
    """
    Task to run the fitting algorithm.

    This implements a "run" method, this executes the fitting algorithm.
    As no "mock" method is defined, when this task is called in "MOCK"
    mode within a solver, it will do nothing.
    """

    def __init__(self, params: ParameterFrame):
        super().__init__(params)

    def run(self, setup_result: Dict[str, float]) -> Dict[str, float]:
        """Run the fit."""
        initial_guess = (
            setup_result["sigma"],
            setup_result["mu"],
            setup_result["vertical_offset"],
        )
        opt, _ = curve_fit(
            gaussian, self.params["x"], self.params["y"], p0=initial_guess
        )
        return {
            "sigma": opt[0],
            "mu": opt[1],
            "vertical_offset": opt[2],
        }


class GaussFitTeardown(Task):
    """
    Task to teardown the solver.

    This will typically be used to free resources (e.g., delete files),
    but it's used here to convert the fitting parameters to coordinates.
    This finishes the solver by effectively removing the noise from the
    original data.
    """

    def __init__(self, params: ParameterFrame):
        super().__init__(params)

    def run(self, run_result: Dict[str, float]) -> np.ndarray:
        """Run the teardown procedure."""
        return gaussian(self.params["x"], **run_result)

    def mock(self, run_result: Dict[str, float]) -> np.ndarray:
        """
        Run the teardown procedure in 'mock' mode.

        This is equivalent to the 'run' mode in this case.
        """
        return self.run(run_result)


# %%[markdown]
# Now defining the solver is easy.
# We set the abstract properties to the relevant tasks we've written,
# then the solver is ready to execute for any of our run modes.


# %%
class GaussFitSolver(SolverABC):
    """
    Solver for removing noise from some Gaussian data using a fit.
    """

    setup_cls = GaussFitSetup
    run_cls = GaussFitRun
    teardown_cls = GaussFitTeardown
    run_mode_cls = GaussFitRunMode

    def execute(self, run_mode: RunMode) -> np.ndarray:
        """
        Execute the setup, run, and teardown tasks of this solver.

        As this method is only calling out to its parent class, we don't
        technically need it. But it can be useful to provide typing for
        the return value.
        """
        return super().execute(run_mode)


# %%[markdown]
# We can run the solver in `RUN` mode:

# %%
params = {"x": x, "y": gauss_data}

gauss_solver = GaussFitSolver(params)
result = gauss_solver.execute(GaussFitRunMode.RUN)

plt.plot(params["x"], params["y"])
plt.plot(params["x"], result)

# %%[markdown]
# And we can run the solver in `MOCK` mode:

# %%
result = gauss_solver.execute(GaussFitRunMode.MOCK)

plt.plot(params["x"], params["y"])
plt.plot(params["x"], result)
