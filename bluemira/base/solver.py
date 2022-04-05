# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
General solver class.
"""

import abc
import enum

from bluemira.base.parameter import ParameterFrame


class TaskRunMode(enum.Enum):
    """Enum defining the runmodes for a task."""

    RUN = enum.auto()
    MOCK = enum.auto()
    READ = enum.auto()


class Task(abc.ABC):
    """
    A general task to be called as part of a solver.
    """

    _result = None

    @abc.abstractmethod
    def run(self, params: ParameterFrame):
        """Run the task."""
        return

    @abc.abstractmethod
    def mock(self, params: ParameterFrame):
        """Return a mock value for the task."""
        return

    @abc.abstractmethod
    def read(self, params: ParameterFrame):
        """Return a saved result for the task, usually read from a file."""
        return

    @property
    def result(self):
        """
        Holds the result of the task after it's been run. Is None if the
        task has not been run.
        """
        return self._result


class SolverABC(abc.ABC):
    """
    A base class for general solvers using setup, run and teardown
    stages.

    This interface gives a general way for callers to run external, or
    Bluemira, solvers.

    Parameters
    ----------
    setup_task: Task
        The task to run as setup
    """

    def __init__(self, setup_task: Task, run_task: Task, teardown_task: Task):
        self._setup_task = setup_task
        self._run_task = run_task
        self._teardown_task = teardown_task

    def setup(self, params: ParameterFrame, run_mode: TaskRunMode = TaskRunMode.RUN):
        """
        Set up the solver.

        Typically, this method is used to perform parameter mappings for
        some external code, or deriving dependent parameters. But it can
        be used to perform any required non-computational set up.
        """
        self._run_task_with_mode(self._setup_task, params, run_mode)

    def run(self, params: ParameterFrame, run_mode: TaskRunMode = TaskRunMode.RUN):
        """
        Run the solver.

        This is where computations should be made. This may be something
        like calling a Bluemira problem, or executing some external code
        or process.
        """
        self._run_task_with_mode(self._run_task, params, run_mode)

    def teardown(self, params: ParameterFrame, run_mode: TaskRunMode = TaskRunMode.RUN):
        """
        Clean up the solver.

        This method should perform any clean-up operations required by
        the solver. This may be deleting temporary files, or could
        involve mapping parameters from some external code to Bluemira
        parameters.
        """
        self._run_task_with_mode(self._teardown_task, params, run_mode)

    def _run_task_with_mode(
        self, task: Task, params: ParameterFrame, run_mode: TaskRunMode
    ):
        """Run the given task using the given run mode."""
        if run_mode == TaskRunMode.RUN:
            task.run(params)
        elif run_mode == TaskRunMode.MOCK:
            task.mock(params)
        elif run_mode == TaskRunMode.READ:
            task.read(params)
        else:
            raise ValueError(f"Unrecognised solver run mode '{run_mode}'.")
