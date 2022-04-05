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

To create a generic solver, one must implement:

An Enum subclassing 'RunMode'. This should define the additional run
modes for the solver, e.g., mock, read.

Three classes subclassing 'Task':
    - a 'setup' task
    - a 'run' task
    - a 'teardown' task
Each task must implement a 'run' method, but can also implement an
arbitrary number of other run modes. The tasks do not need to implement
every run mode in the subclassed 'RunMode' enum, but any tasks that do
not implement the selected run mode are skipped.

A solver subclassing 'SolverABC'. The 'setup_cls', 'run_cls', and
'teardown_cls' properties must be set on the class. These will usually
be set to the corresponding Task classes mentioned above.
"""
import abc
import enum
from typing import Any, Callable, Type


class Task(abc.ABC):
    """
    Base class for tasks to be run within a solver.

    Children must override the 'run' method, but other methods, or
    'run modes' can also be defined.
    """

    @abc.abstractmethod
    def run(self, *args, **kwargs):
        """Run the task."""
        pass


class RunMode(enum.Enum):
    """Base enum class for defining run modes within a solver."""

    RUN = enum.auto()

    def to_string(self) -> str:
        """
        Convert the enum value to a string; its name in lower-case.
        """
        return self.name.lower()


class SolverABC(abc.ABC):
    """
    A base class for general solvers using setup, run and teardown
    stages, using multiple runmodes.

    This interface gives a general way for callers to run external, or
    Bluemira, solvers.

    Parameters
    ----------
    run_mode: RunMode
        The enum value for which run mode to use. Typical values would
        be 'run', 'mock' & 'read', but derived classes can define
        arbitrary run modes.
    """

    def __init__(self, run_mode: RunMode, *args, **kwargs) -> None:
        super().__init__()
        self._runmode = run_mode
        self._setup = self.setup_cls(*args, **kwargs)
        self._run = self.run_cls(*args, **kwargs)
        self._teardown = self.teardown_cls(*args, **kwargs)

    @abc.abstractproperty
    def setup_cls(self) -> Type[Task]:
        """
        Class defining the runmodes for the setup stage of the solver.

        Typically, this class performs parameter mappings for some
        external code, or derives dependent parameters. But it can also
        define any required non-computational set up.
        """
        pass

    @abc.abstractproperty
    def run_cls(self) -> Type[Task]:
        """
        Class defining the ru nmodes for the computational stage of the
        solver.

        This class is where computations should be defined. This may be
        something like calling a Bluemira problem, or executing some
        external code or process.
        """
        pass

    @abc.abstractproperty
    def teardown_cls(self) -> Type[Task]:
        """
        Class defining the run modes for the teardown stage of the
        solver.

        This class should perform any clean-up operations required by
        the solver. This may be deleting temporary files, or could
        involve mapping parameters from some external code to Bluemira
        parameters.
        """
        pass

    def execute(self, *args, **kwargs) -> Any:
        """Execute the setup, run, and teardown tasks, in order."""
        self._get_execution_method(self._setup)(*args, **kwargs)
        self._get_execution_method(self._run)(*args, **kwargs)
        self._get_execution_method(self._teardown)(*args, **kwargs)

    def _get_execution_method(self, task: Task) -> Callable:
        """
        Return the method on the task corresponding to this solver's run
        mode (e.g., 'task.run').

        If the method on the task does not exist, return a no-op
        function.
        """
        return getattr(task, self._runmode.to_string(), lambda *args, **kwargs: None)
