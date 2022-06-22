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

1. An :code:`Enum` subclassing :code:`RunMode`.
This should define the additional run modes for the solver,
e.g., mock, read.

2. Three classes subclassing :code:`Task`:

    * a "setup" task
    * a "run" task
    * a "teardown" task

Each task must implement a :code:`run` method, but can also implement an
arbitrary number of other run modes. The tasks do not need to implement
every run mode in the subclassed :code:`RunMode` enum,
but any tasks that do not implement the selected run mode are skipped.

3. A solver subclassing :code:`SolverABC`
This must set the
:code:`setup_cls`, :code:`run_cls`, and :code:`teardown_cls` properties.
These will usually be the corresponding :code:`Task` classes mentioned
above.
"""

import abc
import enum
from typing import Any, Callable, Optional, Type

from bluemira.base.parameter import ParameterFrame


class Task(abc.ABC):
    """
    Base class for tasks to be run within a solver.

    Children must override the :code:`run` method, but other methods, or
    "run modes" can also be defined.
    """

    def __init__(self, params: ParameterFrame) -> None:
        self.params = params

    @abc.abstractmethod
    def run(self):
        """Run the task."""
        pass


class NoOpTask(Task):
    """
    A task that does nothing.

    This can be assigned to a solver to skip any of the setup, run, or
    teardown stages.
    """

    def run(self) -> None:
        """Do nothing."""
        return


class RunMode(enum.Enum):
    """
    Base enum class for defining run modes within a solver.

    Note that no two enumeration's names should be case-insensitively
    equal.
    """

    def to_string(self) -> str:
        """
        Convert the enum name to a string; its name in lower-case.
        """
        return self.name.lower()

    @classmethod
    def from_string(cls, mode_str: str):
        """
        Retrieve an enum value from a case-insensitive string.

        Parameters
        ----------
        mode_str: str
            The run mode's name.
        """
        for run_mode_str, enum_value in cls.__members__.items():
            if run_mode_str.lower() == mode_str.lower():
                return enum_value
        raise ValueError(f"Unknown run mode '{mode_str}'.")


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

    def __init__(self, params: ParameterFrame):
        super().__init__()
        self.params = params
        self._setup = self.setup_cls(self.params)
        self._run = self.run_cls(self.params)
        self._teardown = self.teardown_cls(self.params)

    @abc.abstractproperty
    def setup_cls(self) -> Type[Task]:
        """
        Class defining the run modes for the setup stage of the solver.

        Typically, this class performs parameter mappings for some
        external code, or derives dependent parameters. But it can also
        define any required non-computational set up.
        """
        pass

    @abc.abstractproperty
    def run_cls(self) -> Type[Task]:
        """
        Class defining the run modes for the computational stage of the
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

    @abc.abstractproperty
    def run_mode_cls(self) -> Type[RunMode]:
        """
        Class enumerating the run modes for this solver.

        Common run modes are RUN, MOCK, READ, etc,.
        """
        pass

    def execute(self, run_mode: RunMode) -> Any:
        """Execute the setup, run, and teardown tasks, in order."""
        result = None
        if setup := self._get_execution_method(self._setup, run_mode):
            result = setup()
        if run := self._get_execution_method(self._run, run_mode):
            result = run(result)
        if teardown := self._get_execution_method(self._teardown, run_mode):
            result = teardown(result)
        return result

    def _get_execution_method(self, task: Task, run_mode: RunMode) -> Optional[Callable]:
        """
        Return the method on the task corresponding to this solver's run
        mode (e.g., :code:`task.run`).

        If the method on the task does not exist, return :code:`None`.
        """
        return getattr(task, run_mode.to_string(), None)
