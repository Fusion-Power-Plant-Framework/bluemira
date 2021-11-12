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
The bluemira external code wrapper
"""
from enum import Enum, auto


class RunMode(Enum):
    """Defines the possible runmode"""
    PROMINENCE = auto()
    BATCH = auto()
    MOCK = auto()


class Task():
    """
    A class for any task integration
    """

    # todo: ensure a correspondence between the specified runmode and the implemented
    #  functions

    def __init__(self, runmode):
        self.set_runmode(runmode)

    def set_runmode(self, runmode):
        """Set the runmode"""
        self.runmode = RunMode[runmode]

    def __call__(self, obj=None, *args, **kwargs):
        """
        Call function of object with lowercase name of
        enum
        Parameters
        ----------
        obj: instance
            instance of class the function will come from. If object is not specified,
            self will be used.
        *args
           args of function
        **kwargs
           kwargs of function
        Returns
        -------
        function result
        """
        if obj is None:
            obj = self
        func = getattr(obj, f"_{self.runmode.name.lower()}")
        return func(*args, **kwargs)

    def _prominence(self):
        print("running _prominence")
        # raise NotImplementedError

    def _batch(self):
        print("running _batch")
        # raise NotImplementedError

    def _mock(self):
        print("running _mock")
        # raise NotImplementedError


class ExternalCode(Task):
    """An external code wrapper"""

    class Setup(Task):
        """A class that specified the code setup"""
        pass

    class Run(Task):
        """A class that specified the code run process"""
        pass

    class Teardown(Task):
        """A class that for the teardown"""
        pass
