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
The bluemira transport solver module
"""
import scipy
from bluemira.utilities.tools import get_module

class TransportSolver:
    """
    Transport solver class
    """

    def __init__(self, solver_name, *args, **kwargs):
        self.solver = get_module(solver_name).Solver(*args, **kwargs)
        self.solver.run()

    @property
    def pprime(self):
        """
        Get pprime
        """
        psinorm = self.solver.get_profile("x")
        data = self.solver.get_profile("pprime")
        return scipy.interpolate.UnivariateSpline(psinorm, data, ext=0)

    @property
    def ffprime(self):
        """
        Get ffprime
        """
        psinorm = self.solver.get_profile("x")
        data = self.solver.get_profile("ffprime")
        return scipy.interpolate.UnivariateSpline(psinorm, data, ext=0)


class NoneTransportSolver(TransportSolver):
    """Empty transport solver"""
    def __init__(self, *args, **kwargs):
        self.solver = None

    @TransportSolver.pprime.getter
    def pprime(self):
        return None

    @TransportSolver.ffprime.getter
    def ffprime(self):
        return None
