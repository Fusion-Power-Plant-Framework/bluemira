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

from bluemira.equilibria.fem_fixed_boundary.transport_solver import TransportSolver

from .api import Solver


class PlasmodTransportSolver(TransportSolver):
    """
    Plasmod transport solver class
    """

    def __init__(self, params, build_config):
        self.solver = Solver(params=params, build_config=build_config)
        self.solver.run()

    @property
    def pprime(self):
        """
        Get pprime as function of psi_norm
        """
        psinorm = self.solver.get_profile("x")
        data = self.solver.get_profile("pprime")
        return scipy.interpolate.UnivariateSpline(psinorm, data, ext=0)

    @property
    def ffprime(self):
        """
        Get ffprime as function of psi_norm
        """
        psinorm = self.solver.get_profile("x")
        data = self.solver.get_profile("ffprime")
        return scipy.interpolate.UnivariateSpline(psinorm, data, ext=0)

    @property
    def k95(self):
        """
        Get plasma elongation at 95 % flux surface
        """
        return self.solver.get_scalar("k95")
