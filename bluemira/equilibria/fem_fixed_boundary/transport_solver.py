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
from abc import ABC, abstractmethod

import scipy

from bluemira.codes.plasmod.api import Solver


class TransportSolver(ABC):
    """Abstract transport solver class"""

    # This abstract class specifies all the properties and methods that
    # should be provided by a generic transport solver. For the moment
    # they are limited to pprime and ffprime. It is only a draft version.
    # Todo: check if it is necessary to add other properties or methods.

    def __init__(self, *args, **kwargs):
        self.solver = None

    @property
    @abstractmethod
    def pprime(self):
        """
        Get pprime as function of psi_norm
        """
        pass

    @property
    @abstractmethod
    def ffprime(self):
        """
        Get ffprime as function of psi_norm
        """
        pass


class NoneTransportSolver(TransportSolver):
    """Empty transport solver"""

    def __init__(self, *args, **kwargs):
        self.solver = None

    @TransportSolver.pprime.getter
    def pprime(self):
        """
        Get pprime as function of psi_norm
        """
        return None

    @TransportSolver.ffprime.getter
    def ffprime(self):
        """
        Get ffprime as function of psi_norm
        """
        return None


class PlasmodTransportSolver(TransportSolver):
    """
    Plasmod transport solver class
    """

    def __init__(self, params, build_config):
        self.solver = Solver(params=params, build_config=build_config)
        self.solver.run()
        self._x = None
        self._pprime = None
        self._ffprime = None
        self._kappa_95 = None
        self._delta_95 = None
        self._psi = None

    @property
    def x(self):
        """Get the magnetic coordinate"""
        if self._x is None:
            self._x = self.solver.get_profile("x")
        return self._x

    @property
    def pprime(self):
        """Get pprime as function of the magnetic coordinate"""
        if self._pprime is None:
            data = self.solver.get_profile("pprime")
            self._pprime = scipy.interpolate.UnivariateSpline(self.x, data, ext=0)
        return self._pprime

    @property
    def ffprime(self):
        """Get ffprime as function of the magnetic coordinate"""
        if self._ffprime is None:
            data = self.solver.get_profile("ffprime")
            self._ffprime = scipy.interpolate.UnivariateSpline(self.x, data, ext=0)
        return self._ffprime

    @property
    def kappa_95(self):
        """Get plasma elongation at 95 % flux surface"""
        if self._kappa_95 is None:
            self._kappa_95 = self.solver.get_scalar("k95")
        return self._kappa_95

    @property
    def delta_95(self):
        """Get plasma elongation at 95 % flux surface"""
        if self._delta_95 is None:
            self._delta_95 = self.solver.get_scalar("d95")
        return self._delta_95

    @property
    def psi(self):
        """Get the magnetic coordinate"""
        if self._psi is None:
            self._psi = self.solver.get_profile("psi")
        return self._psi
