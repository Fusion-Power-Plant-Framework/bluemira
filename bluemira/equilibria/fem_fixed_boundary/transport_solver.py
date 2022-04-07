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

import numpy as np
from scipy.interpolate import UnivariateSpline, interp1d

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
        self._psi = None
        self._kappa_95 = None
        self._delta_95 = None
        self._volume_in = None
        self._I_p = None

        # supporting variables
        self.__x_phi = self.solver.get_profile('x')
        self.__psi_plasmod = self.solver.get_profile('psi')
        self.__x_psi = np.sqrt(self.__psi_plasmod / self.__psi_plasmod[-1])

    def _from_phi_to_psi(self, profile):
        """
        Convert the profile to the magnetic coordinate sqrt((psi - psi_ax)/(psi_b -
        psi_ax))
        """
        profile_data = self.solver.get_profile(profile)
        interp_data = interp1d(self.__x_phi, profile_data, kind='linear')
        return interp_data(self.__x_psi)

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
            data = self._from_phi_to_psi("pprime")*1e6
            self._pprime = UnivariateSpline(self.x, data, ext=0)
        return self._pprime

    @property
    def ffprime(self):
        """Get ffprime as function of the magnetic coordinate"""
        if self._ffprime is None:
            data = self._from_phi_to_psi("ffprime")*1e6
            self._ffprime = UnivariateSpline(self.x, data, ext=0)
        return self._ffprime

    @property
    def psi(self):
        """Get the magnetic coordinate"""
        if self._psi is None:
            self._psi = self._from_phi_to_psi("psi")*1e6
        return self._psi

    @property
    def kappa_95(self):
        """Get plasma elongation at 95 % flux surface"""
        if self._kappa_95 is None:
            self._kappa_95 = self.solver.params["k95"]
        return self._kappa_95

    @property
    def delta_95(self):
        """Get plasma elongation at 95 % flux surface"""
        if self._delta_95 is None:
            self._delta_95 = self.solver.params["d95"]
        return self._delta_95

    @property
    def volume_in(self):
        """Get plasma elongation at 95 % flux surface"""
        if self._volume_in is None:
            self._volume_in = self.solver.params["V_p"]
        return self._volume_in

    @property
    def I_p(self):
        """Get plasma elongation at 95 % flux surface"""
        if self._I_p is None:
            self._I_p = self.solver.params["I_p"]*1e6
        return self._I_p


