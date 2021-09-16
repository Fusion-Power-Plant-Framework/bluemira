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
Flux surface utility classes and calculations
"""

import numpy as np
from scipy.integrate import odeint


class FluxSurace:
    pass

    def connection_length(self, eq):
        pass


class OpenFluxSurface(FluxSurace):
    pass


class ClosedFluxSurface(FluxSurace):
    pass

    def safety_factor(self, eq):
        return self.connection_length(eq) / self.geometry.length


def connection_length(eq, x, z):
    dx = x[:-1] - x[1:]
    dz = z[:-1] - z[1:]
    x_mp = x[:-1] + 0.5 * dx
    z_mp = z[:-1] + 0.5 * dz
    Bx = eq.Bx(x_mp, z_mp)
    Bz = eq.Bz(x_mp, z_mp)
    Bt = eq.Bt(x_mp)
    dtheta = np.hypot(dx, dz)
    field_ratio = Bt / np.hypot(Bx, Bz)
    dphi = field_ratio * dtheta
    dl = np.hypot(dtheta, dphi)
    return np.sum(dl)


def connection_length_sm(eq, x, z):
    dx = x[:-1] - x[1:]
    dz = z[:-1] - z[1:]
    x_mp = x[:-1] + 0.5 * dx
    z_mp = z[:-1] + 0.5 * dz
    Bx = eq.Bx(x_mp, z_mp)
    Bz = eq.Bz(x_mp, z_mp)
    Bt = eq.Bt(x_mp)
    dtheta = np.hypot(dx, dz)
    field_ratio = Bt / np.hypot(Bx, Bz)
    dphi = field_ratio * dtheta / x_mp
    dl = dphi * np.sqrt((dx / dphi) ** 2 + (dz / dphi) ** 2 + x[:-1] ** 2)
    return np.sum(dl)


from scipy.integrate import odeint


class FLT:
    def __init__(self, eq):
        self.eq = eq

    def dy_dt(self, xz, angles):
        Bx = self.eq.Bx(*xz[:2])
        Bz = self.eq.Bz(*xz[:2])
        Bt = self.eq.Bt(xz[0])
        B = np.sqrt(Bx ** 2 + Bz ** 2 + Bt ** 2)
        return xz[0] / Bt * np.array([Bx, Bz, B])

    def trace_fl(self, x, z, n_points=100):
        angles = np.linspace(0, 20 * 2 * np.pi, n_points)
        result = odeint(self.dy_dt, np.array([x, z, 0]), angles)
        return result.T


def flux_expansion(eq, x1, z1, x2, z2):
    return x1 * eq.Bp(x1, z1) / (x2 * eq.Bp(x2, z2))
