# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Conductor class"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from bluemira.magnets.cable import Cable
from bluemira.magnets.materials import Material
from bluemira.magnets.utils import (
    parall_k,
    parall_r,
    serie_k,
    serie_r,
)


class Conductor:
    def __init__(
            self,
            cable: Cable,
            mat_jacket: Material,
            mat_ins: Material,
            dx_jacket: float,
            dy_jacket: float,
            dx_ins: float,
            dy_ins: float,
            name: str = "",
    ):
        """
        A generic conductor consisting of a cable surrounded by a jacket and an
        insulator.

        Parameters
        ----------
        cable:
            the conductor's cable
        mat_jacket:
            jacket's material
        mat_ins:
            insulator's material
        dx_jacket:
            x-thickness of the jacket
        dy_jacket:
            y-tickness of the jacket
        dx_ins:
            x-thickness of the insulator
        dy_ins:
            y-tickness of the insulator
        name:
            string identifier
        """
        self.name = name
        self._dx_jacket = dx_jacket
        self._dy_jacket = dy_jacket
        self._dy_ins = dy_ins
        self._dx_ins = dx_ins
        self.mat_ins = mat_ins
        self.mat_jacket = mat_jacket
        self.cable = cable

    @property
    def dx(self):
        """x-dimension of the conductor [m]"""
        return self.dx_ins * 2 + self.dx_jacket * 2 + self.cable.dx

    @property
    def dy(self):
        """y-dimension of the conductor [m]"""
        return self.dy_ins * 2 + self.dy_jacket * 2 + self.cable.dy

    @property
    def dx_jacket(self):
        """Thickness in the x-direction of the jacket [m]"""
        return self._dx_jacket

    @dx_jacket.setter
    def dx_jacket(self, value):
        self._dx_jacket = value

    @property
    def dy_jacket(self):
        """Thickness in the y-direction of the jacket [m]"""
        return self._dy_jacket

    @dy_jacket.setter
    def dy_jacket(self, value):
        self._dy_jacket = value

    @property
    def dx_ins(self):
        """Thickness in the x-direction of the insulator [m]"""
        return self._dx_ins

    @dx_ins.setter
    def dx_ins(self, value):
        self._dx_ins = value

    @property
    def dy_ins(self):
        """Thickness in the y-direction of the jacket [m]"""
        return self._dy_ins

    @dy_ins.setter
    def dy_ins(self, value):
        self._dy_ins = value

    @property
    def area(self):
        """Area of the conductor [m^2]"""
        return self.dx * self.dy

    @property
    def area_jacket(self):
        """Area of the jacket [m^2]"""
        return (self.dx - 2 * self.dx_ins) * (
                self.dy - 2 * self.dy_ins
        ) - self.cable.area

    @property
    def area_ins(self):
        """Area of the insulator [m^2]"""
        return self.area - self.area_jacket - self.cable.area

    def res(self, **kwargs):
        """
        Computes the conductor's equivalent resistivity considering the resistance
        of its strands in parallel.

        Parameters
        ----------
        **kwargs: dict
            Additional parameters for resistance calculations.

        Return
        ------
            float [Ohm m]

        Notes
        -----
        The insulator in not considered into the calculation.
        """
        resistances = np.array([
            self.cable.res(**kwargs) / self.cable.area,
            self.mat_jacket.res(**kwargs) / self.area_jacket,
        ])
        res_tot = parall_r(resistances)
        return res_tot * self.area

    def cp_v(self, **kwargs):
        """
        Computes the conductor's equivalent specific heat considering the specific heats
        of its components in series.

        Parameters
        ----------
        **kwargs: dict
            Additional parameters for resistance calculations.

        Return
        ------
            float [J/K/m]

        Notes
        -----
        The insulator in not considered into the calculation.

        TODO: decide if also the insulator should be considered
        """
        weighted_specific_heat = np.array([
            self.cable.cp_v(**kwargs) * self.cable.area,
            self.mat_jacket.cp_v(**kwargs) * self.area_jacket,
        ])
        return serie_r(weighted_specific_heat) / self.area

    # TODO: add a description of the 0D structural model, otherwise the following
    # function cannot be undestood
    def Kx_topbot_ins(self, **kwargs):
        return self.mat_ins.ym(**kwargs) * self.dy / self.dx_ins

    def Kx_lat_ins(self, **kwargs):
        return self.mat_ins.ym(**kwargs) * self.dy_ins / (self.dx - 2 * self.dx_ins)

    def Kx_lat_jacket(self, **kwargs):
        return (
                self.mat_jacket.ym(**kwargs) * self.dy_jacket / (
                self.dx - 2 * self.dx_ins)
        )

    def Kx_topbot_jacket(self, **kwargs):
        return self.mat_jacket.ym(**kwargs) * self.cable.dy / self.dx_jacket

    def Kx_cable(self, **kwargs):
        return self.cable.Kx(**kwargs)

    def Kx(self, **kwargs):
        return (serie_k([self.Kx_topbot_ins(**kwargs) / 2,
                         parall_k([
                             2 * self.Kx_lat_ins(**kwargs),
                             2 * self.Kx_lat_jacket(**kwargs),
                             serie_k([self.Kx_cable(**kwargs),
                                      self.Kx_topbot_jacket(**kwargs) / 2]),
                         ]),
                         ])
                )

    def Ky_topbot_ins(self, **kwargs):
        return self.mat_ins.ym(**kwargs) * self.dx / self.dy_ins

    def Ky_lat_ins(self, **kwargs):
        return self.mat_ins.ym(**kwargs) * self.dx_ins / (self.dy - 2 * self.dy_ins)

    def Ky_lat_jacket(self, **kwargs):
        return (
                self.mat_jacket.ym(**kwargs) * self.dx_jacket / (
                self.dy - 2 * self.dy_ins)
        )

    def Ky_topbot_jacket(self, **kwargs):
        return self.mat_jacket.ym(**kwargs) * self.cable.dx / self.dy_jacket

    def Ky_cable(self, **kwargs):
        return self.cable.Ky(**kwargs)

    def Ky(self, **kwargs):
        return (serie_k([self.Ky_topbot_ins(**kwargs) / 2,
                         parall_k([
                             2 * self.Ky_lat_ins(**kwargs),
                             2 * self.Ky_lat_jacket(**kwargs),
                             serie_k([self.Ky_cable(**kwargs),
                                      self.Ky_topbot_jacket(**kwargs) / 2]),
                         ]),
                         ])
                )

    def plot(self, xc: float = 0, yc: float = 0, show: bool = False, ax=None):
        """
        Schematic plot of the cable cross-section.

        Parameters
        ----------
        xc:
            x coordinate of the cable center in the considered coordinate system
        yc:
            y coordinate of the cable center in the considered coordinate system
        show:
            if True, the plot is displayed
        ax:
            Matplotlib Axis on which the plot shall be displayed. If None,
            a new figure is created
        """
        if ax is None:
            _, ax = plt.subplots()

        pc = np.array([xc, yc])
        a = self.cable.dx / 2 + self.dx_jacket
        b = self.cable.dy / 2 + self.dy_jacket

        p0 = np.array([-a, -b])
        p1 = np.array([a, -b])
        p2 = np.array([[a, b]])
        p3 = np.array([-a, b])
        points_ext_jacket = np.vstack((p0, p1, p2, p3, p0)) + pc

        c = a + self.dx_ins
        d = b + self.dy_ins

        p0 = np.array([-c, -d])
        p1 = np.array([c, -d])
        p2 = np.array([[c, d]])
        p3 = np.array([-c, d])
        points_ext_ins = np.vstack((p0, p1, p2, p3, p0)) + pc

        ax.fill(points_ext_ins[:, 0], points_ext_ins[:, 1], "red")
        ax.fill(points_ext_jacket[:, 0], points_ext_jacket[:, 1], "blue")

        ax = self.cable.plot(xc=xc, yc=yc, show=False, ax=ax)

        if show:
            ax.set_aspect("equal")
            plt.show()

        return ax


class SquareConductor(Conductor):
    def __init__(
            self,
            cable: Cable,
            mat_jacket: Material,
            mat_ins: Material,
            dx_jacket: float,
            dx_ins: float,
            name: str = "",
    ):
        """
        Representation of a square conductor

        Parameters
        ----------
        cable:
            the conductor's cable
        mat_jacket:
            jacket's material
        mat_ins:
            insulator's material
        dx_jacket:
            x(y)-thickness of the jacket
        dx_ins:
            x(y)-thickness of the insulator
        name:
            string identifier

        """
        dy_jacket = dx_jacket
        dy_ins = dx_ins
        super().__init__(
            cable=cable,
            mat_jacket=mat_jacket,
            mat_ins=mat_ins,
            dx_jacket=dx_jacket,
            dy_jacket=dy_jacket,
            dx_ins=dx_ins,
            dy_ins=dy_ins,
            name=name,
        )

    @property
    def dy_jacket(self):
        return self.dx_jacket

    @property
    def dy_ins(self):
        return self.dx_ins


def _sigma_r_jacket(conductor: Conductor, pressure: float, f_z: float, T: float,
                    B: float):
    saf_jacket = (conductor.cable.dx + 2 * conductor.dx_jacket) / (
            2 * conductor.dx_jacket
    )
    K = parall_k([
        2 * conductor.Ky_lat_ins(T=T, B=B),
        2 * conductor.Ky_lat_jacket(T=T, B=B),
        serie_k([conductor.Ky_cable(T=T, B=B),
                 conductor.Ky_topbot_jacket(T=T, B=B) / 2]),
    ])
    X_jacket = 2 * conductor.Ky_lat_jacket(T=T, B=B) / K
    return pressure * X_jacket * saf_jacket + f_z / conductor.area_jacket


def optimize_jacket_conductor(
        conductor: Conductor,
        pressure: float,
        fz: float,
        T: float,
        B: float,
        allowable_sigma: float,
        bounds: np.array = None,
):
    def sigma_difference(
            dx_jacket: float,
            pressure: float,
            fz: float,
            T: float,
            B: float,
            conductor: Conductor,
            allowable_sigma: float,
    ):
        conductor.dx_jacket = dx_jacket
        sigma_r = _sigma_r_jacket(conductor, pressure, fz, T, B)
        diff = abs(sigma_r - allowable_sigma)
        return diff

    method = None
    if bounds is not None:
        method = "bounded"

    result = minimize_scalar(
        fun=sigma_difference,
        args=(pressure, fz, T, B, conductor, allowable_sigma),
        bounds=bounds,
        method=method,
        options={"xatol": 1e-4},
    )

    if not result.success:
        raise ValueError("dx_jacket optimization did not converge.")
    conductor.dx_jacket = result.x
    print(f"Optimal dx_jacket: {conductor.dx_jacket}")
    print(f"Averaged sigma_r: {_sigma_r_jacket(conductor, pressure, fz, T, B) / 1e6} "
          f"MPa")

    return result