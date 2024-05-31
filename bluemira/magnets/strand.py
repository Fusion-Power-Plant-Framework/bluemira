# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Strand class"""

import numpy as np

from bluemira.magnets.materials import Copper100, Material, Nb3Sn, NbTi
from bluemira.magnets.utils import parall_r, serie_r


class Strand:
    """
    Represents a strand with a circular cross-section.
    """

    def __init__(
            self,
            materials: list[Material],
            percentage: np.array | list[float],
            d_strand: float = 0.82e-3,
    ):
        """
        Initialize a Strand instance.

        Parameters
        ----------
        materials:
            List of materials inside the strand.
        percentage:
            Percentage of each material (with the same ordering of materials).
        d_strand:
            Strand diameter in meters.
        """
        self.materials = materials
        self.percentage = percentage
        self.d_strand = d_strand

    @property
    def area(self) -> float:
        """Returns the area of the strand cross-section in square meters."""
        return np.pi * self.d_strand ** 2 / 4

    def E(self, **kwargs) -> float:
        """
        Returns the Young's modulus for the strand.

        Parameters
        ----------
        **kwargs:
            Additional parameters (e.g., temperature).

        Returns
        -------
        Young's modulus in Pascals.
        """
        return 0.1e9

    def res(self, **kwargs) -> float:
        """
        Calculates the equivalent resistivity based on the parallel connection of strand components.

        Parameters
        ----------
        **kwargs:
            Additional parameters for calculating resistivity.

        Returns
        -------
        Equivalent resistivity in Ohm meters.
        """
        resistances = [
            x.res(**kwargs) / self.area / self.percentage[i]
            for i, x in enumerate(self.materials)
        ]
        res_tot = parall_r(resistances)
        return res_tot * self.area

    def cp_v(self, **kwargs) -> float:
        """
        Calculates the equivalent specific heat based on the series connection of strand components.

        Parameters
        ----------
        **kwargs:
            Additional parameters for calculating specific heat.

        Returns
        -------
        Equivalent specific heat in Joules per Kelvin per meter.
        """
        specific_heat = [
            x.cp_v(**kwargs) * self.percentage[i] for i, x in enumerate(self.materials)
        ]
        return serie_r(specific_heat)

    def Ic(
            self, B: float, T: float, strain: float = 0.55, T_margin: float = 1.5, **kwargs
    ) -> float:
        """
        Returns the critical current.

        Parameters
        ----------
        B:
            Operating magnetic field in Teslas.
        T:
            Operating temperature in Kelvins.
        esps:
            Total applied measured strain in percentage. Default is 0.55.
        T_margin:
            Strand temperature margin in operation in Kelvins. Default is 1.5.
        **kwargs:
            Additional parameters.

        Returns
        -------
        Critical current in Amperes.
        """
        return 0


class Wire_Nb3Sn(Strand):
    """Represents an Nb3Sn strand made of 50% Copper100 and 50% Nb3Sn."""

    # superconducting parameters for the calculation of Ic
    c_ = 1.0
    Ca1 = 50.06
    Ca2 = 0.00
    eps_0a = 0.00312
    eps_m = -0.00059
    Bc20max = 33.24
    Tc0max = 16.34
    p = 0.593
    q = 2.156
    C = 83075 * 1e6

    def __init__(self, d_strand: float = 0.82e-3):
        """
        Initialize a Wire_Nb3Sn instance.

        Parameters
        ----------
        d_strand:
            Strand diameter in meters. Default is 0.82e-3.
        """
        copper_100 = Copper100()
        mat_Nb3Sn = Nb3Sn()
        materials = [copper_100, mat_Nb3Sn]
        percentage = [0.5, 0.5]
        super().__init__(materials=materials, percentage=percentage, d_strand=d_strand)
        # percentage of copper with respect to superconducting material
        self._CunonCu = self.percentage[0] / self.percentage[1]
        # area of superconducting material
        self._superc_area = np.pi * self.d_strand ** 2 / (4 * (1 + self._CunonCu))

    def Ic(
            self, B: float, T: float, strain: float = 0.55, T_margin: float = 1.5, **kwargs
    ) -> float:
        """
        Returns the strand critical current.

        Parameters
        ----------
        B:
            Operating magnetic field in Teslas.
        T:
            Operating temperature in Kelvins.
        eps:
            Total applied measured strain in percentage. Default is 0.55.
        T_margin:
            Strand temperature margin in operation in Kelvins. Default is 1.5.

        Returns
        -------
        Strand current density in Amperes per square meter.
        """
        T_ = T + T_margin
        # Todo: check the sign of eps_m in this equation
        int_eps = -strain / 100  # + eps_m
        eps_sh = self.Ca2 * self.eps_0a / (np.sqrt(self.Ca1 ** 2 - self.Ca2 ** 2))
        s_eps = 1 + (
                self.Ca1
                * (
                        np.sqrt(eps_sh ** 2 + self.eps_0a ** 2)
                        - np.sqrt((int_eps - eps_sh) ** 2 + self.eps_0a ** 2)
                )
                - self.Ca2 * int_eps
        ) / (1 - self.Ca1 * self.eps_0a)
        Bc0_eps = self.Bc20max * s_eps
        Tc0_eps = self.Tc0max * (s_eps) ** (1 / 3)
        t = T_ / Tc0_eps
        BcT_eps = Bc0_eps * (1 - t ** (1.52))
        b = B / BcT_eps
        hT = (1 - t ** (1.52)) * (1 - t ** 2)
        fPb = (b ** self.p) * (1 - b) ** self.q
        return self.c_ * (self.C / B) * s_eps * fPb * hT * self._superc_area


class Wire_NbTi(Strand):
    """Represents an NbTi strand."""

    # superconducting parameters for the calculation of Ic
    n = 1.7
    Bc20_T = 15.19
    Tc0_K = 8.907
    C0 = 3.00e04
    C1 = 0.45
    a1 = 3.2
    b1 = 2.43
    C2 = 0.55
    a2 = 0.65
    b2 = 2
    g1 = 1.8
    g2 = 1.8

    def __init__(self, d_strand: float = 0.82e-3):
        """
        Initialize a Wire_NbTi instance.

        Parameters
        ----------
        d_strand:
            Strand diameter in meters. Default is 0.82e-3.
        """
        copper_100 = Copper100()
        mat_NbTi = NbTi()
        materials = [copper_100, mat_NbTi]
        percentage = [0.5, 0.5]
        super().__init__(materials=materials, percentage=percentage, d_strand=d_strand)
        # percentage of copper with respect to superconducting material
        self._CunonCu = self.percentage[0] / self.percentage[1]
        # area of superconducting material
        self._superc_area = np.pi * self.d_strand ** 2 / (4 * (1 + self._CunonCu))

    def Ic(self, B: float, T: float, T_margin: float = 1.5, **kwargs):
        """
        NbTi critical current.

        Parameters
        ----------
        B:
            Operating magnetic field in Teslas.
        T:
            Operating temperature in Kelvins.
        T_margin:
            Strand temperature margin in operation in Kelvins. Default is 1.5.

        Returns
        -------
        Critical current in Amperes.

        References
        ----------
        - Pinning Properties of Commercial Nb-Ti Wires Described by a 2-Components Model, Luigi Muzzi, Gianluca De Marzi, et al.
        - Fit data from DTT TF strand.
        """
        t = (T + T_margin) / self.Tc0_K
        b = B / self.Bc20_T
        tt = 1 - t ** self.n
        G = (self.a1 / (self.a1 + self.b1)) ** self.a1
        GG = (self.b1 / (self.a1 + self.b1)) ** self.b1
        GGG = G * GG
        F = (self.a2 / (self.a2 + self.b2)) ** self.a2
        FF = (self.b2 / (self.a2 + self.b2)) ** self.b2
        FFF = F * FF
        Jc = (
                     self.C0
                     * self.C1
                     / (B * GGG)
                     * (self.b / tt) ** self.a1
                     * (1 - self.b / tt) ** self.b1
                     * tt ** self.g1
                     + self.C0
                     * self.C2
                     / (B * FFF)
                     * (self.b / tt) ** self.a2
                     * (1 - self.b / tt) ** self.b2
                     * tt ** self.g2
             ) * 1e6
        return Jc * self._superc_area
