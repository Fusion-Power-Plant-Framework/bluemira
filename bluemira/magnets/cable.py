# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Cable class"""

from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from bluemira.magnets.strand import Strand
from bluemira.magnets.utils import parall_r, serie_r


class Cable:
    def __init__(
            self,
            dx: float,
            sc_strand: Strand,
            stab_strand: Strand,
            n_sc_strand: int,
            n_stab_strand: int,
            d_cooling_channel: float,
            void_fraction: float = 0.725,
            cos_theta: float = 0.97,
            name: str = "",
    ):
        """
        Representation of a cable

        Parameters
        ----------
        dx:
            x-dimension of the cable [m]
        sc_strand:
            strand of the superconductor
        stab_strand:
            strand of the stabilizer
        d_cooling_channel:
            diameter of the cooling channel
        n_sc_strand:
            number of superconducting strands
        n_stab_strand:
            number of stabilizer strands
        void_fraction:
            void fraction defined as material_volume/total_volume
        cos_theta:
            corrective factor that consider the twist of the cable
        name:
            cable string identifier

        #TODO: discuss if it is necessary to implement also the cooling material
        """
        self.name = name
        self._dx = dx
        self.sc_strand = sc_strand
        self.stab_strand = stab_strand
        self.void_fraction = void_fraction
        self.d_cooling_channel = d_cooling_channel
        self.n_sc_strand = int(n_sc_strand)
        self._n_stab_strand = int(n_stab_strand)
        self.cos_theta = cos_theta
        self._check_consistency()

    @property
    def n_stab_strand(self):
        """Number of stabilizing strands"""
        return self._n_stab_strand

    @n_stab_strand.setter
    def n_stab_strand(self, value: int):
        self._n_stab_strand = int(np.ceil(value))

    def res(self, **kwargs):
        """
        Cable's equivalent resistivity, computed as the parallel between
        strands' resistivity

        Parameters
        ----------
        Return
        ------
            float [Ohm m]
        """
        resistances = np.array([
            self.sc_strand.res(**kwargs) / self.area_sc,
            self.stab_strand.res(**kwargs) / self.area_stab,
        ])
        res_tot = parall_r(resistances)
        return res_tot * self.area

    def cp_v(self, **kwargs):
        """
        Strand's equivalent Specific Heat, compute the series between strand's components

        Parameters
        ----------
        Return
        ------
            float [J/K/m]
        """
        weighted_specific_heat = np.array([
            self.sc_strand.cp_v(**kwargs) * self.area_sc,
            self.stab_strand.cp_v(**kwargs) * self.area_stab,
        ])
        return serie_r(weighted_specific_heat) / (self.area_sc + self.area_stab)

    def _check_consistency(self):
        """Check consistency and return True if all checks are passed."""
        if self.dx <= self.d_cooling_channel or self.dy <= self.d_cooling_channel:
            print("WARNING: inconsistency between dx, dy and d_cooling_channel")
            return False
        return True

    @property
    def area_stab(self):
        """Area of the stabilizer"""
        return self.stab_strand.area * self.n_stab_strand

    @property
    def area_sc(self):
        """Area of the superconductor"""
        return self.sc_strand.area * self.n_sc_strand

    @property
    def area_cc(self):
        """Area of the cooling"""
        return self.d_cooling_channel ** 2 / 4 * np.pi

    @property
    def area(self):
        """Area of the cable considering the void fraction"""
        return (
                self.area_sc + self.area_stab
        ) / self.void_fraction / self.cos_theta + self.area_cc

    @property
    def dx(self):
        return self._dx

    @dx.setter
    def dx(self, value: float):
        self._dx = value

    @property
    def dy(self):
        """y-dimension of the cable [m]"""
        return self.area / self.dx

    def ym(self, **kwargs):
        return 0

    def Kx(self, **kwargs):
        """Total equivalent stiffness along x-axis"""
        return self.ym(**kwargs) * self.dy / self.dx

    def Ky(self, **kwargs):
        """Total equivalent stiffness along y-axis"""
        return self.ym(**kwargs) * self.dx / self.dy

    def optimize_n_stab_ths(
            self,
            t0: float,
            tf: float,
            initial_temperature: float,
            target_temperature: float,
            B: Callable,
            I: Callable,
            bounds: np.ndarray = None,
            show: bool = False,
    ):
        """
        Optimize the number of stabilizer strand in the superconducting cable using a
        0-D hot spot criteria

        Parameters
        ----------
            t0: float
                initial time
            tf: float
                final time
            initial_temperature: float
                temperature [K] at initial time
            target_temperature: float
                target temperature [K] at final time
            B : Callable
                The magnetic field [T] as time function
            I : Callable
                The current [A] flowing through the conductor as time function
            bounds: np.ndarray
                lower and upper limits for the number of strand in the cable
            show: bool
                if True the behavior of temperature as function of time is plotted

        Returns
        -------
            None

        Notes
        -----
            The number of stabilizer strands in the cable is directly modified. An
            error is raised in case the optimization process did not converge.
        """
        return optimize_n_stab_cable(
            self, t0, tf, initial_temperature, target_temperature, B, I, bounds, show
        )

    def plot(self, xc: float = 0, yc: float = 0, show: bool = False, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots()

        pc = np.array([xc, yc])
        a = self.dx / 2
        b = self.dy / 2

        p0 = np.array([-a, -b])
        p1 = np.array([a, -b])
        p2 = np.array([[a, b]])
        p3 = np.array([-a, b])

        points_ext = np.vstack((p0, p1, p2, p3, p0)) + pc
        points_cc = (
                np.array([
                    np.array([np.cos(theta), np.sin(theta)]) * self.d_cooling_channel / 2
                    for theta in np.linspace(0, np.radians(360), 19)
                ])
                + pc
        )

        ax.fill(points_ext[:, 0], points_ext[:, 1], "gold")
        ax.fill(points_cc[:, 0], points_cc[:, 1], "r")

        if show:
            plt.show()
        return ax


class SquareCable(Cable):
    def __init__(
            self,
            sc_strand: Strand,
            stab_strand: Strand,
            n_sc_strand: int,
            n_stab_strand: int,
            d_cooling_channel: float,
            void_fraction: float = 0.725,
            cos_theta: float = 0.97,
            name: str = "",
    ):
        """
        Representation of a square cable

        Parameters
        ----------
        sc_strand:
            strand of the superconductor
        stab_strand:
            strand of the stabilizer
        d_cooling_channel:
            diameter of the cooling channel
        n_sc_strand:
            number of superconducting strands
        n_stab_strand:
            number of stabilizer strands
        void_fraction:
            void fraction defined as material_volume/total_volume
        cos_theta:
            corrective factor that consider the twist of the cable
        name:
            cable string identifier

        #todo decide if it is the case to add also the cooling material
        """
        dx = 0.1
        super().__init__(
            dx=dx,
            sc_strand=sc_strand,
            stab_strand=stab_strand,
            n_sc_strand=n_sc_strand,
            n_stab_strand=n_stab_strand,
            d_cooling_channel=d_cooling_channel,
            void_fraction=void_fraction,
            cos_theta=cos_theta,
            name=name,
        )

    @property
    def dx(self):
        return np.sqrt(self.area)


class DummySquareCableHTS(SquareCable):
    """
    Dummy square cable with young's moduli set to 120 GPa

    Parameters
    ----------
    sc_strand:
        strand of the superconductor
    stab_strand:
        strand of the stabilizer
    d_cooling_channel:
        diameter of the cooling channel
    n_sc_strand:
        number of superconducting strands
    n_stab_strand:
        number of stabilizer strands
    void_fraction:
        void fraction defined as material_volume/total_volume
    cos_theta:
        corrective factor that consider the twist of the cable
    name:
        cable string identifier

    #todo decide if it is the case to add also the cooling material
    """

    def ym(self, **kwargs):
        return 120e9


class DummySquareCableLTS(SquareCable):
    """
    Dummy square cable with young's moduli set to 120 GPa

    Parameters
    ----------
    sc_strand:
        strand of the superconductor
    stab_strand:
        strand of the stabilizer
    d_cooling_channel:
        diameter of the cooling channel
    n_sc_strand:
        number of superconducting strands
    n_stab_strand:
        number of stabilizer strands
    void_fraction:
        void fraction defined as material_volume/total_volume
    cos_theta:
        corrective factor that consider the twist of the cable
    name:
        cable string identifier

    #todo decide if it is the case to add also the cooling material
    """

    def ym(self, **kwargs):
        return 0.1e9




def _heat_balance_model_cable(t, T, B: Callable, I: Callable, cable: Cable):
    """
    Calculate the derivative of temperature (dT/dt) for the heat balance problem.

    Parameters
    ----------
        t : float
            The current time in seconds.
        T : float
            The current temperature in Celsius.
        B : Callable
            The magnetic field [T] as time function
        I : Callable
            The current [A] flowing through the conductor as time function
        cable : Cable
            the superconducting cable

    Returns
    -------
        dTdt : float
            The derivative of temperature with respect to time (dT/dt).
    """
    # Calculate the rate of heat generation (Joule dissipation)
    if isinstance(T, np.ndarray):
        T = T[0]

    Q_gen = (I(t) / cable.area) ** 2 * cable.res(B=B(t), T=T)

    # Calculate the rate of heat absorption by conductor components
    Q_abs = cable.cp_v(T=T)

    # Calculate the derivative of temperature with respect to time (dT/dt)
    dTdt = Q_gen / Q_abs

    return dTdt


def _temperature_evolution(
        t0: float,
        tf: float,
        initial_temperature: float,
        B: Callable,
        I: Callable,
        cable: Cable,
):
    solution = solve_ivp(
        _heat_balance_model_cable,
        [t0, tf],
        [initial_temperature],
        args=(B, I, cable),
        dense_output=True,
    )

    if not solution.success:
        raise ValueError("Temperature evolution did not converged")

    return solution


def optimize_n_stab_cable(
        cable: Cable,
        t0: float,
        tf: float,
        initial_temperature: float,
        target_temperature: float,
        B: Callable,
        I: Callable,
        bounds: np.ndarray = None,
        show: bool = False,
):
    """
    Optimize the number of stabilizer strand in a superconducting cable using a 0-D hot spot criteria

    Parameters
    ----------
        cable: Cable
            the superconducting cable
        t0: float
            initial time
        tf: float
            final time
        initial_temperature: float
            temperature [K] at initial time
        target_temperature: float
            target temperature [K] at final time
        B : Callable
            The magnetic field [T] as time function
        I : Callable
            The current [A] flowing through the conductor as time function
        bounds: np.ndarray
            lower and upper limits for the number of strand in the cable
        show: bool
            if True the behavior of temperature as function of time is plotted

    Returns
    -------
        None

    Notes
    -----
        The number of stabilizer strands in the cable is directly modified. An error is raised in case the optimization
        process did not converge.
    """

    def final_temperature_difference(
            n_stab: int,
            t0: float,
            tf: float,
            initial_temperature: float,
            target_temperature: float,
            B: Callable,
            I: Callable,
            cable: Cable,
    ):
        cable.n_stab_strand = n_stab

        solution = _temperature_evolution(
            t0=t0, tf=tf, initial_temperature=initial_temperature, B=B, I=I, cable=cable
        )
        final_T = float(solution.y[0][-1])
        diff = abs(final_T - target_temperature)
        return diff

    method = None
    if bounds is not None:
        method = "bounded"

    result = minimize_scalar(
        fun=final_temperature_difference,
        args=(t0, tf, initial_temperature, target_temperature, B, I, cable),
        bounds=bounds,
        method=method,
    )

    if not result.success:
        raise ValueError(
            "n_stab optimization did not converge. Check your input parameters or initial bracket."
        )

    solution = _temperature_evolution(t0, tf, initial_temperature, B, I, cable)
    final_temperature = solution.y[0][-1]

    print(f"Optimal n_stab: {cable.n_stab_strand}")
    print(f"Final temperature with optimal n_stab: {final_temperature} Kelvin")

    if show:
        _, ax = plt.subplots()
        ax.plot(solution.t, solution.y[0], "r")
        time_steps = np.linspace(t0, tf, 100)
        ax.plot(time_steps, solution.sol(time_steps)[0], "b")
        plt.show()

    return result
