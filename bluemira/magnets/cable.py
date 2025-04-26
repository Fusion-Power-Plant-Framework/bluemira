# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Cable class"""

from abc import ABC, abstractmethod
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from bluemira.base.look_and_feel import bluemira_error, bluemira_print, bluemira_warn
from bluemira.magnets.strand import Strand, SuperconductingStrand
from bluemira.magnets.utils import parall_r, serie_r


class ABCCable(ABC):
    """
    Abstract base class for cable representations.

    This class models a generic superconducting cable with both stabilizer and
    superconducting strands, incorporating void fractions and geometrical details.
    """

    def __init__(
        self,
        sc_strand: SuperconductingStrand,
        stab_strand: Strand,
        n_sc_strand: int,
        n_stab_strand: int,
        d_cooling_channel: float,
        void_fraction: float = 0.725,
        cos_theta: float = 0.97,
        name: str = "Cable",
    ):
        """
        Representation of a cable. Only the x-dimension of the cable is given as
        input. The y-dimension is calculated on the basis of the cable design.

        Notes
        -----
        Cooling material not implemented.

        Parameters
        ----------
        sc_strand : SuperconductingStrand
            The superconducting strand.
        stab_strand : Strand
            The stabilizer strand.
        n_sc_strand : int
            Number of superconducting strands.
        n_stab_strand : int
            Number of stabilizing strands.
        d_cooling_channel : float
            Diameter of the cooling channel [m].
        void_fraction : float
            Ratio of material volume to total volume [unitless].
        cos_theta : float
            Correction factor for twist in the cable layout.
        name : str
            Identifier for the cable instance.
        """
        # initialize private variables
        self._d_cooling_channel = None
        self._void_fraction = None
        self._n_sc_strand = None
        self._n_stab_strand = None
        self._cos_theta = None
        self._shape = None

        # assign
        self.name = name
        self.sc_strand = sc_strand
        self.stab_strand = stab_strand
        self.void_fraction = void_fraction
        self.d_cooling_channel = d_cooling_channel
        self.n_sc_strand = n_sc_strand
        self.n_stab_strand = n_stab_strand
        self.cos_theta = cos_theta

    @property
    @abstractmethod
    def dx(self):
        """Cable dimension in the x-direction [m]."""

    @property
    @abstractmethod
    def dy(self):
        """Cable dimension in the y-direction [m]."""

    @property
    def aspect_ratio(self):
        """
        Compute the aspect ratio of the cable cross-section.
        """
        return self.dx / self.dy

    @property
    def n_sc_strand(self):
        """Number of superconducting strands"""
        return self._n_sc_strand

    @n_sc_strand.setter
    def n_sc_strand(self, value: int):
        """
        Set the number of superconducting strands.

        Raises
        ------
        ValueError
            If the value is not positive.
        """
        if value <= 0:
            msg = f"The number of superconducting strands must be positive, got {value}"
            bluemira_error(msg)
            raise ValueError(msg)
        self._n_sc_strand = int(np.ceil(value))

    @property
    def n_stab_strand(self):
        """Number of stabilizing strands"""
        return self._n_stab_strand

    @n_stab_strand.setter
    def n_stab_strand(self, value: int):
        """
        Set the number of stabilizer strands.

        Raises
        ------
        ValueError
            If the value is negative.
        """
        if value < 0:
            msg = f"The number of stabilizing strands must be positive, got {value}"
            bluemira_error(msg)
            raise ValueError(msg)
        self._n_stab_strand = int(np.ceil(value))

    @property
    def d_cooling_channel(self):
        """Diameter of the cooling channel [m]."""
        return self._d_cooling_channel

    @d_cooling_channel.setter
    def d_cooling_channel(self, value: float):
        """
        Set the cooling channel diameter.

        Raises
        ------
        ValueError
            If the value is negative.
        """
        if value < 0:
            msg = f"diameter of the cooling channel must be positive, got {value}"
            bluemira_error(msg)
            raise ValueError(msg)

        self._d_cooling_channel = value

    @property
    def void_fraction(self):
        """Void fraction of the cable."""
        return self._void_fraction

    @void_fraction.setter
    def void_fraction(self, value: float):
        if value < 0 or value > 1:
            msg = f"void_fraction must be between 0 and 1, got {value}"
            bluemira_error(msg)
            raise ValueError(msg)

        self._void_fraction = value

    @property
    def cos_theta(self):
        """Correction factor for strand orientation (twist)."""
        return self._cos_theta

    @cos_theta.setter
    def cos_theta(self, value: float):
        if value <= 0 or value > 1:
            msg = f"cos theta must be in the interval ]0, 1], got {value}"
            bluemira_error(msg)
            raise ValueError(msg)

        self._cos_theta = value

    def rho(self, **kwargs):
        """
        Compute the average mass density of the cable [kg/m³].

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments forwarded to strand property evaluations.

        Returns
        -------
        float
            Averaged mass density in kg/m³.
        """
        return (
            self.sc_strand.rho(**kwargs) * self.area_sc
            + self.stab_strand.rho(**kwargs) * self.area_stab
        ) / (self.area_sc + self.area_stab)

    def erho(self, **kwargs):
        """
        Computes the cable's equivalent resistivity considering the resistance
        of its strands in parallel.

        Parameters
        ----------
        **kwargs: dict
            Additional parameters for resistance calculations.

        Returns
        -------
            float [Ohm m]
        """
        resistances = np.array([
            self.sc_strand.erho(**kwargs) / self.area_sc,
            self.stab_strand.erho(**kwargs) / self.area_stab,
        ])
        res_tot = parall_r(resistances)
        return res_tot * self.area

    def Cp(self, **kwargs):  # noqa: N802
        """
        Computes the cable's equivalent specific heat considering the specific heats
        of its strands in series.

        Parameters
        ----------
        **kwargs: dict
            Additional parameters for specific heat calculations.

        Returns
        -------
            float [J/K/m]
        """
        weighted_specific_heat = np.array([
            self.sc_strand.Cp(**kwargs) * self.area_sc * self.sc_strand.rho(**kwargs),
            self.stab_strand.Cp(**kwargs)
            * self.area_stab
            * self.stab_strand.rho(**kwargs),
        ])
        return serie_r(weighted_specific_heat) / (
            self.area_sc * self.sc_strand.rho(**kwargs)
            + self.area_stab * self.stab_strand.rho(**kwargs)
        )

    @property
    def area_stab(self):
        """Area of the stabilizer region"""
        return self.stab_strand.area * self.n_stab_strand

    @property
    def area_sc(self):
        """Area of the superconductor region"""
        return self.sc_strand.area * self.n_sc_strand

    @property
    def area_cc(self):
        """Area of the cooling channel"""
        return self.d_cooling_channel**2 / 4 * np.pi

    @property
    def area(self):
        """Area of the cable considering the void fraction"""
        return (
            self.area_sc + self.area_stab
        ) / self.void_fraction / self.cos_theta + self.area_cc

    def E(self, **kwargs):  # noqa: N802
        """
        Return the effective Young's modulus of the cable [Pa].

        This is a default placeholder implementation in the base class.
        Subclasses may use `kwargs` to modify behavior.

        Parameters
        ----------
        **kwargs :
            Arbitrary keyword arguments (ignored here).

        Returns
        -------
        float
            Default Young's modulus (0).
        """
        raise NotImplementedError("E for Cable is not implemented.")

    def _heat_balance_model_cable(
        self,
        t: float,
        temperature: float,
        B_fun: Callable,
        I_fun: Callable,  # noqa: N803
    ):
        """
        Calculate the derivative of temperature (dT/dt) for a 0D heat balance problem.

        Parameters
        ----------
            t : float
                The current time in seconds.
            temperature : float
                The current temperature in Celsius.
            B_fun : Callable
                The magnetic field [T] as time function
            I_fun : Callable
                The current [A] flowing through the conductor as time function

        Returns
        -------
            dTdt : float
                The derivative of temperature with respect to time (dT/dt).
        """
        # Calculate the rate of heat generation (Joule dissipation)
        if isinstance(temperature, np.ndarray):
            temperature = temperature[0]

        operational_point = {"B": B_fun(t), "temperature": temperature}

        Q_gen = (I_fun(t) / self.area) ** 2 * self.erho(**operational_point)  # noqa:N806

        # Calculate the rate of heat absorption by conductor components
        Q_abs = self.Cp(**operational_point) * self.rho(**operational_point)  # noqa:N806

        # Calculate the derivative of temperature with respect to time (dT/dt)
        # dTdt = Q_gen / Q_abs

        return Q_gen / Q_abs

    def _temperature_evolution(
        self,
        t0: float,
        tf: float,
        initial_temperature: float,
        B_fun: Callable,
        I_fun: Callable,  # noqa: N803
    ):
        solution = solve_ivp(
            self._heat_balance_model_cable,
            [t0, tf],
            [initial_temperature],
            args=(B_fun, I_fun),
            dense_output=True,
        )

        if not solution.success:
            raise ValueError("Temperature evolution did not converged")

        return solution

    def optimize_n_stab_ths(
        self,
        t0: float,
        tf: float,
        initial_temperature: float,
        target_temperature: float,
        B_fun: Callable,
        I_fun: Callable,  # noqa: N803
        bounds: np.ndarray = None,
        *,
        show: bool = False,
    ):
        """
        Optimize the number of stabilizer strand in the superconducting cable using a
        0-D hot spot criteria.

        Parameters
        ----------
        t0:
            Initial time [s].
        tf:
            Final time [s].
        initial_temperature:
            Temperature [K] at initial time.
        target_temperature:
            Target temperature [K] at final time.
        B_fun :
            Magnetic field [T] as a time-dependent function.
        I_fun :
            Current [A] as a time-dependent function.
        bounds:
            Lower and upper limits for the number of stabilizer strands.
        show:
            If True, the behavior of temperature over time is plotted.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            The result of the optimization process.

        Raises
        ------
        ValueError
            If the optimization process does not converge.

        Notes
        -----
        - The number of stabilizer strands in the cable is modified directly.
        - Cooling material contribution is neglected when applying the hot spot criteria.
        """

        def final_temperature_difference(
            n_stab: int,
            t0: float,
            tf: float,
            initial_temperature: float,
            target_temperature: float,
            B_fun: Callable,
            I_fun: Callable,  # noqa: N803
        ):
            """
            Compute the absolute temperature difference at final time between the
            simulated and target temperatures.

            This method modifies the private attribute `_n_stab_strand` to update the
            cable configuration, simulates the temperature evolution over time, and
            returns the absolute difference between the final temperature and the
            specified target.

            Parameters
            ----------
            n_stab : int
                Number of stabilizer strands to set temporarily for this simulation.
            t0 : float
                Initial time of the simulation [s].
            tf : float
                Final time of the simulation [s].
            initial_temperature : float
                Temperature at the start of the simulation [K].
            target_temperature : float
                Desired temperature at the end of the simulation [K].
            B_fun : Callable
                Magnetic field as a time-dependent function [T].
            I_fun : Callable
                Current as a time-dependent function [A].

            Returns
            -------
            float
                Absolute difference between the simulated final temperature and the
                target temperature [K].

            Notes
            -----
            - This method is typically used as a cost function for optimization routines
              (e.g., minimizing the temperature error by tuning `n_stab`).
            - It modifies the internal state `self._n_stab_strand`, which may affect
              subsequent evaluations unless restored.
            """
            self._n_stab_strand = n_stab

            solution = self._temperature_evolution(
                t0=t0,
                tf=tf,
                initial_temperature=initial_temperature,
                B_fun=B_fun,
                I_fun=I_fun,
            )
            final_temperature = float(solution.y[0][-1])
            # diff = abs(final_temperature - target_temperature)
            return abs(final_temperature - target_temperature)

        method = None
        if bounds is not None:
            method = "bounded"

        result = minimize_scalar(
            fun=final_temperature_difference,
            args=(t0, tf, initial_temperature, target_temperature, B_fun, I_fun),
            bounds=bounds,
            method=method,
        )

        if not result.success:
            raise ValueError(
                "n_stab optimization did not converge. Check your input parameters "
                "or initial bracket."
            )

        # Here we re-ensure the n_stab_strand to be an integer
        self.n_stab_strand = self._n_stab_strand

        solution = self._temperature_evolution(t0, tf, initial_temperature, B_fun, I_fun)
        final_temperature = solution.y[0][-1]

        if final_temperature > target_temperature:
            bluemira_error(
                f"Final temperature ({final_temperature:.2f} K) exceeds target "
                f"temperature "
                f"({target_temperature} K) even with maximum n_stab = "
                f"{self.n_stab_strand}."
            )
            raise ValueError(
                "Optimization failed to keep final temperature ≤ target. "
                "Try increasing the upper bound of n_stab or adjusting cable parameters."
            )
        bluemira_print(f"Optimal n_stab: {self.n_stab_strand}")
        bluemira_print(
            f"Final temperature with optimal n_stab: {final_temperature:.2f} Kelvin"
        )

        if show:
            # _, ax = plt.subplots()
            # ax.plot(solution.t, solution.y[0], "r*")
            # time_steps = np.linspace(t0, tf, 100)
            # ax.plot(time_steps, solution.sol(time_steps)[0], "b")
            # plt.grid(True)
            # plt.xlabel("Time [s]")
            # plt.ylabel("Temperature [K]")
            # plt.title("Quench temperature evoltuion")
            #
            # # *** Additional info ***
            # additional_info = [f"Hot spot temp. = {target_temperature} [K]",
            #                    f"Initial temp. = {initial_temperature} [K]",
            #                    f"Sc. strand = {self.sc_strand.__class__.__name__}",
            #                    f"n. sc. strand = {self.n_sc_strand}",
            #                    f"Stab. strand = {self.stab_strand.__class__.__name__}",
            #                    f"n. stab. strand = {self.n_stab_strand}"]
            #
            # additional_info = '\n'.join(additional_info)
            # plt.text(50, 80, additional_info)
            # plt.show()

            _, ax = plt.subplots()

            # Plot the main solution
            ax.plot(solution.t, solution.y[0], "r*")
            time_steps = np.linspace(t0, tf, 100)
            ax.plot(time_steps, solution.sol(time_steps)[0], "b")
            plt.grid(visible=True)
            plt.xlabel("Time [s]")
            plt.ylabel("Temperature [K]")
            plt.title("Quench temperature evolution")

            # Create secondary axis on the right (no ticks or labels needed)
            ax2 = ax.twinx()  # This creates a new y-axis that shares the same x-axis
            ax2.set_yticks([])  # Remove y-axis ticks
            ax2.set_ylabel("")  # Remove y-axis label

            # Plot additional info next to the right y-axis
            additional_info = [
                f"Hot spot temp. = {target_temperature} [K]",
                f"Initial temp. = {initial_temperature} [K]",
                f"Sc. strand = {self.sc_strand.__class__.__name__}",
                f"n. sc. strand = {self.n_sc_strand}",
                f"Stab. strand = {self.stab_strand.__class__.__name__}",
                f"n. stab. strand = {self.n_stab_strand}",
            ]
            additional_info = "\n".join(additional_info)

            # Set text position right after the right y-axis
            ax2.text(
                1.05,
                0.5,
                additional_info,
                transform=ax2.transAxes,
                verticalalignment="center",
                fontsize=10,
            )

            plt.show()

        return result

    # OD homogenized structural properties
    @abstractmethod
    def Kx(self, **kwargs):  # noqa: N802
        """Total equivalent stiffness along x-axis"""

    @abstractmethod
    def Ky(self, **kwargs):  # noqa: N802
        """Total equivalent stiffness along y-axis"""

    def plot(self, xc: float = 0, yc: float = 0, *, show: bool = False, ax=None):
        """
        Plot a schematic view of the cable cross-section.

        This method visualizes the outer shape of the cable and the cooling channel,
        assuming a rectangular or elliptical layout based on `dx`, `dy`, and
        `d_cooling_channel`. It draws the cable centered at (xc, yc) within the
        current coordinate system.

        Parameters
        ----------
        xc : float, optional
            x-coordinate of the cable center in the plot [m]. Default is 0.
        yc : float, optional
            y-coordinate of the cable center in the plot [m]. Default is 0.
        show : bool, optional
            If True, the plot is rendered immediately with `plt.show()`.
            Default is False.
        ax : matplotlib.axes.Axes or None, optional
            The matplotlib Axes object to draw on. If None, a new figure and
            Axes are created internally.

        Returns
        -------
        matplotlib.axes.Axes
            The Axes object with the cable plot, which can be further customized
            or saved.

        Notes
        -----
        - The outer shape is drawn as a gold-colored rectangle centered at `(xc, yc)`.
        - The cooling channel is drawn as a red circle.
        - This method does not clear or modify existing content in `ax`.
        - Use `show=True` to render the figure when not embedding in another
            plot context.
        """
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
        ax.set_aspect("equal")

        if show:
            plt.show()
        return ax

    def __str__(self):
        """
        Return a human-readable summary of the cable configuration.

        Includes geometric properties, void and twist factors, and a string
        representation of both the superconducting and stabilizer strands.

        Returns
        -------
        str
            A formatted multiline string describing the cable.
        """
        return (
            f"name: {self.name}\n"
            f"dx: {self.dx}\n"
            f"dy: {self.dy}\n"
            f"aspect ratio: {self.aspect_ratio}\n"
            f"d cooling channel: {self.d_cooling_channel}\n"
            f"void fraction: {self.void_fraction}\n"
            f"cos(theta): {self.cos_theta}\n"
            f"----- sc strand -------\n"
            f"sc strand: {self.sc_strand!s}\n"
            f"----- stab strand -------\n"
            f"stab strand: {self.stab_strand!s}\n"
            f"-----------------------\n"
            f"n sc strand: {self.n_sc_strand}\n"
            f"n stab strand: {self.n_stab_strand}"
        )

    def to_dict(self) -> dict:
        """
        Return a dictionary with all base cable parameters.

        This method serializes the cable configuration into a dictionary format,
        which can be useful for saving, logging, or exporting the data.

        Returns
        -------
        dict
            A dictionary containing:
            - "name" (str): Name of the cable.
            - "n_sc_strand" (int): Number of superconducting strands.
            - "n_stab_strand" (int): Number of stabilizer strands.
            - "d_cooling_channel" (float): Diameter of the cooling channel [m].
            - "void_fraction" (float): Fraction of void (non-material) volume in the
            cable.
            - "cos_theta" (float): Cosine of the winding angle theta.
            - "sc_strand" (dict): Dictionary with parameters of the superconducting
            strand.
            - "stab_strand" (dict): Dictionary with parameters of the stabilizer strand.
        """
        return {
            "name": self.name,
            "n_sc_strand": self.n_sc_strand,
            "n_stab_strand": self.n_stab_strand,
            "d_cooling_channel": self.d_cooling_channel,
            "void_fraction": self.void_fraction,
            "cos_theta": self.cos_theta,
            "sc_strand": self.sc_strand.to_dict(),
            "stab_strand": self.stab_strand.to_dict(),
        }

    @classmethod
    @abstractmethod
    def from_dict(cls, cable_dict: dict) -> "ABCCable":
        """
        Construct a cable object from a dictionary configuration.

        This method must be implemented by all concrete cable subclasses to handle
        their specific parameters.

        Parameters
        ----------
        cable_dict : dict
            Dictionary containing the cable configuration, including strand definitions.

        Returns
        -------
        ABCCable
            A fully instantiated cable object of the appropriate type.
        """


class RectangularCable(ABCCable):
    """
    Cable with a rectangular cross-section.

    The x-dimension is provided directly. The y-dimension is derived based on
    the total area and x-dimension.
    """

    def __init__(
        self,
        dx: float,
        sc_strand: SuperconductingStrand,
        stab_strand: Strand,
        n_sc_strand: int,
        n_stab_strand: int,
        d_cooling_channel: float,
        void_fraction: float = 0.725,
        cos_theta: float = 0.97,
        name: str = "RectangularCable",
    ):
        """
        Representation of a cable. Only the x-dimension of the cable is given as
        input. The y-dimension is calculated on the basis of the cable design.

        Notes
        -----
        Cooling material not implemented.

        Parameters
        ----------
        dx : float
            Cable width in the x-direction [m].
        sc_strand : SuperconductingStrand
            Superconducting strand.
        stab_strand : Strand
            Stabilizer strand.
        n_sc_strand : int
            Number of superconducting strands.
        n_stab_strand : int
            Number of stabilizer strands.
        d_cooling_channel : float
            Cooling channel diameter [m].
        void_fraction : float, optional
            Void fraction (material_volume / total_volume).
        cos_theta : float, optional
            Correction factor for strand twist.
        name : str, optional
            Name of the cable.
        """
        super().__init__(
            sc_strand=sc_strand,
            stab_strand=stab_strand,
            n_sc_strand=n_sc_strand,
            n_stab_strand=n_stab_strand,
            d_cooling_channel=d_cooling_channel,
            void_fraction=void_fraction,
            cos_theta=cos_theta,
            name=name,
        )

        # initialize private variables
        self._dx = None

        # assign
        self.dx = dx

    @property
    def dx(self):
        """Cable dimension in the x direction [m]"""
        return self._dx

    @dx.setter
    def dx(self, value: float):
        """
        Set cable width in x-direction.

        Raises
        ------
        ValueError
            If value is not positive.
        """
        if value <= 0:
            msg = "dx must be positive"
            bluemira_error(msg)
            raise ValueError(msg)
        self._dx = value

    @property
    def dy(self):
        """Cable dimension in the y direction [m]"""
        return self.area / self.dx

    # Decide if this function shall be a setter.
    # Defined as "normal" function to underline that it modifies dx.
    def set_aspect_ratio(self, value: float) -> None:
        """Modify dx in order to get the given aspect ratio"""
        self.dx = np.sqrt(value * self.area)

    # OD homogenized structural properties
    def Kx(self, **kwargs):  # noqa: N802
        """
        Compute the total equivalent stiffness along the x-axis.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments for material or geometric modifiers.

        Returns
        -------
        float
            Homogenized stiffness in the x-direction [Pa].
        """
        return self.E(**kwargs) * self.dy / self.dx

    def Ky(self, **kwargs):  # noqa: N802
        """
        Compute the total equivalent stiffness along the y-axis.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments for material or geometric modifiers.

        Returns
        -------
        float
            Homogenized stiffness in the y-direction [Pa].
        """
        return self.E(**kwargs) * self.dx / self.dy

    def to_dict(self) -> dict:
        """
        Serialize the rectangular cable configuration to a dictionary.

        This includes all base cable properties (strand configuration, geometry,
        cooling, etc.) and shape-specific parameters like `dx` and `aspect_ratio`.

        Returns
        -------
        dict
            A complete dictionary representation of the rectangular cable, including:
            - type : str
            - name : str
            - dx : float
            - aspect_ratio : float
            - n_sc_strand : int
            - n_stab_strand : int
            - d_cooling_channel : float
            - void_fraction : float
            - cos_theta : float
            - sc_strand : dict
            - stab_strand : dict
        """
        data = super().to_dict()
        data["type"] = "rectangular"
        data["dx"] = self.dx
        data["aspect_ratio"] = self.aspect_ratio
        return data

    @classmethod
    def from_dict(cls, config: dict) -> "RectangularCable":
        """
        Construct a `RectangularCable` instance from a dictionary.

        This method deserializes the cable and its nested strands. It accepts either
        a direct width (`dx`) or an aspect ratio (`aspect_ratio`) to compute geometry.

        Behavior:
        - If both `dx` and `aspect_ratio` are given, a warning is issued and
        `aspect_ratio`
          is applied (overwriting dx).
        - If only `aspect_ratio` is given, a default `dx = 0.01` m is used.
        - If only `dx` is given, it is used as-is.
        - If neither is provided, a `ValueError` is raised.

        Parameters
        ----------
        config : dict
            Required:
            - n_sc_strand : int
            - n_stab_strand : int
            - d_cooling_channel : float
            - sc_strand : dict
            - stab_strand : dict

            Optional:
            - dx : float
            - aspect_ratio : float
            - name : str
            - void_fraction : float
            - cos_theta : float

        Returns
        -------
        RectangularCable
            A fully configured rectangular cable instance.

        Raises
        ------
        ValueError
            If neither `dx` nor `aspect_ratio` is provided in the configuration.
        """
        sc_strand = SuperconductingStrand.from_dict(None, config["sc_strand"])
        stab_strand = Strand.from_dict(None, config["stab_strand"])

        dx = config.get("dx")
        aspect_ratio = config.get("aspect_ratio")

        # Handle geometry logic
        if dx is not None and aspect_ratio is not None:
            bluemira_warn(
                "Both 'dx' and 'aspect_ratio' specified. Aspect ratio will override dx."
            )

        if aspect_ratio is not None:
            dx = 0.01  # default dx if only aspect ratio is provided

        if dx is None:
            raise ValueError("At least one of 'dx' or 'aspect_ratio' must be specified.")

        # Construct cable
        cable = cls(
            dx=dx,
            sc_strand=sc_strand,
            stab_strand=stab_strand,
            n_sc_strand=config["n_sc_strand"],
            n_stab_strand=config["n_stab_strand"],
            d_cooling_channel=config["d_cooling_channel"],
            void_fraction=config.get("void_fraction", 0.725),
            cos_theta=config.get("cos_theta", 0.97),
            name=config.get("name", "RectangularCable"),
        )

        if aspect_ratio is not None:
            cable.set_aspect_ratio(aspect_ratio)

        return cable


class DummyRectangularCableHTS(RectangularCable):
    """
    Dummy rectangular cable with young's moduli set to 120 GPa.
    """

    name = "DummyRectangularCableHTS"

    def E(self, **kwargs):  # noqa: N802, PLR6301, ARG002
        """
        Return the Young's modulus of the cable material.

        This is a constant value specific to the implementation. Subclasses may override
        this method to provide a temperature- or field-dependent modulus. The `kwargs`
        parameter is unused here but retained for interface consistency.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments (unused in this implementation).

        Returns
        -------
        float
            Young's modulus in Pascals [Pa].
        """
        return 120e9


class DummyRectangularCableLTS(RectangularCable):
    """
    Dummy square cable with young's moduli set to 0.1 GPa
    """

    name = "DummyRectangularCableLTS"

    def E(self, **kwargs):  # noqa: N802, PLR6301, ARG002
        """
        Return the Young's modulus of the cable material.

        This implementation returns a fixed value (0.1 GPa). Subclasses may override
        this method with more sophisticated behavior. `kwargs` are included for
        compatibility but not used in this implementation.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments (not used here).

        Returns
        -------
        float
            Young's modulus in Pascals [Pa].
        """
        return 0.1e9


class SquareCable(ABCCable):
    """
    Cable with a square cross-section.

    Both dx and dy are derived from the total cross-sectional area.
    """

    def __init__(
        self,
        sc_strand: SuperconductingStrand,
        stab_strand: Strand,
        n_sc_strand: int,
        n_stab_strand: int,
        d_cooling_channel: float,
        void_fraction: float = 0.725,
        cos_theta: float = 0.97,
        name: str = "SquareCable",
    ):
        """
        Representation of a square cable.

        Notes
        -----
        No geometrical dimensions are given. They are extrapolated from the cable
        design parameters.

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

        Notes
        -----
        Cooling material not implemented
        """
        super().__init__(
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
        """Cable dimension in the x direction [m]"""
        return np.sqrt(self.area)

    @property
    def dy(self):
        """Cable dimension in the y direction [m]"""
        return self.dx

    # OD homogenized structural properties
    def Kx(self, **kwargs):  # noqa: N802
        """
        Compute the total equivalent stiffness along the x-axis.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments passed to the material stiffness function `E`.

        Returns
        -------
        float
            Homogenized stiffness in the x-direction [Pa].
        """
        return self.E(**kwargs) * self.dy / self.dx

    def Ky(self, **kwargs):  # noqa: N802
        """
        Compute the total equivalent stiffness along the y-axis.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments passed to the material stiffness function `E`.

        Returns
        -------
        float
            Homogenized stiffness in the y-direction [Pa].
        """
        return self.E(**kwargs) * self.dx / self.dy

    def to_dict(self) -> dict:
        """
        Serialize the square cable configuration to a dictionary.

        Includes all base cable properties (strand configuration, cooling, geometry,
        etc.)
        and adds the cable type identifier.

        Returns
        -------
        dict
            A complete dictionary representation of the square cable, including:
            - type : str
            - name : str
            - n_sc_strand : int
            - n_stab_strand : int
            - d_cooling_channel : float
            - void_fraction : float
            - cos_theta : float
            - sc_strand : dict
            - stab_strand : dict
        """
        data = super().to_dict()
        data["type"] = "square"
        return data

    @classmethod
    def from_dict(cls, config: dict) -> "SquareCable":
        """
        Construct a `SquareCable` instance from a dictionary.

        This method deserializes both the cable's structural parameters and
        its nested strand configurations.

        Parameters
        ----------
        config : dict
            Dictionary containing the square cable configuration. Required keys:
            - n_sc_strand : int
            - n_stab_strand : int
            - d_cooling_channel : float
            - sc_strand : dict
            - stab_strand : dict

            Optional keys:
            - name : str
            - void_fraction : float
            - cos_theta : float

        Returns
        -------
        SquareCable
            A new instance of the `SquareCable` class with populated fields.
        """
        sc_strand = SuperconductingStrand.from_dict("sc_strand", config["sc_strand"])
        stab_strand = Strand.from_dict("stab_strand", config["stab_strand"])

        return cls(
            sc_strand=sc_strand,
            stab_strand=stab_strand,
            n_sc_strand=config["n_sc_strand"],
            n_stab_strand=config["n_stab_strand"],
            d_cooling_channel=config["d_cooling_channel"],
            void_fraction=config.get("void_fraction", 0.725),
            cos_theta=config.get("cos_theta", 0.97),
            name=config.get("name", "SquareCable"),
        )


class DummySquareCableHTS(SquareCable):
    """
    Dummy square cable with Young's modulus set to 120 GPa.
    """

    name = "DummySquareCableHTS"

    def E(self, **kwargs):  # noqa: N802, PLR6301, ARG002
        """
        Return the Young's modulus for the HTS dummy cable.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments (unused in this implementation).

        Returns
        -------
        float
            Young's modulus in Pascals [Pa].
        """
        return 120e9


class DummySquareCableLTS(SquareCable):
    """
    Dummy square cable with Young's modulus set to 0.1 GPa.
    """

    name = "DummySquareCableLTS"

    def E(self, **kwargs):  # noqa: N802, PLR6301, ARG002
        """
        Return the Young's modulus for the LTS dummy cable.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments (unused in this implementation).

        Returns
        -------
        float
            Young's modulus in Pascals [Pa].
        """
        return 0.1e9


class RoundCable(ABCCable):
    """
    A cable with round cross-section for superconducting applications.

    This cable type includes superconducting and stabilizer strands arranged
    around a central cooling channel.
    """

    def __init__(
        self,
        sc_strand: SuperconductingStrand,
        stab_strand: Strand,
        n_sc_strand: int,
        n_stab_strand: int,
        d_cooling_channel: float,
        void_fraction: float = 0.725,
        cos_theta: float = 0.97,
        name: str = "RoundCable",
    ):
        """
        Representation of a round cable

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
        """
        super().__init__(
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
        """Cable dimension in the x direction [m] (i.e. cable's diameter)"""
        return np.sqrt(self.area * 4 / np.pi)

    @property
    def dy(self):
        """Cable dimension in the y direction [m] (i.e. cable's diameter)"""
        return self.dx

    # OD homogenized structural properties
    # A structural analysis should be performed to check how much the rectangular
    #  approximation is fine also for the round cable.
    def Kx(self, **kwargs):  # noqa: N802
        """
        Compute the equivalent stiffness of the cable along the x-axis.

        This is a homogenized 1D structural property derived from the Young's modulus
        and the cable's geometry. The stiffness reflects the effective resistance
        to deformation in the x-direction.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments forwarded to the `E` method. These may include
            temperature, magnetic field, or other conditions if supported by the
            subclass.

        Returns
        -------
        float
            Equivalent stiffness in the x-direction [Pa].
        """
        return self.E(**kwargs) * self.dy / self.dx

    def Ky(self, **kwargs):  # noqa: N802
        """
        Compute the equivalent stiffness of the cable along the y-axis.

        Similar to `Kx`, this stiffness is derived from the effective Young's modulus
        and the geometric proportions of the cable, representing resistance to
        deformation in the y-direction.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments forwarded to the `E` method. These may include
            temperature, magnetic field, or other conditions if supported by the
            subclass.

        Returns
        -------
        float
            Equivalent stiffness in the y-direction [Pa].
        """
        return self.E(**kwargs) * self.dx / self.dy

    def plot(self, xc: float = 0, yc: float = 0, *, show: bool = False, ax=None):
        """
        Schematic plot of the cable cross-section.

        Parameters
        ----------
        xc : float, optional
            x-coordinate of the cable center [m]. Default is 0.
        yc : float, optional
            y-coordinate of the cable center [m]. Default is 0.
        show : bool, optional
            If True, the plot is displayed immediately using `plt.show()`.
            Default is False.
        ax : matplotlib.axes.Axes or None, optional
            Axis to plot on. If None, a new figure and axis are created.

        Returns
        -------
        matplotlib.axes.Axes
            The axis object containing the cable plot, useful for further customization
            or saving.
        """
        if ax is None:
            _, ax = plt.subplots()

        pc = np.array([xc, yc])

        points_ext = (
            np.array([
                np.array([np.cos(theta), np.sin(theta)]) * self.dx / 2
                for theta in np.linspace(0, np.radians(360), 19)
            ])
            + pc
        )

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

    def to_dict(self) -> dict:
        """
        Serialize the round cable configuration to a dictionary.

        This includes all base cable properties (strand configuration, cooling,
        etc.) and adds the type identifier.

        Returns
        -------
        dict
            A complete dictionary representation of the round cable, including:
            - type : str
            - name : str
            - n_sc_strand : int
            - n_stab_strand : int
            - d_cooling_channel : float
            - void_fraction : float
            - cos_theta : float
            - sc_strand : dict
            - stab_strand : dict
        """
        data = super().to_dict()
        data["type"] = "round"
        return data

    @classmethod
    def from_dict(cls, config: dict) -> "RoundCable":
        """
        Construct a `RoundCable` instance from a dictionary.

        This method deserializes both the cable's structural parameters and
        its nested strand configurations.

        Parameters
        ----------
        config : dict
            Dictionary containing the round cable configuration. Required keys:
            - n_sc_strand : int
            - n_stab_strand : int
            - d_cooling_channel : float
            - sc_strand : dict
            - stab_strand : dict

            Optional keys:
            - name : str
            - void_fraction : float
            - cos_theta : float

        Returns
        -------
        RoundCable
            A new instance of the `RoundCable` class with populated fields.
        """
        sc_strand = SuperconductingStrand.from_dict("sc_strand", config["sc_strand"])
        stab_strand = Strand.from_dict("stab_strand", config["stab_strand"])

        return cls(
            sc_strand=sc_strand,
            stab_strand=stab_strand,
            n_sc_strand=config["n_sc_strand"],
            n_stab_strand=config["n_stab_strand"],
            d_cooling_channel=config["d_cooling_channel"],
            void_fraction=config.get("void_fraction", 0.725),
            cos_theta=config.get("cos_theta", 0.97),
            name=config.get("name", "RoundCable"),
        )


class DummyRoundCableHTS(RoundCable):
    """
    Dummy round cable with Young's modulus set to 120 GPa.

    This class provides a simplified round cable configuration for high-temperature
    superconducting (HTS) analysis with a fixed stiffness value.
    """

    name = "DummyRoundCableHTS"

    def E(self, **kwargs):  # noqa: N802, PLR6301, ARG002
        """
        Return the Young's modulus for the HTS dummy round cable.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments (unused in this implementation).

        Returns
        -------
        float
            Young's modulus in Pascals [Pa].
        """
        return 120e9


class DummyRoundCableLTS(RoundCable):
    """
    Dummy round cable with Young's modulus set to 0.1 GPa.

    This class provides a simplified round cable configuration for low-temperature
    superconducting (LTS) analysis with a fixed, softer stiffness value.
    """

    name = "DummyRoundCableLTS"

    def E(self, **kwargs):  # noqa: N802, PLR6301, ARG002
        """
        Return the Young's modulus for the LTS dummy round cable.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments (unused in this implementation).

        Returns
        -------
        float
            Young's modulus in Pascals [Pa].
        """
        return 0.1e9
