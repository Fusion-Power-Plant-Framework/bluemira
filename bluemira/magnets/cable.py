# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Cable class"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matproplib import OperationalConditions
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from bluemira.base.look_and_feel import bluemira_error, bluemira_print, bluemira_warn
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.base.parameter_frame.typed import ParameterFrameLike
from bluemira.magnets.registry import RegistrableMeta
from bluemira.magnets.strand import (
    Strand,
    SuperconductingStrand,
    create_strand_from_dict,
)
from bluemira.magnets.utils import reciprocal_summation, summation

# ------------------------------------------------------------------------------
# Global Registries
# ------------------------------------------------------------------------------
CABLE_REGISTRY = {}


# ------------------------------------------------------------------------------
# Cable Class
# ------------------------------------------------------------------------------
@dataclass
class CableParams(ParameterFrame):
    """
    Parameters needed for the TF cable
    """

    n_sc_strand: Parameter[int]
    """Number of superconducting strands."""
    n_stab_strand: Parameter[int]
    """Number of stabilizing strands."""
    d_cooling_channel: Parameter[float]
    """Diameter of the cooling channel [m]."""
    void_fraction: Parameter[float] = 0.725
    """Ratio of material volume to total volume [unitless]."""
    cos_theta: Parameter[float] = 0.97
    """Correction factor for twist in the cable layout."""


class ABCCable(ABC, metaclass=RegistrableMeta):
    """
    Abstract base class for superconducting cables.

    Defines the general structure and common methods for cables
    composed of superconducting and stabilizer strands.

    Notes
    -----
    - This class is abstract and cannot be instantiated directly.
    - Subclasses must define `dx`, `dy`, `Kx`, `Ky`, and `from_dict`.
    """

    _registry_ = CABLE_REGISTRY
    _name_in_registry_: str | None = None  # Abstract base classes should NOT register
    param_cls: type[CableParams] = CableParams

    def __init__(
        self,
        sc_strand: SuperconductingStrand,
        stab_strand: Strand,
        params: ParameterFrameLike,
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
        super().__init__(params)  # fix when split into builders and designers
        # initialize private variables
        self._shape = None  # remove?

        # assign
        # Setting self.name triggers automatic instance registration
        self.name = name
        self.sc_strand = sc_strand
        self.stab_strand = stab_strand

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

    def rho(self, op_cond: OperationalConditions):
        """
        Compute the average mass density of the cable [kg/m³].

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        float
            Averaged mass density in kg/m³.
        """
        return (
            self.sc_strand.rho(op_cond) * self.area_sc
            + self.stab_strand.rho(op_cond) * self.area_stab
        ) / (self.area_sc + self.area_stab)

    def erho(self, op_cond: OperationalConditions):
        """
        Computes the cable's equivalent resistivity considering the resistance
        of its strands in parallel.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
            float [Ohm m]
        """
        resistances = np.array([
            self.sc_strand.erho(op_cond) / self.area_sc,
            self.stab_strand.erho(op_cond) / self.area_stab,
        ])
        res_tot = reciprocal_summation(resistances)
        return res_tot * self.area

    def Cp(self, op_cond: OperationalConditions):  # noqa: N802
        """
        Computes the cable's equivalent specific heat considering the specific heats
        of its strands in series.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
            float [J/K/m]
        """
        weighted_specific_heat = np.array([
            self.sc_strand.Cp(op_cond) * self.area_sc * self.sc_strand.rho(op_cond),
            self.stab_strand.Cp(op_cond)
            * self.area_stab
            * self.stab_strand.rho(op_cond),
        ])
        return summation(weighted_specific_heat) / (
            self.area_sc * self.sc_strand.rho(op_cond)
            + self.area_stab * self.stab_strand.rho(op_cond)
        )

    @property
    def area_stab(self):
        """Area of the stabilizer region"""
        return self.stab_strand.area * self.params.n_stab_strand.value

    @property
    def area_sc(self):
        """Area of the superconductor region"""
        return self.sc_strand.area * self.params.n_sc_strand.value

    @property
    def area_cc(self):
        """Area of the cooling channel"""
        return self.params.d_cooling_channel.value**2 / 4 * np.pi

    @property
    def area(self):
        """Area of the cable considering the void fraction"""
        return (
            self.area_sc + self.area_stab
        ) / self.params.void_fraction.value / self.params.cos_theta.value + self.area_cc

    def E(self, op_cond: OperationalConditions):  # noqa: N802
        """
        Return the effective Young's modulus of the cable [Pa].

        This is a default placeholder implementation in the base class.
        Subclasses may use `kwargs` to modify behavior.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

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

        op_cond = OperationalConditions(temperature=temperature, magnetic_field=B_fun(t))

        Q_gen = (I_fun(t) / self.area) ** 2 * self.erho(op_cond)  # noqa:N806

        # Calculate the rate of heat absorption by conductor components
        Q_abs = self.Cp(op_cond) * self.rho(op_cond)  # noqa:N806

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
            self.params.n_stab_strand.value = n_stab

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
        self.params.n_stab_strand.value = int(np.ceil(self.params.n_stab_strand.value))

        solution = self._temperature_evolution(t0, tf, initial_temperature, B_fun, I_fun)
        final_temperature = solution.y[0][-1]

        if final_temperature > target_temperature:
            bluemira_error(
                f"Final temperature ({final_temperature:.2f} K) exceeds target "
                f"temperature "
                f"({target_temperature} K) even with maximum n_stab = "
                f"{self.params.n_stab_strand.value}."
            )
            raise ValueError(
                "Optimization failed to keep final temperature ≤ target. "
                "Try increasing the upper bound of n_stab or adjusting cable parameters."
            )
        bluemira_print(f"Optimal n_stab: {self.params.n_stab_strand.value}")
        bluemira_print(
            f"Final temperature with optimal n_stab: {final_temperature:.2f} Kelvin"
        )

        if show:
            _, (ax_temp, ax_ib) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

            # --- Plot Temperature Evolution ---
            ax_temp.plot(solution.t, solution.y[0], "r*", label="Simulation points")
            time_steps = np.linspace(t0, tf, 100)
            ax_temp.plot(
                time_steps, solution.sol(time_steps)[0], "b", label="Interpolated curve"
            )
            ax_temp.grid(visible=True)
            ax_temp.set_ylabel("Temperature [K]", fontsize=10)
            ax_temp.set_title("Quench temperature evolution", fontsize=11)
            ax_temp.legend(fontsize=9)

            ax_temp.tick_params(axis="y", labelcolor="k", labelsize=9)

            # Insert text box with additional info
            info_text = (
                f"Target T: {target_temperature:.2f} K\n"
                f"Initial T: {initial_temperature:.2f} K\n"
                f"SC Strand: {self.sc_strand.name}\n"
                f"n. sc. strand = {self.params.n_sc_strand.value}\n"
                f"Stab. strand = {self.stab_strand.name}\n"
                f"n. stab. strand = {self.params.n_stab_strand.value}\n"
            )
            props = {"boxstyle": "round", "facecolor": "white", "alpha": 0.8}
            ax_temp.text(
                0.65,
                0.5,
                info_text,
                transform=ax_temp.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=props,
            )

            # --- Plot I_fun(t) and B_fun(t) ---
            time_steps_fine = np.linspace(t0, tf, 300)
            I_values = [I_fun(t) for t in time_steps_fine]  # noqa: N806
            B_values = [B_fun(t) for t in time_steps_fine]

            ax_ib.plot(time_steps_fine, I_values, "g", label="Current [A]")
            ax_ib.set_ylabel("Current [A]", color="g", fontsize=10)
            ax_ib.tick_params(axis="y", labelcolor="g", labelsize=9)
            ax_ib.grid(visible=True)

            ax_ib_right = ax_ib.twinx()
            ax_ib_right.plot(
                time_steps_fine, B_values, "m--", label="Magnetic field [T]"
            )
            ax_ib_right.set_ylabel("Magnetic field [T]", color="m", fontsize=10)
            ax_ib_right.tick_params(axis="y", labelcolor="m", labelsize=9)

            # Labels
            ax_ib.set_xlabel("Time [s]", fontsize=10)
            ax_ib.tick_params(axis="x", labelsize=9)

            # Combined legend for both sides
            lines, labels = ax_ib.get_legend_handles_labels()
            lines2, labels2 = ax_ib_right.get_legend_handles_labels()
            ax_ib.legend(lines + lines2, labels + labels2, loc="best", fontsize=9)

            plt.tight_layout()
            plt.show()

        return result

    # OD homogenized structural properties
    @abstractmethod
    def Kx(self, op_cond: OperationalConditions):  # noqa: N802
        """Total equivalent stiffness along x-axis"""

    @abstractmethod
    def Ky(self, op_cond: OperationalConditions):  # noqa: N802
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
                np.array([np.cos(theta), np.sin(theta)])
                * self.params.d_cooling_channel.value
                / 2
                for theta in np.linspace(0, np.radians(360), 19)
            ])
            + pc
        )

        ax.fill(points_ext[:, 0], points_ext[:, 1], "gold", snap=False)
        ax.fill(points_cc[:, 0], points_cc[:, 1], "r", snap=False)
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
            f"d cooling channel: {self.params.d_cooling_channel.value}\n"
            f"void fraction: {self.params.void_fraction.value}\n"
            f"cos(theta): {self.params.cos_theta.value}\n"
            f"----- sc strand -------\n"
            f"sc strand: {self.sc_strand!s}\n"
            f"----- stab strand -------\n"
            f"stab strand: {self.stab_strand!s}\n"
            f"-----------------------\n"
            f"n sc strand: {self.params.n_sc_strand.value}\n"
            f"n stab strand: {self.params.n_stab_strand.value}"
        )

    def to_dict(self) -> dict:
        """
        Serialize the cable instance to a dictionary.

        Returns
        -------
        dict
            Dictionary containing cable and strand configuration.
        """
        return {
            "name_in_registry": getattr(
                self, "_name_in_registry_", self.__class__.__name__
            ),
            "name": self.name,
            "n_sc_strand": self.params.n_sc_strand.value,
            "n_stab_strand": self.params.n_stab_strand.value,
            "d_cooling_channel": self.params.d_cooling_channel.value,
            "void_fraction": self.params.void_fraction.value,
            "cos_theta": self.params.cos_theta.value,
            "sc_strand": self.sc_strand.to_dict(),
            "stab_strand": self.stab_strand.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        cable_dict: dict[str, Any],
        name: str | None = None,
    ) -> "ABCCable":
        """
        Deserialize a cable instance from a dictionary.

        Parameters
        ----------
        cls : type
            Class to instantiate (Cable or subclass).
        cable_dict : dict
            Dictionary containing serialized cable data.
        name : str
            Name for the new instance. If None, attempts to use the 'name' field from
            the dictionary.

        Returns
        -------
        ABCCable
            Instantiated cable object.

        Raises
        ------
        ValueError
            If name_in_registry mismatch or duplicate instance name.
        """
        name_in_registry = cable_dict.get("name_in_registry")
        expected_name_in_registry = getattr(cls, "_name_in_registry_", cls.__name__)

        if name_in_registry != expected_name_in_registry:
            raise ValueError(
                f"Cannot create {cls.__name__} from dictionary with name_in_registry "
                f"'{name_in_registry}'. Expected '{expected_name_in_registry}'."
            )

        # Deserialize strands
        sc_strand_data = cable_dict["sc_strand"]
        if isinstance(sc_strand_data, Strand):
            sc_strand = sc_strand_data
        else:
            sc_strand = create_strand_from_dict(strand_dict=sc_strand_data)

        stab_strand_data = cable_dict["stab_strand"]
        if isinstance(stab_strand_data, Strand):
            stab_strand = stab_strand_data
        else:
            stab_strand = create_strand_from_dict(strand_dict=stab_strand_data)

        # how to resolve this with ParameterFrame?
        return cls(
            sc_strand=sc_strand,
            stab_strand=stab_strand,
            n_sc_strand=cable_dict["n_sc_strand"],
            n_stab_strand=cable_dict["n_stab_strand"],
            d_cooling_channel=cable_dict["d_cooling_channel"],
            void_fraction=cable_dict.get("void_fraction", 0.725),
            cos_theta=cable_dict.get("cos_theta", 0.97),
            name=name or cable_dict.get("name"),
        )


@dataclass
class RectangularCableParams(ParameterFrame):
    """
    Parameters needed for the TF cable
    """

    dx: Parameter[float]
    """Cable width in the x-direction [m]."""
    n_sc_strand: Parameter[int]
    """Number of superconducting strands."""
    n_stab_strand: Parameter[int]
    """Number of stabilizing strands."""
    d_cooling_channel: Parameter[float]
    """Diameter of the cooling channel [m]."""
    void_fraction: Parameter[float] = 0.725
    """Ratio of material volume to total volume [unitless]."""
    cos_theta: Parameter[float] = 0.97
    """Correction factor for twist in the cable layout."""


class RectangularCable(ABCCable):
    """
    Cable with a rectangular cross-section.

    The x-dimension is provided directly. The y-dimension is derived based on
    the total area and x-dimension.
    """

    _name_in_registry_ = "RectangularCable"

    def __init__(
        self,
        sc_strand: SuperconductingStrand,
        stab_strand: Strand,
        params: ParameterFrameLike,
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
            params=params,
            name=name,
        )

    @property
    def dy(self):
        """Cable dimension in the y direction [m]"""
        return self.area / self.params.dx.value

    # Decide if this function shall be a setter.
    # Defined as "normal" function to underline that it modifies dx.
    def set_aspect_ratio(self, value: float) -> None:
        """Modify dx in order to get the given aspect ratio"""
        self.params.dx.value = np.sqrt(value * self.area)

    # OD homogenized structural properties
    def Kx(self, op_cond: OperationalConditions):  # noqa: N802
        """
        Compute the total equivalent stiffness along the x-axis.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        float
            Homogenized stiffness in the x-direction [Pa].
        """
        return self.E(op_cond) * self.dy /         self.params.dx.value

    def Ky(self, op_cond: OperationalConditions):  # noqa: N802
        """
        Compute the total equivalent stiffness along the y-axis.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        float
            Homogenized stiffness in the y-direction [Pa].
        """
        return self.E(op_cond) * self.params.dx.value / self.dy

    def to_dict(self) -> dict:
        """
        Serialize the rectangular cable into a dictionary.

        Returns
        -------
        dict
            Dictionary including rectangular cable parameters.
        """
        data = super().to_dict()
        data.update({
            "dx": self.params.dx.value,
            "aspect_ratio": self.aspect_ratio,
        })
        return data

    @classmethod
    def from_dict(
        cls,
        cable_dict: dict[str, Any],
        name: str | None = None,
    ) -> "RectangularCable":
        """
        Deserialize a RectangularCable from a dictionary.

        Behavior:
        - If both 'dx' and 'aspect_ratio' are provided, a warning is issued and
        aspect_ratio is applied.
        - If only 'aspect_ratio' is provided, dx and dy are calculated accordingly.
        - If only 'dx' is provided, it is used as-is.
        - If neither is provided, raises a ValueError.

        Parameters
        ----------
        cls : type
            Class to instantiate (Cable or subclass).
        cable_dict : dict
            Dictionary containing serialized cable data.
        name : str
            Name for the new instance. If None, attempts to use the 'name' field from
            the dictionary.

        Returns
        -------
        RectangularCable
            Instantiated rectangular cable object.

        Raises
        ------
        ValueError
            If neither 'dx' nor 'aspect_ratio' is provided.
        """
        name_in_registry = cable_dict.get("name_in_registry")
        expected_name_in_registry = getattr(cls, "_name_in_registry_", cls.__name__)

        if name_in_registry != expected_name_in_registry:
            raise ValueError(
                f"Cannot create {cls.__name__} from dictionary with name_in_registry "
                f"'{name_in_registry}'. Expected '{expected_name_in_registry}'."
            )

        # Deserialize strands
        sc_strand_data = cable_dict["sc_strand"]
        if isinstance(sc_strand_data, Strand):
            sc_strand = sc_strand_data
        else:
            sc_strand = create_strand_from_dict(strand_dict=sc_strand_data)

        stab_strand_data = cable_dict["stab_strand"]
        if isinstance(stab_strand_data, Strand):
            stab_strand = stab_strand_data
        else:
            stab_strand = create_strand_from_dict(strand_dict=stab_strand_data)

        # Geometry parameters
        dx = cable_dict.get("dx")
        aspect_ratio = cable_dict.get("aspect_ratio")

        if dx is not None and aspect_ratio is not None:
            bluemira_warn(
                "Both 'dx' and 'aspect_ratio' specified. Aspect ratio will override dx "
                "after creation."
            )

        if aspect_ratio is not None and dx is None:
            # Default dx if only aspect ratio is provided. It will be recalculated at
            # the end when set_aspect_ratio is called
            dx = 0.01

        if dx is None:
            raise ValueError(
                "Serialized RectangularCable must include at least 'dx' or "
                "'aspect_ratio'."
            )

        # Base cable parameters
        n_sc_strand = cable_dict["n_sc_strand"]
        n_stab_strand = cable_dict["n_stab_strand"]
        d_cooling_channel = cable_dict["d_cooling_channel"]
        void_fraction = cable_dict.get("void_fraction", 0.725)
        cos_theta = cable_dict.get("cos_theta", 0.97)

        # how to handle with parameterframe?
        # Create cable
        cable = cls(
            dx=dx,
            sc_strand=sc_strand,
            stab_strand=stab_strand,
            n_sc_strand=n_sc_strand,
            n_stab_strand=n_stab_strand,
            d_cooling_channel=d_cooling_channel,
            void_fraction=void_fraction,
            cos_theta=cos_theta,
            name=name or cable_dict.get("name"),
        )

        # Adjust aspect ratio if needed
        if aspect_ratio is not None:
            cable.set_aspect_ratio(aspect_ratio)

        return cable


class DummyRectangularCableHTS(RectangularCable):
    """
    Dummy rectangular cable with young's moduli set to 120 GPa.
    """

    _name_in_registry_ = "DummyRectangularCableHTS"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("name", "DummyRectangularCableHTS")
        super().__init__(*args, **kwargs)

    def E(self, op_cond: OperationalConditions):  # noqa: N802, PLR6301, ARG002
        """
        Return the Young's modulus of the cable material.

        This is a constant value specific to the implementation. Subclasses may override
        this method to provide a temperature- or field-dependent modulus. The `kwargs`
        parameter is unused here but retained for interface consistency.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

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

    _name_in_registry_ = "DummyRectangularCableLTS"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("name", "DummyRectangularCableLTS")
        super().__init__(*args, **kwargs)

    def E(self, op_cond):  # noqa: N802, PLR6301, ARG002
        """
        Return the Young's modulus of the cable material.

        This implementation returns a fixed value (0.1 GPa). Subclasses may override
        this method with more sophisticated behavior. `kwargs` are included for
        compatibility but not used in this implementation.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

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

    _name_in_registry_ = "SquareCable"
    param_cls: type[CableParams] = CableParams

    def __init__(
        self,
        sc_strand: SuperconductingStrand,
        stab_strand: Strand,
        params: ParameterFrameLike,
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
            params=params,
            name=name,
        )

    # replace dx and dy with dl?
    @property
    def dx(self):
        """Cable dimension in the x direction [m]"""
        return np.sqrt(self.area)

    @property
    def dy(self):
        """Cable dimension in the y direction [m]"""
        return self.dx

    # OD homogenized structural properties
    def Kx(self, op_cond: OperationalConditions):  # noqa: N802
        """
        Compute the total equivalent stiffness along the x-axis.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        float
            Homogenized stiffness in the x-direction [Pa].
        """
        return self.E(op_cond) * self.dy / self.dx

    def Ky(self, op_cond: OperationalConditions):  # noqa: N802
        """
        Compute the total equivalent stiffness along the y-axis.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        float
            Homogenized stiffness in the y-direction [Pa].
        """
        return self.E(op_cond) * self.dx / self.dy

    def to_dict(self) -> dict:
        """
        Serialize the SquareCable.

        Returns
        -------
        dict
            Serialized dictionary.
        """
        return super().to_dict()

    @classmethod
    def from_dict(
        cls,
        cable_dict: dict[str, Any],
        name: str | None = None,
    ) -> "SquareCable":
        """
        Deserialize a SquareCable from a dictionary.

        Parameters
        ----------
        cls : type
            Class to instantiate (Cable or subclass).
        cable_dict : dict
            Dictionary containing serialized cable data.
        name : str
            Name for the new instance. If None, attempts to use the 'name' field from
            the dictionary.

        Returns
        -------
        SquareCable
            Instantiated square cable.

        Raises
        ------
        ValueError
            If unique_name is False and a duplicate name is detected in the instance
            cache.
        """
        name_in_registry = cable_dict.get("name_in_registry")
        expected_name_in_registry = getattr(cls, "_name_in_registry_", cls.__name__)

        if name_in_registry != expected_name_in_registry:
            raise ValueError(
                f"Cannot create {cls.__name__} from dictionary with name_in_registry "
                f"'{name_in_registry}'. Expected '{expected_name_in_registry}'."
            )

        sc_strand = create_strand_from_dict(strand_dict=cable_dict["sc_strand"])
        stab_strand = create_strand_from_dict(strand_dict=cable_dict["stab_strand"])

        # how to handle this?
        return cls(
            sc_strand=sc_strand,
            stab_strand=stab_strand,
            n_sc_strand=cable_dict["n_sc_strand"],
            n_stab_strand=cable_dict["n_stab_strand"],
            d_cooling_channel=cable_dict["d_cooling_channel"],
            void_fraction=cable_dict.get("void_fraction", 0.725),
            cos_theta=cable_dict.get("cos_theta", 0.97),
            name=name or cable_dict.get("name"),
        )


class DummySquareCableHTS(SquareCable):
    """
    Dummy square cable with Young's modulus set to 120 GPa.
    """

    _name_in_registry_ = "DummySquareCableHTS"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("name", "DummySquareCableHTS")
        super().__init__(*args, **kwargs)

    def E(self, op_cond: OperationalConditions):  # noqa: N802, PLR6301, ARG002
        """
        Return the Young's modulus for the HTS dummy cable.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

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

    _name_in_registry_ = "DummySquareCableLTS"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("name", "DummySquareCableLTS")
        super().__init__(*args, **kwargs)

    def E(self, op_cond: OperationalConditions):  # noqa: N802, PLR6301, ARG002
        """
        Return the Young's modulus for the LTS dummy cable.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

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

    _name_in_registry_ = "RoundCable"
    param_cls: type[CableParams] = CableParams

    def __init__(
        self,
        sc_strand: SuperconductingStrand,
        stab_strand: Strand,
        params: ParameterFrameLike,
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
            params=params,
            name=name,
        )

    # replace dx and dy with dr?
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
    def Kx(self, op_cond: OperationalConditions):  # noqa: N802
        """
        Compute the equivalent stiffness of the cable along the x-axis.

        This is a homogenized 1D structural property derived from the Young's modulus
        and the cable's geometry. The stiffness reflects the effective resistance
        to deformation in the x-direction.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        float
            Equivalent stiffness in the x-direction [Pa].
        """
        return self.E(op_cond) * self.dy / self.dx

    def Ky(self, op_cond: OperationalConditions):  # noqa: N802
        """
        Compute the equivalent stiffness of the cable along the y-axis.

        Similar to `Kx`, this stiffness is derived from the effective Young's modulus
        and the geometric proportions of the cable, representing resistance to
        deformation in the y-direction.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        float
            Equivalent stiffness in the y-direction [Pa].
        """
        return self.E(op_cond) * self.dx / self.dy

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
                np.array([np.cos(theta), np.sin(theta)])
                * self.params.d_cooling_channel.value
                / 2
                for theta in np.linspace(0, np.radians(360), 19)
            ])
            + pc
        )

        ax.fill(points_ext[:, 0], points_ext[:, 1], "gold", snap=False)
        ax.fill(points_cc[:, 0], points_cc[:, 1], "r", snap=False)

        if show:
            plt.show()
        return ax

    def to_dict(self) -> dict:
        """
        Serialize the RoundCable.

        Returns
        -------
        dict
            Serialized dictionary.
        """
        return super().to_dict()

    @classmethod
    def from_dict(
        cls,
        cable_dict: dict[str, Any],
        name: str | None = None,
    ) -> "RoundCable":
        """
        Deserialize a RoundCable from a dictionary.

        Parameters
        ----------
        cls : type
            Class to instantiate (Cable or subclass).
        cable_dict : dict
            Dictionary containing serialized cable data.
        name : str
            Name for the new instance. If None, attempts to use the 'name' field from
            the dictionary.

        Returns
        -------
        RoundCable
            Instantiated square cable.

        Raises
        ------
        ValueError
            If unique_name is False and a duplicate name is detected in the instance
            cache.
        """
        name_in_registry = cable_dict.get("name_in_registry")
        expected_name_in_registry = getattr(cls, "_name_in_registry_", cls.__name__)

        if name_in_registry != expected_name_in_registry:
            raise ValueError(
                f"Cannot create {cls.__name__} from dictionary with name_in_registry "
                f"'{name_in_registry}'. Expected '{expected_name_in_registry}'."
            )

        sc_strand = create_strand_from_dict(strand_dict=cable_dict["sc_strand"])
        stab_strand = create_strand_from_dict(strand_dict=cable_dict["stab_strand"])

        # how to handle?
        return cls(
            sc_strand=sc_strand,
            stab_strand=stab_strand,
            n_sc_strand=cable_dict["n_sc_strand"],
            n_stab_strand=cable_dict["n_stab_strand"],
            d_cooling_channel=cable_dict["d_cooling_channel"],
            void_fraction=cable_dict.get("void_fraction", 0.725),
            cos_theta=cable_dict.get("cos_theta", 0.97),
            name=name or cable_dict.get("name"),
        )


class DummyRoundCableHTS(RoundCable):
    """
    Dummy round cable with Young's modulus set to 120 GPa.

    This class provides a simplified round cable configuration for high-temperature
    superconducting (HTS) analysis with a fixed stiffness value.
    """

    _name_in_registry_ = "DummyRoundCableHTS"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("name", "DummyRoundCableHTS")
        super().__init__(*args, **kwargs)

    def E(self, op_cond: OperationalConditions):  # noqa: N802, PLR6301, ARG002
        """
        Return the Young's modulus for the HTS dummy round cable.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

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

    _name_in_registry_ = "DummyRoundCableLTS"

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("name", "DummyRoundCableLTS")
        super().__init__(*args, **kwargs)

    def E(self, op_cond: OperationalConditions):  # noqa: N802, PLR6301, ARG002
        """
        Return the Young's modulus for the LTS dummy round cable.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        float
            Young's modulus in Pascals [Pa].
        """
        return 0.1e9


def create_cable_from_dict(
    cable_dict: dict,
    name: str | None = None,
):
    """
    Factory function to create a Cable or its subclass from a serialized dictionary.

    Parameters
    ----------
    cable_dict : dict
        Dictionary with serialized cable data. Must include a 'name_in_registry' field.
    name : str, optional
        If given, overrides the name from the dictionary.

    Returns
    -------
    ABCCable
        Instantiated cable object.

    Raises
    ------
    ValueError
        If 'name_in_registry' is missing or no matching class is found.
    """
    name_in_registry = cable_dict.get("name_in_registry")
    if name_in_registry is None:
        raise ValueError(
            "Serialized cable dictionary must contain a 'name_in_registry' field."
        )

    cls = CABLE_REGISTRY.get(name_in_registry)
    if cls is None:
        raise ValueError(
            f"No registered cable class with registration name '{name_in_registry}'. "
            "Available classes are: " + ", ".join(CABLE_REGISTRY.keys())
        )

    return cls.from_dict(name=name, cable_dict=cable_dict)
