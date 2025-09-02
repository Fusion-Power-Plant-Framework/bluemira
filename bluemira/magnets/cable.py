# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Cable class"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matproplib import OperationalConditions
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from bluemira.base.look_and_feel import (
    bluemira_debug,
    bluemira_error,
    bluemira_print,
    bluemira_warn,
)
from bluemira.magnets.strand import (
    Strand,
    SuperconductingStrand,
    create_strand_from_dict,
)
from bluemira.magnets.utils import reciprocal_summation, summation

if TYPE_CHECKING:
    from collections.abc import Callable


class ABCCable(ABC):
    """
    Abstract base class for superconducting cables.

    Defines the general structure and common methods for cables
    composed of superconducting and stabilizer strands.

    Notes
    -----
    - This class is abstract and cannot be instantiated directly.
    - Subclasses must define `dx`, `dy`, `Kx`, `Ky`, and `from_dict`.
    """

    _name_in_registry_: str | None = None  # Abstract base classes should NOT register

    def __init__(
        self,
        sc_strand: SuperconductingStrand,
        stab_strand: Strand,
        n_sc_strand: int,
        n_stab_strand: int,
        d_cooling_channel: float,
        void_fraction: float,
        cos_theta: float,
        name: str = "Cable",
        **props,
    ):
        """
        Representation of a cable. Only the x-dimension of the cable is given as
        input. The y-dimension is calculated on the basis of the cable design.

        Notes
        -----
        Cooling material not implemented.

        Parameters
        ----------
        sc_strand:
            The superconducting strand.
        stab_strand:
            The stabilizer strand.
        n_sc_strand:
            Number of superconducting strands.
        n_stab_strand:
            Number of stabilizing strands.
        d_cooling_channel:
            Diameter of the cooling channel [m].
        void_fraction:
            Ratio of material volume to total volume [unitless].
        cos_theta:
            Correction factor for twist in the cable layout.
        name:
            Identifier for the cable instance.
        """
        # assign
        # Setting self.name triggers automatic instance registration
        self.name = name
        self.sc_strand = sc_strand
        self.stab_strand = stab_strand
        self.n_sc_strand = n_sc_strand
        self.n_stab_strand = n_stab_strand
        self.d_cooling_channel = d_cooling_channel
        self.void_fraction = void_fraction
        self.cos_theta = cos_theta

        youngs_modulus: Callable[[Any, OperationalConditions], float] | float | None = (
            props.pop("E", None)
        )
        if youngs_modulus is not None:
            if "E" in vars(type(self)):
                bluemira_debug("E already defined in class, ignoring")
            else:
                self.E = (
                    youngs_modulus
                    if callable(youngs_modulus)
                    else lambda self, op_cond, v=youngs_modulus: youngs_modulus
                )

        for k, v in props.items():
            setattr(self, k, v if callable(v) else lambda *arg, v=v, **kwargs: v)  # noqa: ARG005
        self._props = list(props.keys()) + (
            [] if "E" in vars(type(self)) or youngs_modulus is None else ["E"]
        )

    @property
    @abstractmethod
    def dx(self):
        """Half Cable dimension in the x-direction [m]."""

    @property
    @abstractmethod
    def dy(self):
        """Half Cable dimension in the y-direction [m]."""

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
        op_cond:
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

    def erho(self, op_cond: OperationalConditions) -> float:
        """
        Computes the cable's equivalent resistivity considering the resistance
        of its strands in parallel.

        Parameters
        ----------
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            resistivity [Ohm m]
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
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Specific heat capacity [J/K/m]
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
    def area_stab(self) -> float:
        """Area of the stabilizer region"""
        return self.stab_strand.area * self.n_stab_strand

    @property
    def area_sc(self) -> float:
        """Area of the superconductor region"""
        return self.sc_strand.area * self.n_sc_strand

    @property
    def area_cc(self) -> float:
        """Area of the cooling channel"""
        return self.d_cooling_channel**2 / 4 * np.pi

    @property
    def area(self) -> float:
        """Area of the cable considering the void fraction"""
        return (
            self.area_sc + self.area_stab
        ) / self.void_fraction / self.cos_theta + self.area_cc

    def E(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Return the effective Young's modulus of the cable [Pa].

        This is a default placeholder implementation in the base class.
        Subclasses may use `kwargs` to modify behavior.

        Parameters
        ----------
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Default Young's modulus (0).
        """
        raise NotImplementedError("E for Cable is not implemented.")

    def _heat_balance_model_cable(
        self,
        t: float,
        temperature: float,
        B_fun: Callable,
        I_fun: Callable,  # noqa: N803
    ) -> float:
        """
        Calculate the derivative of temperature (dT/dt) for a 0D heat balance problem.

        Parameters
        ----------
        t:
            The current time in seconds.
        temperature:
            The current temperature in Celsius.
        B_fun:
            The magnetic field [T] as time function
        I_fun:
            The current [A] flowing through the conductor as time function

        Returns
        -------
        :
            The derivative of temperature with respect to time (dT/dt).
        """
        # Calculate the rate of heat generation (Joule dissipation)
        if isinstance(temperature, np.ndarray):
            temperature = temperature.item()

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

    def optimise_n_stab_ths(
        self,
        t0: float,
        tf: float,
        initial_temperature: float,
        target_temperature: float,
        B_fun: Callable[[float], float],
        I_fun: Callable[[float], float],  # noqa: N803
        bounds: np.ndarray | None = None,
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
        :
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
            B_fun: Callable[[float], float],
            I_fun: Callable[[float], float],  # noqa: N803
        ) -> float:
            """
            Compute the absolute temperature difference at final time between the
            simulated and target temperatures.

            This method modifies the private attribute `_n_stab_strand` to update the
            cable configuration, simulates the temperature evolution over time, and
            returns the absolute difference between the final temperature and the
            specified target.

            Parameters
            ----------
            n_stab:
                Number of stabilizer strands to set temporarily for this simulation.
            t0:
                Initial time of the simulation [s].
            tf:
                Final time of the simulation [s].
            initial_temperature:
                Temperature at the start of the simulation [K].
            target_temperature:
                Desired temperature at the end of the simulation [K].
            B_fun:
                Magnetic field as a time-dependent function [T].
            I_fun:
                Current as a time-dependent function [A].

            Returns
            -------
            :
                Absolute difference between the simulated final temperature and the
                target temperature [K].

            Notes
            -----
            - This method is typically used as a cost function for optimization routines
              (e.g., minimizing the temperature error by tuning `n_stab`).
            - It modifies the internal state `self._n_stab_strand`, which may affect
              subsequent evaluations unless restored.
            """
            self.n_stab_strand = n_stab

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

        result = minimize_scalar(
            fun=final_temperature_difference,
            args=(t0, tf, initial_temperature, target_temperature, B_fun, I_fun),
            bounds=bounds,
            method=None if bounds is None else "bounded",
        )

        if not result.success:
            raise ValueError(
                "n_stab optimization did not converge. Check your input parameters "
                "or initial bracket."
            )

        # Here we re-ensure the n_stab_strand to be an integer
        self.n_stab_strand = int(np.ceil(self.n_stab_strand))

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

        @dataclass
        class StabilisingStrandRes:
            solution: Any
            info_text: str

        return StabilisingStrandRes(
            solution,
            (
                f"Target T: {target_temperature:.2f} K\n"
                f"Initial T: {initial_temperature:.2f} K\n"
                f"SC Strand: {self.sc_strand.name}\n"
                f"n. sc. strand = {self.n_sc_strand}\n"
                f"Stab. strand = {self.stab_strand.name}\n"
                f"n. stab. strand = {self.n_stab_strand}\n"
            ),
        )

    # OD homogenized structural properties
    @abstractmethod
    def Kx(self, op_cond: OperationalConditions):  # noqa: N802
        """Total equivalent stiffness along x-axis"""

    @abstractmethod
    def Ky(self, op_cond: OperationalConditions):  # noqa: N802
        """Total equivalent stiffness along y-axis"""

    def plot(
        self, xc: float = 0, yc: float = 0, *, show: bool = False, ax=plt.Axes | None
    ):
        """
        Plot a schematic view of the cable cross-section.

        This method visualizes the outer shape of the cable and the cooling channel,
        assuming a rectangular or elliptical layout based on `dx`, `dy`, and
        `d_cooling_channel`. It draws the cable centered at (xc, yc) within the
        current coordinate system.

        Parameters
        ----------
        xc:
            x-coordinate of the cable center in the plot [m]. Default is 0.
        yc:
            y-coordinate of the cable center in the plot [m]. Default is 0.
        show:
            If True, the plot is rendered immediately with `plt.show()`.
            Default is False.
        ax:
            The matplotlib Axes object to draw on. If None, a new figure and
            Axes are created internally.

        Returns
        -------
        :
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

        p0 = np.array([-self.dx, -self.dy])
        p1 = np.array([self.dx, -self.dy])
        p2 = np.array([[self.dx, self.dy]])
        p3 = np.array([-self.dx, self.dy])

        points_ext = np.vstack((p0, p1, p2, p3, p0)) + pc
        points_cc = (
            np.array([
                np.array([np.cos(theta), np.sin(theta)]) * self.d_cooling_channel / 2
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

    def __str__(self) -> str:
        """
        Return a human-readable summary of the cable configuration.

        Includes geometric properties, void and twist factors, and a string
        representation of both the superconducting and stabilizer strands.

        Returns
        -------
        :
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

    def to_dict(self) -> dict[str, str | float | int | dict[str, Any]]:
        """
        Serialize the cable instance to a dictionary.

        Returns
        -------
        dict
            Dictionary containing cable and strand configuration.
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
            **{k: getattr(k)() for k in self._props},
        }

    @classmethod
    def from_dict(
        cls,
        cable_dict: dict[str, Any],
        name: str | None = None,
    ) -> ABCCable:
        """
        Deserialize a cable instance from a dictionary.

        Parameters
        ----------
        cable_dict:
            Dictionary containing serialized cable data.
        name:
            Name for the new instance. If None, attempts to use the 'name' field from
            the dictionary.

        Returns
        -------
        :
            Instantiated cable object.

        Raises
        ------
        ValueError
            If name_in_registry mismatch or duplicate instance name.
        """
        name_in_registry = cable_dict.pop("name_in_registry", None)
        expected_name_in_registry = getattr(cls, "_name_in_registry_", cls.__name__)

        if name_in_registry != expected_name_in_registry:
            raise ValueError(
                f"Cannot create {cls.__name__} from dictionary with name_in_registry "
                f"'{name_in_registry}'. Expected '{expected_name_in_registry}'."
            )

        # Deserialize strands
        sc_strand_data = cable_dict.pop("sc_strand")
        if isinstance(sc_strand_data, Strand):
            sc_strand = sc_strand_data
        else:
            sc_strand = create_strand_from_dict(strand_dict=sc_strand_data)

        stab_strand_data = cable_dict.pop("stab_strand")
        if isinstance(stab_strand_data, Strand):
            stab_strand = stab_strand_data
        else:
            stab_strand = create_strand_from_dict(strand_dict=stab_strand_data)

        # how to resolve this with ParameterFrame?
        return cls(
            sc_strand=sc_strand,
            stab_strand=stab_strand,
            n_sc_strand=cable_dict.pop("n_sc_strand"),
            n_stab_strand=cable_dict.pop("n_stab_strand"),
            d_cooling_channel=cable_dict.pop("d_cooling_channel"),
            void_fraction=cable_dict.pop("void_fraction"),
            cos_theta=cable_dict.pop("cos_theta"),
            name=name or cable_dict.pop("name", None),
            **cable_dict,
        )


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
        n_sc_strand: int,
        n_stab_strand: int,
        d_cooling_channel: float,
        void_fraction: float,
        cos_theta: float,
        dx: float,
        name: str = "RectangularCable",
        **props,
    ):
        """
        Representation of a cable. Only the x-dimension of the cable is given as
        input. The y-dimension is calculated on the basis of the cable design.

        Notes
        -----
        Cooling material not implemented.

        Parameters
        ----------
        sc_strand:
            Superconducting strand.
        stab_strand:
            Stabilizer strand.
        n_sc_strand:
            Number of superconducting strands.
        n_stab_strand:
            Number of stabilizing strands.
        d_cooling_channel:
            Diameter of the cooling channel [m].
        void_fraction:
            Ratio of material volume to total volume [unitless].
        cos_theta:
            Correction factor for twist in the cable layout.
        dx:
            Cable half-width in the x-direction [m].
        name:
            Name of the cable
        props:
            extra properties
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
            **props,
        )
        self.dx = dx

    @property
    def dy(self) -> float:
        """Half Cable dimension in the y direction [m]"""
        return self.area / self.dx / 4

    # Decide if this function shall be a setter.
    # Defined as "normal" function to underline that it modifies dx.
    def set_aspect_ratio(self, value: float):
        """Modify dx in order to get the given aspect ratio"""
        self.dx = np.sqrt(value * self.area) / 2

    # OD homogenized structural properties
    def Kx(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Compute the total equivalent stiffness along the x-axis.

        Parameters
        ----------
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Homogenized stiffness in the x-direction [Pa].
        """
        return self.E(op_cond) * self.dy / self.dx

    def Ky(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Compute the total equivalent stiffness along the y-axis.

        Parameters
        ----------
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Homogenized stiffness in the y-direction [Pa].
        """
        return self.E(op_cond) * self.dx / self.dy

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the rectangular cable into a dictionary.

        Returns
        -------
        :
            Dictionary including rectangular cable parameters.
        """
        data = super().to_dict()
        data.update({
            "dx": self.dx,
            "aspect_ratio": self.aspect_ratio,
        })
        return data

    @classmethod
    def from_dict(
        cls,
        cable_dict: dict[str, Any],
        name: str | None = None,
    ) -> RectangularCable:
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
        cable_dict:
            Dictionary containing serialized cable data.
        name:
            Name for the new instance. If None, attempts to use the 'name' field from
            the dictionary.

        Returns
        -------
        :
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
        sc_strand_data = cable_dict.pop("sc_strand")
        if isinstance(sc_strand_data, Strand):
            sc_strand = sc_strand_data
        else:
            sc_strand = create_strand_from_dict(strand_dict=sc_strand_data)

        stab_strand_data = cable_dict.pop("stab_strand")
        if isinstance(stab_strand_data, Strand):
            stab_strand = stab_strand_data
        else:
            stab_strand = create_strand_from_dict(strand_dict=stab_strand_data)

        # Geometry parameters
        dx = cable_dict.pop("dx", None)
        aspect_ratio = cable_dict.pop("aspect_ratio")

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
        n_sc_strand = cable_dict.pop("n_sc_strand")
        n_stab_strand = cable_dict.pop("n_stab_strand")
        d_cooling_channel = cable_dict.pop("d_cooling_channel")
        void_fraction = cable_dict.pop("void_fraction")
        cos_theta = cable_dict.pop("cos_theta")

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
            name=name or cable_dict.pop("name", None),
            **cable_dict,
        )

        # Adjust aspect ratio if needed
        if aspect_ratio is not None:
            cable.set_aspect_ratio(aspect_ratio)

        return cable


class SquareCable(ABCCable):
    """
    Cable with a square cross-section.

    Both dx and dy are derived from the total cross-sectional area.
    """

    _name_in_registry_ = "SquareCable"

    def __init__(
        self,
        sc_strand: SuperconductingStrand,
        stab_strand: Strand,
        n_sc_strand: int,
        n_stab_strand: int,
        d_cooling_channel: float,
        void_fraction: float,
        cos_theta: float,
        name: str = "SquareCable",
        **props,
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
        n_sc_strand:
            Number of superconducting strands.
        n_stab_strand:
            Number of stabilizing strands.
        d_cooling_channel:
            Diameter of the cooling channel [m].
        void_fraction:
            Ratio of material volume to total volume [unitless].
        cos_theta:
            Correction factor for twist in the cable layout.
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
            **props,
        )

    @property
    def dx(self) -> float:
        """Half Cable dimension in the x direction [m]"""
        return np.sqrt(self.area / 4)

    @property
    def dy(self) -> float:
        """Half Cable dimension in the y direction [m]"""
        return self.dx

    # OD homogenized structural properties
    def Kx(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Compute the total equivalent stiffness along the x-axis.

        Parameters
        ----------
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Homogenized stiffness in the x-direction [Pa].
        """
        return self.E(op_cond)

    def Ky(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Compute the total equivalent stiffness along the y-axis.

        Parameters
        ----------
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Homogenized stiffness in the y-direction [Pa].
        """
        return self.E(op_cond)

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the SquareCable.

        Returns
        -------
        :
            Serialized dictionary.
        """
        return super().to_dict()

    @classmethod
    def from_dict(
        cls,
        cable_dict: dict[str, Any],
        name: str | None = None,
    ) -> SquareCable:
        """
        Deserialize a SquareCable from a dictionary.

        Parameters
        ----------
        cable_dict:
            Dictionary containing serialized cable data.
        name:
            Name for the new instance. If None, attempts to use the 'name' field from
            the dictionary.

        Returns
        -------
        :
            Instantiated square cable.

        Raises
        ------
        ValueError
            If unique_name is False and a duplicate name is detected in the instance
            cache.
        """
        sc_strand = create_strand_from_dict(strand_dict=cable_dict.pop("sc_strand"))
        stab_strand = create_strand_from_dict(strand_dict=cable_dict.pop("stab_strand"))

        # how to handle this?
        return cls(
            sc_strand=sc_strand,
            stab_strand=stab_strand,
            n_sc_strand=cable_dict.pop("n_sc_strand"),
            n_stab_strand=cable_dict.pop("n_stab_strand"),
            d_cooling_channel=cable_dict.pop("d_cooling_channel"),
            void_fraction=cable_dict.pop("void_fraction"),
            cos_theta=cable_dict.pop("cos_theta"),
            name=name or cable_dict.pop("name", None),
        )


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
        void_fraction: float,
        cos_theta: float,
        name: str = "RoundCable",
        **props,
    ):
        """
        Representation of a round cable

        Parameters
        ----------
        sc_strand:
            strand of the superconductor
        stab_strand:
            strand of the stabilizer
        n_sc_strand:
            Number of superconducting strands.
        n_stab_strand:
            Number of stabilizing strands.
        d_cooling_channel:
            Diameter of the cooling channel [m].
        void_fraction:
            Ratio of material volume to total volume [unitless].
        cos_theta:
            Correction factor for twist in the cable layout.
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
            **props,
        )

    @property
    def dx(self) -> float:
        """Half Cable dimension in the x direction [m] (i.e. cable's radius)"""
        return np.sqrt(self.area / np.pi)

    @property
    def dy(self) -> float:
        """Half Cable dimension in the y direction [m] (i.e. cable's radius)"""
        return self.dx

    # OD homogenized structural properties
    # A structural analysis should be performed to check how much the rectangular
    #  approximation is fine also for the round cable.
    def Kx(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Compute the equivalent stiffness of the cable along the x-axis.

        This is a homogenized 1D structural property derived from the Young's modulus
        and the cable's geometry. The stiffness reflects the effective resistance
        to deformation in the x-direction.

        Parameters
        ----------
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Equivalent stiffness in the x-direction [Pa].
        """
        return self.E(op_cond)

    def Ky(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Compute the equivalent stiffness of the cable along the y-axis.

        Similar to `Kx`, this stiffness is derived from the effective Young's modulus
        and the geometric proportions of the cable, representing resistance to
        deformation in the y-direction.

        Parameters
        ----------
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Equivalent stiffness in the y-direction [Pa].
        """
        return self.E(op_cond)

    def plot(
        self,
        xc: float = 0,
        yc: float = 0,
        *,
        show: bool = False,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """
        Schematic plot of the cable cross-section.

        Parameters
        ----------
        xc:
            x-coordinate of the cable center [m]. Default is 0.
        yc:
            y-coordinate of the cable center [m]. Default is 0.
        show:
            If True, the plot is displayed immediately using `plt.show()`.
            Default is False.
        ax:
            Axis to plot on. If None, a new figure and axis are created.

        Returns
        -------
        :
            The axis object containing the cable plot, useful for further customization
            or saving.
        """
        if ax is None:
            _, ax = plt.subplots()

        pc = np.array([xc, yc])

        points_ext = (
            np.array([
                np.array([np.cos(theta), np.sin(theta)]) * self.dx
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

        ax.fill(points_ext[:, 0], points_ext[:, 1], "gold", snap=False)
        ax.fill(points_cc[:, 0], points_cc[:, 1], "r", snap=False)

        if show:
            plt.show()
        return ax

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the RoundCable.

        Returns
        -------
        :
            Serialized dictionary.
        """
        return super().to_dict()

    @classmethod
    def from_dict(
        cls,
        cable_dict: dict[str, Any],
        name: str | None = None,
    ) -> RoundCable:
        """
        Deserialize a RoundCable from a dictionary.

        Parameters
        ----------
        cable_dict:
            Dictionary containing serialized cable data.
        name:
            Name for the new instance. If None, attempts to use the 'name' field from
            the dictionary.

        Returns
        -------
        :
            Instantiated square cable.

        Raises
        ------
        ValueError
            If unique_name is False and a duplicate name is detected in the instance
            cache.
        """
        sc_strand = create_strand_from_dict(strand_dict=cable_dict.pop("sc_strand"))
        stab_strand = create_strand_from_dict(strand_dict=cable_dict.pop("stab_strand"))

        # how to handle?
        return cls(
            sc_strand=sc_strand,
            stab_strand=stab_strand,
            n_sc_strand=cable_dict.pop("n_sc_strand"),
            n_stab_strand=cable_dict.pop("n_stab_strand"),
            d_cooling_channel=cable_dict.pop("d_cooling_channel"),
            void_fraction=cable_dict.pop("void_fraction"),
            cos_theta=cable_dict.pop("cos_theta"),
            name=name or cable_dict.pop("name", None),
            **cable_dict,
        )


def create_cable_from_dict(
    cable_dict: dict,
    name: str | None = None,
) -> ABCCable:
    """
    Factory function to create a Cable or its subclass from a serialized dictionary.

    Parameters
    ----------
    cable_dict:
        Dictionary with serialized cable data. Must include a 'name_in_registry' field.
    name:
        If given, overrides the name from the dictionary.

    Returns
    -------
    :
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
