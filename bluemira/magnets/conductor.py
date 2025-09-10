# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Conductor class"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from bluemira.base.look_and_feel import bluemira_debug
from bluemira.magnets.cable import ABCCable
from bluemira.magnets.utils import reciprocal_summation, summation

if TYPE_CHECKING:
    from matproplib import OperationalConditions
    from matproplib.material import Material


class Conductor:
    """
    A generic conductor consisting of a cable surrounded by a jacket and an
    insulator.
    """

    def __init__(
        self,
        cable: ABCCable,
        mat_jacket: Material,
        mat_ins: Material,
        dx_jacket: float,
        dy_jacket: float,
        dx_ins: float,
        dy_ins: float,
        name: str = "Conductor",
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
            x-thickness of the jacket [m].
        dy_jacket:
            y-thickness of the jacket [m].
        dx_ins:
            x-thickness of the insulator [m].
        dy_ins:
            y-thickness of the insulator [m].
        name:
            string identifier
        """
        self.name = name
        self.dx_jacket = dx_jacket
        self._dy_jacket = dy_jacket
        self.dx_ins = dx_ins
        self._dy_ins = dy_ins
        self.mat_ins = mat_ins
        self.mat_jacket = mat_jacket
        self.cable = cable

    @property
    def dy_jacket(self):
        """y-thickness of the jacket [m]"""
        return self._dy_jacket

    @property
    def dy_ins(self):
        """y-thickness of the ins [m]"""
        return self._dy_ins

    @property
    def dx(self):
        """Half x-dimension of the conductor [m]"""
        return self.dx_ins + self.dx_jacket + self.cable.dx

    @property
    def dy(self):
        """Half y-dimension of the conductor [m]"""
        return self.dy_ins + self.dy_jacket + self.cable.dy

    @property
    def area(self):
        """Area of the conductor [m^2]"""
        return self.dx * self.dy * 4

    @property
    def area_jacket(self):
        """Area of the jacket [m^2]"""
        return 4 * (self.cable.dx + self.dx_jacket) * (self.cable.dy + self.dy_jacket)

    @property
    def area_ins(self):
        """
        Area of the insulator [m^2]

        Returns
        -------
        :
            area [m²]
        """
        return self.area - self.area_jacket - self.cable.area

    def to_dict(self) -> dict[str, Any]:
        """
        Serialize the conductor instance to a dictionary.

        Returns
        -------
        :
            Dictionary with serialized conductor data.
        """
        return {
            "name": self.name,
            "cable": self.cable.to_dict(),
            "mat_jacket": self.mat_jacket.name,
            "mat_ins": self.mat_ins.name,
            "dx_jacket": self.dx_jacket,
            "dy_jacket": self.dy_jacket,
            "dx_ins": self.dx_ins,
            "dy_ins": self.dy_ins,
        }

    @classmethod
    def from_dict(
        cls,
        conductor_dict: dict[str, Any],
        name: str | None = None,
    ) -> Conductor:
        """
        Deserialize a Conductor instance from a dictionary.

        Parameters
        ----------
        conductor_dict:
            Dictionary containing serialized conductor data.
        name:
            Name for the new instance. If None, attempts to use the 'name' field from
            the dictionary.

        Returns
        -------
        :
            A fully reconstructed Conductor instance.

        Raises
        ------
        ValueError
            If the 'name_in_registry' field does not match the expected class
            registration name,
            or if the name already exists and unique_name is False.
        """
        # Deserialize cable
        cable = create_cable_from_dict(
            cable_dict=conductor_dict["cable"],
        )

        # Resolve jacket material
        mat_jacket = conductor_dict["mat_jacket"]

        # Resolve insulation material
        mat_ins = conductor_dict["mat_ins"]

        # Instantiate
        return cls(
            cable=cable,
            mat_jacket=mat_jacket,
            mat_ins=mat_ins,
            dx_jacket=conductor_dict["dx_jacket"],
            dy_jacket=conductor_dict["dy_jacket"],
            dx_ins=conductor_dict["dx_ins"],
            dy_ins=conductor_dict["dy_ins"],
            name=name or conductor_dict.get("name"),
        )

    def erho(self, op_cond: OperationalConditions) -> float:
        """
        Computes the conductor's equivalent resistivity considering the resistance
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

        Notes
        -----
        The insulator in not considered into the calculation.
        """
        resistances = np.array([
            self.cable.erho(op_cond) / self.cable.area,
            self.mat_jacket.electrical_resistivity(op_cond) / self.area_jacket,
        ])
        res_tot = reciprocal_summation(resistances)
        return res_tot * self.area

    def Cp(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Computes the conductor's equivalent specific heat considering the specific heats
        of its components in series.

        Parameters
        ----------
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Specific heat capacity [J/K/m]

        Notes
        -----
        The insulator in not considered into the calculation.
        """
        weighted_specific_heat = np.array([
            self.cable.Cp(op_cond) * self.cable.area,
            self.mat_jacket.specific_heat_capacity(op_cond) * self.area_jacket,
        ])
        return summation(weighted_specific_heat) / self.area

    def _Kx_topbot_ins(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Equivalent stiffness of the top/bottom insulator in the x-direction.

        Returns
        -------
        :
            Axial stiffness [N/m]
        """
        return self.mat_ins.youngs_modulus(op_cond) * 2 * self.cable.dy / self.dx_ins

    def _Kx_lat_ins(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Equivalent stiffness of the lateral insulator in the x-direction.

        Returns
        -------
        :
            Axial stiffness [N/m]
        """
        return self.mat_ins.youngs_modulus(op_cond) * self.dy_ins / (2 * self.dx)

    def _Kx_lat_jacket(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Equivalent stiffness of the lateral jacket in the x-direction.

        Returns
        -------
        :
            Axial stiffness [N/m]
        """
        return (
            self.mat_jacket.youngs_modulus(op_cond)
            * self.dy_jacket
            / (2 * self.dx - 2 * self.dx_ins)
        )

    def _Kx_topbot_jacket(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Equivalent stiffness of the top/bottom jacket in the x-direction.

        Returns
        -------
        :
            Axial stiffness [N/m]
        """
        return (
            self.mat_jacket.youngs_modulus(op_cond) * 2 * self.cable.dy / self.dx_jacket
        )

    def Kx(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Equivalent stiffness of the conductor in the x-direction.

        Returns
        -------
        :
            Axial stiffness [N/m]
        """
        return summation([
            self._Kx_lat_ins(op_cond),
            self._Kx_lat_jacket(op_cond),
            reciprocal_summation([
                self._Kx_topbot_ins(op_cond),
                self._Kx_topbot_jacket(op_cond),
                self.cable.Kx(op_cond),
                self._Kx_topbot_jacket(op_cond),
                self._Kx_topbot_ins(op_cond),
            ]),
            self._Kx_lat_jacket(op_cond),
            self._Kx_lat_ins(op_cond),
        ])

    def _Ky_topbot_ins(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Equivalent stiffness of the top/bottom insulator in the y-direction.

        Returns
        -------
        :
            Axial stiffness [N/m]
        """
        return self.mat_ins.youngs_modulus(op_cond) * 2 * self.cable.dx / self.dy_ins

    def _Ky_lat_ins(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Equivalent stiffness of the lateral insulator in the y-direction.

        Returns
        -------
        :
            Axial stiffness [N/m]
        """
        return self.mat_ins.youngs_modulus(op_cond) * self.dx_ins / (2 * self.dy)

    def _Ky_lat_jacket(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Equivalent stiffness of the lateral jacket in the y-direction.

        Returns
        -------
        :
            Axial stiffness [N/m]
        """
        return (
            self.mat_jacket.youngs_modulus(op_cond)
            * self.dx_jacket
            / (2 * self.dy - 2 * self.dy_ins)
        )

    def _Ky_topbot_jacket(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Equivalent stiffness of the top/bottom jacket in the y-direction.

        Returns
        -------
        :
            Axial stiffness [N/m]
        """
        return (
            self.mat_jacket.youngs_modulus(op_cond) * 2 * self.cable.dx / self.dy_jacket
        )

    def Ky(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Equivalent stiffness of the conductor in the y-direction.

        Returns
        -------
        :
            Axial stiffness [N/m]
        """
        return summation([
            self._Ky_lat_ins(op_cond),
            self._Ky_lat_jacket(op_cond),
            reciprocal_summation([
                self._Ky_topbot_ins(op_cond),
                self._Ky_topbot_jacket(op_cond),
                self.cable.Ky(op_cond),
                self._Ky_topbot_jacket(op_cond),
                self._Ky_topbot_ins(op_cond),
            ]),
            self._Ky_lat_jacket(op_cond),
            self._Ky_lat_ins(op_cond),
        ])

    def _tresca_sigma_jacket(
        self,
        pressure: float,
        f_z: float,
        op_cond: OperationalConditions,
        direction: str = "x",
    ) -> float:
        """
        Calculate the radial stress in the jacket when the conductor is subjected to a
        pressure
        along a specified direction and a force perpendicular to the conductor
        cross-section.
        The Tresca criterion is used for this calculation.

        Parameters
        ----------
        pressure:
            The pressure applied along the specified direction (Pa).
        f_z:
            The force applied in the z direction, perpendicular to the conductor
            cross-section (N).
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.
        direction:
            The direction along which the pressure is applied ('x' or 'y'). Default is
            'x'.

        Returns
        -------
        :
            The calculated Tresca stress in the jacket (Pa).

        Raises
        ------
        ValueError
            If the specified direction is not 'x' or 'y'.
        """
        if direction not in {"x", "y"}:
            raise ValueError("Invalid direction: choose either 'x' or 'y'.")

        if direction == "x":
            saf_jacket = (self.cable.dx + self.dx_jacket) / (self.dx_jacket)

            K = summation([  # noqa: N806
                2 * self._Ky_lat_ins(op_cond),
                2 * self._Ky_lat_jacket(op_cond),
                reciprocal_summation([
                    self.cable.Ky(op_cond),
                    self._Ky_topbot_jacket(op_cond) / 2,
                ]),
            ])

            X_jacket = 2 * self._Ky_lat_jacket(op_cond) / K  # noqa: N806

        else:
            saf_jacket = (self.cable.dy + self.dy_jacket) / (self.dy_jacket)

            K = summation([  # noqa: N806
                2 * self._Kx_lat_ins(op_cond),
                2 * self._Kx_lat_jacket(op_cond),
                reciprocal_summation([
                    self.cable.Kx(op_cond),
                    self._Kx_topbot_jacket(op_cond) / 2,
                ]),
            ])

            X_jacket = 2 * self._Kx_lat_jacket(op_cond) / K  # noqa: N806

        # tresca_stress
        return pressure * X_jacket * saf_jacket + f_z / self.area_jacket

    def optimize_jacket_conductor(
        self,
        pressure: float,
        f_z: float,
        op_cond: OperationalConditions,
        allowable_sigma: float,
        bounds: np.ndarray | None = None,
        direction: str = "x",
    ):
        """
        Optimize the jacket dimension of a conductor based on allowable stress using
        the Tresca criterion.

        Parameters
        ----------
        pressure:
            The pressure applied along the specified direction (Pa).
        f_z:
            The force applied in the z direction, perpendicular to the conductor
            cross-section (N).
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material properties.
        allowable_sigma:
            The allowable stress (Pa) for the jacket material.
        bounds:
            Optional bounds for the jacket thickness optimization (default is None).
        direction:
            The direction along which the pressure is applied ('x' or 'y'). Default is
            'x'.

        Returns
        -------
        :
            The result of the optimization process containing information about the
            optimal jacket thickness.

        Raises
        ------
        ValueError
            If the optimization process did not converge.

        Notes
        -----
        This function uses the Tresca yield criterion to optimize the thickness of the
        jacket surrounding the conductor.
        This function directly update the conductor's jacket thickness along the x
        direction to the optimal value.
        """

        def sigma_difference(
            jacket_thickness: float,
            pressure: float,
            fz: float,
            op_cond: OperationalConditions,
            allowable_sigma: float,
            direction: str = "x",
        ) -> float:
            """
            Objective function for optimizing conductor jacket thickness based on the
            Tresca yield criterion.

            This function computes the absolute difference between the calculated Tresca
            stress in the jacket and the allowable stress. It is used as a fitness
            function during scalar minimization to determine the optimal jacket
            thickness.

            Parameters
            ----------
            jacket_thickness:
                Proposed thickness of the conductor jacket [m] in the direction
                perpendicular to the applied pressure.
            pressure:
                Magnetic or mechanical pressure applied along the specified direction
                [Pa].
            fz:
                Axial or vertical force applied perpendicular to the cross-section [N].
            op_cond:
                Operational conditions including temperature, magnetic field, and strain
                at which to calculate the material property.
            allowable_sigma:
                Maximum allowed stress for the jacket material [Pa].
            direction:
                Direction of the applied pressure. Can be either 'x' (horizontal) or
                'y' (vertical). Default is 'x'.

            Returns
            -------
            :
                Absolute difference between the calculated Tresca stress and the
                allowable stress [Pa].

            Raises
            ------
            ValueError
                If the `direction` is not 'x' or 'y'.

            Notes
            -----
            - This function updates the conductor's internal jacket dimension (
            `dx_jacket` or `dy_jacket`) with the trial value `jacket_thickness`.
            - It is intended for use with scalar optimization algorithms such as
              `scipy.optimize.minimize_scalar`.
            """
            if direction not in {"x", "y"}:
                raise ValueError("Invalid direction: choose either 'x' or 'y'.")

            if direction == "x":
                self.dx_jacket = jacket_thickness
            else:
                self.dy_jacket = jacket_thickness

            sigma_r = self._tresca_sigma_jacket(pressure, fz, op_cond, direction)

            # Normal difference
            diff = abs(sigma_r - allowable_sigma)

            # Penalty if stress exceeds allowable
            if sigma_r > allowable_sigma:
                penalty = 1e6 + (sigma_r - allowable_sigma) * 1e6
                return diff + penalty

            return diff

        debug_msg = ["Method optimize_jacket_conductor:"]

        if direction == "x":
            debug_msg.append(f"Previous dx_jacket: {self.dx_jacket}")
        else:
            debug_msg.append(f"Previous dy_jacket: {self.dy_jacket}")

        method = "bounded" if bounds is not None else None

        if method == "bounded":
            debug_msg.append(f"bounds: {bounds}")

        result = minimize_scalar(
            fun=sigma_difference,
            args=(pressure, f_z, op_cond, allowable_sigma),
            bounds=bounds,
            method=method,
            options={"xatol": 1e-4},
        )

        if not result.success:
            raise ValueError("Optimization of the jacket conductor did not converge.")
        if direction == "x":
            self.dx_jacket = result.x
            debug_msg.append(f"Optimal dx_jacket: {self.dx_jacket}")
        else:
            self.dy_jacket = result.x
            debug_msg.append(f"Optimal dy_jacket: {self.dy_jacket}")
        debug_msg.append(
            f"Averaged sigma in the {direction}-direction: "
            f"{self._tresca_sigma_jacket(pressure, f_z, op_cond) / 1e6} MPa\n"
            f"Allowable stress in the {direction}-direction: {allowable_sigma / 1e6} "
            f"MPa."
        )
        debug_msg = "\n".join(debug_msg)
        bluemira_debug(debug_msg)

        return result

    def plot(self, xc: float = 0, yc: float = 0, *, show: bool = False, ax=None):
        """
        Plot a schematic cross-section of the conductor, including cable, jacket,
        and insulator layers.

        This method visualizes the hierarchical geometry of the conductor centered
        at a given position. The jacket and insulator are drawn as rectangles,
        while the internal cable uses its own plotting method.

        Parameters
        ----------
        xc:
            X-coordinate of the conductor center in the reference coordinate system.
            Default is 0.
        yc:
            Y-coordinate of the conductor center in the reference coordinate system.
            Default is 0.
        show:
            If True, the figure is rendered immediately using `plt.show()`.
            Default is False.
        ax:
            Axis on which to render the plot. If None, a new figure and axis will be
            created internally.

        Returns
        -------
        ax:
            The axis containing the rendered plot.

        Notes
        -----
        - The conductor consists of three nested parts:
            1. Inner cable (delegated to `self.cable.plot()`),
            2. Jacket layer (blue rectangle),
            3. Insulation layer (red rectangle).
        - The aspect ratio of the plot is set to 'equal' if `show=True` to ensure
          accurate representation of geometry.
        """
        if ax is None:
            _, ax = plt.subplots()

        pc = np.array([xc, yc])
        a = self.cable.dx + self.dx_jacket
        b = self.cable.dy + self.dy_jacket

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

        ax.fill(points_ext_ins[:, 0], points_ext_ins[:, 1], "red", snap=False)
        ax.fill(points_ext_jacket[:, 0], points_ext_jacket[:, 1], "blue", snap=False)

        ax = self.cable.plot(xc=xc, yc=yc, show=False, ax=ax)

        if show:
            ax.set_aspect("equal")
            plt.show()

        return ax

    def __str__(self) -> str:
        """
        Generate a human-readable string representation of the conductor.

        Returns
        -------
        :
            A multi-line summary of the conductor's key dimensions and its nested
            cable description. This includes:
              - Total x and y dimensions,
              - Jacket and insulator thicknesses,
              - Delegated string representation of the internal cable object.
        """
        return (
            f"name: {self.name}\n"
            f"dx: {self.dx}\n"
            f"dy: {self.dy}\n"
            f"------- cable -------\n"
            f"cable: {self.cable!s}\n"
            f"---------------------\n"
            f"dx_jacket: {self.dx_jacket}\n"
            f"dy_jacket: {self.dy_jacket}\n"
            f"dx_ins: {self.dx_ins}\n"
            f"dy_ins: {self.dy_ins}"
        )


class SymmetricConductor(Conductor):  # jm -    actually worthwhile or just set up
    #         conductor with dx = dy and don't duplicate?
    """
    Representation of a symmetric conductor in which both jacket and insulator
    mantain a constant thickness (i.e. dy_jacket = dx_jacket and dy_ins = dx_ins).
    """

    def __init__(
        self,
        cable: ABCCable,
        mat_jacket: Material,
        mat_ins: Material,
        dx_jacket: float,
        dx_ins: float,
        name: str = "SymmetricConductor",
    ):
        """
        Representation of a symmetric conductor in which both jacket and insulator
        mantain a constant thickness (i.e. dy_jacket = dx_jacket and dy_ins = dx_ins).

        Parameters
        ----------
        cable:
            the conductor's cable
        mat_jacket:
            jacket's material
        mat_ins:
            insulator's material
        dx_jacket:
            x-thickness of the jacket [m].
        dx_ins:
            x-thickness of the insulator [m].
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
        """
        y-thickness of the jacket [m].

        Returns
        -------
        float

        Notes
        -----
        Assumes the same value as `dx_jacket`, ensuring symmetry in both directions.
        """
        return self.dx_jacket

    @property
    def dy_ins(self):
        """
        y-thickness of the insulator [m].

        Returns
        -------
        float

        Notes
        -----
        Assumes the same value as `dx_ins`, ensuring symmetry in both directions.
        """
        return self.dx_ins

    def to_dict(self) -> dict:
        """
        Serialize the symmetric conductor instance to a dictionary.

        Returns
        -------
        dict
            Dictionary with serialized symmetric conductor data.
        """
        return {
            "name": self.name,
            "cable": self.cable.to_dict(),
            "mat_jacket": self.mat_jacket.name,
            "mat_ins": self.mat_ins.name,
            "dx_jacket": self.dx_jacket,
            "dx_ins": self.dx_ins,
        }

    @classmethod
    def from_dict(
        cls,
        conductor_dict: dict[str, Any],
        name: str | None = None,
    ) -> SymmetricConductor:
        """
        Deserialize a SymmetricConductor instance from a dictionary.

        Parameters
        ----------
        conductor_dict:
            Dictionary containing serialized conductor data.
        name:
            Name for the new instance.

        Returns
        -------
        :
            A fully reconstructed SymmetricConductor instance.

        Raises
        ------
        ValueError
            If the 'name_in_registry' does not match the expected registration name.
        """
        # Deserialize cable
        cable = create_cable_from_dict(
            cable_dict=conductor_dict["cable"],
        )

        # Resolve jacket material
        mat_jacket = conductor_dict["mat_jacket"]

        # Resolve insulation material
        mat_ins = conductor_dict["mat_ins"]

        # Instantiate
        return cls(
            cable=cable,
            mat_jacket=mat_jacket,
            mat_ins=mat_ins,
            dx_jacket=conductor_dict["dx_jacket"],
            dx_ins=conductor_dict["dx_ins"],
            name=name or conductor_dict.get("name"),
        )


def create_conductor_from_dict(
    conductor_dict: dict,
    name: str | None = None,
) -> Conductor:
    """
    Factory function to create a Conductor (or subclass) from a serialized dictionary.

    Parameters
    ----------
    conductor_dict:
        Serialized conductor dictionary, must include 'name_in_registry' field.
    name:
        Name to assign to the created conductor. If None, uses the name in the
        dictionary.

    Returns
    -------
    :
        A fully instantiated Conductor (or subclass) object.

    Raises
    ------
    ValueError
        If no class is registered with the given name_in_registry.
    """
    name_in_registry = conductor_dict.get("name_in_registry")
    if name_in_registry is None:
        raise ValueError("Conductor dictionary must include 'name_in_registry' field.")

    conductor_cls = CONDUCTOR_REGISTRY.get(name_in_registry)
    if conductor_cls is None:
        available = list(CONDUCTOR_REGISTRY.keys())
        raise ValueError(
            f"No registered conductor class with name_in_registry '{name_in_registry}'. "
            f"Available: {available}"
        )

    return conductor_cls.from_dict(
        name=name,
        conductor_dict=conductor_dict,
    )
