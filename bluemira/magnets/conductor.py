# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Conductor class"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from bluemira.base.look_and_feel import bluemira_debug
from bluemira.magnets.cable import ABCCable
from bluemira.magnets.utils import (
    parall_k,
    parall_r,
    serie_k,
    serie_r,
)
from bluemira.materials.material import MassFractionMaterial


class Conductor:
    """
    A generic conductor consisting of a cable surrounded by a jacket and an
    insulator.
    """

    def __init__(
        self,
        cable: ABCCable,
        mat_jacket: MassFractionMaterial,
        mat_ins: MassFractionMaterial,
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
        """
        Area of the insulator [m^2]

        Returns
        -------
            float [m²]
        """
        return self.area - self.area_jacket - self.cable.area

    def erho(self, **kwargs):
        """
        Computes the conductor's equivalent resistivity considering the resistance
        of its strands in parallel.

        Parameters
        ----------
        **kwargs: dict
            Additional parameters for resistance calculations.

        Returns
        -------
            float [Ohm m]

        Notes
        -----
        The insulator in not considered into the calculation.
        """
        resistances = np.array([
            self.cable.erho(**kwargs) / self.cable.area,
            self.mat_jacket.erho(**kwargs) / self.area_jacket,
        ])
        res_tot = parall_r(resistances)
        return res_tot * self.area

    def Cp(self, **kwargs):  # noqa: N802
        """
        Computes the conductor's equivalent specific heat considering the specific heats
        of its components in series.

        Parameters
        ----------
        **kwargs: dict
            Additional parameters for resistance calculations.

        Returns
        -------
            float [J/K/m]

        Notes
        -----
        The insulator in not considered into the calculation.
        """
        weighted_specific_heat = np.array([
            self.cable.Cp(**kwargs) * self.cable.area,
            self.mat_jacket.Cp(**kwargs) * self.area_jacket,
        ])
        return serie_r(weighted_specific_heat) / self.area

    def _Kx_topbot_ins(self, **kwargs):  # noqa: N802
        """
        Equivalent stiffness of the top/bottom insulator in the x-direction.

        Returns
        -------
        float
            Axial stiffness [N/m]
        """
        return self.mat_ins.E(**kwargs) * self.cable.dy / self.dx_ins

    def _Kx_lat_ins(self, **kwargs):  # noqa: N802
        """
        Equivalent stiffness of the lateral insulator in the x-direction.

        Returns
        -------
        float
            Axial stiffness [N/m]
        """
        return self.mat_ins.E(**kwargs) * self.dy_ins / self.dx

    def _Kx_lat_jacket(self, **kwargs):  # noqa: N802
        """
        Equivalent stiffness of the lateral jacket in the x-direction.

        Returns
        -------
        float
            Axial stiffness [N/m]
        """
        return self.mat_jacket.E(**kwargs) * self.dy_jacket / (self.dx - 2 * self.dx_ins)

    def _Kx_topbot_jacket(self, **kwargs):  # noqa: N802
        """
        Equivalent stiffness of the top/bottom jacket in the x-direction.

        Returns
        -------
        float
            Axial stiffness [N/m]
        """
        return self.mat_jacket.E(**kwargs) * self.cable.dy / self.dx_jacket

    def _Kx_cable(self, **kwargs):  # noqa: N802
        """
        Equivalent stiffness of the cable in the x-direction.

        Returns
        -------
        float
            Axial stiffness [N/m]
        """
        return self.cable.Kx(**kwargs)

    def Kx(self, **kwargs):  # noqa: N802
        """
        Equivalent stiffness of the conductor in the x-direction.

        Returns
        -------
        float
            Axial stiffness [N/m]
        """
        return parall_k([
            self._Kx_lat_ins(**kwargs),
            self._Kx_lat_jacket(**kwargs),
            serie_k([
                self._Kx_topbot_ins(**kwargs),
                self._Kx_topbot_jacket(**kwargs),
                self._Kx_cable(**kwargs),
                self._Kx_topbot_jacket(**kwargs),
                self._Kx_topbot_ins(**kwargs),
            ]),
            self._Kx_lat_jacket(**kwargs),
            self._Kx_lat_ins(**kwargs),
        ])

    def _Ky_topbot_ins(self, **kwargs):  # noqa: N802
        """
        Equivalent stiffness of the top/bottom insulator in the y-direction.

        Returns
        -------
        float
            Axial stiffness [N/m]
        """
        return self.mat_ins.E(**kwargs) * self.cable.dx / self.dy_ins

    def _Ky_lat_ins(self, **kwargs):  # noqa: N802
        """
        Equivalent stiffness of the lateral insulator in the y-direction.

        Returns
        -------
        float
            Axial stiffness [N/m]
        """
        return self.mat_ins.E(**kwargs) * self.dx_ins / self.dy

    def _Ky_lat_jacket(self, **kwargs):  # noqa: N802
        """
        Equivalent stiffness of the lateral jacket in the y-direction.

        Returns
        -------
        float
            Axial stiffness [N/m]
        """
        return self.mat_jacket.E(**kwargs) * self.dx_jacket / (self.dy - 2 * self.dy_ins)

    def _Ky_topbot_jacket(self, **kwargs):  # noqa: N802
        """
        Equivalent stiffness of the top/bottom jacket in the y-direction.

        Returns
        -------
        float
            Axial stiffness [N/m]
        """
        return self.mat_jacket.E(**kwargs) * self.cable.dx / self.dy_jacket

    def _Ky_cable(self, **kwargs):  # noqa: N802
        """
        Equivalent stiffness of the cable in the y-direction.

        Returns
        -------
        float
            Axial stiffness [N/m]
        """
        return self.cable.Ky(**kwargs)

    def Ky(self, **kwargs):  # noqa: N802
        """
        Equivalent stiffness of the conductor in the y-direction.

        Returns
        -------
        float
            Axial stiffness [N/m]
        """
        return parall_k([
            self._Ky_lat_ins(**kwargs),
            self._Ky_lat_jacket(**kwargs),
            serie_k([
                self._Ky_topbot_ins(**kwargs),
                self._Ky_topbot_jacket(**kwargs),
                self._Ky_cable(**kwargs),
                self._Ky_topbot_jacket(**kwargs),
                self._Ky_topbot_ins(**kwargs),
            ]),
            self._Ky_lat_jacket(**kwargs),
            self._Ky_lat_ins(**kwargs),
        ])

    def _tresca_sigma_jacket(
        self,
        pressure: float,
        f_z: float,
        temperature: float,
        B: float,
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
        pressure :
            The pressure applied along the specified direction (Pa).
        f_z :
            The force applied in the z direction, perpendicular to the conductor
            cross-section (N).
        T :
            The operating temperature (K).
        B :
            The operating magnetic field (T).
        direction :
            The direction along which the pressure is applied ('x' or 'y'). Default is
            'x'.

        Returns
        -------
            The calculated Tresca stress in the jacket (Pa).

        Raises
        ------
        ValueError
            If the specified direction is not 'x' or 'y'.
        """
        operational_point = {"temperature": temperature, "B": B}
        if direction not in {"x", "y"}:
            raise ValueError("Invalid direction: choose either 'x' or 'y'.")

        if direction == "x":
            saf_jacket = (self.cable.dx + 2 * self.dx_jacket) / (2 * self.dx_jacket)

            K = parall_k([  # noqa: N806
                2 * self._Ky_lat_ins(**operational_point),
                2 * self._Ky_lat_jacket(**operational_point),
                serie_k([
                    self._Ky_cable(**operational_point),
                    self._Ky_topbot_jacket(**operational_point) / 2,
                ]),
            ])

            X_jacket = 2 * self._Ky_lat_jacket(**operational_point) / K  # noqa: N806

        else:
            saf_jacket = (self.cable.dy + 2 * self.dy_jacket) / (2 * self.dy_jacket)

            K = parall_k([  # noqa: N806
                2 * self._Kx_lat_ins(**operational_point),
                2 * self._Kx_lat_jacket(**operational_point),
                serie_k([
                    self._Kx_cable(**operational_point),
                    self._Kx_topbot_jacket(**operational_point) / 2,
                ]),
            ])

            X_jacket = 2 * self._Kx_lat_jacket(**operational_point) / K  # noqa: N806

        # tresca_stress = pressure * X_jacket * saf_jacket + f_z / self.area_jacket

        return pressure * X_jacket * saf_jacket + f_z / self.area_jacket

    def optimize_jacket_conductor(
        self,
        pressure: float,
        f_z: float,
        temperature: float,
        B: float,
        allowable_sigma: float,
        bounds: np.ndarray | None = None,
        direction: str = "x",
    ):
        """
        Optimize the jacket dimension of a conductor based on allowable stress using
        the Tresca criterion.

        Parameters
        ----------
        pressure :
            The pressure applied along the specified direction (Pa).
        f_z :
            The force applied in the z direction, perpendicular to the conductor
            cross-section (N).
        temperature :
            The operating temperature (K).
        B :
            The operating magnetic field (T).
        allowable_sigma :
            The allowable stress (Pa) for the jacket material.
        bounds :
            Optional bounds for the jacket thickness optimization (default is None).
        direction :
            The direction along which the pressure is applied ('x' or 'y'). Default is
            'x'.

        Returns
        -------
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
            temperature: float,
            B: float,
            allowable_sigma: float,
            direction: str = "x",
        ):
            """
            Objective function for optimizing conductor jacket thickness based on the
            Tresca yield criterion.

            This function computes the absolute difference between the calculated Tresca
            stress in the jacket and the allowable stress. It is used as a fitness
            function during scalar minimization to determine the optimal jacket
            thickness.

            Parameters
            ----------
            jacket_thickness : float
                Proposed thickness of the conductor jacket [m] in the direction
                perpendicular to the applied pressure.
            pressure : float
                Magnetic or mechanical pressure applied along the specified direction
                [Pa].
            fz : float
                Axial or vertical force applied perpendicular to the cross-section [N].
            temperature : float
                Operating temperature of the conductor [K].
            B : float
                Magnetic field at the conductor location [T].
            allowable_sigma : float
                Maximum allowed stress for the jacket material [Pa].
            direction : str, optional
                Direction of the applied pressure. Can be either 'x' (horizontal) or
                'y' (vertical). Default is 'x'.

            Returns
            -------
            float
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

            sigma_r = self._tresca_sigma_jacket(pressure, fz, temperature, B, direction)
            # diff = abs(sigma_r - allowable_sigma)
            return abs(sigma_r - allowable_sigma)

        method = "bounded" if bounds is not None else None

        result = minimize_scalar(
            fun=sigma_difference,
            args=(pressure, f_z, temperature, B, allowable_sigma),
            bounds=bounds,
            method=method,
            options={"xatol": 1e-4},
        )

        if not result.success:
            raise ValueError("Optimization of the jacket conductor did not converge.")
        self.dx_jacket = result.x
        if direction == "x":
            bluemira_debug(f"Optimal dx_jacket: {self.dx_jacket}")
        else:
            bluemira_debug(f"Optimal dy_jacket: {self.dy_jacket}")
        bluemira_debug(
            f"Averaged sigma in the {direction}-direction: "
            f"{self._tresca_sigma_jacket(pressure, f_z, temperature, B) / 1e6} MPa"
        )

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
        xc : float, optional
            X-coordinate of the conductor center in the reference coordinate system.
            Default is 0.
        yc : float, optional
            Y-coordinate of the conductor center in the reference coordinate system.
            Default is 0.
        show : bool, optional
            If True, the figure is rendered immediately using `plt.show()`.
            Default is False.
        ax : matplotlib.axes.Axes or None, optional
            Axis on which to render the plot. If None, a new figure and axis will be
            created internally.

        Returns
        -------
        ax : matplotlib.axes.Axes
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

    def __str__(self):
        """
        Generate a human-readable string representation of the conductor.

        Returns
        -------
        str
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


class SymmetricConductor(Conductor):
    """
    Representation of a symmetric conductor in which both jacket and insulator
    mantain a constant thickness (i.e. dy_jacket = dx_jacket and dy_ins = dx_ins).
    """

    def __init__(
        self,
        cable: ABCCable,
        mat_jacket: MassFractionMaterial,
        mat_ins: MassFractionMaterial,
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
