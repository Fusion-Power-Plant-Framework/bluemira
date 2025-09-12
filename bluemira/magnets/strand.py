# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Strand definition and builder module.

Includes:
- Strand and SuperconductingStrand classes (material + geometry + Ic/Jc)
- Automatic class and instance registration mechanisms
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matproplib import OperationalConditions
from matproplib.material import MaterialFraction, mixture

from bluemira import display
from bluemira.base.look_and_feel import bluemira_error
from bluemira.display.plotter import PlotOptions
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_circle


class Strand:
    """
    Represents a strand with a circular cross-section, composed of a homogenised
    mixture of materials.

    This class automatically registers itself and its instances.
    """

    def __init__(
        self,
        materials: list[MaterialFraction],
        d_strand: float,
        operating_temperature: float,
        name: str | None = "Strand",
    ):
        """
        Initialise a Strand instance.

        Parameters
        ----------
        materials:
            Materials composing the strand with their fractions.
        d_strand:
            Strand diameter [m].
        operating_temperature: float
            Operating temperature [K].

        name:
            Name of the strand. Defaults to "Strand".
        """
        self.d_strand = d_strand
        self.operating_temperature = operating_temperature
        self.materials = materials

        self.name = name
        self._shape = None
        # Create homogenised material
        self._homogenised_material = mixture(
            name=name,
            materials=materials,
            fraction_type="mass",
        )

    @property
    def materials(self) -> list[MaterialFraction]:
        """
        List of MaterialFraction materials composing the strand.

        Returns
        -------
        :
            Materials and their fractions.
        """
        return self._materials

    @materials.setter
    def materials(self, new_materials: list[MaterialFraction]):
        """
        Set a new list of materials for the strand.

        Parameters
        ----------
        new_materials:
            New materials to set.

        Raises
        ------
        TypeError
            If new_materials is not a list or contains invalid elements.
        """
        if not isinstance(new_materials, list):
            raise TypeError(
                f"materials must be a list, got {type(new_materials).__name__}."
            )

        for item in new_materials:
            if not isinstance(item, MaterialFraction):
                raise TypeError(
                    f"Each item in materials must be a MaterialFraction, got "
                    f"{type(item).__name__}."
                )

        self._materials = new_materials

    @property
    def area(self) -> float:
        """
        Cross-sectional area of the strand.

        Returns
        -------
        :
            Area [m²].
        """
        return np.pi * (self.d_strand**2) / 4

    @property
    def shape(self) -> BluemiraFace:
        """
        2D geometric representation of the strand.

        Returns
        -------
        :
            Circular face of the strand.
        """
        if self._shape is None:
            self._shape = BluemiraFace([make_circle(self.d_strand)])
        return self._shape

    def E(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Young's modulus of the strand material.

        Parameters
        ----------
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Young's modulus [Pa].
        """
        return self._homogenised_material.youngs_modulus(op_cond)

    def rho(self, op_cond: OperationalConditions) -> float:
        """
        Density of the strand material.

        Parameters
        ----------
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Density [kg/m³].
        """
        return self._homogenised_material.density(op_cond)

    def erho(self, op_cond: OperationalConditions) -> float:
        """
        Electrical resistivity of the strand material.

        Parameters
        ----------
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Electrical resistivity [Ohm·m].
        """
        # Treat parallel calculation for resistivity
        if len(self._homogenised_material.mixture_fraction) > 1:
            # If multiple materials, calculate resistivity in parallel
            return 1 / sum(
                m.fraction / m.material.electrical_resistivity(op_cond)
                for m in self._homogenised_material.mixture_fraction
            )
        return self._homogenised_material.electrical_resistivity(op_cond)

    def Cp(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Specific heat capacity of the strand material.

        Parameters
        ----------
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Specific heat [J/kg/K].
        """
        # Treat volume/specific heat capacity calculation
        if len(self._homogenised_material.mixture_fraction) > 1:
            # Match dw-Cp (even if multiplied by density later, this is still different
            # to a normal homogenised mixture)
            density = self._homogenised_material.density(op_cond)
            return (
                sum(
                    mf.fraction
                    * mf.material.specific_heat_capacity(op_cond)
                    * mf.material.density(op_cond)
                    for mf in self._homogenised_material.mixture_fraction
                )
                / density
            )
        return self._homogenised_material.specific_heat_capacity(op_cond)

    def plot(
        self, ax: plt.Axes | None = None, *, show: bool = True, **kwargs
    ) -> plt.Axes:
        """
        Plot a 2D cross-section of the strand.

        Parameters
        ----------
        ax:
            Axis to plot on.
        show:
            Whether to show the plot immediately.
        kwargs:
            Additional arguments passed to the plot function.

        Returns
        -------
        matplotlib.axes.Axes
            Matplotlib axis with the plot.
        """
        plot_options = PlotOptions()
        plot_options.view = "xy"
        return display.plot_2d(
            self.shape, options=plot_options, ax=ax, show=show, **kwargs
        )

    def __str__(self) -> str:
        """
        String representation of the strand.

        Returns
        -------
        :
            Description of the strand.
        """
        return (
            f"name = {self.name}\n"
            f"d_strand = {self.d_strand}\n"
            f"materials = {self.materials}\n"
            f"shape = {self.shape}\n"
        )

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise the strand instance to a dictionary.

        Returns
        -------
        :
            Dictionary with serialised strand data.
        """
        return {
            "name": self.name,
            "d_strand": self.d_strand,
            "temperature": self.operating_temperature,
            "materials": [
                {
                    "material": m.material,
                    "fraction": m.fraction,
                }
                for m in self.materials
            ],
        }


class SuperconductingStrand(Strand):
    """
    Represents a superconducting strand with a circular cross-section.

    Includes methods to compute critical current (Ic) and critical current
    density (Jc) based on the superconducting material.

    Automatically registered using the RegistrableMeta metaclass.
    """

    def __init__(
        self,
        materials: list[MaterialFraction],
        d_strand: float,
        operating_temperature: float,
        name: str | None = "SuperconductingStrand",
    ):
        """
        Initialise a superconducting strand.

        Parameters
        ----------
        materials:
            Materials composing the strand with their fractions. One material must be
            a supercoductor.
        d_strand:
            Strand diameter [m].
        operating_temperature: float
            Operating temperature [K].
        name:
            Name of the strand. Defaults to "Strand".
        """
        super().__init__(
            materials=materials,
            d_strand=d_strand,
            operating_temperature=operating_temperature,
            name=name,
        )
        self._sc = self._check_materials()

    def _check_materials(self) -> MaterialFraction:
        """
        Ensure there is exactly one superconducting material.

        Returns
        -------
        :
            The identified superconducting material.

        Raises
        ------
        ValueError
            If no superconducting material or multiple are found.
        """
        sc = None
        for material in self.materials:
            if material.material.is_superconductor:
                if sc is None:
                    sc = material
                else:
                    msg = (
                        f"Only one superconductor material can be defined per "
                        f"superconducting strand. Found multiple: {sc} and {material}."
                    )
                    bluemira_error(msg)
                    raise ValueError(msg)

        if sc is None:
            msg = "No superconducting material found in strand."
            bluemira_error(msg)
            raise ValueError(msg)

        return sc

    @property
    def sc_area(self) -> float:
        """
        Cross-sectional area of the superconducting material.

        Returns
        -------
        :
            Superconducting area [m²].
        """
        return self.area * self._sc.fraction

    def Jc(self, op_cond: OperationalConditions) -> float:  # noqa:N802
        """
        Critical current density of the superconducting material.

        Parameters
        ----------
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Critical current density [A/m²].
        """
        if op_cond.strain is None:
            op_cond.strain = 0.0055
        return self._sc.material.critical_current_density(op_cond)

    def Ic(self, op_cond: OperationalConditions) -> float:  # noqa:N802
        """
        Critical current based on Jc and superconducting area.

        Parameters
        ----------
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Critical current [A].
        """
        return self.Jc(op_cond) * self.sc_area

    def plot_Ic_B(  # noqa:N802
        self,
        B: np.ndarray,
        temperature: float,
        ax: plt.Axes | None = None,
        *,
        show: bool = True,
        **kwargs,
    ) -> plt.Axes:
        """
        Plot critical current Ic as a function of magnetic field B.

        Parameters
        ----------
        B:
            Array of magnetic field values [T].
        temperature:
            Operating temperature [K].
        ax:
            Axis to plot on. If None, a new figure is created.
        show:
            Whether to immediately show the plot.
        kwargs:
            Additional arguments passed to Ic calculation.

        Returns
        -------
        :
            Axis with the plotted Ic vs B curve.
        """
        if ax is None:
            _, ax = plt.subplots()

        op_conds = [
            OperationalConditions(
                temperature=temperature,
                magnetic_field=Bi,
                strain=kwargs.get("eps", 0.0055),
            )
            for Bi in B
        ]
        ic_values = [self.Ic(op) for op in op_conds]
        ax.plot(B, ic_values)
        ax.set_title(
            f"Critical Current for {self.__class__.__name__}\nTemperature = "
            f"{temperature} K"
        )
        ax.set_xlabel("Magnetic Field B [T]")
        ax.set_ylabel("Critical Current Ic [A]")
        ax.grid(visible=True)

        if show:
            plt.show()

        return ax
