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
from bluemira.magnets.registry import RegistrableMeta

# ------------------------------------------------------------------------------
# Global Registries
# ------------------------------------------------------------------------------

STRAND_REGISTRY = {}

# ------------------------------------------------------------------------------
# Strand Class
# ------------------------------------------------------------------------------


class Strand(metaclass=RegistrableMeta):
    """
    Represents a strand with a circular cross-section, composed of a homogenized
    mixture of materials.

    This class automatically registers itself and its instances.
    """

    _registry_ = STRAND_REGISTRY
    _name_in_registry_ = "Strand"

    def __init__(
        self,
        materials: list[MaterialFraction],
        d_strand: float = 0.82e-3,
        temperature: float | None = None,
        name: str | None = "Strand",
    ):
        """
        Initialize a Strand instance.

        Parameters
        ----------
        materials : list of MaterialFraction
            Materials composing the strand with their fractions.
        d_strand : float, optional
            Strand diameter in meters (default 0.82e-3).
        temperature : float, optional
            Operating temperature [K].
        name : str or None, optional
            Name of the strand. Defaults to "Strand".
        """
        self._d_strand = None
        self._shape = None
        self._materials = None
        self._temperature = None

        self.d_strand = d_strand
        self.materials = materials
        self.name = name
        self.temperature = temperature

        # Create homogenised material
        self._homogenised_material = mixture(
            name=name,
            materials=materials,
            fraction_type="mass",
        )

    @property
    def materials(self) -> list:
        """
        List of MaterialFraction materials composing the strand.

        Returns
        -------
        list of MaterialFraction
            Materials and their fractions.
        """
        return self._materials

    @materials.setter
    def materials(self, new_materials: list):
        """
        Set a new list of materials for the strand.

        Parameters
        ----------
        new_materials : list of MaterialFraction
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
    def temperature(self) -> float | None:
        """
        Operating temperature of the strand.

        Returns
        -------
        float or None
            Temperature in Kelvin.
        """
        return self._temperature

    @temperature.setter
    def temperature(self, value: float | None):
        """
        Set a new operating temperature for the strand.

        Parameters
        ----------
        value : float or None
            New operating temperature in Kelvin.

        Raises
        ------
        ValueError
            If temperature is negative.
        TypeError
            If temperature is not a float or None.
        """
        if value is not None:
            if not isinstance(value, (float, int)):
                raise TypeError(
                    f"temperature must be a float or int, got {type(value).__name__}."
                )

            if value < 0:
                raise ValueError("Temperature cannot be negative.")

        self._temperature = float(value) if value is not None else None

    @property
    def d_strand(self) -> float:
        """
        Diameter of the strand.

        Returns
        -------
        Parameter
            Diameter [m].
        """
        return self._d_strand

    @d_strand.setter
    def d_strand(self, d: float):
        """
        Set the strand diameter and reset shape if changed.

        Parameters
        ----------
        d : float or Parameter
            New strand diameter.

        Raises
        ------
        ValueError
            If diameter is non-positive.
        TypeError
            If diameter is not a float number.
        """
        if not isinstance(d, (float, int)):
            raise TypeError(f"d_strand must be a float, got {type(d).__name__}")
        if d <= 0:
            raise ValueError("d_strand must be positive.")

        if self.d_strand is None or d != self.d_strand:
            self._d_strand = float(d)
            self._shape = None

    @property
    def area(self) -> float:
        """
        Cross-sectional area of the strand.

        Returns
        -------
        float
            Area [m²].
        """
        return np.pi * (self.d_strand**2) / 4

    @property
    def shape(self) -> BluemiraFace:
        """
        2D geometric representation of the strand.

        Returns
        -------
        BluemiraFace
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
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        float
            Young's modulus [Pa].
        """
        return self._homogenised_material.youngs_modulus(op_cond)

    def rho(self, op_cond: OperationalConditions) -> float:
        """
        Density of the strand material.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        float
            Density [kg/m³].
        """
        return self._homogenised_material.density(op_cond)

    def erho(self, op_cond: OperationalConditions) -> float:
        """
        Electrical resistivity of the strand material.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        float
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
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        float
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

    def plot(self, ax=None, *, show: bool = True, **kwargs):
        """
        Plot a 2D cross-section of the strand.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis to plot on.
        show : bool, optional
            Whether to show the plot immediately.
        kwargs : dict
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
        str
            Description of the strand.
        """
        return (
            f"name = {self.name}\n"
            f"d_strand = {self.d_strand}\n"
            f"materials = {self.materials}\n"
            f"shape = {self.shape}\n"
        )

    def to_dict(self) -> dict:
        """
        Serialize the strand instance to a dictionary.

        Returns
        -------
        dict
            Dictionary with serialized strand data.
        """
        return {
            "name_in_registry": getattr(
                self, "_name_in_registry_", self.__class__.__name__
            ),
            "name": self.name,
            "d_strand": self.d_strand,
            "temperature": self.temperature,
            "materials": [
                {
                    "material": m.material,
                    "fraction": m.fraction,
                }
                for m in self.materials
            ],
        }

    @classmethod
    def from_dict(
        cls,
        strand_dict: dict[str, Any],
        name: str | None = None,
    ) -> "Strand":
        """
        Deserialize a Strand instance from a dictionary.

        Parameters
        ----------
        cls : type
            Class to instantiate (Strand or subclass).
        strand_dict : dict
            Dictionary containing serialized strand data.
        name : str
            Name for the new instance. If None, attempts to use the 'name' field from
            the dictionary.

        Returns
        -------
        Strand
            A new instantiated Strand object.

        Raises
        ------
        TypeError
            If the materials in the dictionary are not valid MaterialFraction instances.
        ValueError
            If the name_in_registry in the dictionary does not match the expected
            class registration name.
        """
        # Validate registration name
        name_in_registry = strand_dict.get("name_in_registry")
        expected_name_in_registry = getattr(cls, "_name_in_registry_", cls.__name__)

        if name_in_registry != expected_name_in_registry:
            raise ValueError(
                f"Cannot create {cls.__name__} from dictionary with name_in_registry "
                f"'{name_in_registry}'. "
                f"Expected '{expected_name_in_registry}'."
            )

        # Deserialize materials
        material_mix = []
        for m in strand_dict["materials"]:
            material_data = m["material"]
            if isinstance(material_data, str):
                raise TypeError(
                    "Material data must be a Material instance, not a string - "
                    "TEMPORARY."
                )
            material_obj = material_data

            material_mix.append(
                MaterialFraction(material=material_obj, fraction=m["fraction"])
            )

        return cls(
            materials=material_mix,
            temperature=strand_dict.get("temperature"),
            d_strand=strand_dict.get("d_strand"),
            name=name or strand_dict.get("name"),
        )


# ------------------------------------------------------------------------------
# SuperconductingStrand Class
# ------------------------------------------------------------------------------


class SuperconductingStrand(Strand):
    """
    Represents a superconducting strand with a circular cross-section.

    Includes methods to compute critical current (Ic) and critical current
    density (Jc) based on the superconducting material.

    Automatically registered using the RegistrableMeta metaclass.
    """

    _name_in_registry_ = "SuperconductingStrand"

    def __init__(
        self,
        materials: list[MaterialFraction],
        d_strand: float = 0.82e-3,
        temperature: float | None = None,
        name: str | None = "SuperconductingStrand",
    ):
        """
        Initialize a superconducting strand.

        Parameters
        ----------
        materials : list of MaterialFraction
            Materials composing the strand with their fractions. One material must be
            a supercoductor.
        d_strand : float, optional
            Strand diameter in meters (default 0.82e-3).
        temperature : float, optional
            Operating temperature [K].
        name : str or None, optional
            Name of the strand. Defaults to "Strand".
        """
        super().__init__(
            materials=materials,
            d_strand=d_strand,
            temperature=temperature,
            name=name,
        )
        self._sc = self._check_materials()

    def _check_materials(self) -> MaterialFraction:
        """
        Ensure there is exactly one superconducting material.

        Returns
        -------
        MaterialFraction
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
        float
            Superconducting area [m²].
        """
        return self.area * self._sc.fraction

    def Jc(self, op_cond: OperationalConditions) -> float:  # noqa:N802
        """
        Critical current density of the superconducting material.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        float
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
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        float
            Critical current [A].
        """
        return self.Jc(op_cond) * self.sc_area

    def plot_Ic_B(  # noqa:N802
        self,
        B: np.ndarray,
        temperature: float,
        ax=None,
        *,
        show: bool = True,
        **kwargs,
    ):
        """
        Plot critical current Ic as a function of magnetic field B.

        Parameters
        ----------
        B : np.ndarray
            Array of magnetic field values [T].
        temperature : float
            Operating temperature [K].
        ax : matplotlib.axes.Axes, optional
            Axis to plot on. If None, a new figure is created.
        show : bool, optional
            Whether to immediately show the plot.
        kwargs : dict
            Additional arguments passed to Ic calculation.

        Returns
        -------
        matplotlib.axes.Axes
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


# ------------------------------------------------------------------------------
# Supporting functions
# ------------------------------------------------------------------------------
def create_strand_from_dict(
    strand_dict: dict[str, Any],
    name: str | None = None,
):
    """
    Factory function to create a Strand or its subclass from a serialized dictionary.

    Parameters
    ----------
    strand_dict : dict
        Dictionary with serialized strand data. Must include a 'name_in_registry' field
        corresponding to a registered class.
    name : str, optional
        If given, overrides the name from the dictionary.

    Returns
    -------
    Strand
        An instance of the appropriate Strand subclass.

    Raises
    ------
    ValueError
        If 'name_in_registry' is missing from the dictionary.
        If no matching registered class is found.
    """
    name_in_registry = strand_dict.get("name_in_registry")
    if name_in_registry is None:
        raise ValueError(
            "Serialized strand dictionary must contain a 'name_in_registry' field."
        )

    cls = STRAND_REGISTRY.get(name_in_registry)
    if cls is None:
        raise ValueError(
            f"No registered strand class with registration name '{name_in_registry}'. "
            "Available classes are: " + ", ".join(STRAND_REGISTRY.keys())
        )

    return cls.from_dict(name=name, strand_dict=strand_dict)
