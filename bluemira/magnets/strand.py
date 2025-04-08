# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Strand class"""

import matplotlib.pyplot as plt
import numpy as np

from bluemira import display
from bluemira.base.look_and_feel import bluemira_error
from bluemira.base.parameter_frame import Parameter
from bluemira.display.plotter import PlotOptions
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_circle
from bluemira.materials.material import Superconductor
from bluemira.materials.mixtures import HomogenisedMixture, MixtureFraction


class Strand(HomogenisedMixture):
    """
    Represents a strand with a circular cross-section, composed of a homogenized
    mixture of materials.
    """

    def __init__(
        self,
        name: str,
        materials: list[MixtureFraction],
        material_id: int | None = None,
        temperature: float | None = None,
        d_strand: float | Parameter | None = 0.82e-3,
    ):
        """
        Initialize a Strand instance with a homogenized material mixture.

        Parameters
        ----------
        name : str
            The name of the strand.
        materials : list of MixtureFraction
            List of materials composing the strand with their fractions.
        material_id : int or None, optional
            Index of the primary material (default is None).
        temperature : float or None, optional
            Operating temperature of the strand [K].
        d_strand : float or Parameter, optional
            Diameter of the strand cross-section in meters (default: 0.82e-3).
        """
        percent_type: str = "vo"
        packing_fraction = 1
        enrichment = None
        super().__init__(
            name=name,
            materials=materials,
            material_id=material_id,
            percent_type=percent_type,
            packing_fraction=packing_fraction,
            enrichment=enrichment,
            temperature=temperature,
        )
        self._d_strand = None
        self.d_strand = d_strand
        self._shape = None

    @property
    def d_strand(self):
        """
        Strand diameter.

        Returns
        -------
        Parameter
            Diameter of the strand [m].
        """
        return self._d_strand

    @d_strand.setter
    def d_strand(self, d: float | Parameter):
        """
        Set the strand diameter, ensuring it is positive and different from
        the current value. Triggers geometry reset if updated.

        Parameters
        ----------
        d : float or Parameter
            New strand diameter.

        Raises
        ------
        ValueError
            If the diameter is non-positive or identical to the current one.
        """
        if type(d) is float:
            d = Parameter("diameter", d, "m")

        if d.value < 0:
            msg = "Strand diameter must be positive."
            bluemira_error(msg)
            raise ValueError(msg)
        if self.d_strand is None or d.value != self.d_strand.value:
            self._d_strand = d
            self._shape = None
        else:
            msg = "The new value for the strand diameter is equal to the old one."

    @property
    def area(self) -> float:
        """
        Compute the cross-sectional area of the strand.

        Returns
        -------
        float
            Area of the strand [m²].
        """
        return np.pi * self.d_strand.value**2 / 4

    @property
    def shape(self):
        """
        Returns the 2D geometric representation of the strand.

        Returns
        -------
        BluemiraFace
            Circular face representing the strand geometry.
        """
        if self._shape is None:
            diameter = self.d_strand.value
            self._shape = BluemiraFace([make_circle(diameter)])
        return self._shape

    def plot(self, ax=None, *, show: bool = True, **kwargs):
        """
        Plot a 2D view of the strand cross-section.

        Parameters
        ----------
        ax : matplotlib.axes.Axes or None, optional
            Axis object to plot on. If None, a new figure is created.
        show : bool, optional
            Whether to display the plot immediately.
        kwargs : dict
            Additional keyword arguments passed to the plotting function.

        Returns
        -------
        matplotlib.axes.Axes
            Axis with the plot rendered.
        """
        plot_options = PlotOptions()
        plot_options.view = "xy"
        return display.plot_2d(
            self.shape, options=plot_options, ax=ax, show=show, **kwargs
        )

    def __str__(self):
        """
        Generate a formatted string summarizing the strand.

        Returns
        -------
        str
            Human-readable description of the strand including name, diameter,
            material, and shape.
        """
        return (
            f"name = {self.name}\n"
            f"d_strand = {self.d_strand}\n"
            f"material = {self.material}\n"
            f"shape = {self.shape}\n"
        )


class SuperconductingStrand(Strand, Superconductor):
    """
    Represents a superconducting strand with a circular cross-section.
    """

    def __init__(
        self,
        name: str,
        materials: list[MixtureFraction],
        material_id: int | None = None,
        temperature: float | None = None,
        d_strand: float | Parameter | None = 0.82e-3,
    ):
        """
        Initialize a superconducting strand.

        Parameters
        ----------
        name : str
            The name of the strand.
        materials : list of MixtureFraction
            List of materials composing the strand, including one superconductor.
        material_id : int or None, optional
            Index of the primary material.
        temperature : float or None, optional
            Operating temperature of the strand [K].
        d_strand : float or Parameter, optional
            Diameter of the strand cross-section [m].
        """
        super().__init__(
            name=name,
            materials=materials,
            material_id=material_id,
            temperature=temperature,
        )
        self._sc = self._check_materials()
        self._d_strand = None
        self.d_strand = d_strand
        self._shape = None

    def _check_materials(self):
        """
        Validates the presence of exactly one superconducting material.

        Returns
        -------
        MixtureFraction
            The identified superconducting material.

        Raises
        ------
        ValueError
            If no or multiple superconducting materials are found.
        """
        sc = None
        for material in self.materials:
            m = material.material
            if isinstance(m, Superconductor):
                if sc is None:
                    sc = material
                else:
                    msg = (
                        f"Only one superconductor material can be defined per "
                        f"strand. At least two have been found: {sc} and {material}."
                    )
                    bluemira_error(msg)
                    raise ValueError(msg)
        if sc is None:
            msg = "No superconductor material found."
            bluemira_error(msg)
            raise ValueError(msg)
        return sc

    @property
    def sc_area(self):
        """
        Compute the superconducting portion of the strand's area.

        Returns
        -------
        float
            Area of the superconducting material [m²].
        """
        return self.area * self._sc.fraction

    def Jc(self, **kwargs) -> float:  # noqa:N802
        """
        Return the critical current density of the superconducting material.

        Parameters
        ----------
        kwargs : dict
            Additional inputs to the Jc model (e.g., B, temperature).

        Returns
        -------
        float
            Critical current density [A/m²].
        """
        return self._sc.material.Jc(**kwargs)

    def Ic(self, **kwargs) -> float:  # noqa:N802
        """
        Compute the total critical current based on Jc and sc_area.

        Parameters
        ----------
        kwargs : dict
            Additional arguments forwarded to the Jc computation.

        Returns
        -------
        float
            Critical current [A].
        """
        return self.Jc(**kwargs) * self.sc_area

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
        Plot the critical current as a function of magnetic field.

        Parameters
        ----------
        B : np.ndarray
            Magnetic field values [T].
        temperature : float
            Operating temperature [K].
        ax : matplotlib.axes.Axes or None, optional
            Axis to plot on. A new one is created if None.
        show : bool, optional
            Whether to show the plot immediately.
        kwargs : dict
            Additional arguments passed to the `Ic()` method.

        Returns
        -------
        matplotlib.axes.Axes
            The axis with the Ic-B curve plotted.
        """
        if ax is None:
            _, ax = plt.subplots()

        Ic_sc = [self.Ic(B=Bi, temperature=temperature, **kwargs) for Bi in B]  # noqa:N806
        ax.plot(B, Ic_sc)
        # Adding the plot title and axis labels
        plt.title(
            f"Critical current for {self.__class__.__name__}\n"
            f"Temperature = {temperature} [K]"
        )  # Title
        plt.xlabel("B [T]")  # X-axis label
        plt.ylabel("Ic [A]")  # Y-axis label
        # Enabling the grid
        plt.grid(visible=True)
        if show:
            plt.show()
        return ax
