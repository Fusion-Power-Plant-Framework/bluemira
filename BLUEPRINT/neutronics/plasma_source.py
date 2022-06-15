# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
Implementation of a plasma neutron source based on parametric-plasma-source

For more information / citation see
[here](https://doi.org/10.1016/j.fusengdes.2012.02.025)
"""

import math

import numpy as np
from scipy.interpolate import griddata


class PlasmaSource:
    """
    A class to represent a plasma neutron source
    """

    def __init__(
        self,
        ion_density_pedestal,
        ion_density_separatrix,
        ion_density_origin,
        ion_temperature_pedestal,
        ion_temperature_separatrix,
        ion_temperature_origin,
        pedestal_radius,
        ion_density_peaking_factor,
        ion_temperature_peaking_factor,
        ion_temperature_beta,
        minor_radius,
        major_radius,
        elongation,
        triangularity,
        shafranov_shift,
        plasma_mode,
        number_of_radial_bins=100,
        number_of_angular_bins=100,
    ):
        self.ion_density_pedestal = ion_density_pedestal
        self.ion_density_separatrix = ion_density_separatrix
        self.ion_density_origin = ion_density_origin
        self.ion_temperature_pedestal = ion_temperature_pedestal
        self.ion_temperature_separatrix = ion_temperature_separatrix
        self.ion_temp_origin = ion_temperature_origin
        self.pedestal_radius = pedestal_radius / 100.0
        self.ion_density_peaking_factor = ion_density_peaking_factor
        self.ion_temperature_peaking_factor = ion_temperature_peaking_factor
        self.ion_temperature_beta = ion_temperature_beta
        self.minor_radius = minor_radius / 100.0
        self.major_radius = major_radius / 100.0
        self.elongation = elongation
        self.triangularity = triangularity
        self.shafranov_shift = shafranov_shift / 100.0
        self.plasma_mode = plasma_mode

        self.number_of_radial_bins = number_of_radial_bins
        self.number_of_angular_bins = number_of_angular_bins

        self.points = []
        self.strength_values = []
        self.average_energy_values = []

        self.setup_plasma_source_xz()

    @property
    def plasma_mode(self):
        """
        The mode of the plasma (H, A, or L)
        """
        return self._plasma_mode

    @plasma_mode.setter
    def plasma_mode(self, value):
        allowed_values = ["H", "A", "L"]
        if value in allowed_values:
            self._plasma_mode = value
        else:
            raise ValueError(
                f"PlasmaSource.plasma_mode must be one of {allowed_values}, "
                f"provided value was '{value}'."
            )

    def setup_plasma_source_xz(self):
        """
        Setup the plasma source in x-z

        Uses the radial and angular bin widths to generate a 2-d grid in x-z, which
        can then be interpolated to get a derived value for any point in x-z.

        Initalises arrays for source strength and average energy calculations.
        """
        bin_width = self.minor_radius / self.number_of_radial_bins
        angular_width = (2.0 * np.pi) / self.number_of_angular_bins

        self.points = []
        self.values = []

        for idx in range(self.number_of_radial_bins + 1):
            radius = bin_width * idx
            ion_density = self.ion_density(radius)
            ion_temperature = self.ion_temperature(radius)
            strength_at_radius = ion_density**2 * self.dt_cross_section(
                ion_temperature
            )
            ion_kt_at_radius = math.sqrt(ion_temperature * 0.001)
            neutron_average_energy_at_radius = 14.08 + (
                (5.59 / 2.35) * (ion_kt_at_radius)
            )
            for jdx in range(self.number_of_angular_bins):
                alpha = angular_width * jdx
                x, z = self.convert_r_alpha_to_xz(radius, alpha)
                self.points += [[x, z]]
                self.strength_values += [strength_at_radius]
                self.average_energy_values += [neutron_average_energy_at_radius]

    def convert_r_alpha_to_xz(self, r, alpha):
        """
        Convert the provided (r,alpha) coordinate to (x,z)

        Parameters
        ----------
        r : float
            The radius from the magnetic center of the plasma [m].
        alpha : float
            The angle above the center of the plasma around the magnetic axis [rad].

        Returns
        -------
        x : float
            The corresponding x coordinate [m].
        z : float
            The corresponding z coordinate [m].
        """
        shift = self.shafranov_shift * (1.0 - (r / self.minor_radius) ** 2)
        shifted_radius = self.major_radius + shift

        x = shifted_radius + r * math.cos(alpha + (self.triangularity * math.sin(alpha)))
        z = self.elongation * r * math.sin(alpha)

        return x, z

    def get_source_strength_xz(self, x, z, interpolation_method="linear"):
        """
        Get the source strength by interpolating the x-z grid

        Requires the source to be initialised to obtain the discretised x-z grid.

        Parameters
        ----------
        x : float or np.array(N, M, float64)
            The x coordinate or N x M grid of x coordinates [m].
        z : float or np.array(N, M, float64)
            The z coordinate or N x M grid of z coordinates  [m].
        interpolation_method : str, optional
            The interpolation method to use in the `scipy.interpolation.griddata`
            calculation.
            One of nearest, linear, or  cubic, by default "linear".

        Returns
        -------
        source_strength : float or np.array(N, M, float64)
            The interpolated source strength [n / m^3].
        """
        source_strength = griddata(
            self.points, self.strength_values, (x, z), method=interpolation_method
        )
        return (
            float(source_strength) if np.ndim(source_strength) == 0 else source_strength
        )

    def get_average_source_energy_xz(self, x, z, interpolation_method="linear"):
        """
        Get the average source energy by interpolating the x-z grid

        Requires the source to be initialised to obtain the discretised x-z grid.

        Parameters
        ----------
        x : float or np.array(N, M, float64)
            The x coordinate or N x M grid of x coordinates [m].
        z : float or np.array(N, M, float64)
            The z coordinate or N x M grid of x coordinates [m].
        interpolation_method : str, optional
            The interpolation method to use in the `scipy.interpolation.griddata`
            calculation.
            One of nearest, linear, or  cubic, by default "linear".

        Returns
        -------
        average_source_energy : float or np.array(N, M, float64)
            The interpolated average source energy [MeV].
        """
        average_source_energy = griddata(
            self.points, self.average_energy_values, (x, z), method=interpolation_method
        )
        return (
            float(average_source_energy)
            if np.ndim(average_source_energy) == 0
            else average_source_energy
        )

    def ion_density(self, radius):
        """
        The ion density at the provided radius.

        Parameters
        ----------
        radius : float
            The radius from the magnetic axis [m].

        Returns
        -------
        ion_density : float
            The ion density [i / m^3].
        """
        ion_density = 0.0
        if self.plasma_mode == "L":
            ion_density = self.ion_density_origin * (
                1.0 - (radius / self.minor_radius) ** 2
            )
        else:
            if radius <= self.pedestal_radius:
                ion_density += self.ion_density_pedestal
                product = 1.0 - (radius / self.pedestal_radius) ** 2
                product = math.pow(product, self.ion_density_peaking_factor)
                ion_density += (
                    self.ion_density_origin - self.ion_density_pedestal
                ) * product
            else:
                ion_density += self.ion_density_separatrix
                product = self.ion_density_pedestal - self.ion_density_separatrix
                ion_density += (
                    product
                    * (self.minor_radius - radius)
                    / (self.minor_radius - self.pedestal_radius)
                )
        return ion_density

    def ion_temperature(self, radius):
        """
        The ion temperature at the provided radius.

        Parameters
        ----------
        radius : float
            The radius from the magnetic axis [m].

        Returns
        -------
        ion_temperature : float
            The ion temperature [keV].
        """
        ion_temperature = 0.0
        if self.plasma_mode == "L":
            ion_temperature = self.ion_temp_origin * (
                1.0
                - math.pow(
                    radius / self.minor_radius, self.ion_temperature_peaking_factor
                )
            )
        else:
            if radius <= self.pedestal_radius:
                ion_temperature += self.ion_temperature_pedestal
                product = 1.0 - math.pow(
                    radius / self.pedestal_radius, self.ion_temperature_beta
                )
                product = math.pow(product, self.ion_temperature_peaking_factor)
                ion_temperature += (
                    self.ion_temp_origin - self.ion_temperature_pedestal
                ) * product
            else:
                ion_temperature += self.ion_temperature_separatrix
                product = self.ion_temperature_pedestal - self.ion_temperature_separatrix
                ion_temperature += (
                    product
                    * (self.minor_radius - radius)
                    / (self.minor_radius - self.pedestal_radius)
                )
        return ion_temperature

    def dt_cross_section(self, ion_temperature):
        """
        The deuterium-tritium cross section at the provided ion temperature.

        Parameters
        ----------
        ion_temperature : float
            The ion temperature [keV].

        Returns
        -------
        cross_section : float
            The deuterium-tritium cross section.
        """
        c = [
            2.5663271e-18,
            19.983026,
            2.5077133e-2,
            2.5773408e-3,
            6.1880463e-5,
            6.6024089e-2,
            8.1215505e-3,
        ]

        u = 1.0 - ion_temperature * (
            c[2] + ion_temperature * (c[3] - c[4] * ion_temperature)
        ) / (1.0 + ion_temperature * (c[5] + c[6] * ion_temperature))

        dt = c[0] / (math.pow(u, 5.0 / 6.0) * math.pow(ion_temperature, 2.0 / 3.0))
        dt *= math.exp(-1.0 * c[1] * math.pow(u / ion_temperature, 1.0 / 3.0))

        return dt

    def radial_source_strength(self, radius):
        """
        The source strength at the provided radius.

        Parameters
        ----------
        radius : float
            The radium from the magnetic axis [m].

        Returns
        -------
        source_strength : float
            The source strength [n / m^3]
        """
        ion_density = self.ion_density(radius)
        ion_temperature = self.ion_temperature(radius)
        source_strength = ion_density**2 * self.dt_cross_section(ion_temperature)
        return source_strength
