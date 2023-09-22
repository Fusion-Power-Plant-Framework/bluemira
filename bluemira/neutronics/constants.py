# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
"""constants used for the neutronics module"""
from periodictable import elements

from bluemira.base.constants import N_AVOGADRO, raw_uc

# Amount of energy released in a single dt fusion reaction, in MeV.
energy_per_dt = raw_uc(17.58, "MeV", "J")
# Amount of energy carried away by the neutron
dt_neutron_energy = energy_per_dt * (4 / 5)  # [J]

# Energy required to displace an Fe atom in Fe. See docstring of DPACoefficients
dpa_Fe_threshold_eV = 40  # Source cites 40 eV.


# The following material science constants are in cgs.
Fe_molar_mass_g = elements.isotope("Fe").mass
Fe_density_g_cc = elements.isotope("Fe").density


class DPACoefficients:
    """
    Get the coefficients required
        to convert the number of damage into the number of displacements.
    number of atoms in region = avogadro * density * volume / molecular mass
    number of atoms in 1 cc   = avogadro * density          / molecular mass
    dpa_per_second_of_operation = src_rate * displacements / atoms
    dpa_fpy = dpa_per_second_of_operation / S_TO_YEAR

    taken from [1]_.
    .. [1] Shengli Chena, David Bernard
       On the calculation of atomic displacements using damage energy
       Results in Physics 16 (2020) 102835
       https://doi.org/10.1016/j.rinp.2019.102835
    """

    def __init__(
        self,
        density_g_cc: float = Fe_density_g_cc,
        molar_mass_g: float = Fe_molar_mass_g,
        dpa_threshold_eV: float = dpa_Fe_threshold_eV,
    ):
        """
        Parameters
        ----------
        density_g_cc: float [g/cm^2]
            density of the wall material,
            where the damage (in DPA) would be calculated later.
        molar_mass_g: float [g/mole]
            molar mass of the wall material,
            where the damage (in DPA) would be calculated later.
        dpa_threshold_eV: float [eV/count]
            the average amount of energy dispersed
            by displacing one atom in the wall material's lattice.

        Attributes/values
        -----------------
        atoms_per_cc: number density, given in cgs.
        displacements_per_damage_eV:
        """
        self.atoms_per_cc = N_AVOGADRO * density_g_cc / molar_mass_g
        self.displacements_per_damage_eV = 0.8 / (2 * dpa_threshold_eV)
