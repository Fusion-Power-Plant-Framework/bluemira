"""constants used for the neutronics module"""
# Manually set constants
energy_per_dt_MeV = 17.58  # probably good to put this in bluemira anyways
dpa_Fe_threshold_eV = 40  # Energy required to displace an Fe atom in Fe. See docstring of get_dpa_coefs. Source cites 40 eV.

avogadro = BMUnitRegistry.Quantity("N_A").to_base_units().magnitude
Fe_molar_mass_g = elements.isotope("Fe").mass
Fe_density_g_cc = elements.isotope("Fe").density


class DPACoefficients:
    """
    Get the coefficients required
        to convert the number of damage into the number of displacements.
    number of atoms in region = avogadro * density * volume / molecular mass
    number of atoms in 1 cc   = avogadro * density          / molecular mass
    dpa_fpy = displacements / atoms * s_in_yr * src_rate

    taken from [1]_.
    .. [1] Shengli Chena, David Bernard
       On the calculation of atomic displacements using damage energy
       Results in Physics 16 (2020) 102835
       https://doi.org/10.1016/j.rinp.2019.102835
    """

    def __init__(
        self,
        density_g_cc=Fe_density_g_cc,
        molar_mass_g=Fe_molar_mass_g,
        dpa_threshold_eV=dpa_Fe_threshold_eV,
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
        """
        self.atoms_per_cc = avogadro * density_g_cc / molar_mass_g
        self.displacements_per_damage_eV = 0.8 / (2 * dpa_threshold_eV)
