# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Neutronics sources"""

import numpy as np
import openmc
from openmc_plasma_source import tokamak_source

from bluemira.base.constants import raw_uc
from bluemira.codes.openmc.params import PlasmaSourceParameters
from bluemira.radiation_transport.neutronics.constants import dt_neutron_energy


def make_tokamak_source(
    source_parameters: PlasmaSourceParameters,
) -> list[openmc.Source]:
    """Make a tokamak neutron source using a PlasmaSourceParameters.
    Some parameters are hard coded, while the rest are rest in from the params.json
    and stored in the PlasmaSourceParameters

    Parameters
    ----------
    source_parameters:
        PlasmaSourceParameters

    Returns
    -------

    """
    return tokamak_source(
        # tokamak geometry: Check units
        major_radius=raw_uc(source_parameters.major_radius, "m", "cm"),
        minor_radius=raw_uc(source_parameters.minor_radius, "m", "cm"),
        elongation=source_parameters.elongation,
        triangularity=source_parameters.triangularity,
        mode="H",
        # plasma geometry: ion stuff
        ion_density_centre=1.09e20,
        ion_density_pedestal=1.09e20,
        ion_density_peaking_factor=source_parameters.peaking_factor,  # Check this one!
        ion_density_separatrix=3e19,
        ion_temperature_centre=source_parameters.temperature,  # Change the name in source param
        ion_temperature_pedestal=6.09e3,
        ion_temperature_separatrix=0.1e3,
        ion_temperature_peaking_factor=8.06,
        ion_temperature_beta=6,
        # shaping
        shafranov_factor=source_parameters.shaf_shift,  # Check if it's relative v.s. absolute
        pedestal_radius=0.8 * raw_uc(source_parameters.minor_radius, "m", "cm"),
        # plasma composition
        fuel={"D": 0.5, "T": 0.5},
    )


def make_ring_source(source_parameters: PlasmaSourceParameters) -> openmc.Source:
    """Create the ring source"""  # noqa: DOC201
    return create_ring_source(
        source_parameters.major_radius, source_parameters.shaf_shift
    )


def create_ring_source(major_r_cm: float, shaf_shift_cm: float) -> openmc.Source:
    """
    Creating simple line ring source lying on the Z=0 plane,
    at r = major radius + shafranov shift,
    producing 14.1 MeV neutrons with no variation in energy.
    A more accurate source will slightly affect the wall loadings and dpa profiles.

    Parameters
    ----------
    major_r_cm:
        major radius [cm]
    shaf_shift_cm:
        shafranov shift [cm]
    """  # noqa: DOC201
    ring_source = openmc.IndependentSource()
    source_radii_cm = openmc.stats.Discrete([major_r_cm + shaf_shift_cm], [1])
    source_z_values = openmc.stats.Discrete([0], [1])
    source_angles = openmc.stats.Uniform(a=0.0, b=2 * np.pi)
    ring_source.space = openmc.stats.CylindricalIndependent(
        r=source_radii_cm, phi=source_angles, z=source_z_values, origin=(0.0, 0.0, 0.0)
    )
    ring_source.angle = openmc.stats.Isotropic()
    ring_source.energy = openmc.stats.Discrete(
        [raw_uc(dt_neutron_energy, "J", "eV")], [1]
    )

    return ring_source
