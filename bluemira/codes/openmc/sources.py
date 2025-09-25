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
    source: openmc.Source
        D-T fusion source for OpenMC

    Notes
    -----
    The same source material referenced by openmc_plasma_source is used:
    .. doi:: 10.1016/j.fusengdes.2012.02.025
      :title: Fausser et al, 'Tokamak D-T neutron source models for different
              plasma physics confinement modes', Fus. Eng. and Design,
    """
    return tokamak_source(
        # tokamak geometry
        major_radius=raw_uc(source_parameters.major_radius, "m", "cm"),
        minor_radius=raw_uc(source_parameters.minor_radius, "m", "cm"),
        elongation=source_parameters.elongation,
        triangularity=source_parameters.triangularity,
        mode="H",
        # plasma geometry: ion stuff
        ion_density_centre=source_parameters.ion_density_core,
        ion_density_pedestal=source_parameters.ion_density_ped,
        ion_density_peaking_factor=source_parameters.ion_density_alpha,
        ion_density_separatrix=source_parameters.ion_density_sep,
        ion_temperature_centre=source_parameters.ion_temperature_core,
        ion_temperature_pedestal=source_parameters.ion_temperature_ped,
        ion_temperature_separatrix=source_parameters.ion_temperature_sep,
        ion_temperature_peaking_factor=source_parameters.ion_density_alpha,
        ion_temperature_beta=source_parameters.ion_temperature_beta,
        # shaping
        shafranov_factor=source_parameters.shaf_shift,
        pedestal_radius=source_parameters.pedestal_radius,
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
