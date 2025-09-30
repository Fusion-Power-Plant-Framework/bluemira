# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Neutronics sources"""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
import openmc
from tokamak_neutron_source import (
    TokamakNeutronSource,
    FluxMap,
    TransportInformation,
    FractionalFuelComposition,
    Reactions,
)
from tokamak_neutron_source.profile import ParabolicPedestalProfile

from bluemira.base.constants import raw_uc
from bluemira.codes.openmc.params import PlasmaSourceParameters
from bluemira.radiation_transport.neutronics.constants import dt_neutron_energy

if TYPE_CHECKING:
    from eqdsk import EQDSKInterface


def make_tokamak_source(
    eq: EQDSKInterface,
    source_parameters: PlasmaSourceParameters,
    cell_side_length: float = 0.1,
) -> list[openmc.Source]:
    """
    Make a tokamak neutron source using an equilibrium and PlasmaSourceParameters
    for PROCESS parabolic-pedestal profiles.

    Parameters
    ----------
    eq:
        Equilibrium description
    source_parameters:
        PlasmaSourceParameters
    cell_side_length:
        The dimension of the squares with which to discretise the neutron source

    Returns
    -------
    source:
        Fusion source for OpenMC
    """
    rho_profile = np.linspace(0, 1, 50)
    transport = TransportInformation.from_parameterisations(
        ion_temperature_profile=ParabolicPedestalProfile(
            source_parameters.electron_temperature_core,
            source_parameters.electron_temperature_ped,
            source_parameters.electron_temperature_sep,
            source_parameters.electron_temperature_alpha,
            source_parameters.electron_temperature_beta,
            source_parameters.rho_pedestal,
        ),
        fuel_density_profile=ParabolicPedestalProfile(
            source_parameters.electron_density_core,
            source_parameters.electron_density_ped,
            source_parameters.electron_density_sep,
            source_parameters.electron_density_alpha,
            2.0,  # Hard-coded as 2.0 in PROCESS
            source_parameters.rho_pedestal,
        ),
        rho_profile=rho_profile,
        fuel_composition=FractionalFuelComposition(D=0.5, T=0.5),
    )
    flux_map = FluxMap.from_eqdsk(eq)

    source = TokamakNeutronSource(
        transport,
        flux_map,
        source_type=[Reactions.D_T, Reactions.D_D],
        cell_side_length=cell_side_length,
    )
    return source.to_openmc_source()


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
