# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Example of how to use the neutronics module"""

# %% [markdown]
# # Example of how to use the neutronics module

# %%
from pathlib import Path
from typing import Tuple

import numpy as np

from bluemira.base.constants import raw_uc
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.tools import make_polygon
from bluemira.neutronics.make_materials import BlanketType
from bluemira.neutronics.neutronics_axisymmetric import TBRHeatingSimulation
from bluemira.neutronics.params import (
    OpenMCSimulationRuntimeParameters,
    TokamakOperationParametersBase,
    get_preset_physical_properties,
)

# %% [markdown]
# Since nuclear data (data about the probability of nuclear reactions) is required in
# order to run the neutronics simulations, we must download the
# cross-section data (see nuclear_data_downloader) to a local directory,
# and let OpenMC know where that directory is.

# %%
CROSS_SECTION_XML = str(
    Path("~/code/bluemira/bluemira_openmc_data/cross_sections.xml").expanduser()
)
# %% [markdown]
# In this script we can also demonstrate openmc's ability to calculate the volume
# stochastically (rather than analytically). Specifically, in this script we
# toggle this parameter on/off using the following variable:

# %%
volume_calculation = False

# %% [markdown]
# For each of the three Tokamak designs:
# - WCLL (Water Cooled Lithium Lead breeding blanket)
# - DCLL (Dual Coolant Lithium Lead breeding blanket)
# - HCPB (Helium Cooled Pebble Bed breeding blanket),
#
# bluemira provides a list of preset parameters in `get_preset_physical_properties`
# so that you can a simple generic tokamak of that design.
#
# Using these presets parameters, we can set up a simulation for OpenMC to run.

# %%
# set up the variables to be used for the openmc simulation
# allowed blanket_type so far = {'WCLL', 'DCLL', 'HCPB'}
breeder_materials, plasma_geometry, tokamak_geometry = get_preset_physical_properties(
    BlanketType.HCPB
)

runtime_variables = OpenMCSimulationRuntimeParameters(
    cross_section_xml=CROSS_SECTION_XML,
    particles=100000,  # 16800 takes 5 seconds,  1000000 takes 280 seconds.
    batches=2,
    photon_transport=True,
    electron_treatment="ttb",
    run_mode="plot",
    openmc_write_summary=False,
    parametric_source=True,
    # only used if stochastic_volume_calculation is turned on.
    volume_calc_particles=int(4e8),
)

operation_variable = TokamakOperationParametersBase(
    reactor_power=1998e6,  # [W]
    temperature=raw_uc(15.4, "keV", "K"),
    peaking_factor=1.508,  # [dimensionless]
    shaf_shift=0.0,  # [m]
    vertical_shift=0.0,  # [m]
)

# set up a DEMO-like reactor, and run OpenMC simualtion
tbr_heat_sim = TBRHeatingSimulation(
    runtime_variables,
    operation_variable,
    breeder_materials,
    plasma_geometry,
    tokamak_geometry,
)
blanket_wire = make_polygon(Coordinates(np.load("blanket_face.npy")))
divertor_wire = make_polygon(Coordinates(np.load("divertor_face.npy")))

fw_coordinates = Coordinates(
    {
        "x": [
            9.94,
            10.89,
            11.90,
            12.19,
            12.03,
            11.34,
            9.69,
            7.53,
            6.38,
            5.95,
            5.93,
            6.21,
            6.84,
            7.26,
            7.90,
            8.79,
            9.94,
        ],
        "z": [
            -5.27,
            -3.57,
            -1.62,
            0,
            1.57,
            3.05,
            4.57,
            5.26,
            3.95,
            1.32,
            -1.91,
            -3.51,
            -5.01,
            -5.86,
            -5.78,
            -6.46,
            -5.27,
        ],
    }
)

from dataclasses import dataclass


@dataclass(frozen=True)
class TRegion:
    index: int
    direction: np.ndarray


BBDIV = TRegion(7, np.array([0, 0, 1]))
IBDIV = TRegion(12, np.array([-1, 0, 0]))
OBDIV = TRegion(16, np.array([1, 0, 0]))


import matplotlib.pyplot as plt

f, ax = plt.subplots()
ax.plot(*fw_coordinates.xz)
ax.set_aspect("equal")
for r in [BBDIV, IBDIV, OBDIV]:
    ax.quiver(
        fw_coordinates.x[r.index],
        fw_coordinates.z[r.index],
        r.direction[0],
        r.direction[2],
    )
plt.show()


@dataclass(frozen=True)
class FWDeconstruction:
    coordinates: Coordinates
    tregions: Tuple[TRegion]


fw_deconstruction = FWDeconstruction(fw_coordinates, (IBDIV, BBDIV, OBDIV))

tbr_heat_sim.setup(
    fw_deconstruction,
    divertor_wire,
    new_major_radius=9.00,  # [m]
    new_aspect_ratio=3.10344,  # [dimensionless]
    new_elong=1.792,  # [dimensionless]
    plot_geometry=True,
)

# %%
# get the TBR, component heating, first wall dpa, and photon heat flux
tbr_heat_sim.run()
# %%
# Takes ~ 45s on a modern laptop
if volume_calculation:
    tbr_heat_sim.calculate_volume_stochastically()

# %%
results = tbr_heat_sim.get_result()

print(results)
