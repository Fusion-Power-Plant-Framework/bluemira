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
"""Example of how to use the neutronics module"""

# %% [markdown]
# # Example of how to use the neutronics module

# %%
from pathlib import Path

import numpy as np

from bluemira.base.constants import raw_uc
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.tools import make_polygon
from bluemira.neutronics.make_materials import BlanketType
from bluemira.neutronics.params import (
    OpenMCSimulationRuntimeParameters,
    TokamakOperationParameters,
    get_preset_physical_properties,
)
from bluemira.neutronics.quick_tbr_heating import TBRHeatingSimulation

# %%
CROSS_SECTION_XML = str(
    Path("~/Documents/BLUEPRINT/bluemira_openmc_data/cross_sections.xml").expanduser()
)

# %%
# set up the variables to be used for the openmc simulation
# allowed blanket_type so far = {'WCLL', 'DCLL', 'HCPB'}
breeder_materials, tokamak_geometry = get_preset_physical_properties(BlanketType.HCPB)

runtime_variables = OpenMCSimulationRuntimeParameters(
    cross_section_xml=CROSS_SECTION_XML,
    particles=100000,  # 16800 takes 5 seconds,  1000000 takes 280 seconds.
    batches=2,
    photon_transport=True,
    electron_treatment="ttb",
    run_mode="fixed source",
    openmc_write_summary=False,
    parametric_source=True,
    # only used if stochastic_volume_calculation is turned on.
    volume_calc_particles=int(4e8),
)

operation_variable = TokamakOperationParameters(
    reactor_power=1998e6,  # [W]
    temperature=round(raw_uc(15.4, "keV", "K"), 5),
    peaking_factor=1.508,  # [dimensionless]
    shaf_shift=0.0,  # [m]
    vertical_shift=0.0,  # [m]
)

# set up a DEMO-like reactor, and run OpenMC simualtion
tbr_heat_sim = TBRHeatingSimulation(
    runtime_variables, operation_variable, breeder_materials, tokamak_geometry
)
blanket_wire = make_polygon(Coordinates(np.load("blanket_face.npy")))
divertor_wire = make_polygon(Coordinates(np.load("divertor_face.npy")))
tbr_heat_sim.setup(
    blanket_wire,
    divertor_wire,
    new_major_radius=9.00,  # [m]
    new_aspect_ratio=3.10344,  # [dimensionless]
    new_elong=1.792,  # [dimensionless]
    plot_geometry=True,
)

# %%
tbr_heat_sim.run()
# get the TBR, component heating, first wall dpa, and photon heat flux
results = tbr_heat_sim.get_result()

print(results)
# tbr_heat_sim.calculate_volume_stochastically()
# # don't do this because it takes a long time.
