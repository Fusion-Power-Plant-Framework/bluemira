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
Example core and scraper-off layer radiation source calculation
"""

# %%
import os

import matplotlib.pyplot as plt

import bluemira.codes.process as process
from bluemira.base.config import Configuration
from bluemira.base.file import get_bluemira_path
from bluemira.base.parameter import ParameterFrame
from bluemira.equilibria import Equilibrium
from bluemira.geometry._deprecated_loop import Loop
from bluemira.radiation_transport.advective_transport import ChargedParticleSolver
from bluemira.radiation_transport.radiation_profile import (
    CoreRadiation,
    DNScrapeOffLayerRadiation,
)

# %%[markdown]
# Read equilibrium

# %%
read_path = get_bluemira_path("equilibria", subfolder="data")
eq_name = "DN-DEMO_eqref.json"
eq_name = os.sep.join([read_path, eq_name])
eq = Equilibrium.from_eqdsk(eq_name, load_large_file=True)

# %%[markdown]
# Get first wall shape

# %%
read_path = get_bluemira_path("radiation_transport/test_data", subfolder="tests")
fw_name = "DN_fw_shape.json"
fw_name = os.sep.join([read_path, fw_name])
fw_shape = Loop.from_file(fw_name)

# %%[markdown]
# Run particle solver

# %%
p_solver_params = ParameterFrame()
solver = ChargedParticleSolver(p_solver_params, eq, dx_mp=0.001)
solver.analyse(first_wall=fw_shape)

# %%[markdown]
# Run PROCESS solver

# %%
PROCESS_PATH = ""
binary = f"{PROCESS_PATH}process"

new_params = {
    "kappa": 1.6969830041844367,
}
params = Configuration(new_params)

build_config = {
    "mode": "run",
    "binary": binary,
}

process_solver = process.Solver(
    params=params,
    build_config=build_config,
)
process_solver.run()

# %%[markdown]
# Get impurity fractions

# %%
impurity_content = {
    "H": process_solver.get_species_fraction("H"),
    "He": process_solver.get_species_fraction("He"),
    "Xe": process_solver.get_species_fraction("Xe"),
    "W": process_solver.get_species_fraction("W"),
}

# %%[markdown]
# Get impurity data: temperature reference
# and radiative loss function reference

# %%
impurity_data = {
    "H": {
        "T_ref": process_solver.get_species_data("H")[0],
        "L_ref": process_solver.get_species_data("H")[1],
    },
    "He": {
        "T_ref": process_solver.get_species_data("He")[0],
        "L_ref": process_solver.get_species_data("He")[1],
    },
    "Xe": {
        "T_ref": process_solver.get_species_data("Xe")[0],
        "L_ref": process_solver.get_species_data("Xe")[1],
    },
    "W": {
        "T_ref": process_solver.get_species_data("W")[0],
        "L_ref": process_solver.get_species_data("W")[1],
    },
}

# %%[markdown]
# Eventually customise plasma parameters to deviate from default values

# %%
plasma_params = ParameterFrame(
    # fmt: off
    [
        ["kappa", "Elongation", 3, "dimensionless", None, "Input"],
    ]
    # fmt: on
)

# %%[markdown]
# Run core radiation source calculation for Spherical Tokamak

# %%
stcore = CoreRadiation(solver, impurity_content, impurity_data, plasma_params)

# %%[markdown]
# Build radiation profile at the midplane and 2D core radiation map

# %%
stcore.build_mp_rad_profile()
stcore.build_core_radiation_map()

# %%[markdown]
# Run scrape-off layer radiation source calculation for Spherical Tokamak

# %%
stsol = DNScrapeOffLayerRadiation(
    solver, impurity_content, impurity_data, plasma_params, fw_shape
)

# %%[markdown]
# Build temperature and density poloidal profiles for the scrape-off layer.
# The output contains for lists for the four sectors: LFS lower divertor,
# LFS upper divertor, HFS lower divertor, HFS upper divertor

# %%
t_and_n_sol_profiles = stsol.build_sol_profiles(fw_shape)

# %%[markdown]
# Build radiation source profiles for the scrape-off layer.
# The output contains for lists for the four sectors.

# %%
rad_sector_profiles = stsol.build_sol_rad_distribution(*t_and_n_sol_profiles)

# %%[markdown]
# Build 2D scrape-off layer radiation map

# %%
stsol.build_sol_radiation_map(*rad_sector_profiles, fw_shape)

# %%[markdown]
# Plot results

# %%
plt.show()
