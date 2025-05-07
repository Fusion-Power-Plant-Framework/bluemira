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

"""
Example single null first wall particle heat flux
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria import Equilibrium
from bluemira.geometry.coordinates import Coordinates
from bluemira.radiation_transport.advective_transport import ChargedParticleSolver

# %% [markdown]
# # Single null first wall particle heat flux
#
# First we load an up equilibrium.
# If you would like to view the double null version change the variable below

# %%
DOUBLE_NULL = True
read_path = get_bluemira_path("equilibria", subfolder="data")
eq_name = "DN-DEMO_eqref.json" if DOUBLE_NULL else "EU-DEMO_EOF.json"
eq_name = Path(read_path, eq_name)
eq = Equilibrium.from_eqdsk(eq_name, from_cocos=3, qpsi_positive=False)

# %% [markdown]
#
# Now we load a first wall geometry, so that the solver can determine where the flux
# surfaces intersect the first wall.

# %%
read_path = get_bluemira_path("radiation_transport/test_data", subfolder="tests")
fw_name = "DN_fw_shape.json" if DOUBLE_NULL else "first_wall.json"
fw_name = Path(read_path, fw_name)
fw_shape = Coordinates.from_json(fw_name)

# %% [markdown]
#
# Then we define some input `Parameter`s for the solver.

# %%
if DOUBLE_NULL:
    params = {
        "P_sep_particle": 150,
        "f_p_sol_near": 0.65,
        "fw_lambda_q_near_omp": 0.002,
        "fw_lambda_q_far_omp": 0.1,
        "fw_lambda_q_near_imp": 0.002,
        "fw_lambda_q_far_imp": 0.1,
        "f_lfs_lower_target": 0.9 * 0.5,
        "f_hfs_lower_target": 0.1 * 0.5,
        "f_lfs_upper_target": 0.9 * 0.5,
        "f_hfs_upper_target": 0.1 * 0.5,
    }
else:
    params = {
        "P_sep_particle": 150,
        "f_p_sol_near": 0.60,
        "fw_lambda_q_near_omp": 0.01,
        "fw_lambda_q_far_omp": 0.05,
        "f_lfs_lower_target": 0.75,
        "f_hfs_lower_target": 0.25,
        "f_lfs_upper_target": 0,
        "f_hfs_upper_target": 0,
    }

# %% [markdown]
#
# Finally, we initialise the `ChargedParticleSolver` and run it.

# %%
solver = ChargedParticleSolver(params, eq, dx_mp=0.0001)
x, z, hf = solver.analyse(first_wall=fw_shape)

# Make the plot
ax = solver.plot()
fig = ax.figure

fig.savefig("charged_particles.png", dpi=300, bbox_inches="tight")
plt.close(fig)
