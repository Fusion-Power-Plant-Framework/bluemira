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
An example to run PROCESS
"""

# %% [markdown]
# # Solvers
# This example shows how to use File interface solvers.
# Using PROCESS as the external solver

# %%
from bluemira.base.file import get_bluemira_path
from bluemira.base.logs import set_log_level
from bluemira.codes import process

# %% [markdown]
# ## Configuring a solver
#
# PROCESS is one of the codes bluemira can use to compliment a reactor design.
# As with any of the external codes bluemira uses, a solver object is created.
# The solver object abstracts away most of the complexities of running different
# programs within bluemira.
#
# ### Setting up
#
# #### Logging
# To enable debug logging run the below cell

# %%
set_log_level("DEBUG")

# %% [markdown]
# #### Binary Location
# Firstly if process is not in your system path we need to provide the
# binary location to the solver.

# %%
# PROCESS_PATH = "/home/process/lives/here"
PROCESS_PATH = ""
binary = f"{PROCESS_PATH}process"

# %% [markdown]
# #### Creating the solver object
# bluemira-PROCESS parameter names have been mapped across where possible.
# Some example parameters have been set here in `new_params`
# before being converted into a bluemira configuration store.

# %%
params = {
    "A": 3.1,
    "R_0": 9.002,
    "I_p": 17.75e6,
    "B_0": 5.855,
    "V_p": -2500,
    "v_burn": -1.0e6,
    "kappa_95": 1.652,
    "delta_95": 0.333,
    "delta": 0.38491934960310104,
    "kappa": 1.6969830041844367,
    "tk_ts": 0,
}


# %% [markdown]
# Finally the `build_config` dictionary collates the configuration settings for
# the solver

# %%
build_config = {
    "binary": binary,
    "run_dir": get_bluemira_path(subfolder="generated_data"),
}

# %% [markdown]
# Now we can create the solver object with the parameters and build configuration

# %%
process_solver = process.Solver(params=params, build_config=build_config)

# %% [markdown]
# #### Running the solver
# Call the `execute` method of the solver, using one of the solver's
# run modes.

# %%
process_solver.execute(process_solver.run_mode_cls.RUN)
