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
Fusion reactivity example
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from bluemira.display import plot_defaults
from bluemira.plasma_physics.reactions import ReactivityMethod, reactivity

# %%
plot_defaults()
temperature = np.linspace(10e6, 100e6, 1000)  # [K]

# %% [markdown]
# # Comparison of reactivity within the most complete model: Bosch-Hale

# %%
sigma_v_DT = reactivity(temperature, "D-T")
sigma_v_DD1 = reactivity(temperature, "D-D1")
sigma_v_DD2 = reactivity(temperature, "D-D2")
sigma_v_DHe3 = reactivity(temperature, "D-He3")  # noqa: N816
f, ax = plt.subplots()
ax.loglog(temperature, sigma_v_DT, label="DT")
ax.loglog(temperature, sigma_v_DD1, label="DD1 ($\\rightarrow$ $^3$He)")
ax.loglog(temperature, sigma_v_DD2, label="DD2 ($\\rightarrow$ T)", ls="--")
ax.loglog(temperature, sigma_v_DHe3, label="DHe3")
ax.set_title("All reactivities according to Bosch-Hale")
ax.set_xlabel("T [K]")
ax.set_ylabel("$\\sigma_{v}$ [$m^{3}/s$]")
ax.grid(which="both")
ax.legend()
plt.show()

# %% [markdown]
# # DT reactivites comparison

# %%
sigma_v_DT_bh = reactivity(temperature, "D-T", ReactivityMethod.BOSCH_HALE)
sigma_v_DT_pm = reactivity(temperature, "D-T", ReactivityMethod.PLASMOD)
sigma_v_DT_jn = reactivity(temperature, "D-T", ReactivityMethod.JOHNER)

f, ax = plt.subplots()
ax.plot(temperature, sigma_v_DT_bh, label="Bosch-Hale")
ax.plot(temperature, sigma_v_DT_pm, label="Plasmod", ls="--")
ax.plot(temperature, sigma_v_DT_jn, label="Johner", ls="-.")
ax.set_title("DT reactivity")
ax.set_xlabel("T [K]")
ax.set_ylabel("$\\sigma_{v}$ [$m^{3}/s$]")
ax.grid(which="both")
ax.legend()
plt.show()

# %% [markdown]
# # DD reactivities comparison

# %%
sigma_v_DD1_bh = reactivity(temperature, "D-D1", ReactivityMethod.BOSCH_HALE)
sigma_v_DD2_bh = reactivity(temperature, "D-D2", ReactivityMethod.BOSCH_HALE)
sigma_v_DD_pm = reactivity(temperature, "D-D1", ReactivityMethod.PLASMOD)

f, ax = plt.subplots()
ax.plot(temperature, sigma_v_DD1_bh, label="Bosch-Hale:\n D-D $\\rightarrow$ $^3$He")
ax.plot(temperature, sigma_v_DD2_bh, label="Bosch-Hale:\n D-D $\\rightarrow$ T", ls="--")
ax.plot(temperature, sigma_v_DD_pm, label="Plasmod:\nD-D (unspecified)", ls="-.")
ax.set_title("DD reactivities")
ax.set_xlabel("T [K]")
ax.set_ylabel("$\\sigma_{v}$ [$m^{3}/s$]")
ax.grid(which="both")
ax.legend()
plt.show()
