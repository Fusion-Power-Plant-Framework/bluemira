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
from bluemira.plasma_physics.reactions import reactivity

# %% [markdown]
# # Fusion Reactivity example
#
# Let's plot the reactivity of a couple of well-known fusion reactions.

# %%
plot_defaults()

temperature = np.linspace(10e6, 100e6, 1000)  # [K]

sigma_v_DT = reactivity(temperature, "D-T")
sigma_v_DD = reactivity(temperature, "D-D")
sigma_v_DHe3 = reactivity(temperature, "D-He3")  # noqa: N816

f, ax = plt.subplots()
ax.loglog(temperature, sigma_v_DT, label="D-T")
ax.loglog(temperature, sigma_v_DD, label="D-D")
ax.loglog(temperature, sigma_v_DHe3, label="D-He3")

ax.grid(which="both")
ax.set_xlabel("T [K]")
ax.set_ylabel("$\\sigma_{v}$ [$m^{3}/s$]")
ax.legend()
plt.show()
