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
An example to show the use of the raw unit converters
"""


# %% [markdown]
# # Units Example
# ## Raw conversion
#
# In some situations it may be useful to convert a raw value to a different unit.
# It is recommended that all conversions,
# however simple, are made using `bluemira.constants.raw_uc`.
# This is how all unit conversions are performed internally.
#
# Using `raw_uc` makes it less likely bugs would be introduced in the event of a
# base unit change.

# %%
import bluemira.base.constants as const

# %%
print(const.raw_uc(1, "um^3", "m^3"))
# gas flow rate conversion @OdegC
print(const.raw_uc(1, "mol/s", "Pa m^3/s"))
print(const.gas_flow_uc(1, "mol/s", "Pa m^3/s"))
# gas flow rate conversion @25degC
print(const.gas_flow_uc(1, "mol/s", "Pa m^3/s", gas_flow_temperature=298.15))
# boltzmann constant conversion
print(const.raw_uc(1, "eV", "K"))

# %% [markdown]
# ## Raw Temperature conversion with checks from different units
# The explicit temperature conversion routines guard against temperatures
# below absolute zero

# %%
try:
    const.to_kelvin(-300)
except ValueError as v:
    print(v)

print(const.to_celsius(10, unit="rankine"))
