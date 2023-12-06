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
An example of how to use Parameters and ParameterFrames within bluemira.
"""

# %% [markdown]
# # Parameters and ParameterFrames in bluemira
# `Parameters` and `ParameterFrames` are the mechanism bluemira uses to contain
# metadata about a given value.
# Each `Parameter` must have a unit associated with its value, and can have a
# source, description and long_name.
#
# `ParameterFrame`s implicitly convert parameter values' units to the
# [base units of bluemira](
# https://bluemira.readthedocs.io/en/latest/conventions.html#unit-convention).

# %%
from dataclasses import dataclass

from pint.errors import DimensionalityError

from bluemira.base.parameter_frame import Parameter, ParameterFrame

# %% [markdown]
# ## Parameters and Units
# First, I make a small `ParameterFrame` and compare the two methods of creating it.


# %%
@dataclass
class MyParameterFrame(ParameterFrame):
    """A ParameterFrame"""

    R_0: Parameter[float]
    A: Parameter[float]


# the unit "" is equivalent to "dimensionless"
mypf = MyParameterFrame.from_dict({
    "A": {"value": 5, "unit": ""},
    "R_0": {"value": 8, "unit": "m"},
})
mypf2 = MyParameterFrame.from_dict({
    "A": {"value": 5, "unit": ""},
    "R_0": {"value": 8, "unit": "m"},
})

print(mypf)
print(mypf2)
# Both frames equal
assert mypf == mypf2  # noqa: S101

# %% [markdown]
# Trying to set a unit with the wrong dimension

# %%
mydiffval = MyParameterFrame.from_dict({
    "A": {"value": 6, "unit": "m"},
    "R_0": {"value": 8, "unit": "m"},
})

try:
    mypf.update_from_frame(mydiffval)
except DimensionalityError as de:
    print(de)

# %% [markdown]
# Changing a value of a parameter with a compatible but different unit

# %%
mypf.update_from_dict({"R_0": {"value": 6000, "unit": "mm"}})

print(mypf)

# %% [markdown]
#
# Accessing the value of a parameter in a different unit

# %%
print(mypf.R_0.value_as("cm"))  # 600
