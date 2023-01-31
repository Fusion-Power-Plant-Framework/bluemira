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
mypf = MyParameterFrame.from_dict(
    {"A": {"value": 5, "unit": ""}, "R_0": {"value": 8, "unit": "m"}}
)
mypf2 = MyParameterFrame.from_dict(
    {"A": {"value": 5, "unit": ""}, "R_0": {"value": 8, "unit": "m"}}
)

print(mypf)
print(mypf2)
# Both frames equal
assert mypf == mypf2  # noqa: S101

# %% [markdown]
# Trying to set a unit with the wrong dimension

# %%
mydiffval = MyParameterFrame.from_dict(
    {"A": {"value": 6, "unit": "m"}, "R_0": {"value": 8, "unit": "m"}}
)

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
