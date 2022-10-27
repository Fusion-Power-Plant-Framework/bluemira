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
An example of how to use Parameters and ParameterFrames within bluemira.
"""

# %%[markdown]
# # Parameters and ParameterFrames in bluemira
# `Parameters` and `ParameterFrames` are the mechanism bluemira uses to contain
# metadata about a given value.
# Each `Parameter` must have a unit associated with the value and can have a
# source, description and long_name.
#
# The mechanics of the unit system in bluemira are fairly staight forward
# It provides an implicit interface to convert units to the base units of bluemira.

# %%
from dataclasses import dataclass

from pint.errors import DimensionalityError

import bluemira.base.constants as const
from bluemira.base.parameter_frame import Parameter, ParameterFrame

# %%[markdown]
# ## Parameters and Units
# First I make a small ParameterFrame and compare the two methods of creating a frame

# %%


@dataclass
class MyParameterFrame(ParameterFrame):
    """A ParameterFrame"""

    R_0: Parameter[float]
    A: Parameter[float]


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

# %%[markdown]
# Trying to set a unit with the wrong dimension

# %%
mydiffval = MyParameterFrame.from_dict(
    {"A": {"value": 6, "unit": "m"}, "R_0": {"value": 8, "unit": "m"}}
)

try:
    mypf.update_from_frame(mydiffval)
except DimensionalityError as de:
    print(de)

# %%[markdown]
# Changing a value of a parameter with a compatible but different unit

# %%

mypf.update_values({"R_0": {"value": 6000, "unit": "mm"}})

print(mypf)

# %%[markdown]

# Accessing the value of a parameter in a different unit

# %%

print(mypf.R_0.value_as("cm"))  # 600

# %%[markdown]
# ## Raw conversion

# In some situations it may be useful to convert a raw value to a different unit.
# It is recommended (and enforced within bluemira core) that all conversions however
# simplistic are converted this way.
#
# Any possible future base unit changes can be achieved by searching for a given unit.

# %%

print(const.raw_uc(1, "um^3", "m^3"))
# gas flow rate conversion @OdegC
print(const.raw_uc(1, "mol/s", "Pa m^3/s"))
print(const.gas_flow_uc(1, "mol/s", "Pa m^3/s"))
# gas flow rate conversion @25degC
print(const.gas_flow_uc(1, "mol/s", "Pa m^3/s", gas_flow_temperature=298.15))
# boltzmann constant conversion
print(const.raw_uc(1, "eV", "K"))

# %%[markdown]
# ## Raw Temperature conversion with checks from different units

# %%

try:
    const.to_kelvin(-300)
except ValueError as v:
    print(v)

print(const.to_celsius(10, unit="rankine"))
