# %%
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
An example to show the use of the raw unit converters
"""


# %% [markdown]
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
