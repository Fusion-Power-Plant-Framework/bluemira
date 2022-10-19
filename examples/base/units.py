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
An example of how to use Units within bluemira.
"""

# %%[markdown]
# # Unit conversion in bluemira
# The mechanics of the unit system in bluemira are fairly staight forward
# It aims to provide a useful user interface to convert units
# internally in most situations the units are up to the developer

# %%
from dataclasses import dataclass

from pint.errors import DimensionalityError

import bluemira.base.constants as const
from bluemira.base.parameter_frame import Parameter, ParameterFrame, parameter_frame

# %%[markdown]
# ## Raw conversion

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
# ## Raw Temperature conversion with checks

# %%

try:
    const.to_kelvin(-300)
except ValueError as v:
    print(v)

print(const.to_celsius(10, unit="rankine"))

# %%[markdown]
# ## Parameters and Units
# First I make a small ParameterFrame

# %%


@dataclass
class MyParameterFrame(ParameterFrame):
    """A ParameterFrame"""

    A: Parameter[float]


# this works the same
@parameter_frame
class MyDecoratedFrame:
    """A ParameterFrame made with a decorator"""

    A: Parameter[float]


mypf = MyParameterFrame.from_dict({"A": {"value": 5, "unit": ""}})
mydecpf = MyDecoratedFrame.from_dict({"A": {"value": 5, "unit": ""}})

print(mypf)
print(mydecpf)
# Both frames equal
assert all([dparam == param for dparam, param in zip(mydecpf, mypf)])  # noqa: S101

# %%[markdown]
# Trying to set a unit with the wrong dimension

# %%
mydiffval = MyDecoratedFrame.from_dict({"A": {"value": 6, "unit": "m"}})

try:
    mypf.update_from_frame(mydiffval)
except DimensionalityError as de:
    print(de)

# %%[markdown]
# Changing a value of a parameter with a compatible but different unit

# %%

mypf.update_values({"A": {"value": 6, "unit": ""}})

print(mypf)
