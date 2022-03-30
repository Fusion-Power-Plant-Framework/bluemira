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

from pint.errors import DimensionalityError

import bluemira.base.config as cfg
import bluemira.base.constants as const
import bluemira.base.parameter as param

# %%[markdown]
# ## Raw conversion

# %%

print(const.raw_uc(1, "um^3", "m^3"))
# gas flow rate conversion @OdegC
print(const.raw_uc(1, "mol/s", "Pa m^3/s"))
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
# First I grab the default configuration

# %%

pf = cfg.Configuration()

# %%[markdown]
# Trying to set a unit with the wrong dimension

# %%

param1 = param.Parameter(var="A", value=5, unit="m")
try:
    pf.update_kw_parameters({"A": param1})
except DimensionalityError as de:
    print(de)

# %%[markdown]
# Changing a value of a parameter with a compatible but different unit

# %%

print(pf.I_p)
param2 = param.Parameter(var="I_p", value=5, unit="uA", source="very small input")

# %%

pf.set_parameter("I_p", param2)
print(pf.I_p)
