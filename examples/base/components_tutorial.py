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
An example of how to use Components to represent a set of objects in a reactor.
"""

# %%
from anytree import RenderTree

from bluemira.base.components import Component, MagneticComponent, PhysicalComponent

# %%[markdown]
# Example of a Tree structure
# Definition of some Components as groups. These do not have a physical shape / material
# but represent common systems within a reactor (or indeed the reactor itself).

# %%
reactor = Component("Reactor")

magnets = Component("Magnets", parent=reactor)
tf_coils = Component("TFCoils", parent=magnets)
pf_coils = Component("PFCoils", parent=magnets)

# %%[markdown]
# Definition of some sub-components as physical components
# Note: it is not necessary to store the component in a variable. It is already
# stored as child of the parent, if any.

# %%
for i in range(6):
    MagneticComponent(
        "PF" + str(i),
        shape="pf_shape" + str(i),
        material="pf_material" + str(i),
        conductor="pf_conductor" + str(i),
        parent=pf_coils,
    )

# %%[markdown]
# Do the same for the CS coils

# %%
cs_coils = Component("CSCoils", parent=magnets)
for i in range(6):
    MagneticComponent(
        "CS" + str(i),
        shape="cs_shape" + str(i),
        material="cs_material" + str(i),
        conductor="cs_conductor" + str(i),
        parent=cs_coils,
    )

# %%[markdown]
# Adding in vessel components

# %%
in_vessel = Component("InVessel", parent=reactor)
blanket = PhysicalComponent(
    "Blanket",
    shape="BB_shape",
    material="BB_material",
    parent=in_vessel,
)
divertor = PhysicalComponent(
    "Divertor",
    shape="Div_shape",
    material="Div_material",
    parent=in_vessel,
)
vessel = PhysicalComponent(
    "Vessel",
    shape="VV_shape",
    material="VV_material",
    parent=in_vessel,
)

# %%[markdown]
# Printing the tree

# %%
print(RenderTree(reactor))
