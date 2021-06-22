#  bluemira is an integrated inter-disciplinary design tool for future fusion
#  reactors. It incorporates several modules, some of which rely on other
#  codes, to carry out a range of typical conceptual fusion reactor design
#  activities.
#  #
#  Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
#                     D. Short
#  #
#  bluemira is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#  #
#  bluemira is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#  Lesser General Public License for more details.
#  #
#  You should have received a copy of the GNU Lesser General Public
#  License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
An example of how to use Components to represent a set of objects in a reactor.
"""

# %%
from anytree import RenderTree
import bluemira.components as bm_comp

# %%[markdown]
# Example of a Tree structure
# Definition of some Components as groups. These do not have a physical shape / material
# but represent common systems within a reactor (or indeed the reactor itself).

# %%
reactor = bm_comp.GroupingComponent("Reactor", config={}, inputs={})

magnets = bm_comp.GroupingComponent("Magnets", config={}, inputs={}, parent=reactor)
tf_coils = bm_comp.GroupingComponent("TFCoils", config={}, inputs={}, parent=magnets)
pf_coils = bm_comp.GroupingComponent("PFCoils", config={}, inputs={}, parent=magnets)

# %%[markdown]
# Definition of some sub-components as physical components
# Note: it is not necessary to store the component in a variable. It is already
# stored as child of the parent, if any.

# %%
for i in range(6):
    bm_comp.MagneticComponent(
        "PF" + str(i),
        config={},
        inputs={},
        shape="pf_shape" + str(i),
        material="pf_material" + str(i),
        conductor="pf_conductor" + str(i),
        parent=pf_coils,
    )

# %%[markdown]
# Do the same for the CS coils

# %%
cs_coils = bm_comp.GroupingComponent("CSCoils", config={}, inputs={}, parent=magnets)
for i in range(6):
    bm_comp.MagneticComponent(
        "CS" + str(i),
        config={},
        inputs={},
        shape="cs_shape" + str(i),
        material="cs_material" + str(i),
        conductor="cs_conductor" + str(i),
        parent=cs_coils,
    )

# %%[markdown]
# Adding in vessel components

# %%
in_vessel = bm_comp.GroupingComponent("InVessel", config={}, inputs={}, parent=reactor)
blanket = bm_comp.PhysicalComponent(
    "Blanket",
    config={},
    inputs={},
    shape="BB_shape",
    material="BB_material",
    parent=in_vessel,
)
divertor = bm_comp.PhysicalComponent(
    "Divertor",
    config={},
    inputs={},
    shape="Div_shape",
    material="Div_material",
    parent=in_vessel,
)
vessel = bm_comp.PhysicalComponent(
    "Vessel",
    config={},
    inputs={},
    shape="VV_shape",
    material="VV_material",
    parent=in_vessel,
)

# %%[markdown]
# Printing the tree

# %%
print(RenderTree(reactor))
