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

# %%
from anytree import NodeMixin, RenderTree
import bluemira.components.Base as Comp

# %%[markdown]
# # Example of a Tree structure
# # Definition of some Components as groups

# %%
reactor = Comp.Component("Reactor")

magnets = Comp.Component("Magnets", parent=reactor)
tf_coils = Comp.Component("TFCoils", parent=magnets)
pf_coils = Comp.Component("PFCoils", parent=magnets)

# %%[markdown]
# # Definition of some sub-components as physical components
# # Note: it is not necessary to store the component in a variable. It is already
# # stored as child of the parent, if any.

# %%
for i in range(6):
    Comp.PhysicalComponent("PF" + str(i), shape="pf_shape" + str(i),
                           material="pf_material" + str(i), parent=pf_coils)

# %%[markdown]
# # Do the same for the CS coils

# %%
cs_coils = Comp.Component("CSCoils", parent=magnets)
for i in range(6):
    Comp.PhysicalComponent("CS" + str(i), shape="cs_shape" + str(i),
                           material="cs_material" + str(i), parent=cs_coils)

# %%[markdown]
# # Adding in vessel components

# %%
in_vessel = Comp.Component("InVessel", parent=reactor)
blanket = Comp.PhysicalComponent("Blanket", shape="BB_shape", material="BB_material",
                                 parent=in_vessel)
divertor = Comp.PhysicalComponent("Divertor", shape="Div_shape",
                                  material="Div_material", parent=in_vessel)
vessel = Comp.PhysicalComponent("Vessel", shape="VV_shape",
                                  material="VV_material", parent=in_vessel)

# %%[markdown]
# # Printing the tree

# %%
print(RenderTree(reactor))