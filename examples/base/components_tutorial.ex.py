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
An example of how to use Components to represent a set of objects in a reactor.
"""

# %%
from anytree import RenderTree

from bluemira.base.components import Component, MagneticComponent, PhysicalComponent

# %% [markdown]
# # Components
#
# Example of a Tree structure
# Definition of some Components as groups. These do not have a physical shape / material
# but represent common systems within a reactor (or indeed the reactor itself).

# %%
reactor = Component("Reactor")

magnets = Component("Magnets", parent=reactor)
tf_coils = Component("TFCoils", parent=magnets)
pf_coils = Component("PFCoils", parent=magnets)

# %% [markdown]
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

# %% [markdown]
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

# %% [markdown]
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

# %% [markdown]
# Printing the tree

# %%
print(RenderTree(reactor))
