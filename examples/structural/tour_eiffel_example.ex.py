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
Eiffel tower structural example
"""

# %%
import matplotlib.pyplot as plt
from matproplib.conditions import STPConditions

from bluemira.materials.basic import SS316
from bluemira.structural.crosssection import RectangularBeam
from bluemira.structural.model import FiniteElementModel

plt.close("all")

model = FiniteElementModel()
ss316 = SS316()
op_cond = STPConditions()

# %% [markdown]
# # Structural Example


# %%
def make_platform(width, elevation, cross_section, *, elements=True, internodes=False):
    """
    Make a square platform at a certain elevation.
    """
    if not internodes:
        model.add_node(-width / 2, -width / 2, elevation)
        i = model.geometry.nodes[-1].id_number
        model.add_node(width / 2, -width / 2, elevation)
        model.add_node(width / 2, width / 2, elevation)
        model.add_node(-width / 2, width / 2, elevation)

        if elements:
            for j in range(3):
                model.add_element(i + j, i + j + 1, cross_section, ss316, op_cond)
            model.add_element(i + j + 1, i, cross_section, ss316, op_cond)
    if internodes:
        model.add_node(-width / 2, -width / 2, elevation)
        i = model.geometry.nodes[-1].id_number
        model.add_node(0, -width / 2, elevation)
        model.add_node(width / 2, -width / 2, elevation)
        model.add_node(width / 2, 0, elevation)
        model.add_node(width / 2, width / 2, elevation)
        model.add_node(0, width / 2, elevation)
        model.add_node(-width / 2, width / 2, elevation)
        model.add_node(-width / 2, 0, elevation)
        if elements:
            for j in range(7):
                model.add_element(i + j, i + j + 1, cross_section, ss316, op_cond)
            model.add_element(i + j + 1, i, cross_section, ss316, op_cond)


# %%
cs1 = RectangularBeam(5, 5)
cs2 = RectangularBeam(4, 4)
cs3 = RectangularBeam(2, 2)
cs4 = RectangularBeam(1, 1)

# %%
make_platform(100, 0, cs1, elements=False)
for i in range(4):
    model.add_support(i, dx=True, dy=True, dz=True, rx=True, ry=True, rz=True)

# %%
make_platform(60, 56, cs2, internodes=True)
model.add_element(0, 4, cs1, ss316, op_cond)
model.add_element(0, 5, cs1, ss316, op_cond)
model.add_element(0, 11, cs1, ss316, op_cond)
model.add_element(1, 5, cs1, ss316, op_cond)
model.add_element(1, 6, cs1, ss316, op_cond)
model.add_element(1, 7, cs1, ss316, op_cond)
model.add_element(2, 7, cs1, ss316, op_cond)
model.add_element(2, 8, cs1, ss316, op_cond)
model.add_element(2, 9, cs1, ss316, op_cond)
model.add_element(3, 9, cs1, ss316, op_cond)
model.add_element(3, 10, cs1, ss316, op_cond)
model.add_element(3, 11, cs1, ss316, op_cond)

# %%
make_platform(37, 116, cs3)

# %%
model.add_element(4, 12, cs2, ss316, op_cond)
model.add_element(5, 12, cs2, ss316, op_cond)
model.add_element(11, 12, cs2, ss316, op_cond)
model.add_element(5, 13, cs2, ss316, op_cond)
model.add_element(6, 13, cs2, ss316, op_cond)
model.add_element(7, 13, cs2, ss316, op_cond)
model.add_element(7, 14, cs2, ss316, op_cond)
model.add_element(8, 14, cs2, ss316, op_cond)
model.add_element(9, 14, cs2, ss316, op_cond)
model.add_element(9, 15, cs2, ss316, op_cond)
model.add_element(10, 15, cs2, ss316, op_cond)
model.add_element(11, 15, cs2, ss316, op_cond)

# %%
make_platform(12, 196, cs3)

# %%
model.add_element(12, 16, cs3, ss316, op_cond)
model.add_element(13, 17, cs3, ss316, op_cond)
model.add_element(14, 18, cs3, ss316, op_cond)
model.add_element(15, 19, cs3, ss316, op_cond)

# %%
make_platform(6, 276, cs3)

model.add_element(16, 20, cs4, ss316, op_cond)
model.add_element(17, 21, cs4, ss316, op_cond)
model.add_element(18, 22, cs4, ss316, op_cond)
model.add_element(19, 23, cs4, ss316, op_cond)


# %%
model.add_node(0, 0, 316)

# %%
for i in range(4):
    model.add_element(20 + i, 24, cs4, ss316, op_cond)

# %%
model.add_node(0, 0, 324)
model.add_element(24, 25, cs4, ss316, op_cond)

# %%
model.plot(show_cross_sections=True)

# %%
model.add_gravity_loads()
model.add_node_load(24, 1e6, "Fy")


# %%
result = model.solve()

# %%

result.plot(stress=True)

plt.show()
