# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.
"""
Eiffel tower beams example
"""

# %%
import matplotlib.pyplot as plt
from BLUEPRINT.beams.model import FiniteElementModel
from BLUEPRINT.beams.crosssection import RectangularBeam
from BLUEPRINT.beams.material import SS316

plt.close("all")

model = FiniteElementModel()


# %%
def make_platform(width, elevation, cross_section, elements=True, internodes=False):
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
                model.add_element(i + j, i + j + 1, cross_section, SS316)
            model.add_element(i + j + 1, i, cross_section, SS316)
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
                model.add_element(i + j, i + j + 1, cross_section, SS316)
            model.add_element(i + j + 1, i, cross_section, SS316)


# %%
cs1 = RectangularBeam(3, 3)
cs2 = RectangularBeam(1.5, 1.5)
cs3 = RectangularBeam(1, 1)
cs4 = RectangularBeam(0.5, 0.5)
cs5 = RectangularBeam(0.25, 0.25)

# %%
make_platform(124.9, 0, cs1, elements=False)
for i in range(4):
    model.add_support(i, *6 * [True])

# %%
make_platform(70.69, 57.64, cs2, internodes=True)
model.add_element(0, 4, cs1, SS316)
model.add_element(0, 5, cs1, SS316)
model.add_element(0, 11, cs1, SS316)
model.add_element(1, 5, cs1, SS316)
model.add_element(1, 6, cs1, SS316)
model.add_element(1, 7, cs1, SS316)
model.add_element(2, 7, cs1, SS316)
model.add_element(2, 8, cs1, SS316)
model.add_element(2, 9, cs1, SS316)
model.add_element(3, 9, cs1, SS316)
model.add_element(3, 10, cs1, SS316)
model.add_element(3, 11, cs1, SS316)

# %%
make_platform(40.96, 115.73, cs3)

# %%
model.add_element(4, 12, cs2, SS316)
model.add_element(5, 12, cs2, SS316)
model.add_element(11, 12, cs2, SS316)
model.add_element(5, 13, cs2, SS316)
model.add_element(6, 13, cs2, SS316)
model.add_element(7, 13, cs2, SS316)
model.add_element(7, 14, cs2, SS316)
model.add_element(8, 14, cs2, SS316)
model.add_element(9, 14, cs2, SS316)
model.add_element(9, 15, cs2, SS316)
model.add_element(10, 15, cs2, SS316)
model.add_element(11, 15, cs2, SS316)

# %%
make_platform(18.65, 276.13, cs4)

# %%
model.add_element(12, 16, cs3, SS316)
model.add_element(13, 17, cs3, SS316)
model.add_element(14, 18, cs3, SS316)
model.add_element(15, 19, cs3, SS316)

# %%
model.add_node(0, 0, 316)

# %%
for i in range(4):
    model.add_element(16 + i, 20, cs3, SS316)

# %%
model.add_node(0, 0, 324)
model.add_element(20, 21, cs5, SS316)

# %%
model.plot()

# %%
model.add_gravity_loads()
model.add_distributed_load(40, -1000, "Fz")
model.add_distributed_load(43, -1000, "Fz")

# %%
result = model.solve()

# %%
result.plot(stress=True)

plt.show()
