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
Plotting module examples
"""

import numpy as np

# %%
from bluemira.geometry.plane import BluemiraPlane

# %%[markdown]
# Creation of a plane and respective placement

# %%
factor = np.random.uniform(1, 100)
base = np.random.random((1, 3))[0] * factor
axis = np.random.random((1, 3))[0] * factor

plane = BluemiraPlane(base=base, axis=axis)
placement = plane.to_placement()
xy_plane = placement.xy_plane()
yz_plane = placement.yz_plane()
xz_plane = placement.xz_plane()

dir_z = np.array([0, 0, 1])
dir_z1 = placement.mult_vec(dir_z) - placement.mult_vec(np.array([0, 0, 0]))

print(f"dir_z = {dir_z}, dir_z1 = {dir_z1}, plane.axis = {plane.axis}")
print(np.allclose(dir_z1, plane.axis))
print(np.allclose(xy_plane.base, plane.base))
print(np.allclose(xy_plane.axis, plane.axis))

plane = placement.xy_plane()
placement2 = plane.to_placement()
print(np.allclose(plane.axis, placement.mult_vec([0, 0, 1]) - placement.base))
plane = placement.xz_plane()
print(np.allclose(plane.axis, placement.mult_vec([0, -1, 0]) - placement.base))
plane = placement.yz_plane()
print(np.allclose(plane.axis, placement.mult_vec([1, 0, 0]) - placement.base))


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
Plotting module examples
"""

import numpy as np

# %%
from bluemira.geometry.plane import BluemiraPlane

# %%[markdown]
# Create of a random plane

# %%
factor = np.random.uniform(1, 100)
base = np.random.random((1, 3))[0] * factor
axis = np.random.random((1, 3))[0] * factor
plane = BluemiraPlane(base=base, axis=axis)

# %%[markdown]
# Convert a plane to a placement.

# %%
placement = plane.to_placement()

# %%[markdown]
# Extract the xy, yz, and xz planes for the placement

# %%
xy_plane = placement.xy_plane()

# %%[markdown]
# Check that all has been created correctly

# %%
dir_z = np.array([0, 0, 1])
dir_z1 = placement.mult_vec(dir_z) - placement.mult_vec(np.array([0, 0, 0]))

print(f"dir_z = {dir_z}, dir_z1 = {dir_z1}, plane.axis = {plane.axis}")
print(np.allclose(dir_z1, plane.axis))
print(np.allclose(xy_plane.base, plane.base))
print(np.allclose(xy_plane.axis, plane.axis))
