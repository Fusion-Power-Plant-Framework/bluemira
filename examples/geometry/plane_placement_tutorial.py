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
plane = BluemiraPlane(axis=[0, 1, 1])
placement = plane.to_placement()
xy_plane = placement.xy_plane()

print(np.allclose(xy_plane.base, plane.base))
print(np.allclose(xy_plane.axis, plane.axis))

# %%[markdown]
# Change plane base and axis

# %%
plane.base = [1, -1, 5]
plane.axis = [1, 1, 0]
placement = plane.to_placement()
xy_plane = placement.xy_plane()

print(np.allclose(xy_plane.base, plane.base))
print(np.allclose(xy_plane.axis, plane.axis))
