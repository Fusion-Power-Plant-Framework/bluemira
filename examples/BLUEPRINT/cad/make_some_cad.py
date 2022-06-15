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
Simple CAD objects for testing OCC CAD display.
"""

from BLUEPRINT.cad.cadtools import (
    extrude,
    make_axis,
    make_face,
    make_shell,
    revolve,
    show_CAD,
)

# %%
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.shell import Shell

# %%[markdown]
# # Make Some CAD
#
# A simple example that generates a cube and a shell and displays the CAD.
#
# First make a cube.

# %%
square = Loop(x=[2, 4, 4, 2, 2], z=[2, 2, 4, 4, 2])

face = make_face(square)

cube = extrude(face, vec=[0, 2, 0])

# %%[markdown]
# Now make a shell and revolve it to make it 3D.

# %%
loop = Loop(
    x=[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 5, 4, 3, 2, 1],
    y=[7, 7, 8, 8, 9, 10, 8, 7, 6, 5, 4, 3, 1, 2, 3, 4, 7],
)

shell = Shell(loop, loop.offset(0.5))

face = make_shell(shell)

axis = make_axis([0, 0, 0], [0, 1, 0])

revolution = revolve(face, axis, 30)

# %%[markdown]
# Finally display the CAD.
# This will open in a pop-up window.

# %%
show_CAD(cube, revolution)
