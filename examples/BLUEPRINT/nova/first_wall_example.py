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
First wall example
"""

# %%[markdown]
# # First Wall Example

# %%
import os

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import get_bluemira_path
from bluemira.display.auto_config import plot_defaults
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.limiter import Limiter
from BLUEPRINT.geometry.boolean import (
    boolean_2d_difference,
    boolean_2d_union,
    convex_hull,
)
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.nova.firstwall import Paneller

plot_defaults()

# %%[markdown]
# Let's load some Equilibrium objects, so that we have something to work with

# %%
read_path = get_bluemira_path("equilibria", subfolder="data/bluemira")
eof_name = "EU-DEMO_EOF.json"
sof_name = "EU-DEMO_SOF.json"

sof_name = os.sep.join([read_path, sof_name])
eof_name = os.sep.join([read_path, eof_name])

sof = Equilibrium.from_eqdsk(sof_name)
eof = Equilibrium.from_eqdsk(eof_name)

# %%[markdown]
# The basic idea here is to ignore the heat fluxes due to particles and
# radiation when designing the first wall (for now), and assume that the particle
# heat fluxes will dominate. The particles will predominantly follow magnetic
# field lines, so we can use these as an ersatz for where the heat fluxes
# will be highest.
#
# It's important to remember the plasma will move around to some extent, and
# without knowing too much about this we can add some geometric criteria to the
# problem, too.
#
# We then draw a preliminary first wall shape based on some geometry and magnetic
# criteria:
#
# *  some geometrical offset to the last closed flux surface (LCFS)
# *  some normalised psi offset to the LCFS
# *  apply the above to the start and end of flat-top equilibria
#
#
# We're going to step through some of these steps now, to see what this looks
# like in practice.
#
# Let's define some values:

# %%
dx = 0.125  # [m] Geometrical offset to LCFS
psi_n = 1.06  # [-] Normalised psi of the desired boundary flux surface

# %%[markdown]
# Now let's extract some geometry from the Equilibrium objects

# %%
geometry_offset_loops = []
flux_offsets_loops = []

for equilibrium in [sof, eof]:
    # Get the geometry of the LCFS
    lcfs = equilibrium.get_LCFS()
    dx_loop = lcfs.offset(dx)
    geometry_offset_loops.append(dx_loop)
    # Get a flux offset loop
    psi_n_loop = equilibrium.get_flux_surface(psi_n)
    flux_offsets_loops.append(psi_n_loop)

# %%[markdown]
# Let's have a look at these

# %%
f, ax = plt.subplots()

for loop in geometry_offset_loops:
    loop.plot(ax, fill=False, edgecolor="b")

for loop in flux_offsets_loops:
    loop.plot(ax, fill=False, edgecolor="r")

# %%[markdown]
# We're going to ignore the divertor region for this exercise, as this should
# be treated separately. Let's chop all our geometries so that we ignore
# anything below the X-point
#
# Let's get the active X-points of the equilibria:
#
# *  first retrieve all O-points and X-points
# *  pick the first X-point (active, "strongest")

# %%
sof_opoint, sof_xpoints = sof.get_OX_points()
eof_opoint, eof_xpoints = eof.get_OX_points()

sof_xpoint = sof_xpoints[0]
eof_xpoint = eof_xpoints[0]

# %%[markdown]
# Now get the lowest point of the X-point

# %%
z_xpoint = min(sof_xpoint.z, eof_xpoint.z)

# %%[markdown]
# And trim all the loops accordingly

# %%
loops = geometry_offset_loops + flux_offsets_loops
clipped_loops = []
for loop in loops:
    clip = np.where(loop.z > z_xpoint)
    new_loop = Loop(loop.x[clip], z=loop.z[clip])
    clipped_loops.append(new_loop)

# %%[markdown]
# Let's have a look at these

# %%
f, ax = plt.subplots()

for loop in clipped_loops:
    loop.plot(ax, fill=False, edgecolor="b")

# %%[markdown]
# Now let's imagine our first wall is not allowed inside any of these areas.
# There are two ways of going about things here:
#
# *  Boolean union
# *  Convex hull
#
#
# Let's do both and see what the difference is.
#
# ## Boolean union

# %%
union = clipped_loops[0]
for loop in clipped_loops[1:]:
    loop.close()  # Need to close the open Loops for this operation
    union = boolean_2d_union(union, loop)[0]

# %%[markdown]
# ## Convex hull

# %%
hull = convex_hull(clipped_loops)

# %%
f, (ax1, ax2) = plt.subplots(1, 2)
union.plot(ax1, facecolor="r")
hull.plot(ax2, facecolor="r")
for loop in clipped_loops:
    loop.plot(ax1, fill=False, edgecolor="b")
    loop.plot(ax2, fill=False, edgecolor="b")

ax1.set_title("Boolean union")
ax2.set_title("Convex hull")

# %%[markdown]
# So they are pretty similar, but the subtleties are important. A convex hull
# in 2-D can be thought of an elastic band wrapping itself around the points.
# This means no re-entrant profiles.
# A boolean union will give a smaller area, and include re-entrant angles.
#
# For this example, we'll stick with the convex hull approach.
#
# Let's make an open Loop, assuming that the divertor will be managed elsewhere.
# There are lots of ways of doing this, the below is one way:

# %%
hull.interpolate(200)
z_min = min(hull.z)
div_box = Loop(x=[0, 20, 20, 0, 0], z=[-10, -10, z_min + 0.1, z_min + 0.1, -10])

count = 0
for i, point in enumerate(hull.d2.T):
    if div_box.point_inside(point):
        if count > 2:
            hull.reorder(i, 0)
            hull.open_()
            break
        count += 1

hull = boolean_2d_difference(hull, div_box)[0]

hull.plot()

# %%[markdown]
# Now, flux conformal walls don't really make sense, because the plasma shape
# can never really be constant. Having very curvy walls is usually quite
# expensive, too. So we tend to panel the walls, to make modules that are
# cheaper to manufacture.
#
# We don't want:
# *  the modules to be too sharply angled to each other (so we'll set a maximum
#    turning angle: angle)
# *  the modules to be too small (so we'll limit them with: dx_min)
# *  the modules to be too big (so we'll limit them with: dx_max)

# %%
paneller = Paneller(hull.x, hull.z, angle=20, dx_min=0.5, dx_max=2.5)
paneller.optimise()

x, z = paneller.d2
fw_loop = Loop(x=x, z=z)

# %%[markdown]
# So let's look at the final result

# %%

# Add an arbirtary divertor shape
x_div = [6.5, 7, 7.5, 8, 8.5, 9]
z_div = [-6.5, -6.6, -6, -6, -6.6, -6.5]
x = np.append(fw_loop.x, x_div)
z = np.append(fw_loop.z, z_div)
fw_loop = Loop(x=x, z=z)
fw_loop.close()


eof.limiter = Limiter([(x, z) for x, z in zip(fw_loop.x, fw_loop.z)])

f, ax = plt.subplots()

eof.plot(ax)
sof.plot(ax)
