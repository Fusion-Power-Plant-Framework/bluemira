# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                    J. Morris, D. Short
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
Some examples of using bluemira geometry objects.
"""

# %%
import bluemira.geometry as geo
from bluemira.codes import _freecadapi as cadapi
from bluemira.display import plot_2d, show_cad
from bluemira.display.plotter import DEFAULT_PLOT_OPTIONS
from operator import itemgetter

DEFAULT_PLOT_OPTIONS["plane"] = "xy"

# %%[markdown]
# # A simple tutorial for the bluemira geometric module
#
# ## 1. Creation of a bluemira wire

# %%
pntslist = [(1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
bmwire = geo.tools.make_polygon(pntslist)
print(f"{bmwire=}")
plot_2d(bmwire)

# %%[markdown]
# ### Same result using cadapi

# %%
wire = cadapi.make_polygon(pntslist)
print(f"Freecad wire: {wire}, length: {wire.Length}, isClosed: {wire.isClosed()}")
bmwire = geo.wire.BluemiraWire(wire, "bmwire")
print(f"{bmwire=}")
plot_2d(bmwire)

# %%[markdown]
# ## 2. Creation of a closed bluemira wire

# %%
bmwire = geo.tools.make_polygon(pntslist, closed=True)
print(bmwire)
plot_2d(bmwire)

# %%[markdown]
# ## 3. Make some operations on bluemira wire

# %%
ndiscr = 10

# %%[markdown]
# ### 3.1 Discretize in {ndiscr} points

# %%
points = bmwire.discretize(ndiscr)
print(f"{points}")
plot_2d(points)

# %%[markdown]
# ### 3.2 Discretize considering the edges

# %%
points = bmwire.discretize(ndiscr, byedges=True)
print(f"{points=}")
plot_2d(points)

# %%[markdown]
# ## 4. Creation of a bluemira face

# %%
bmface = geo.face.BluemiraFace(bmwire, "bmface")
print(f"{bmface=}")
plot_2d(bmface)

# %%[markdown]
# ## 5. Scaling a bluemira wire
#
# **NOTE**: scale function modifies the original object
#
# ### 5.1 Scale a BluemiraWire

# %%
print(f"Original object: {bmwire}")
bmwire.scale(2)
print(f"Scaled object: {bmwire}")

# %%[markdown]
# **NOTE**: since bmface is connected to bmwire, a scale operation on bmwire also affect
# bmface

# %%[markdown]
# ### 5.2 Scale a BluemiraFace

# %%
print(f"Original object: {bmface}")
bmface.scale(2)
print(f"Scaled object: {bmface}")

# %%[markdown]
# ## 6. Saving as a STEP file

# %%
shapes = [bmwire._shape, bmface._shape]
print(shapes)
cadapi.save_as_STEP(shapes, "geo_shapes")

# %%[markdown]
# ## 7. Closing a BluemiraWire
#
# ### 7.1 When boundary is list(Part.Wire)

# %%
pntslist = [(1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
wire = cadapi.make_polygon(pntslist, closed=False)
bmwire_nc = geo.wire.BluemiraWire(wire)
print(f"Before close: {bmwire_nc}")
print(f"Boundary: {bmwire_nc.boundary}")
bmwire_nc.close()
print(f"After close: {bmwire_nc}")
print(f"Boundary: {bmwire_nc.boundary}")

# %%[markdown]
# ### 7.2 When boundary is list(BluemiraWire)

# %%
bmwire_nc = geo.wire.BluemiraWire(geo.wire.BluemiraWire(wire))
print(bmwire_nc)
bmwire_nc.close()
print(bmwire_nc)
print(bmwire_nc.boundary)

# %%[markdown]
# ## 8. Translate

# %%
bmface.translate((5.0, 2.0, 0.0))
plot_2d(bmface)
geo.tools.save_as_STEP([bmwire, bmface], "geo_translate")

# %%[markdown]
# ## 9. Bezier spline curve

# %%
pntslist = [(1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
bmwire_nc = geo.tools.make_bspline(pntslist, closed=False)
print(bmwire_nc)
plot_2d(bmwire_nc)

# %%[markdown]
# ### Same result using cadapi

# %%
wire = cadapi.make_bspline(pntslist, closed=False)
bmwire_nc = geo.wire.BluemiraWire(wire)
print(bmwire_nc)
plot_2d(bmwire_nc)
geo.tools.save_as_STEP([bmwire_nc], "geo_bspline")

# %%[markdown]
# ## 10. Revolve

# %%
bmsolid = geo.tools.revolve_shape(bmface, direction=(0.0, 1.0, 0.0))
show_cad(bmsolid)
geo.tools.save_as_STEP([bmsolid], "geo_revolve")

# %%[markdown]
# ## 11. Face and solid with hole

# %%
pntslist_out = [(1.0, 1.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
delta = 0.25
pntslist_in = [
    (1.0 - delta, 1.0 - delta, 0.0),
    (0.0 + delta, 1.0 - delta, 0.0),
    (0.0 + delta, 0.0 + delta, 0.0),
    (1.0 - delta, 0.0 + delta, 0.0),
]
wire_out = geo.tools.make_polygon(pntslist_out, closed=True)
wire_in = geo.tools.make_polygon(pntslist_in, closed=True)
bmface = geo.face.BluemiraFace([wire_out, wire_in])
plot_2d(bmface)
geo.tools.save_as_STEP([bmface], "geo_face_with_hole")

# %%
bmsolid = geo.tools.revolve_shape(bmface, direction=(0.0, 1.0, 0.0))
show_cad(bmsolid)
geo.tools.save_as_STEP([bmsolid], "geo_solid_with_hole")

# %%[markdown]
# ## 12. Solid creation from Shell

# %%
vertexes = [
    (0.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (1.0, 1.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 1.0),
    (0.0, 1.0, 1.0),
]

# faces creation
faces = []
v_index = [
    (0, 1, 2, 3),
    (5, 4, 7, 6),
    (0, 4, 5, 1),
    (1, 5, 6, 2),
    (2, 6, 7, 3),
    (3, 7, 4, 0),
]
for ind, value in enumerate(v_index):
    wire = geo.tools.make_polygon(list(itemgetter(*value)(vertexes)), closed=True)
    faces.append(geo.face.BluemiraFace(wire, "face" + str(ind)))

# shell creation
shell = geo.shell.BluemiraShell(faces, "shell")

# solid creation from shell
solid = geo.solid.BluemiraSolid(shell, "solid")
print(solid)
show_cad(solid)
geo.tools.save_as_STEP([solid], "geo_cube")
