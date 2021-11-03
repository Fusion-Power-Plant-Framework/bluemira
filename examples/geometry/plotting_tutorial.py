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

# %%
import matplotlib.pyplot as plt

import bluemira.geometry.tools
from bluemira.base.components import PhysicalComponent, GroupingComponent
import bluemira.display as display
from bluemira.display._matplotlib_plot import (
    PointsPlotter,
    WirePlotter,
    FacePlotter,
)
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.face import BluemiraFace

# %%[markdown]
# ## Setup
#
# Creation of a closed wire and respective face
#
# PrincetonD parametrisation is used as example.
#
# Note: the curve is generated on the x-z plane

# %%
p = PrincetonD()
p.adjust_variable("x1", 4, lower_bound=3, upper_bound=5)
p.adjust_variable("x2", 16, lower_bound=10, upper_bound=20)
p.adjust_variable("dz", 0, lower_bound=0, upper_bound=0)
wire = p.create_shape()
face = BluemiraFace(wire)


# %%[markdown]
# ## Default plotting
#
# We can display the BluemiraWire and BluemiraFace in the following way, using the
# default settings.

# %%
display.plot_2d(wire)
display.plot_3d(wire)
display.show_cad(face)


# %%[markdown]
#
# Discretise the wire to an array of points.

# %%
points = p.create_array(n_points=10).T

# %%[markdown]
# ## Points Plot
#
# Simple plot of the obtained points.
#
# A PointsPlotter is created specifying size, edge and face colors.
#
# Note: 2D plot of points is always made on the first 2 coordinates. For this reason
# the plot is shown as a cloud of points on a line

# %%
pplotter = PointsPlotter(poptions={"s": 30, "facecolors": "red", "edgecolors": "black"})
pplotter.plot_2d(points)

# %%[markdown]
# ## 3D Scatter Plot
#
# A plot of the same points, but in 3D this time.

# %%
pplotter.plot_3d(points)

# %%[markdown]
# ## Wire Plot
#
# A WirePlotter is used with the default setup with:
#
# - plane = xz (this is the projection plane, not a section plane)
# - point size = 10
# - ndiscr = 10
# - plot title

# %%
wplotter = WirePlotter(plane="xz")
wplotter.options.poptions["s"] = 20
wplotter.options.ndiscr = 5
wplotter.plot_2d(wire)


# %%[markdown]
# ## 3D Curve Plot
#
# A plot of the same wire, but in 3D this time.

# %%
wplotter.plot_3d(wire)

# %%[markdown]
# ## Wire Plot with Matplotlib Default Options
#
# In this example poptions is set to an empty dict. The default matplotlib are used.

# %%
wplotter.options.poptions = {}
wplotter.plot_2d(wire)
# The plot is immediately shown by default, so it is not possible to act on the plot
# setup e.g. following commands would not work
# wplotter.ax.set_title(f"Wire plot")
# wplotter.show_plot()

# %%[markdown]
# ## Wire plot with some modifications
#
# In this example, we disable the automatic display of the plot (show=False), and apply
# a title to the plot

# %%
wplotter.options.poptions = {}
wplotter.plot_2d(wire, show=False)
wplotter.ax.set_title(f"Wire plot")
wplotter.show_plot_2d()

# %%[markdown]
# ## Face Plot
#
# A FacePlotter is used with the default setup with:
#
# - plane = xz (this is the projection plane, not a section plane)
# - ndiscr = 30
# - plot title

# %%
fplotter = FacePlotter(plane="xz")
fplotter.options.ndiscr = 30
fplotter.plot_2d(face, show=False)
fplotter.ax.set_title("Face plot without points")
fplotter.show_plot_2d()

# %%[markdown]
# ## Face Plot with Points Enabled
#
# We've set the points to be disabled by default, but we can activate them again for
# individual plotters.

# %%
fplotter = FacePlotter(plane="xz")
fplotter.options.ndiscr = 30
fplotter.options.show_points = True
fplotter.plot_2d(face, show=False)
fplotter.ax.set_title("Face plot with points")
fplotter.show_plot_2d()

# %%[markdown]
# ## Make a Second Face
#
# A second geometry is created, surrounding our original face.

# %%
p2 = PrincetonD()
p2.adjust_variable("x1", 3.5, lower_bound=3, upper_bound=5)
p2.adjust_variable("x2", 17, lower_bound=10, upper_bound=20)
p2.adjust_variable("dz", 0, lower_bound=0, upper_bound=0)
wire2 = p2.create_shape()
face2 = BluemiraFace(wire2)

# %%[markdown]
# ## Combined Face Plot
#
# Face and face2 are plotted using the same FacePlotter. Since no plot options have
# been changed, the two faces will be plotted in the same way (e.g. same color).

# %%
fplotter2 = FacePlotter(plane="xz")
fplotter2.options.show_points = True
fplotter2.options.foptions = {"color": "blue"}
fplotter2.plot_2d(face, show=False)
fplotter2.plot_2d(face2, ax=fplotter2.ax, show=False)
fplotter2.ax.set_title("Both faces in blue")
fplotter2.show_plot_2d()
print(f"fplotter2.options: {fplotter2.options.as_dict()}")

# %%[markdown]
# ## Combined Face Plot with Different Colours
#
# Plot both face with different colour.
#
# Note: if face is plotted before face2, face2 will be "covered" by face.

# %%
fplotter2.options.foptions = {"color": "blue"}
fplotter2.plot_2d(face2, show=False)
fplotter2.options.foptions = {"color": "green"}
fplotter2.plot_2d(face, ax=fplotter2.ax, show=False)
fplotter2.ax.set_title("Both faces with different colors")
fplotter2.show_plot_2d()

# %%[markdown]
# ## Face with Hole
#
# A third face is create as difference between face and face2 (a BluemiraFace object
# has been created using wire2 as outer boundary and wire as inner boundary).
#
# Note:
# - when plotting points, it can happen that markers are not centred properly as
#       described in https://github.com/matplotlib/matplotlib/issues/11836
# - face3 is created with a wire deepcopy in order to be able to modify face and face2
# (and thus wire and wire2) without modifying face3

# %%
face3 = BluemiraFace([wire2.deepcopy(), wire.deepcopy()])
fplotter3 = FacePlotter(plane="xz")
fplotter3.options.show_points = True
fplotter3.plot_2d(face3, show=False)
fplotter3.ax.set_title("Face with hole - points enabled")
fplotter3.show_plot_2d()

fplotter3.options.foptions["color"] = "blue"
fplotter3.options.show_points = False
fplotter3.plot_2d(face3, show=False, ax=None)
fplotter3.ax.set_title("Face with hole - points disabled - blue")
fplotter3.show_plot_2d()

# %%[markdown]
# ## Perform Some Face Operations
#
# Scale and move our face

# %%
bari = face.center_of_mass
face.scale(0.5)
new_bari = face.center_of_mass
diff = bari - new_bari
v = (diff[0], diff[1], diff[2])
face.translate(v)


# %%[markdown]
# ## Wires and Faces
#
# Create and plot a couple of Wires and then create and plot the corresponding Faces.

# %%
points = [[0, 0, 0], [1, 0, 0], [1, 0, 3], [0, 0, 3]]
wire = bluemira.geometry.tools.make_polygon(points, closed=True)
wire1 = wire.deepcopy()
wire1.translate((3, 0, 5))
wplotter.plot_2d(wire, show=False)
wplotter.ax.set_title("wire")
# wplotter.show_plot_2d()

wplotter.plot_2d(wire1, show=False)
wplotter.ax.set_title("wire1")
wplotter.show_plot_2d()


# %%[markdown]
# ## Plots with Matplotlib Default Point Options
#
# Plot the points on a boundary of a face with matplotlib defaults.
#
# Note that, since poptions = {}, points color is automatically changed by matplotlib.

# %%
wface = BluemiraFace(wire)
w1face = BluemiraFace(wire1)
wplotter.plot_2d(wface.boundary[0])
print(f"test_boundary wplotter options: {wplotter.options.as_dict()}")
wplotter.plot_2d(w1face.boundary[0], ax=wplotter.ax)
print(f"test_boundary wplotter options: {wplotter.options.as_dict()}")
wplotter.ax.set_title("test boundary from faces - matplotlib default poptions")
wplotter.show_plot_2d()

# %%[markdown]
# ## Plot with Matplotlib Default Wire Options
#
# Plot the boundary of a face with matplotlib defaults.
#
# Note that, since woptions = {}, wire color is automatically changed by matplotlib

# %%
wplotter.options.woptions = {}
wplotter.plot_2d(wface.boundary[0])
print(f"test_boundary wplotter options: {wplotter.options.as_dict()}")
wplotter.plot_2d(w1face.boundary[0], ax=wplotter.ax)
print(f"test_boundary wplotter options: {wplotter.options.as_dict()}")
wplotter.ax.set_title(
    "test boundary from faces - matplotlib default poptions and " "woptions"
)
wplotter.show_plot_2d()

# %%[markdown]
# ## PhysicalComponent Plot
#
# Creates a `PhysicalComponent` and plots it in the xz plane

# %%
c = PhysicalComponent("Comp", face)
c.plot_2d_options.plane = "xz"
c.plot_2d_options.ndiscr = 30
ax = c.plot_2d(show=False)
ax.set_title("test component plot")
plt.gca().set_aspect("equal")
plt.show(block=True)

# %%[markdown]
# ## GroupingComponent Plot
#
# Creates a `GroupingComponent` and plots it in the xz plane using matplotlib defaults.

# %%
bluemira.display._matplotlib_plot.DEFAULT["foptions"] = {}
bluemira.display._matplotlib_plot.DEFAULT["woptions"] = {}
group = GroupingComponent("Components")
c1 = PhysicalComponent("Comp1", face, parent=group)
c2 = PhysicalComponent("Comp2", wface, parent=group)
c3 = PhysicalComponent("Comp3", w1face, parent=group)
group.plot_2d()

# %%[markdown]
# ## Component and BluemiraGeo Combined Plot
#
# Plots a component on the same axes as a BluemiraFace.

# %%
wplotter.options.woptions["color"] = "red"
ax = wplotter.plot_2d(wface.boundary[0])
fplotter.options.foptions["color"] = "green"
fplotter.options.woptions["color"] = "black"
ax = fplotter.plot_2d(w1face, ax=ax)
ax = c.plot_2d(ax=ax)
ax.set_title("test component + bluemirageo plot")
plt.gca().set_aspect("equal")
plt.show(block=True)

# %%[markdown]
# Show the options from our combined plot

# %%
print(f"wire plotter options: {wplotter.options.as_dict()}")
print(f"face plotter options: {fplotter.options.as_dict()}")
print(f"component plotter options: {c.plot_2d_options.as_dict()}")

# %%[markdown]
# ## CAD Display
#
# Displays a GroupingComponent in a bluemira display window.

# %%
group.show_cad()
