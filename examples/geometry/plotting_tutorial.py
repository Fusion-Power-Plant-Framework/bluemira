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

import bluemira.geometry.tools
from bluemira.base._matplotlib_plot import (
    PointsPlotter,
    WirePlotter,
    FacePlotter,
    FaceCompoundPlotter,
)
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.face import BluemiraFace

# creation of a closed wire and respective face
# PrincetonD parametrization is used as example.
# Note: the curve is generated into the xz plane
p = PrincetonD()
p.adjust_variable("x1", 4, lower_bound=3, upper_bound=5)
p.adjust_variable("x2", 16, lower_bound=10, upper_bound=20)
p.adjust_variable("dz", 0, lower_bound=0, upper_bound=0)
wire = p.create_shape()
face = BluemiraFace(wire)

# discretize the wire
points = p.create_array(n_points=10).T

# simple plot of the obtained points
# a PointsPlotter is created specifying size, edge and face colors.
# Note: 2D plot of points is always made on the first 2 coordinates. For this reason
# the plot is shown as a points cloud on a line
print("points plot")
pplotter = PointsPlotter(poptions={"s": 30, "facecolors": "red", "edgecolors": "black"})
pplotter.plot2d(points, show=True, block=True)

# inital test for 3d scatter plot
pplotter.plot3d(points, show=True, block=True)

# plot the wire
# a WirePlotter is used with the default setup with:
# - plane = xz (this is the projection plane, not a section plane)
# - point size = 10
# - ndiscr = 10
# - plot title
print("wire plot")
wplotter = WirePlotter(plane="xz")
wplotter.options._options['poptions']["s"] = 20
wplotter.options._options['ndiscr'] = 10
wplotter.plot2d(wire, show=False, block=True)
wplotter.ax.set_title(f"Wire plot, ndiscr: {wplotter.options._options['ndiscr']}")
wplotter.show_plot2d()


# inital test for 3d curve plot
wplotter.plot3d(wire, show=True, block=True)

# in this example poptions is set to an empty dict. The default matplotlib are used.
print("wire plot other options")
wplotter.options._options['poptions'] = {}
wplotter.plot2d(wire, show=True, block=True)
# The plot is immediately shown, so it is not possible to act on the plot setup
# e.g. following commands would not work
# wplotter.ax.set_title(f"Wire plot")
# wplotter.show_plot()

# Just disabling points plotting from now on modifying directly the DEFAULT
# dictionary. Not really a good pratice, but it is easy in this case.
bluemira.base._matplotlib_plot.DEFAULT["flag_points"] = False

# face plot
fplotter = FacePlotter(plane="xz")
fplotter.options._options['ndiscr'] = 30
fplotter.plot2d(face, show=False, block=True)
fplotter.ax.set_title("Face plot without points")
fplotter.show_plot2d()

# face plot - points enabled - just to check
fplotter = FacePlotter(plane="xz")
fplotter.options._options['ndiscr'] = 30
fplotter.options._options['flag_points'] = True
fplotter.plot2d(face, show=False, block=True)
fplotter.ax.set_title("Face plot with points")
fplotter.show_plot2d()

# a second geometry is created (it contains the first face)
p2 = PrincetonD()
p2.adjust_variable("x1", 3.5, lower_bound=3, upper_bound=5)
p2.adjust_variable("x2", 17, lower_bound=10, upper_bound=20)
p2.adjust_variable("dz", 0, lower_bound=0, upper_bound=0)
wire2 = p2.create_shape()
face2 = BluemiraFace(wire2)

# face and face2 are plotted using the same FacePlotter. Since no plot options have
# been changed, the two faces will be plotted in the same way (e.g. same color).
# Note: internal points are not plotted 'cause they are covered by fill plot.
fplotter2 = FacePlotter(plane="xz")
fplotter2.options._options['flag_points'] = True
fplotter2.options._options['foptions'] = {"color": "blue"}
fplotter2.plot2d(face, show=False, block=True)
fplotter2.plot2d(face2, ax=fplotter2.ax, show=False, block=True)
fplotter2.ax.set_title("Both faces in blue")
fplotter2.show_plot2d()
print(f"fplotter2.options: {fplotter2.options.asdict()}")

# plot both face with different color.
# Note: if face is plotte before face2, face2 will be "covered" by face.
fplotter2.options._options['foptions'] = {"color": "blue"}
fplotter2.plot2d(face2, show=False, block=True)
fplotter2.options._options['foptions'] = {"color": "green"}
fplotter2.plot2d(face, ax=fplotter2.ax, show=False, block=True)
fplotter2.ax.set_title("Both faces with different colors")
fplotter2.show_plot2d()

# a third face is create as difference between face and face2 (a BluemiraFace object
# has been created using wire2 as outer boundary and wire as inner boundary
# Note:
# - when plotting points, it can happen that markers are not centered properly as
#       described in https://github.com/matplotlib/matplotlib/issues/11836
# - face3 is created with a wire deepcopy in order to be able to modify face and face2
# (and thus wire and wire2) without modifying face3
face3 = BluemiraFace([wire2.deepcopy(), wire.deepcopy()])
fplotter3 = FacePlotter(plane="xz")
fplotter3.plot2d(face3)
fplotter3.ax.set_title("Face with hole - points enabled")
fplotter3.show_plot2d()

fplotter3.options._options['foptions']['color'] = "blue"
fplotter3.plot2d(face3, ax=None)
fplotter3.ax.set_title("Face with hole - points disabled - blue")
fplotter3.show_plot2d()

# some operations on face
bari = face.center_of_mass
face.scale(0.5)
new_bari = face.center_of_mass
diff = bari - new_bari
v = (diff[0], diff[1], diff[2])
face.translate(v)

# creation of a face compound plotter
# color test with palettes
cplotter = FaceCompoundPlotter(palette="Blues_r")
cplotter.set_plane('xz')
cplotter.plot2d([face3, face])
cplotter.ax.set_title("Compound plot - test in Blue_r")
cplotter.show_plot2d()

cplotter = FaceCompoundPlotter(plane="xz", palette="light:#105ba4")
cplotter.plot2d([face3, face])
cplotter.ax.set_title("Compound plot - test with single color light:#105ba4")
cplotter.show_plot2d()

points = [[0,0,0], [1,0,0], [1,0,3], [0,0,3]]
wire = bluemira.geometry.tools.make_polygon(points, closed=True)
wire1 = wire.deepcopy()
wire1.translate((3,0,5))
wplotter.plot2d(wire, show=False, block=False)
wplotter.ax.set_title("wire")
wplotter.show_plot2d()
wplotter.plot2d(wire1, show=False, block=False)
wplotter.ax.set_title("wire1")
wplotter.show_plot2d()

wface = BluemiraFace(wire)
w1face = BluemiraFace(wire1)
cplotter.plot2d([wface, w1face])
cplotter.ax.set_title("test faces")
cplotter.show_plot2d()

# plot of faces boundary. Note that, since poptions = {}, points color is
# automatically changed by matplotlib
wplotter.plot2d(wface.boundary[0])
print(f"test_boundary wplotter options: {wplotter.options.asdict()}")
wplotter.plot2d(w1face.boundary[0], ax=wplotter.ax)
print(f"test_boundary wplotter options: {wplotter.options.asdict()}")
wplotter.ax.set_title("test boundary from faces - matplotlib default poptions")
wplotter.show_plot2d()

# plot of faces boundary. Note that, since poptions = {}, points color is
# automatically changed by matplotlib
wplotter.options._options['woptions'] = {}
wplotter.plot2d(wface.boundary[0])
print(f"test_boundary wplotter options: {wplotter.options.asdict()}")
wplotter.plot2d(w1face.boundary[0], ax=wplotter.ax)
print(f"test_boundary wplotter options: {wplotter.options.asdict()}")
wplotter.ax.set_title("test boundary from faces - matplotlib default poptions and "
                      "woptions")
wplotter.show_plot2d()

# plot of a component
import matplotlib.pyplot as plt
from bluemira.base.components import PhysicalComponent, GroupingComponent
c = PhysicalComponent("Comp", face)
c._plotter2d.options._options['plane'] = 'xz'
c._plotter2d.options._options['ndiscr'] = 30
ax = c.plot2d(show=False)
ax.set_title("test component plot")
plt.gca().set_aspect("equal")
plt.show(block=True)

# plot of a group of components
bluemira.base._matplotlib_plot.DEFAULT["foptions"] = {}
bluemira.base._matplotlib_plot.DEFAULT["woptions"] = {}
group = GroupingComponent("Components")
c1 = PhysicalComponent("Comp1", face, parent=group)
c2 = PhysicalComponent("Comp2", wface, parent=group)
c3 = PhysicalComponent("Comp3", w1face, parent=group)
group.plot2d(show=True, block=True)

# combined plot of Componennt and BluemiraGeo instances
wplotter.options._options['woptions']['color'] = 'red'
ax = wplotter.plot2d(wface.boundary[0])
fplotter.options._options['foptions']['color'] = 'green'
fplotter.options._options['woptions']['color'] = 'black'
ax = fplotter.plot2d(w1face, ax=ax)
ax = c.plot2d(ax=ax)
ax.set_title("test component + bluemirageo plot")
plt.gca().set_aspect("equal")
plt.show(block=True)

# just a check that the options dict is modified correctly
print(wplotter.options.asdict())
print(fplotter.options.asdict())
print(c.plot2d_options.asdict())

#plot CAD
group.plotcad()
