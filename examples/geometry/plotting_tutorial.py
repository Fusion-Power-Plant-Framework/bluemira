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

from bluemira.plotting.plotter import (
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
points = p.create_array(n_points=10)

# simple plot of the obtained points
# a PointsPlotter is created specifying size, edge and face colors.
# Note: 2D plot of points is always made on the first 2 coordinates. For this reason
# the plot is shown as a points cloud on a line
pplotter = PointsPlotter(poptions={"s": 30, "facecolors": "red", "edgecolors": "black"})
pplotter(points, show=True, block=True)

# plot the wire
# a WirePlotter is used with the default setup with:
# - plane = xz (this is the projection plane, not a section plane)
# - point size = 10
# - ndiscr = 10
# - plot title
wplotter = WirePlotter(plane="xz")
wplotter.change_poptions(("s", 10))
ndiscr = 10
wplotter(wire, show=False, block=True, ndiscr=ndiscr, byedges=True)
wplotter.ax.set_title(f"Wire plot, ndiscr: {ndiscr}")
wplotter.show_plot()

# in this exaple points plot is disabled.
wplotter.change_poptions({})
wplotter(wire, show=True, block=True, ndiscr=10, byedges=True)
# The plot is immediately shown, so it is not possible to act on the plot setup
# e.g. following commands would not work
# wplotter.ax.set_title(f"Wire plot, ndiscr: {ndiscr}")
# wplotter.show_plot()

# face plot
fplotter = FacePlotter(plane="xz")
fplotter.plot_points = False
fplotter(face, show=False, block=True, ndiscr=10, byedges=True)
fplotter.ax.set_title("Face plot")
fplotter.show_plot()

# a second geometry is created (it contains the first face)
p2 = PrincetonD()
p2.adjust_variable("x1", 3.5, lower_bound=3, upper_bound=5)
p2.adjust_variable("x2", 17, lower_bound=10, upper_bound=20)
p2.adjust_variable("dz", 0, lower_bound=0, upper_bound=0)
wire2 = p2.create_shape()
face2 = BluemiraFace(wire2)

# face and face2 are plotted using the same FacePlotter. Since no plot options have
# been changed, the two faces will be plotted in the same way (e.g. same color).
fplotter2 = FacePlotter(plane="xz")
fplotter2.plot_points = True
fplotter2.options["foptions"] = {"color": "blue"}
fplotter2(face, show=False, block=True, ndiscr=100, byedges=True)
fplotter2(face2, ax=fplotter2.ax, show=False, block=True, ndiscr=100, byedges=True)
fplotter2.ax.set_title("Both faces in blue")
fplotter2.show_plot()

# plot both face with different color.
# Note: if face is plotte before face2, face2 will be "covered" by face.
fplotter2.options["foptions"] = {"color": "blue"}
fplotter2(face2, show=False, block=True, ndiscr=100, byedges=True)
fplotter2.options["foptions"] = {"color": "green"}
fplotter2(face, ax=fplotter2.ax, show=False, block=True, ndiscr=100, byedges=True)
fplotter2.ax.set_title("Both faces with different colors")
fplotter2.show_plot()

# a third face is create as difference between face and face2 (a BluemiraFace object
# has been created using wire2 as outer boundary and wire as inner boundary
# Note:
# - when plotting points, it can happen that markers are not centered properly as
#       described in https://github.com/matplotlib/matplotlib/issues/11836
# - face3 is created with a wire deepcopy in order to be able to modify face and face2
# (and thus wire and wire2) without modifying face3
face3 = BluemiraFace([wire2.deepcopy(), wire.deepcopy()])
fplotter3 = FacePlotter(plane="xz")
fplotter3.plot_points = True
fplotter3(face3, ndiscr=100, byedges=True)
fplotter3.ax.set_title("Face with hole - points enabled")
fplotter3.show_plot()

fplotter3.plot_points = False
fplotter3.change_foptions(("color", "blue"))
fplotter3(face3, ax=None, ndiscr=100, byedges=True)
fplotter3.ax.set_title("Face with hole - points disabled - blue")
fplotter3.show_plot()

# some operations on face
bari = face.center_of_mass
face.scale(0.5)
new_bari = face.center_of_mass
diff = bari - new_bari
v = (diff[0], diff[1], diff[2])
face.translate(v)

# creation of a face compound plotter
# color test with palettes
cplotter = FaceCompoundPlotter(plane="xz", palette="Blues_r")
cplotter.plot_points = False
cplotter([face3, face], ndiscr=100, byedges=True)
cplotter.ax.set_title("Compound plot - test in Blue_r")
cplotter.show_plot()

cplotter = FaceCompoundPlotter(plane="xz", palette="light:#105ba4")
cplotter.plot_points = False
cplotter([face3, face], ndiscr=100, byedges=True)
cplotter.ax.set_title("Compound plot - test with single color light:#105ba4")
cplotter.show_plot()
