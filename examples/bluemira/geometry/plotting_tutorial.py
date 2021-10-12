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

from bluemira.plotting.plotter import PointsPlotter, WirePlotter, FacePlotter
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane

p = PrincetonD()
p.adjust_variable("x1", 4, lower_bound=3, upper_bound=5)
p.adjust_variable("x2", 16, lower_bound=10, upper_bound=20)
p.adjust_variable("dz", 0, lower_bound=0, upper_bound=0)

wire = p.create_shape()
face = BluemiraFace(wire)

array = p.create_array(n_points=200)

pplotter = PointsPlotter(poptions={"s": 30, "facecolors": "red", "edgecolors": "black"})
pplotter.plot(array, show=True, block=True)

wplotter = WirePlotter()
wplotter.plot(wire, show=True, block=True)

wplotter.options["poptions"] = {}
wplotter.plot(wire, show=True, block=True)

# Todo: FacePlotter need adjustments. In 3D only the wires are showed. Not sure if it
#  is possible to "simply" show a 3D face in matplotlib
fplotter = FacePlotter()
fplotter.plot(face, show=True, block=True)

# make a plane
plane = BluemiraPlane([0,0,1], [0,1,0], 90)
print(plane.to_matrix())
print(plane.base)
print(plane.rotation)
