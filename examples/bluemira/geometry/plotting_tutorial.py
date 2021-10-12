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

from bluemira.plotting.plotter import PointsPlotter, WirePlotter, FacePlotter
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.face import BluemiraFace

p = PrincetonD()
p.adjust_variable("x1", 4, lower_bound=3, upper_bound=5)
p.adjust_variable("x2", 16, lower_bound=10, upper_bound=20)
p.adjust_variable("dz", 0, lower_bound=0, upper_bound=0)

wire = p.create_shape()
face = BluemiraFace(wire)

array = p.create_array(n_points=10)

# pplotter = PointsPlotter(poptions={"s": 30, "facecolors": "red", "edgecolors": "black"})
# pplotter.plot(array, show=True, block=True)
#
# wplotter = WirePlotter()
# wplotter.plot(wire, show=True, block=True)
#
# wplotter.options["poptions"] = {}
# wplotter.plot(wire, show=True, block=True)

# fplotter = FacePlotter(plane='xz')
# fplotter.options["plot_flag"]["poptions"] = False
# fplotter.plot(face, show=True, block=True)

p2 = PrincetonD()
p2.adjust_variable("x1", 3.5, lower_bound=3, upper_bound=5)
p2.adjust_variable("x2", 17, lower_bound=10, upper_bound=20)
p2.adjust_variable("dz", 0, lower_bound=0, upper_bound=0)

wire2 = p2.create_shape()
face2 = BluemiraFace(wire2)

fplotter2 = FacePlotter(plane='xz')
fplotter2.options["plot_flag"]["poptions"] = True
fplotter2.options["foptions"] = {"color": "blue"}
# ax = fplotter2.plot(face, show=True, block=True)


fplotter3 = FacePlotter(plane='xz')
fplotter3.plot_points = False
face3 = BluemiraFace([wire2, wire])
fplotter3.plot(face3, show=True, block=True, ndiscr=100)
