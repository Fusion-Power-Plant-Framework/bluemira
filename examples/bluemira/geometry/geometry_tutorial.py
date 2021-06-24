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

import bluemira.geometry as geo

# Note: this tutorial must to be translated into a set of pytests

if __name__ == "__main__":
    print("This is a simple tutorial for the geometric module")

    print("1. Creation of a closed wire")
    pntslist = [(1., 1., 0.), (0., 1., 0.), (0., 0., 0.), (1., 0., 0.)]
    wire = geo.freecadapi.make_polygon(pntslist, closed=True)
    print("wire: {}, length: {}, isClosed: {}".format(wire, wire.Length,
                                                      wire.isClosed()))
    print("2. Creation of a bluemira wire")
    bmwire = geo.wire.BluemiraWire(wire, "bmwire")
    print(bmwire)

    print("3. Make some operations on bluemira wire")
    ndiscr = 10
    print("3.1 Discretize in {} points".format(ndiscr))
    points = bmwire.discretize(ndiscr)
    print(points)
    print("3.2 Discretize considering the edges")
    points = bmwire.discretize(ndiscr, byedges=True)
    print(points)

    print("4. Creation of a bluemira face")
    bmface = geo.face.BluemiraFace(bmwire, "bmface")
    print(bmface)

    print("5. Test of scale function.")
    print("Note: scale function modifies the original object")
    print("5.1 Scale a BluemiraWire")
    print("Original object: {}".format(bmwire))
    bmwire.scale(2)
    print("Scaled object: {}".format(bmwire))
    print("NOTE: since bmface is connected to bmwire, a scale operation on bmwire will"
          " affect also bmface")
    print("5.1 Scale a BluemiraFace")
    print("Original object: {}".format(bmface))
    bmface.scale(2)
    print("Scaled object: {}".format(bmface))

    print("6. Test Save as STEP file.")
    shapes = [bmwire._shape, bmface._shape]
    print(shapes)
    geo.freecadapi.save_as_STEP(shapes)

    print("7. Test BluemiraWire.close")
    print("7.1 when boundary is list(Part.Wire)")
    pntslist = [(1., 1., 0.), (0., 1., 0.), (0., 0., 0.), (1., 0., 0.)]
    wire = geo.freecadapi.make_polygon(pntslist, closed=False)
    bmwire_nc = geo.wire.BluemiraWire(wire)
    print(bmwire_nc)
    bmwire_nc.close()
    print(bmwire_nc)
    print(bmwire_nc.boundary)

    print("7.2 when boundary is list(BluemiraWire)")
    bmwire_nc = geo.wire.BluemiraWire(geo.wire.BluemiraWire(wire))
    print(bmwire_nc)
    bmwire_nc.close()
    print(bmwire_nc)
    print(bmwire_nc.boundary)

    print("7. Test Translate")

