# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
Analytical expressions for the field inside an arbitrarily shaped winding pack
with arbitrarily shaped cross-section, following equations as described in:


"""
import numpy as np

from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import distance_to, make_polygon
from bluemira.magnetostatics.baseclass import (
    SourceGroup,
)
from bluemira.magnetostatics.trapezoidal_prism import TrapezoidalPrismCurrentSource


def polyhedral_discretised_sources(
    normal,
    trap_vec,
    wire,
    length,
    alpha,
    beta,
    current,
    rows,
):
    """
    Function to approximate a polyhedral current source using a
    set of discretised trapezoidal prism current sources.
    """
    points = wire.vertexes.T
    # create source group
    sources = SourceGroup([])
    # ensure vector parallel to trapezoidal is normalised
    trap_vec = trap_vec / np.linalg.norm(trap_vec)
    perp_vec = np.cross(normal, trap_vec)
    # ensure vector perpendicular to trapezoidal is normalised
    perp_vec = perp_vec / np.linalg.norm(perp_vec)
    theta_l = np.deg2rad(beta)
    theta_u = np.deg2rad(alpha)
    face = BluemiraFace(boundary=wire)
    j = current / face.area
    main_length = length
    vals1 = []
    vals2 = []
    for p in points:
        vals1 += [np.dot(p, trap_vec)]
        vals2 += [np.dot(p, perp_vec)]
    tmin = points[vals1.index(min(vals1)), :]
    tmax = points[vals1.index(max(vals1)), :]
    tdist = np.dot(tmax - tmin, trap_vec)
    offset = tdist / rows
    pmin = points[vals2.index(min(vals2)), :]
    pmax = points[vals2.index(max(vals2)), :]
    pdist = np.dot(pmax - pmin, perp_vec)

    for i in range(rows):
        vdist = i * offset + offset / 2
        cen = tmin + vdist * trap_vec
        up = cen + pdist * perp_vec
        low = cen - pdist * perp_vec
        x = np.array([low[0], up[0]])
        y = np.array([low[1], up[1]])
        z = np.array([low[2], up[2]])
        coords = Coordinates({"x": x, "y": y, "z": z})
        line = make_polygon(coords, closed=False)
        dist, vectors = distance_to(wire, line)
        if np.round(dist, 4) > 0:
            print("no intersect between line and wire")
        else:
            p1 = np.array(vectors[0][0])
            p2 = np.array(vectors[1][0])
            o = np.multiply(0.5, (p1 + p2))
            width = np.linalg.norm(p2 - p1)
            area = width * offset
            current = j * area
            dz_l = np.dot((o - tmin), trap_vec) * np.tan(theta_l)
            dz_u = np.dot((o - tmin), trap_vec) * np.tan(theta_u)
            length = main_length + dz_l + dz_u
            source = TrapezoidalPrismCurrentSource(
                o,
                length * normal,
                perp_vec,
                trap_vec,
                offset / 2,
                width / 2,
                theta_u,
                theta_l,
                current,
            )
            sources.add_to_group([source])

    return sources
