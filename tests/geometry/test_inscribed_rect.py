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
import matplotlib.pyplot as plt
import numpy as np
import pytest

import tests
from bluemira.geometry._deprecated_loop import Loop
from bluemira.geometry._deprecated_tools import make_circle_arc
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.inscribed_rect import _rect, inscribed_rect_in_poly
from bluemira.geometry.tools import boolean_cut, make_circle, make_polygon


class TestInscribedRectangle:
    square = Loop(x=[2, 4, 4, 2, 2], z=[0, 0, 2, 2, 0])
    diamond = Loop(x=[6, 8, 10, 8, 6], z=[8, 6, 8, 10, 8])
    circle_xz = make_circle_arc(2, 4, -4)
    circle = Loop(x=circle_xz[0], z=circle_xz[1])
    circle_xz_offset = make_circle_arc(0.6, 5, -5)
    circle_sm = Loop(x=circle_xz_offset[0], z=circle_xz_offset[1])
    complex_shape = BluemiraFace(make_circle(2, center=(4, 0, -4), axis=(0, 1, 0)))
    circle_sm = BluemiraFace(make_circle(0.6, center=(5, 0, -5), axis=(0, 1, 0)))
    for i in [(0, 0, 0), (-2, 0, 2), (-2, 0, 0), (0, 0, 2)]:
        c_s = circle_sm.deepcopy()
        c_s.translate(i)
        complex_shape = boolean_cut(complex_shape, c_s)[0]
    # Convert back to Loop
    complex_shape = Loop(
        *complex_shape.boundary[0].discretize(byedges=True, ndiscr=100).xz
    )

    shapes = [square, diamond, circle, complex_shape]
    convex = [True, True, True, False]

    aspectratios = np.logspace(-1, 1, num=5)

    @pytest.mark.parametrize("shape, convex", zip(shapes, convex))
    def test_inscribed_rectangle(self, shape, convex):
        x = y = 5
        self.r = False
        # Random points in a rectangular grid of the shape
        points = np.random.random((2, x, y))
        points[0] *= np.ptp(shape.x)
        points[1] *= np.ptp(shape.z)
        points[0] += np.min(shape.x)
        points[1] += np.min(shape.z)

        if tests.PLOTTING:
            f, ax = plt.subplots()
            shape.plot(ax, linewidth=0.1)

        shape_face = BluemiraFace(make_polygon(shape.xyz))
        for i in range(x):
            for j in range(y):
                point = points[:, i, j]
                if shape.point_inside(point, include_edges=False):
                    for k in self.aspectratios:
                        dx, dz = inscribed_rect_in_poly(
                            shape.x,
                            shape.z,
                            point[0],
                            point[1],
                            aspectratio=k,
                            convex=convex,
                        )
                        sq = _rect(point[0], point[1], dx, dz)
                        assert len(sq.x) == 5
                        try:
                            tf = boolean_cut(
                                BluemiraFace(make_polygon(sq.xyz)), shape_face
                            )
                            tf = [
                                Loop(*seg.discretize(byedges=True, ndiscr=50).xz)
                                for seg in tf
                            ]
                        except ValueError:
                            tf = None

                        if tests.PLOTTING:
                            ax.plot(*point, marker="o")
                            sq.plot(ax, linewidth=0.1)

                        if tf is not None:
                            # Some overlaps are points or lines of 0 area
                            if not all([t.area == 0.0 for t in tf]):
                                self.assertion_error_creator(
                                    "Overlap", [dx, dz, point, k, convex]
                                )
                                if tests.PLOTTING:
                                    for t in tf:
                                        t.plot(
                                            ax,
                                            facecolor="r",
                                            edgecolor="r",
                                            linewidth=1,
                                            zorder=40,
                                        )
                        if not np.allclose(dx / dz, k):
                            self.assertion_error_creator("Aspect", [dx, dz, dx / dz, k])

        if tests.PLOTTING:
            plt.show()
        if self.r is not False:
            raise AssertionError(self.r)

    def assertion_error_creator(self, name, list_obj):
        if self.r is False:
            self.r = {name: [list_obj]}
        else:
            if name not in self.r:
                self.r = {name: []}
            self.r[name].append(list_obj)

    def test_zero(self):
        assert inscribed_rect_in_poly(self.diamond.x, self.diamond.z, 0, 0) == (0, 0)
