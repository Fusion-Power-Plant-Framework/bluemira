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
Created on Fri Aug  2 13:08:34 2019

@author: matti
"""
import os
import pickle  # noqa :S403

import matplotlib.pyplot as plt
import numpy as np
import pytest

import tests
from bluemira.base.file import get_bluemira_path
from bluemira.geometry.error import GeometryError
from BLUEPRINT.geometry.boolean import (
    boolean_2d_common,
    boolean_2d_common_loop,
    boolean_2d_difference,
    boolean_2d_difference_loop,
    boolean_2d_union,
    boolean_2d_xor,
    convex_hull,
    entagram,
    simplify_loop,
)
from BLUEPRINT.geometry.geomtools import make_box_xz
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.shell import Shell


class TestBooleanSimple:
    @classmethod
    def setup_class(cls):
        loop = Loop(*entagram(1))
        loop2 = Loop([-0.5, 0.5, 0.5, -0.5, -0.5], [-0.5, -0.5, 0.5, 0.5, -0.5])
        cls.loop = loop
        cls.loop2 = loop2

    @pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
    @pytest.mark.longrun
    def test_ops(self):
        f, ax = plt.subplots(2, 5)
        self.loop.plot(ax[0, 0], facecolor="b")
        self.loop2.plot(ax[0, 0], facecolor="r")
        common = boolean_2d_common(self.loop, self.loop2)
        for c in common:
            c.plot(ax[0, 1])
        ax[0, 1].set_title("Boolean common")
        union = boolean_2d_union(self.loop, self.loop2)[0]
        union.plot(ax[1, 1])
        ax[1, 1].set_title("Boolean union")
        difference = boolean_2d_difference(self.loop, self.loop2)
        for d in difference:
            d.plot(ax[0, 2])
        ax[0, 2].set_title("Boolean difference P-S")
        difference = boolean_2d_difference(self.loop2, self.loop)
        for d in difference:
            d.plot(ax[1, 2])
        ax[1, 2].set_title("Boolean difference S-P")

        difference = boolean_2d_xor(self.loop, self.loop2)
        for d in difference:
            d.plot(ax[0, 4])
        ax[0, 4].set_title("Boolean XOR mine P, S")
        difference = boolean_2d_xor(self.loop2, self.loop)
        for d in difference:
            d.plot(ax[1, 4])
        ax[1, 4].set_title("Boolean XOR mine S, P")

    def test_simplify(self):
        loop = Loop([0, 2, 2, 2, 2.2, 2.2, 2, 0, 0], [0, 0, 2, 2.2, 2.2, 2, 2, 2, 0])
        loops = simplify_loop(loop)
        assert loops.area == 4
        loop = Loop([0, 2, 2, 2, 2.2, 2.2, 0, 0], [0, 0, 2, 2.2, 2.2, 2, 2, 0])
        loops = simplify_loop(loop)
        assert loops.area == 4
        # Twisted polygon will not work!


# =============================================================================
#         S = Loop([0, 2, 2, 2, 2.2, 2.2, 2, 0, 0],
#                  [0, 0, 2, 2, 2.2, 2.2, 2, 2, 0])
#         loops = simplify_loop(S)
#         self.assertTrue(loops[0].A == 4)
# =============================================================================


@pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
class TestBooleanOpen:
    @pytest.mark.longrun
    def test_openclip(self):
        fp = get_bluemira_path("BLUEPRINT/geometry/test_data", subfolder="tests")
        fn = os.sep.join([fp, "loopcut_data.pkl"])
        with open(fn, "rb") as f:
            data = pickle.load(f)  # noqa :S301
        loop1 = Loop(data[0][0], data[0][1])  # Closed
        loop2 = Loop(data[0][2], data[0][3])  # Open
        result = boolean_2d_difference(loop2, loop1)

        f, ax = plt.subplots()
        loop1.plot(ax, edgecolor="r", fill=False)
        loop2.plot(ax, edgecolor="b", fill=False)
        result[0].plot(ax, edgecolor="g", fill=False)
        assert result[0].closed is False


class TestBooleanShell:
    @classmethod
    def setup_class(cls):
        square = Loop(x=[4, 6, 6, 4, 4], z=[0, 0, 2, 2, 0])
        square2 = square.translate([-1, 0, 0], update=False)
        square3 = square2.offset(0.5)
        cls.shell = Shell(square2, square3)
        cls.loop = square
        cls.shell2 = Shell(square, square.offset(0.2))

    def test_common(self):

        result = boolean_2d_common(self.shell, self.loop)[0]
        result2 = boolean_2d_common(self.loop, self.shell)[0]
        result.sort_bottom()  # Need to sort to compare for equality
        result2.sort_bottom()
        assert result == result2
        assert (
            result.xyz.all()
            == np.array(
                [
                    [5.0, 5.5, 5.5, 5.0, 5.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 2.0, 2.0, 0.0],
                ]
            ).all()
        )

    def test_union(self):
        result = boolean_2d_union(self.shell, self.loop)[0]
        result2 = boolean_2d_union(self.loop, self.shell)[0]
        result.sort_bottom()
        result2.sort_bottom()
        assert result == result2
        assert (
            result.xyz.all()
            == np.array(
                [
                    [6.0, 5.5, 5.5, 2.5, 2.5, 5.5, 5.5, 6.0, 6.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [2.0, 2.0, 2.5, 2.5, -0.5, -0.5, 0.0, 0.0, 2.0],
                ]
            ).all()
        )

    def test_difference(self):
        result = boolean_2d_difference(self.shell, self.loop)[0]
        result.sort_bottom()
        assert (
            result.xyz.all()
            == np.array(
                [
                    [2.5, 5.5, 5.5, 3.0, 3.0, 5.5, 5.5, 2.5, 2.5],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-0.5, -0.5, 0.0, 0.0, 2.0, 2.0, 2.5, 2.5, -0.5],
                ]
            ).all()
        )

        result2 = boolean_2d_difference(self.loop, self.shell)
        assert len(result2) == 2

    def test_common_shellshell(self):
        result = boolean_2d_common(self.shell, self.shell2)
        result2 = boolean_2d_common(self.shell2, self.shell)

        assert len(result2) == len(result)
        for r1, r2 in zip(result, result2):
            r1.sort_bottom()
            r2.sort_bottom()
            assert r1 == r2

    def test_union_shellshell(self):
        # Warning: this behaviour is not yet used anywhere, and probably isn't
        # what you may be expecting. Need to treat PyPolyTrees better, but this
        # is non-trivial and would be (at present) quite useless.
        result = boolean_2d_union(self.shell, self.shell2)[0]
        result2 = boolean_2d_union(self.shell2, self.shell)[0]

        assert len(result2) == len(result)
        assert result == result2

    def test_difference_shellshell(self):
        result = boolean_2d_difference(self.shell, self.shell2)[0]
        result2 = boolean_2d_difference(self.shell2, self.shell)[0]

        assert result != result2


@pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
class TestCleanloop:
    @pytest.mark.longrun
    def test_hanging_chad(self):
        fp = get_bluemira_path("BLUEPRINT/geometry/test_data", subfolder="tests")
        fn = os.sep.join([fp, "classic_hanging_chad.json"])
        loop = Loop.from_file(fn)
        f, ax = plt.subplots()
        loop.plot(ax)
        simplified_loop = simplify_loop(loop)
        simplified_loop.plot(ax, facecolor="r")
        ax.set_xlim([7, 7.5])
        ax.set_ylim([-5.75, -6])


class TestConvexHull:
    def test_squares(self):
        loop1 = Loop(x=[0, 1, 1, 0, 0], z=[0, 0, 1, 1, 0])
        loop2 = Loop(
            x=[-0.14644661, 0.5, 1.14644661, 0.5, -0.14644661],
            z=[0.5, -0.14644661, 0.5, 1.14644661, 0.5],
        )
        hull = convex_hull([loop1, loop2])

        if tests.PLOTTING:
            f, ax = plt.subplots()
            loop1.plot(ax)
            loop2.plot(ax)
            hull.plot(ax, edgecolor="r", fill=False)
            plt.show()

        assert hull.closed

        assert np.allclose(
            hull.d2,
            np.array(
                [
                    [-0.14644661, 0.0, 0.5, 1.0, 1.14644661, 1.0, 0.5, 0.0, -0.14644661],
                    [0.5, 0.0, -0.14644661, 0.0, 0.5, 1.0, 1.14644661, 1.0, 0.5],
                ]
            ),
        )

    def test_star(self):
        x, z = entagram(4, 8, 3)
        star1 = Loop(x, z)
        x, z = entagram(4, 10, 4)
        star2 = Loop(x, z)
        hull = convex_hull([star1, star2])

        if tests.PLOTTING:
            f, ax = plt.subplots()
            star1.plot(ax)
            star2.plot(ax)
            hull.plot(ax, edgecolor="r", fill=False)
            plt.show()

        assert hull.closed
        assert np.allclose(
            hull.d2,
            np.array(
                [
                    [
                        -4.00000000e00,
                        -3.80422607e00,
                        -2.82842712e00,
                        -2.35114101e00,
                        -7.34788079e-16,
                        2.35114101e00,
                        2.82842712e00,
                        3.80422607e00,
                        4.00000000e00,
                        3.80422607e00,
                        2.82842712e00,
                        2.35114101e00,
                        2.44929360e-16,
                        -2.35114101e00,
                        -2.82842712e00,
                        -3.80422607e00,
                        -4.00000000e00,
                    ],
                    [
                        4.89858720e-16,
                        -1.23606798e00,
                        -2.82842712e00,
                        -3.23606798e00,
                        -4.00000000e00,
                        -3.23606798e00,
                        -2.82842712e00,
                        -1.23606798e00,
                        -9.79717439e-16,
                        1.23606798e00,
                        2.82842712e00,
                        3.23606798e00,
                        4.00000000e00,
                        3.23606798e00,
                        2.82842712e00,
                        1.23606798e00,
                        4.89858720e-16,
                    ],
                ]
            ),
        )

    def test_error(self):
        """
        Different planar dimensions error check
        """
        loop1 = Loop(x=[0, 1, 1, 0, 0], z=[0, 0, 1, 1, 0])
        loop2 = Loop(
            x=[-0.14644661, 0.5, 1.14644661, 0.5, -0.14644661],
            y=[0.5, -0.14644661, 0.5, 1.14644661, 0.5],
        )
        with pytest.raises(GeometryError):
            convex_hull([loop1, loop2])


def test_single_common_loop():
    box_1 = make_box_xz(0, 2, 0, 1)
    box_2 = make_box_xz(1, 3, 0, 1)
    box_3 = boolean_2d_common_loop(box_1, box_2)
    assert isinstance(box_3, Loop)
    assert box_3.area == 1.0

    # Non-overlapping case
    box_1 = make_box_xz(0, 1, 0, 1)
    box_2 = make_box_xz(2, 3, 0, 1)
    with pytest.raises(GeometryError):
        box_3 = boolean_2d_common_loop(box_1, box_2)

    # Multiple common case
    x = [0, 0, 1, 1, 2, 2, 3, 3]
    z = [0, 2, 2, 1, 1, 2, 2, 0]
    u_shape = Loop(x=x, z=z)
    u_shape.close()
    box = make_box_xz(0, 3, 1, 2)
    with pytest.raises(GeometryError):
        common = boolean_2d_common_loop(u_shape, box)


def test_single_difference_loop():

    box_1 = make_box_xz(0, 2, 0, 1)
    box_2 = make_box_xz(1, 2, 0, 1)
    box_3 = boolean_2d_common_loop(box_1, box_2)
    assert isinstance(box_3, Loop)
    assert box_3.area == 1.0

    # Multiple common case
    x = [0, 0, 1, 1, 2, 2, 3, 3]
    z = [0, 2, 2, 1, 1, 2, 2, 0]
    u_shape = Loop(x=x, z=z)
    u_shape.close()
    box = make_box_xz(-1, 4, -1, 2)
    with pytest.raises(GeometryError):
        common = boolean_2d_difference_loop(u_shape, box)

    # Fully-subtracted case
    box = make_box_xz(0, 1, 0, 1)
    with pytest.raises(GeometryError):
        common = boolean_2d_difference_loop(box, box)
