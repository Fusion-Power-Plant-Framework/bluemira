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

import os
import pytest
import numpy as np
import pickle  # noqa (S403)
import matplotlib.pyplot as plt
from BLUEPRINT.base.file import get_BP_path
from BLUEPRINT.base.error import BeamsError
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.shell import Shell
from BLUEPRINT.beams.material import SS316, Concrete
from BLUEPRINT.beams.crosssection import (
    IBeam,
    RectangularBeam,
    CircularBeam,
    CircularHollowBeam,
    CustomCrossSection,
    RapidCustomCrossSection,
    CompositeCrossSection,
    AnalyticalShellComposite,
    MultiCrossSection,
)
import tests


SS316 = SS316()
CONCRETE = Concrete()


class TestIbeam:
    def test_ibeam(self):
        # https://structx.com/Shape_Formulas_013.html
        i_beam = IBeam(1, 1, 0.25, 0.5)
        assert i_beam.area == 750000e-6
        assert i_beam.i_yy == 78125000000e-12
        assert i_beam.i_zz == 46875000000e-12
        assert np.isclose(i_beam.j, 114583333333.3333e-12)

    @pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
    def test_plot(self):
        i_beam = IBeam(1, 1, 0.25, 0.5)
        i_beam.geometry.plot(points=True)

    def test_errors(self):
        props = [
            [1, 1, 0.1, 0],
            [1, 1, 0, 0.1],
            [0, 1, 0.1, 0.1],
            [1, 0, 0.1, 0.1],
            [1, 1, 0.5, 0.1],
            [1, 1, 0.1, 1],
        ]
        for prop in props:
            with pytest.raises(BeamsError):
                IBeam(*prop)

    def test_rectangle(self):
        rect_beam = RectangularBeam(50, 50)
        assert rect_beam.area == 2500
        assert rect_beam.i_zz == 520833.3333333333
        assert rect_beam.i_yy == 520833.3333333333
        assert rect_beam.j == 880208.3333333334

        rect_beam = RectangularBeam(10, 250)
        assert rect_beam.area == 2500
        assert np.isclose(rect_beam.i_zz, 20833.3333)
        assert np.isclose(rect_beam.i_yy, 13020833.33333)
        assert np.isclose(rect_beam.j, 81234.94973767549, rtol=1e-4)


class TestCustomCrossSection:
    def assert_cs_equal(self, cs1, cs2):
        assert np.isclose(cs1.area, cs2.area)
        assert np.isclose(cs1.i_yy, cs2.i_yy)
        assert np.isclose(cs1.i_zz, cs2.i_zz)
        assert np.allclose(cs1.centroid, cs2.centroid)

    def test_rectangle(self):
        props = [[0.41, 0.5], [0.5, 0.4], [50, 600], [0.4, 0.21], [80, 0.1]]
        for prop in props:
            rect_beam = RectangularBeam(*prop)
            loop = rect_beam.geometry
            custom_xs = CustomCrossSection(loop)
            self.assert_cs_equal(custom_xs, rect_beam)

    def test_ibeam(self):
        props = [
            [0.6, 0.6, 0.1, 0.15],
            [0.4, 0.6, 0.1, 0.15],
            [0.6, 0.5, 0.05, 0.15],
            [0.3, 1, 0.1, 0.1],
            [200, 300, 11, 12],
            [0.3, 0.1, 0.025, 0.025],
        ]
        for prop in props:
            i_beam = IBeam(*prop)
            loop = i_beam.geometry

            custom_xs = CustomCrossSection(loop)
            self.assert_cs_equal(custom_xs, i_beam)

    def test_hollow_rectangle(self):
        def make_hollow_rect(w, h, t):
            loop = Loop(
                y=[-w / 2, w / 2, w / 2, -w / 2, -w / 2],
                z=[-h / 2, -h / 2, h / 2, h / 2, -h / 2],
            )
            loop2 = loop.offset(-t)

            shell_r = Shell(loop2, loop)
            return shell_r

        shell = make_hollow_rect(4, 5, 0.1)
        # No assert
        CustomCrossSection(shell)


class TestComposite:
    @pytest.mark.longrun
    def test_composite(self):
        def make_hollow_rect(w, h, t):
            loop = Loop(
                y=[-w / 2, w / 2, w / 2, -w / 2, -w / 2],
                z=[-h / 2, -h / 2, h / 2, h / 2, -h / 2],
            )
            loop2 = loop.offset(-t)

            shell_r = Shell(loop2, loop)
            return shell_r

        rect = RectangularBeam(4 - 0.2, 5 - 0.2).geometry
        shell = make_hollow_rect(4, 5, 0.1)

        # No assert
        comp = CompositeCrossSection([shell, rect], [SS316, CONCRETE])
        if tests.PLOTTING:
            comp.plot()

    def test_fail(self):
        def make_hollow_rect(w, h, t):
            loop = Loop(
                y=[-w / 2, w / 2, w / 2, -w / 2, -w / 2],
                z=[-h / 2, -h / 2, h / 2, h / 2, -h / 2],
            )
            loop2 = loop.offset(-t)

            shell_r = Shell(loop2, loop)
            return shell_r

        rect = RectangularBeam(4 - 0.2, 5 - 0.2).geometry
        shell = make_hollow_rect(4, 5, 0.1)

        with pytest.raises(BeamsError):
            CompositeCrossSection([shell, rect], [SS316])


class TestDuploRectangle:
    @classmethod
    def setup_class(cls):
        path = get_BP_path("BLUEPRINT/structural", subfolder="tests")
        filename = os.sep.join([path, "tf_shell_sections.pkl"])
        with open(filename, "rb") as f:
            cls.shells = pickle.load(f)  # noqa (S301)

    def test_rotation(self):
        for shell in self.shells:
            xs_analytic = AnalyticalShellComposite(shell, [SS316, CONCRETE])
            eizz, eiyy, eizy = (
                xs_analytic.ei_zz,
                xs_analytic.ei_yy,
                xs_analytic.ei_zy,
            )
            # Rotations of 180 and 360 degrees shouldn't change the below props
            xs_analytic.rotate(180)
            assert np.isclose(eizz, xs_analytic.ei_zz)
            assert np.isclose(eiyy, xs_analytic.ei_yy)
            assert np.isclose(eizy, xs_analytic.ei_zy)
            xs_analytic.rotate(180)
            assert np.isclose(eizz, xs_analytic.ei_zz)
            assert np.isclose(eiyy, xs_analytic.ei_yy)
            assert np.isclose(eizy, xs_analytic.ei_zy)

            alphas = [0.35, 14.54, 40.56, 90.6, 176, 270.5, 304.23423662, 361.0]
            for alpha in alphas:
                xs_analytic.rotate(alpha)
                assert not np.isclose(eizz, xs_analytic.ei_zz)
                assert not np.isclose(eiyy, xs_analytic.ei_yy)
                assert not np.isclose(eizy, xs_analytic.ei_zy)

    @pytest.mark.longrun
    def test_comparison(self):
        xs_fes = {"ei_yy": [], "ei_zz": [], "ea": [], "gj": []}
        xs_analytics = {"ei_yy": [], "ei_zz": [], "ea": [], "gj": []}
        for shell in self.shells:
            xs_fe = CompositeCrossSection([shell, shell.inner], [SS316, CONCRETE])
            xs_analytic = AnalyticalShellComposite(shell, [SS316, CONCRETE])
            for key in xs_fes.keys():
                xs_fes[key].append(getattr(xs_fe, key))
                xs_analytics[key].append(getattr(xs_analytic, key))

        for k, v in xs_fes.items():
            xs_fes[k] = np.array(v)

        for k, v in xs_analytics.items():
            xs_analytics[k] = np.array(v)

        for key in ["ei_yy", "ei_zz", "ea"]:
            assert np.allclose(xs_fes[key], xs_analytics[key])

        # GJ is still a very approximate calculation...

        if tests.PLOTTING:
            _, (ax, ax2) = plt.subplots(1, 2)
            colors = ["r", "g", "b", "k"]
            for key, color in zip(["ei_yy", "ei_zz", "ea", "gj"], colors):
                v1 = xs_fes[key]
                v2 = xs_analytics[key]
                ax.plot(v1 / max(v1), color=color, linestyle=":", label=key + " FE")
                ax.plot(
                    v2 / max(v2), color=color, linestyle="--", label=key + " analytic"
                )
                ax2.plot(v1 / v2, color=color, linestyle="-.")
            ax.legend()
            plt.show()


class TestRectangleCustom:
    def test_compare(self):
        sq_beam = RectangularBeam(50, 40)
        custom_beam = RapidCustomCrossSection(sq_beam.geometry, opt_var=42.77)
        for k in ["area", "centroid", "i_yy", "i_zz", "i_zy", "qyy", "qzz", "ry", "rz"]:
            assert np.allclose(getattr(sq_beam, k), getattr(custom_beam, k))
        # J will not be so close...
        assert np.isclose(sq_beam.j, custom_beam.j, rtol=1e-4)


class TestRotation:
    def test_loop_rotation(self):
        circle = CircularBeam(0.5)
        loop = circle.geometry.copy()
        circle.rotate(2345.24)
        loop.rotate(2345.24, p1=[0, *loop.centroid], p2=[1, *loop.centroid])
        assert circle.geometry == loop

    def test_circulars(self):
        circle = CircularBeam(0.5)
        i_zz_c = circle.i_zz
        i_yy_c = circle.i_yy
        i_zy_c = circle.i_zy

        h_circle = CircularHollowBeam(0.45, 0.5)
        i_zz_ch = h_circle.i_zz
        i_yy_ch = h_circle.i_yy
        i_zy_ch = h_circle.i_zy

        assert i_zy_c == 0.0
        assert i_zy_ch == 0.0

        alphas = [0.35, 14.54, 40.56, 90.6, 176, 270.5, 304.23423662, 361.0]
        for alpha in alphas:
            circle.rotate(alpha)
            h_circle.rotate(alpha)
            assert np.isclose(i_zz_c, circle.i_zz)
            assert np.isclose(i_yy_c, circle.i_yy)
            assert np.isclose(i_zy_c, circle.i_zy)
            assert np.isclose(i_zz_ch, h_circle.i_zz)
            assert np.isclose(i_yy_ch, h_circle.i_yy)
            assert np.isclose(i_zy_ch, h_circle.i_zy)

    def test_square(self):
        sq_beam = RectangularBeam(50, 50)
        i_zz_c = sq_beam.i_zz
        i_yy_c = sq_beam.i_yy
        i_zy_c = sq_beam.i_zy
        alphas = [0.35, 14.54, 40.56, 90.6, 176, 270.5, 304.23423662, 361.0]
        for alpha in alphas:
            sq_beam = RectangularBeam(50, 50)
            sq_beam.rotate(alpha)
            assert np.isclose(i_zz_c, sq_beam.i_zz)
            assert np.isclose(i_yy_c, sq_beam.i_yy)
            assert np.isclose(i_zy_c, sq_beam.i_zy)

    def test_rectangle(self):
        r_beam = RectangularBeam(4, 7)
        i_zz_c = r_beam.i_zz
        i_yy_c = r_beam.i_yy
        i_zy_c = r_beam.i_zy
        alphas = [0.35, 14.54, 40.56, 90.6, 176, 270.5, 304.23423662, 361.0]
        for alpha in alphas:
            r_beam = RectangularBeam(4, 7)
            r_beam.rotate(alpha)
            assert not np.isclose(i_zz_c, r_beam.i_zz)
            assert not np.isclose(i_yy_c, r_beam.i_yy)
            assert not np.isclose(i_zy_c, r_beam.i_zy)

        betas = [180, 360]
        for beta in betas:
            r_beam = RectangularBeam(4, 7)
            r_beam.rotate(beta)
            assert np.isclose(i_zz_c, r_beam.i_zz)
            assert np.isclose(i_yy_c, r_beam.i_yy)
            assert np.isclose(i_zy_c, r_beam.i_zy)

    def test_rectangle_values(self):
        """
        Cross-checked with: https://structx.com/Shape_Formulas_033.html
        """
        r_beam = RectangularBeam(0.508, 0.254)
        r_beam.rotate(np.rad2deg(4))
        assert np.isclose(10 ** 12 * r_beam.i_zz, 1582893390.424)

    def test_custom(self):
        loop = Loop(
            y=[0, 1, 1, 2, 2, 3, 3, 4, 4, 3, 3, 2, 2, 1, 1, 0],
            z=[0, 0, -1, -1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 1, 1, 0],
        )
        custom = CustomCrossSection(loop)
        r_custom = RapidCustomCrossSection(loop)
        izz, iyy, izy = custom.i_zz, custom.i_yy, custom.i_zy
        izz_r, iyy_r, izy_r = r_custom.i_zz, r_custom.i_yy, r_custom.i_zy
        assert np.isclose(izz, izz_r)
        assert np.isclose(iyy, iyy_r)
        assert np.isclose(izy, izy_r)
        custom.rotate(180)
        r_custom.rotate(180)
        assert np.isclose(izz, custom.i_zz)
        assert np.isclose(iyy, custom.i_yy)
        assert np.isclose(izy, custom.i_zy)
        assert np.isclose(izz, r_custom.i_zz)
        assert np.isclose(iyy, r_custom.i_yy)
        assert np.isclose(izy, r_custom.i_zy)
        custom.rotate(180)
        r_custom.rotate(180)
        assert np.isclose(izz, custom.i_zz)
        assert np.isclose(iyy, custom.i_yy)
        assert np.isclose(izy, custom.i_zy)
        assert np.isclose(izz, r_custom.i_zz)
        assert np.isclose(iyy, r_custom.i_yy)
        assert np.isclose(izy, r_custom.i_zy)


class TestMultiCrossSection:
    def test_multi(self):
        """
        Check that a multi cross-section recovers Ibeam easy properties.
        Note that J cannot be recovered as the parts are touching.
        """
        base, depth, flange, web = 0.5, 1, 0.1, 0.05
        lower = RectangularBeam(base, flange)
        lower.translate([0, 0, -(depth - flange) / 2])
        upper = RectangularBeam(base, flange)
        upper.translate([0, 0, (depth - flange) / 2])
        middle = RectangularBeam(web, depth - flange * 2)

        multi = MultiCrossSection([lower, middle, upper], centroid=[0, 0])
        ibeam = IBeam(base, depth, flange, web)
        assert np.isclose(multi.area, ibeam.area)
        assert np.isclose(multi.i_zz, ibeam.i_zz)
        assert np.isclose(multi.i_yy, ibeam.i_yy)
        assert np.isclose(multi.i_zy, ibeam.i_zy)


if __name__ == "__main__":
    pytest.main([__file__])
