# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
import pytest
import tests
from unittest.mock import patch

import os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.ticker as ticker
from BLUEPRINT.base.error import GeometryError, EquilibriaError
from BLUEPRINT.base.file import get_BP_path
from BLUEPRINT.geometry.parameterisations import flatD
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.geomtools import circle_seg
from BLUEPRINT.equilibria.coils import Coil, CoilSet, PF_COIL_NAME
from BLUEPRINT.equilibria.positioner import (
    XZLMapper,
    CoilPositioner,
    RegionMapper,
    RegionInterpolator,
)
from BLUEPRINT.equilibria.gridops import Grid

from tests.BLUEPRINT.equilibria.setup_methods import _coilset_setup, _make_square, _make_star


class TestXZLMapper:
    @classmethod
    def setup_class(cls):
        f, cls.ax = plt.subplots()
        fp = get_BP_path("Geometry", subfolder="data")
        tf = Loop.from_file(os.sep.join([fp, "TFreference.json"]))
        tf = tf.offset(2.5)
        clip = np.where(tf.x >= 3.5)
        tf = Loop(tf.x[clip], z=tf.z[clip])
        up = Loop(x=[7.5, 14, 14, 7.5, 7.5], z=[3, 3, 15, 15, 3])
        lp = Loop(x=[10, 10, 15, 22, 22, 15, 10], z=[-6, -10, -13, -13, -8, -8, -6])
        eq = Loop(x=[14, 22, 22, 14, 14], z=[-1.4, -1.4, 1.4, 1.4, -1.4])
        up.plot(cls.ax, fill=False, linestyle="-", edgecolor="r")
        lp.plot(cls.ax, fill=False, linestyle="-", edgecolor="r")
        eq.plot(cls.ax, fill=False, linestyle="-", edgecolor="r")

        cls.zones = [eq, lp, up]
        positioner = CoilPositioner(9, 3.1, 0.33, 1.59, tf, 2.6, 0.5, 6, 5)
        cls.coilset = positioner.make_coilset()
        solenoid = cls.coilset.get_solenoid()
        cls.coilset.set_control_currents(1e6 * np.ones(cls.coilset.n_coils))
        cls.xzl_map = XZLMapper(tf, solenoid.radius, -10, 10, 0.1, CS=False)

    def test_xzl(self):
        l_pos, lb, ub = self.xzl_map.get_Lmap(
            self.coilset, set(self.coilset.get_PF_names())
        )

        self.xzl_map.add_exclusion_zones(self.zones)

        l_pos, lb, ub = self.xzl_map.get_Lmap(
            self.coilset, set(self.coilset.get_PF_names())
        )
        positions = []
        for pos in l_pos:
            positions.append(self.xzl_map.L_to_xz(pos))
        self.coilset.set_positions(positions)
        self.coilset.plot(self.ax)

    def test_2(self):
        lb = [0.9, 0.7, 0.7, 0.5, 0.25, 0.25, 0]
        ub = [1, 0.8, 0.8, 0.6, 0.4, 0.4, 0.2]
        lbn, ubn = self.xzl_map._segment_tracks(lb, ub)
        assert list(lbn) == [0.9, 0.75, 0.7, 0.5, 0.325, 0.25, 0]
        assert list(ubn) == [1, 0.8, 0.75, 0.6, 0.4, 0.325, 0.2]

    def test_n(self):
        lb = [0, 0, 0, 0, 0]
        ub = [1, 1, 1, 1, 1]
        lbn, ubn = self.xzl_map._segment_tracks(lb, ub)
        lbtrue = np.array([0.8, 0.6, 0.4, 0.2, 0])
        ubtrue = np.array([1, 0.8, 0.6, 0.4, 0.2])
        assert np.allclose(lbn, lbtrue)
        assert np.allclose(ubn, ubtrue)

    def test_2n(self):
        lb = [0.95, 0.9, 0.9, 0.9, 0.8, 0.5, 0.5, 0.5, 0]
        ub = [1, 0.95, 0.95, 0.95, 0.9, 0.7, 0.7, 0.7, 0.5]
        lbn, ubn = self.xzl_map._segment_tracks(lb, ub)
        lbtrue = [
            0.95,
            0.9333333333333332,
            0.9166666666666666,
            0.9,
            0.8,
            0.6333333333333333,
            0.5666666666666667,
            0.5,
            0,
        ]
        ubtrue = [
            1,
            0.95,
            0.9333333333333332,
            0.9166666666666666,
            0.9,
            0.7,
            0.6333333333333333,
            0.5666666666666667,
            0.5,
        ]
        assert np.allclose(lbn, np.array(lbtrue))
        assert np.allclose(ubn, np.array(ubtrue))

    def test_n00(self):
        lb = [
            0.9245049621496133,
            0.5469481563527382,
            0.5469481563527382,
            0.5469481563527382,
            0.30272259631747434,
            0.30272259631747434,
            0.0,
            0.0,
        ]
        ub = [
            1.0,
            0.752451839494733,
            0.752451839494733,
            0.752451839494733,
            0.4702461942340625,
            0.4702461942340625,
            0.18361524172039095,
            0.18361524172039095,
        ]
        upper1 = 0.752451839494733
        delta1 = (upper1 - 0.5469481563527382) / 3
        upper2 = 0.4702461942340625
        delta2 = (upper2 - 0.30272259631747434) / 2
        upper3 = 0.18361524172039095
        delta3 = (upper3 - 0.0) / 2
        lbn, ubn = self.xzl_map._segment_tracks(lb, ub)
        # print(lbn, ubn)
        lbtrue = [
            0.9245049621496133,
            upper1 - delta1,
            upper1 - 2 * delta1,
            0.5469481563527382,
            upper2 - delta2,
            0.30272259631747434,
            upper3 - delta3,
            0.0,
        ]
        ubtrue = [
            1.0,
            0.752451839494733,
            upper1 - delta1,
            upper1 - 2 * delta1,
            0.4702461942340625,
            upper2 - delta2,
            0.18361524172039095,
            upper3 - delta3,
        ]
        assert np.allclose(lbn, np.array(lbtrue))
        assert np.allclose(ubn, np.array(ubtrue))


class TestZLMapper:
    @classmethod
    def setup_class(cls):
        """
        Sets up an XZLMapper that with a "normal" set of exclusion zones
        """
        fp = get_BP_path("Geometry", subfolder="data")
        tf = Loop.from_file(os.sep.join([fp, "TFreference.json"]))
        tf = tf.offset(2.5)
        clip = np.where(tf.x >= 3.5)
        tf = Loop(tf.x[clip], z=tf.z[clip])
        up = Loop(x=[7.5, 14, 14, 7.5, 7.5], z=[3, 3, 15, 15, 3])
        lp = Loop(x=[10, 10, 15, 22, 22, 15, 10], z=[-6, -10, -13, -13, -8, -8, -6])
        eq = Loop(x=[14, 22, 22, 14, 14], z=[-1.4, -1.4, 1.4, 1.4, -1.4])

        cls.TF = tf
        cls.zones = [eq, lp, up]
        positioner = CoilPositioner(9, 3.1, 0.33, 1.59, tf, 2.6, 0.5, 6, 5)
        cls.coilset = positioner.make_coilset()
        cls.coilset.set_control_currents(1e6 * np.ones(cls.coilset.n_coils))
        solenoid = cls.coilset.get_solenoid()
        cls.xz_map = XZLMapper(
            tf, solenoid.radius, solenoid.z_min, solenoid.z_max, solenoid.gap, CS=True
        )
        if tests.PLOTTING:
            f, cls.ax = plt.subplots()
            up.plot(cls.ax, fill=False, linestyle="-", edgecolor="r")
            lp.plot(cls.ax, fill=False, linestyle="-", edgecolor="r")
            eq.plot(cls.ax, fill=False, linestyle="-", edgecolor="r")

    def test_cs_zl(self):
        l_pos, lb, ub = self.xz_map.get_Lmap(
            self.coilset, set(self.coilset.get_PF_names())
        )
        self.xz_map.add_exclusion_zones(self.zones)  # au cas ou
        _, _ = self.xz_map.L_to_xz(l_pos[: self.coilset.n_PF])
        xcs, zcs, dzcs = self.xz_map.L_to_zdz(l_pos[self.coilset.n_PF :])
        l_cs = self.xz_map.z_to_L(zcs)
        assert np.allclose(l_cs, l_pos[self.coilset.n_PF :])
        solenoid = self.coilset.get_solenoid()
        z = []
        for c in solenoid.coils:
            z.append(c.z)
        z = np.sort(z)  # [::-1]  # Fixed somewhere else jcrois
        assert np.allclose(z, zcs), z - zcs

        if tests.PLOTTING:
            self.xz_map.plot(ax=self.ax)
            plt.show()


class TestZLMapperEdges:
    @classmethod
    def setup_class(cls):
        """
        Sets up an XZLMapper that will trigger edge cases where a zone covers
        the start or end of a track
        """

        fp = get_BP_path("Geometry", subfolder="data")
        tf = Loop.from_file(os.sep.join([fp, "TFreference.json"]))
        tf = tf.offset(2.5)
        clip = np.where(tf.x >= 3.5)
        tf = Loop(tf.x[clip], z=tf.z[clip])
        up = Loop(x=[0, 14, 14, 0, 0], z=[3, 3, 15, 15, 3])
        lp = Loop(x=[10, 10, 15, 22, 22, 15, 10], z=[-6, -10, -13, -13, -8, -8, -6])
        eq = Loop(x=[14, 22, 22, 14, 14], z=[-1.4, -1.4, 1.4, 1.4, -1.4])
        cls.TF = tf
        cls.zones = [eq, lp, up]
        positioner = CoilPositioner(9, 3.1, 0.33, 1.59, tf, 2.6, 0.5, 6, 5)
        cls.coilset = positioner.make_coilset()
        cls.coilset.set_control_currents(1e6 * np.ones(cls.coilset.n_coils))
        solenoid = cls.coilset.get_solenoid()
        cls.xz_map = XZLMapper(
            tf, solenoid.radius, solenoid.z_min, solenoid.z_max, solenoid.gap, CS=True
        )
        if tests.PLOTTING:
            f, cls.ax = plt.subplots()
            up.plot(cls.ax, fill=False, linestyle="-", edgecolor="r")
            lp.plot(cls.ax, fill=False, linestyle="-", edgecolor="r")
            eq.plot(cls.ax, fill=False, linestyle="-", edgecolor="r")

    def test_cs_zl(self):

        l_pos, lb, ub = self.xz_map.get_Lmap(
            self.coilset, set(self.coilset.get_PF_names())
        )
        self.xz_map.add_exclusion_zones(self.zones)  # au cas ou
        _, _ = self.xz_map.L_to_xz(l_pos[: self.coilset.n_PF])
        xcs, zcs, dzcs = self.xz_map.L_to_zdz(l_pos[self.coilset.n_PF :])
        l_cs = self.xz_map.z_to_L(zcs)
        assert np.allclose(l_cs, l_pos[self.coilset.n_PF :])
        solenoid = self.coilset.get_solenoid()
        z = []
        for c in solenoid.coils:
            z.append(c.z)
        z = np.sort(z)  # [::-1]  # Fixed somewhere else jcrois
        assert np.allclose(z, zcs), z - zcs

        if tests.PLOTTING:
            self.xz_map.plot(ax=self.ax)
            plt.show()


class TestCoilPositioner:
    def test_DEMO_CS(self):  # noqa (N802)
        for n in [3, 5, 7, 9]:
            d_loop = flatD(4, 16, 0)
            d_loop = Loop(x=d_loop[0], z=d_loop[1])
            positioner = CoilPositioner(
                9,
                3.1,
                0.3,
                1.65,
                d_loop,
                2.5,
                0.5,
                6,
                n,
                0.1,
                rtype="Normal",
                cslayout="DEMO",
            )
            coilset = positioner.make_coilset()
            if tests.PLOTTING:
                coilset.plot()  # look good! cba TO test
                plt.show()


class TestRegionMapper:

    square = Loop(**_make_square({"x": 2, "z": 0}, {"x": 4, "z": 2}))
    diamond = Loop(x=[6, 8, 10, 8, 6], z=[8, 6, 8, 10, 8])
    circle_xz = circle_seg(2, h=(4, -4))
    circle = Loop(x=circle_xz[0], z=circle_xz[1])

    shapes = (square, diamond, circle)
    centres = ((3, 1), (8, 8), (4, -4))

    @classmethod
    def setup_class(cls):
        cls._coilset_setup = _coilset_setup
        cls.limits = np.linspace(0, 1, num=101, endpoint=True)

    def _region_L_to_xz_conversion(self):

        for lim in self.limits:
            out1 = self.Rmap.L_to_xz(1, (lim, lim))
            out2 = self.Rmap.L_to_xz("R_1", (lim, lim))

            np.testing.assert_equal(out1[0], out2[0])
            np.testing.assert_equal(out1[1], out2[1])

            yield out1, lim

    def _region_xz_to_L_conversion(self, x, z, lim=None):

        l_values = self.Rmap.xz_to_L(1, x, z)

        np.testing.assert_allclose(l_values[0], lim)
        np.testing.assert_allclose(l_values[0], l_values[1])

        # Overkill
        np.testing.assert_equal(np.logical_and(l_values[0] >= 0, l_values[0] <= 1), True)

        np.testing.assert_equal(np.logical_and(l_values[1] >= 0, l_values[1] <= 1), True)

    @pytest.mark.parametrize("size", [5, 0.8])
    @pytest.mark.parametrize("shape, centre", zip(shapes, centres))
    def test_xz_l_region(self, size, shape, centre):
        rmap = RegionMapper({PF_COIL_NAME.format(1): shape})

        circle_xz = circle_seg(size, h=centre)
        external_coords = Loop(x=circle_xz[0], z=circle_xz[1])
        lv_old = [np.inf, np.inf]

        if tests.PLOTTING and size > 1:
            f, ax = plt.subplots(1, 2)
            external_coords.plot(ax[0], fill=False)
            shape.plot(ax[0], fill=False)
            external_coords.plot(ax[1], fill=False)
            shape.plot(ax[1], fill=False)

        for x, z in zip(external_coords.x[:-1], external_coords.z[:-1]):
            l_values = rmap.xz_to_L(1, x, z)
            xx, zz = rmap.L_to_xz(1, l_values)
            np.testing.assert_equal(
                np.logical_and(l_values[0] >= 0, l_values[0] <= 1), True
            )
            np.testing.assert_equal(
                np.logical_and(l_values[1] >= 0, l_values[1] <= 1), True
            )
            if size > 1:
                if (
                    np.allclose(l_values, [0, 0])
                    or np.allclose(l_values, [1, 1])
                    or np.allclose(l_values, [1, 0])
                    or np.allclose(l_values, [0, 1])
                ):
                    if tests.PLOTTING:
                        ax[1].plot([x, xx], [z, zz], color="tab:orange")
                else:
                    if tests.PLOTTING:
                        ax[0].plot([x, xx], [z, zz], color="tab:blue")
                    np.testing.assert_raises(
                        AssertionError, np.testing.assert_allclose, lv_old, l_values
                    )

                np.testing.assert_raises(
                    AssertionError, np.testing.assert_allclose, [x, z], [xx, zz]
                )
                lv_old = l_values
            else:
                np.testing.assert_allclose(x, xx)
                np.testing.assert_allclose(z, zz)

        if tests.PLOTTING:
            plt.show()

    @pytest.mark.parametrize("shape", shapes)
    def test_l_xz_region(self, shape):

        self.Rmap = RegionMapper({PF_COIL_NAME.format(1): shape})

        # Plane only works for single values
        for (x, z), lim in self._region_L_to_xz_conversion():

            assert shape.point_in_poly([x, z], include_edges=True)

            self._region_xz_to_L_conversion(x, z, lim)

    def test_region_dict(self):

        with pytest.raises(EquilibriaError):
            self.Rmap = RegionMapper(
                [
                    Loop(**_make_square({"x": 13, "z": -9}, {"x": 15, "z": -7})),
                    Loop(**_make_square({"x": 6, "z": -12}, {"x": 8, "z": -10})),
                ]
            )

    def test_wrong_region_dimensions(self):
        sq = _make_square()
        sq = {"x": sq["x"], "y": sq["x"] + 1, "z": sq["z"]}

        loop = Loop(enforce_ccw=False, **sq)
        with pytest.raises(EquilibriaError):
            RegionMapper({PF_COIL_NAME.format(1): loop})

    def test_abstract_region(self):
        RegionMapper({PF_COIL_NAME.format(1): Loop(**_make_square())})

        star = Loop(**_make_star())
        with pytest.raises(GeometryError):
            RegionMapper({PF_COIL_NAME.format(1): star})

    def test_region_naming(self):

        for name in ["abc", "abcR_1", "R_1abc", "R _1"]:
            with pytest.raises(NameError):
                RegionMapper({})._regionname(name)

        assert RegionMapper({})._regionname(1) == "R_1"
        assert RegionMapper({})._regionname("R_1") == "R_1"
        assert RegionMapper({})._regionname("PF_1") == "R_1"

    def test_add_region(self):
        self.Rmap = RegionMapper({})
        assert self.Rmap.no_regions == 0

        self.Rmap.add_region({PF_COIL_NAME.format(1): self.square})

        assert self.Rmap.no_regions == 1
        assert len(self.Rmap.regions) == 1

    def test_lmap(self):
        self._coilset_setup()
        self.Rmap = RegionMapper(
            {
                PF_COIL_NAME.format(5): Loop(
                    **_make_square({"x": 13, "z": -9}, {"x": 15, "z": -7})
                ),
                PF_COIL_NAME.format(6): Loop(
                    **_make_square({"x": 6, "z": -12}, {"x": 8, "z": -10})
                ),
            }
        )

        l_values, lb, ub = self.Rmap.get_Lmap(self.coilset)

        assert all(l_values <= 1)
        assert all(l_values >= 0)

    def test_less_coils_than_regions(self):
        self.Rmap = RegionMapper(
            {
                PF_COIL_NAME.format(1): self.square,
                PF_COIL_NAME.format(2): Loop(
                    **_make_square({"x": 3, "z": 2}, {"x": 4, "z": 3})
                ),
            }
        )

        with patch("BLUEPRINT.equilibria.positioner.bpwarn") as bpwarn:
            self.Rmap.get_Lmap(
                CoilSet(
                    [Coil(3, 3, ctype="PF"), Coil(4, 4, ctype="CS")],
                    9.0,
                )
            )
            bpwarn.assert_called()

    def test_name_converter(self):
        rm = RegionMapper({})

        assert rm._name_converter("R_5") == "PF_5"
        assert rm._name_converter("PF_5", True) == "R_5"

    @pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
    def test_plotting(self):
        self._coilset_setup()
        regions = {}
        for coil in self.coilset.coils.values():
            if coil.ctype == "PF":
                dx = coil.dx * 4
                dz = coil.dz * 4
                xloop = np.array([-dx, dx, dx, -dx, -dx])
                zloop = np.array([-dz, -dz, dz, dz, -dz])
                regions[coil.name] = Loop(x=coil.x + xloop, z=coil.z + zloop)

        self.Rmap = RegionMapper(regions)
        self.Rmap.plot()
        plt.show()

    @pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
    @pytest.mark.parametrize("loop", shapes)
    def test_interpolation_plotting(self, loop):
        zm = RegionMapper({PF_COIL_NAME.format(1): loop})

        nx = nz = 100
        grid = Grid(
            min(loop.x) - 1, max(loop.x) + 1, min(loop.z) - 1, max(loop.z) + 1, nx, nz
        )

        l_values = np.zeros((nx, nz))
        new_x, new_z = np.zeros((nx, nz)), np.zeros((nx, nz))
        for i in range(nx):
            for j in range(nx):
                l1, l2 = zm.xz_to_L(1, grid.x[i, j], grid.z[i, j])
                l_values[i, j] = l1 + l2
                xn, zn = zm.L_to_xz(1, [l1, l2])
                new_x[i, j] = xn
                new_z[i, j] = zn
                # Only makes sense to compare mapping for points inside the Loop
                if zm.regions["R_1"].loop.point_in_poly(
                    [grid.x[i, j], grid.z[i, j]], include_edges=True
                ):
                    assert np.isclose(grid.x[i, j], xn)
                    assert np.isclose(grid.z[i, j], zn)
        f, ax = plt.subplots(1, 2)
        loop.plot(ax[0], fill=False)
        _cf = ax[0].contourf(grid.x, grid.z, l_values)
        self._colourbars(_cf, f, ax[0])
        ax[0].set_title("x-z to L")
        loop.plot(ax[1], fill=False)
        ax[1].set_title("L to x-z")
        _cf = ax[1].contourf(grid.x, grid.z, new_x + new_z)
        self._colourbars(_cf, f, ax[1])
        plt.show()

    def _colourbars(self, _cf, f, ax):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        f.colorbar(_cf, cax=cax, ticks=ticker.MultipleLocator(1))


class TestRegionInterpolator:
    @classmethod
    def setup_class(cls):
        cls._coilset_setup = _coilset_setup

    def test_non_convex_hull(self):
        star = Loop(**_make_star())
        with pytest.raises(GeometryError):
            RegionInterpolator(star)

        with patch.object(RegionInterpolator, "check_loop_feasibility"):
            ri = RegionInterpolator(star)
            with pytest.raises(GeometryError):
                ri.to_L(0.1, 1)
            with pytest.raises(GeometryError):
                ri.to_xz((0.3, 0.25))

    def test_current_sizing(self):
        self._coilset_setup()
        self.Rmap = RegionMapper(
            {
                "PF_5": Loop(**_make_square({"x": 13, "z": -9}, {"x": 15, "z": -7})),
                "PF_6": Loop(**_make_square({"x": 6, "z": -12}, {"x": 8, "z": -10})),
            }
        )

        self.Rmap.get_Lmap(self.coilset)

        clim = self.Rmap.get_size_current_limit()

        assert len(clim) == self.Rmap.no_regions
        for i in clim:
            assert i > 0


if __name__ == "__main__":
    pytest.main([__file__])
