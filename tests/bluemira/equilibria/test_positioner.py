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

import numpy as np
from matplotlib import pyplot as plt

import tests
from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.coils import Coil, CoilSet, SymmetricCircuit
from bluemira.equilibria.positioner import CoilPositioner, RegionMapper, XZLMapper
from bluemira.geometry._deprecated_loop import Loop

DATA_PATH = get_bluemira_path("geometry", subfolder="data")


class TestXZLMapper:
    @classmethod
    def setup_class(cls):
        f, cls.ax = plt.subplots()
        tf = Loop.from_file(os.sep.join([DATA_PATH, "TFreference.json"]))
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
        tf = Loop.from_file(os.sep.join([DATA_PATH, "TFreference.json"]))
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

        tf = Loop.from_file(os.sep.join([DATA_PATH, "TFreference.json"]))
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


class TestRegionMapper:
    @classmethod
    def setup_class(cls):
        coil = Coil(
            x=1.5,
            z=6,
            current=1e6,
            dx=0.25,
            dz=0.5,
            j_max=1e7,
            b_max=100,
            ctype="PF",
            name="PF_2",
        )
        circuit = SymmetricCircuit(coil)

        coil2 = Coil(
            x=4, z=10, current=2e6, dx=1, dz=0.5, j_max=5e6, b_max=50, name="PF_1"
        )

        cls.coilset = CoilSet([coil2, circuit])

    def create_rectangular_region(self):
        max_coil_shifts = {
            "x_shifts_lower": -2.0,
            "x_shifts_upper": 1.0,
            "z_shifts_lower": -1.0,
            "z_shifts_upper": 5.0,
        }

        pfregions = {}
        for coil in self.coilset._ccoils:
            xu = coil.x + max_coil_shifts["x_shifts_upper"]
            xl = coil.x + max_coil_shifts["x_shifts_lower"]
            zu = coil.z + max_coil_shifts["z_shifts_upper"]
            zl = coil.z + max_coil_shifts["z_shifts_lower"]

            rect = Loop(x=[xl, xu, xu, xl, xl], z=[zl, zl, zu, zu, zl])

            pfregions[coil.name] = rect
        return pfregions

    def test_region_mapper(self):
        pfregions = self.create_rectangular_region()
        region_mapper = RegionMapper(pfregions)
        x_init, z_init = self.coilset.get_positions()

        mapped_positions = region_mapper.get_Lmap(self.coilset)[0]
        region_mapper.set_Lmap(mapped_positions)
        x, z = region_mapper.get_xz_arrays()

        assert np.allclose(x_init, x)
        assert np.allclose(z_init, z)
