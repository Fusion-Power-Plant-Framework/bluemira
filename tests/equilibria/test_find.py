# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.find import (
    _in_plasma,
    find_LCFS_separatrix,
    find_local_minima,
    inv_2x2_matrix,
)
from bluemira.equilibria.find_legs import LegFlux, NumNull, SortSplit

DATA = get_bluemira_path("equilibria/test_data", subfolder="tests")


def test_find_local_minima():
    rng = np.random.default_rng()
    for _ in range(10):
        array = np.ones((100, 100))
        i, j = rng.integers(0, 99, 2)
        array[i, j] = 0
        ii, jj = find_local_minima(array)
        assert len(ii) == 1
        assert len(jj) == 1
        assert ii[0] == i
        assert jj[0] == j

    array = np.ones((100, 100))
    array[1, 0] = 0
    array[-2, -2] = 0
    array[-2, 1] = 0
    array[1, -2] = 0
    array[0, 50] = 0
    array[50, 0] = 0

    ii, jj = find_local_minima(array)

    assert len(ii) == 6
    assert len(jj) == 6
    assert (np.sort(ii) == np.array([0, 1, 1, 50, 98, 98])).all()
    assert (np.sort(jj) == np.array([0, 0, 1, 50, 98, 98])).all()


def test_inv_2x2_jacobian():
    a, b, c, d = 3.523, 5.0, 6, 0.2
    inv_jac_true = np.linalg.inv(np.array([[a, b], [c, d]]))
    inv_jac = inv_2x2_matrix(a, b, c, d)
    assert np.allclose(inv_jac_true, inv_jac)


class TestFindLCFSSeparatrix:
    def test_other_grid(self):
        sof = Equilibrium.from_eqdsk(Path(DATA, "eqref_OOB.json"))
        psi = sof.psi()
        o_points, x_points = sof.get_OX_points(psi)
        grid_tol = np.hypot(sof.grid.dx, sof.grid.dz)
        for tolerance in [1e-6, 1e-7, 1e-8, 1e-9]:
            lcfs, separatrix = find_LCFS_separatrix(
                sof.x,
                sof.z,
                sof.psi(),
                o_points=o_points,
                x_points=x_points,
                psi_n_tol=tolerance,
            )
            assert lcfs.closed
            assert not separatrix.closed
            primary_xp = x_points[0]
            distances = lcfs.distance_to([primary_xp.x, 0, primary_xp.z])
            assert np.amin(distances) <= grid_tol
            distances = separatrix.distance_to([primary_xp.x, 0, primary_xp.z])
            assert np.amin(distances) <= grid_tol

    def test_double_null(self):
        sof = Equilibrium.from_eqdsk(Path(DATA, "DN-DEMO_eqref.json"))
        psi = sof.psi()
        o_points, x_points = sof.get_OX_points(psi)
        grid_tol = np.hypot(sof.grid.dx, sof.grid.dz)
        for tolerance in [1e-6, 1e-7, 1e-8, 1e-9]:
            lcfs, separatrix = find_LCFS_separatrix(
                sof.x,
                sof.z,
                sof.psi(),
                o_points=o_points,
                x_points=x_points,
                psi_n_tol=tolerance,
                double_null=True,
            )

            assert lcfs.closed
            primary_xp = x_points[0]
            distances = lcfs.distance_to([primary_xp.x, 0, primary_xp.z])
            assert np.amin(distances) <= grid_tol

            assert isinstance(separatrix, list)
            for loop in separatrix:
                assert not loop.closed
                distances = loop.distance_to([primary_xp.x, 0, primary_xp.z])
                assert np.amin(distances) <= grid_tol


class TestInPlasma:
    def test_recursion(self):
        with open(Path(DATA, "in_plasma_test.json"), "rb") as f:
            data = json.load(f)
        x, z = np.array(data["X"]), np.array(data["Z"])
        lcfs = np.array(data["LCFS"])
        result = np.array(data["result"])
        mask = np.zeros_like(x)

        result2 = _in_plasma(x, z, mask, lcfs)
        assert np.allclose(result, result2)


class TestGetLegs:
    @classmethod
    def setup_class(cls):
        cls.sn_eq = Equilibrium.from_eqdsk(Path(DATA, "eqref_OOB.json"))
        cls.dn_eq = Equilibrium.from_eqdsk(Path(DATA, "DN-DEMO_eqref.json"))
        cls.falsified_dn_eq = deepcopy(cls.sn_eq)

    def test_legflux(self):
        test_sn = LegFlux(self.sn_eq)
        test_dn = LegFlux(self.dn_eq)
        assert test_sn.n_null == NumNull.SN
        assert test_sn.sort_split == SortSplit.X
        assert test_dn.n_null == NumNull.DN
        assert test_dn.sort_split == SortSplit.X
        psi = self.falsified_dn_eq.psi()
        o_points, x_points = self.falsified_dn_eq.get_OX_points(psi=psi)
        _, separatrix = find_LCFS_separatrix(
            self.falsified_dn_eq.x,
            self.falsified_dn_eq.z,
            psi,
            o_points,
            x_points,
            double_null=True,
            psi_n_tol=1e-6,
        )
        test_falsified_dn_eq = LegFlux(self.falsified_dn_eq)
        test_falsified_dn_eq.x_points = x_points[:2]
        test_falsified_dn_eq.separatrix = separatrix
        n_null, sort_split = test_falsified_dn_eq.which_legs()
        assert n_null == NumNull.DN
        assert sort_split == SortSplit.Z

    @pytest.mark.parametrize("n_layers", [2, 3, 5])
    def test_single_null(self, n_layers):
        legflux = LegFlux(self.sn_eq)
        legs = legflux.get_legs(n_layers, 0.2)
        assert len(legs) == 2
        assert "lower_inner" in legs
        assert "lower_outer" in legs
        x_point = self.sn_eq.get_OX_points()[1][0]
        for leg_group in legs.values():
            assert len(leg_group) == n_layers
            for leg in leg_group:
                self.assert_valid_leg(leg, x_point)
                self.assert_valid_leg(leg, x_point)

    def test_single_one_layer(self):
        legflux = LegFlux(self.sn_eq)
        legs = legflux.get_legs(1, 0.0)
        assert len(legs) == 2
        assert "lower_inner" in legs
        assert "lower_outer" in legs
        assert len(legs["lower_inner"]) == 1
        assert len(legs["lower_outer"]) == 1
        x1 = legs["lower_inner"][0].x[0]
        legs = legflux.get_legs(1, 1.0)
        assert len(legs) == 2
        assert "lower_inner" in legs
        assert "lower_outer" in legs
        assert len(legs["lower_inner"]) == 1
        assert len(legs["lower_outer"]) == 1
        x2 = legs["lower_inner"][0].x[0]
        assert np.isclose(x1, x2)

    @pytest.mark.parametrize("n_layers", [2, 3, 5])
    def test_double_null(self, n_layers):
        legflux = LegFlux(self.dn_eq)
        legs = legflux.get_legs(n_layers, 0.2)
        x_points = self.dn_eq.get_OX_points()[1][:2]
        x_points.sort(key=lambda xp: xp.z)
        assert len(legs) == 4
        assert "lower_inner" in legs
        assert "lower_outer" in legs
        assert "upper_inner" in legs
        assert "upper_outer" in legs
        for name, leg_group in legs.items():
            assert len(leg_group) == n_layers
            x_p = x_points[0] if "lower" in name else x_points[1]
            for leg in leg_group:
                self.assert_valid_leg(leg, x_p)

    def test_double_one_layer(self):
        legflux = LegFlux(self.dn_eq)
        legs = legflux.get_legs(1, 0.0)
        assert len(legs) == 4
        assert "lower_inner" in legs
        assert "lower_outer" in legs
        assert len(legs["lower_inner"]) == 1
        assert len(legs["lower_outer"]) == 1
        x1 = legs["lower_inner"][0].x[0]
        legs = legflux.get_legs(1, 1.0)
        assert len(legs) == 4
        assert "lower_inner" in legs
        assert "lower_outer" in legs
        assert len(legs["lower_inner"]) == 1
        assert len(legs["lower_outer"]) == 1
        x2 = legs["lower_inner"][0].x[0]
        assert np.isclose(x1, x2)

    @pytest.mark.parametrize("n_layers", [2, 3, 5])
    def test_double_z_split(self, n_layers):
        psi = self.falsified_dn_eq.psi()
        o_points, x_points = self.falsified_dn_eq.get_OX_points(psi=psi)
        _, separatrix = find_LCFS_separatrix(
            self.falsified_dn_eq.x,
            self.falsified_dn_eq.z,
            psi,
            o_points,
            x_points,
            double_null=True,
            psi_n_tol=1e-6,
        )
        legflux = LegFlux(self.falsified_dn_eq)
        legflux.x_points = x_points[:2]
        legflux.separatrix = separatrix
        legflux.n_null, legflux.sort_split = legflux.which_legs()
        legs = legflux.get_legs(n_layers, 0.2)
        x_points = self.dn_eq.get_OX_points()[1][:2]
        x_points.sort(key=lambda xp: xp.z)
        assert len(legs) == 4
        assert "lower_inner" in legs
        assert "lower_outer" in legs
        assert "upper_inner" in legs
        assert "upper_outer" in legs

    def test_double_z_split_one_layer(self):
        psi = self.falsified_dn_eq.psi()
        o_points, x_points = self.falsified_dn_eq.get_OX_points(psi=psi)
        _, separatrix = find_LCFS_separatrix(
            self.falsified_dn_eq.x,
            self.falsified_dn_eq.z,
            psi,
            o_points,
            x_points,
            double_null=True,
            psi_n_tol=1e-6,
        )
        legflux = LegFlux(self.falsified_dn_eq)
        legflux.x_points = x_points[:2]
        legflux.separatrix = separatrix
        legflux.n_null, legflux.sort_split = legflux.which_legs()
        legs = legflux.get_legs(1, 0.0)
        assert len(legs) == 4
        assert "lower_inner" in legs
        assert "lower_outer" in legs
        assert len(legs["lower_inner"]) == 1
        assert len(legs["lower_outer"]) == 1
        x1 = legs["lower_inner"][0].x[0]
        legs = legflux.get_legs(1, 1.0)
        assert len(legs) == 4
        assert "lower_inner" in legs
        assert "lower_outer" in legs
        assert len(legs["lower_inner"]) == 1
        assert len(legs["lower_outer"]) == 1
        x2 = legs["lower_inner"][0].x[0]
        assert np.isclose(x1, x2)

    def assert_valid_leg(self, leg, x_point, rtol=1e-05):
        assert np.isclose(leg.z[0], x_point.z, rtol=rtol)
