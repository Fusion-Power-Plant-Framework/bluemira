# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from eqdsk.models import Sign

from bluemira.base.constants import EPS
from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.coils import Coil, CoilSet, SymmetricCircuit
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.harmonics.harmonics_approx_functions import (
    PointType,
    coil_harmonic_amplitude_matrix,
    coils_outside_lcfs_sphere,
    collocation_points,
    get_psi_harmonic_amplitudes,
    harmonic_amplitude_marix,
    lcfs_fit_metric,
    spherical_harmonic_approximation,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_constraint_functions import (
    SphericalHarmonicConstraintFunction,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_constraints import (
    SphericalHarmonicConstraint,
)
from bluemira.geometry.coordinates import Coordinates, in_polygon

TEST_PATH = get_bluemira_path("equilibria/test_data", subfolder="tests")


def test_lcfs_fit_metric():
    xa = [1, 2, 2, 1, 1]
    xb = [3, 4, 4, 3, 3]
    xc = [1.5, 2.5, 2.5, 1.5, 1.5]
    za = [1, 1, 2, 2, 1]
    zc = [1.5, 1.5, 2.5, 2.5, 1.5]

    poly1 = Coordinates({"x": xa, "z": za})
    poly2 = Coordinates({"x": xb, "z": za})
    poly3 = Coordinates({"x": xc, "z": zc})
    poly4 = Coordinates({"x": xc, "z": za})

    assert lcfs_fit_metric(poly1, poly1) == 0
    assert lcfs_fit_metric(poly1, poly2) == 1
    assert lcfs_fit_metric(poly1, poly3) == pytest.approx(0.75, rel=0, abs=EPS)
    assert lcfs_fit_metric(poly1, poly4) == pytest.approx(0.5, rel=0, abs=EPS)


def test_harmonic_amplitude_marix():
    r = np.array([1, 1, 1])
    theta = np.array([0, np.pi, 2 * np.pi])
    n = 3
    d = 2
    r_t = 1.0

    test_output = harmonic_amplitude_marix(r, theta, r_t)

    assert test_output.shape == (n, d)

    assert ((test_output - np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])) == 0).all()


def test_coil_harmonic_amplitude_matrix():
    coil = Coil(x=4, z=10, current=2e6, dx=1, dz=0.5, j_max=5.0, b_max=50, name="PF_1")
    circuit = SymmetricCircuit(
        Coil(
            x=1.5,
            z=6,
            current=1e6,
            dx=0.25,
            dz=0.5,
            j_max=10.0,
            b_max=100,
            ctype="PF",
            name="PF_2",
        ),
        Coil(
            x=1.5,
            z=-6,
            current=1e6,
            dx=0.25,
            dz=0.5,
            j_max=10.0,
            b_max=100,
            ctype="PF",
            name="PF_3",
        ),
    )

    sh_coil_names = ["PF_2", "PF_3"]

    coilset = CoilSet(coil, circuit)

    d = 2
    r_t = 1

    test_out_matrx = coil_harmonic_amplitude_matrix(coilset, d, r_t, sh_coil_names)

    assert test_out_matrx.shape[1] == len(sh_coil_names)
    assert test_out_matrx.shape[0] == d


def test_collocation_points():
    n_points = 8
    grid_num = (10, 10)

    x = [1, 1.5, 2, 2.1, 2, 1.5, 1, 0.9, 1]
    z = [-1.8, -1.9, -1.8, 0, 1.8, 1.9, 1.8, 0, -1.8]
    plasma_boundary = Coordinates({"x": x, "z": z})

    point_type_1 = PointType.ARC
    point_type_2 = PointType.ARC_PLUS_EXTREMA
    point_type_3 = PointType.RANDOM
    point_type_4 = PointType.RANDOM_PLUS_EXTREMA
    point_type_5 = PointType.GRID_POINTS

    colloc1 = collocation_points(plasma_boundary, point_type_1, n_points=n_points)
    colloc2 = collocation_points(plasma_boundary, point_type_2, n_points=n_points)
    colloc3 = collocation_points(plasma_boundary, point_type_3, n_points=n_points)
    colloc4 = collocation_points(plasma_boundary, point_type_4, n_points=n_points)
    colloc5 = collocation_points(plasma_boundary, point_type_5, grid_num=grid_num)

    assert colloc1.r.shape[0] == 8
    assert colloc2.r.shape[0] == 12
    assert colloc3.r.shape[0] == 8
    assert colloc4.r.shape[0] == 12
    assert colloc5.r.shape[0] == 64

    for x, z in zip(colloc2.x, colloc2.z, strict=False):
        assert in_polygon(x, z, plasma_boundary.xz.T, include_edges=True)

    for x, z in zip(colloc4.x, colloc4.z, strict=False):
        assert in_polygon(x, z, plasma_boundary.xz.T, include_edges=True)

    for x, z in zip(colloc5.x, colloc5.z, strict=False):
        assert in_polygon(x, z, plasma_boundary.xz.T, include_edges=True)


def test_SphericalHarmonicConstraintFunction():
    coil = Coil(x=4, z=10, current=2e6, dx=1, dz=0.5, j_max=5.0, b_max=50, name="PF_1")

    circuit = SymmetricCircuit(
        Coil(
            x=1.5,
            z=6,
            current=1e6,
            dx=0.25,
            dz=0.5,
            j_max=10.0,
            b_max=100,
            ctype="PF",
            name="PF_2",
        ),
        Coil(
            x=1.5,
            z=-6,
            current=1e6,
            dx=0.25,
            dz=0.5,
            j_max=10.0,
            b_max=100,
            ctype="PF",
            name="PF_3",
        ),
    )

    circuit2 = deepcopy(circuit)
    circuit2["PF_2"].name = "PF_4"
    circuit2["PF_3"].name = "PF_5"
    coilset = CoilSet(coil, circuit, circuit2)

    sh_coil_names = ["PF_1", "PF_2", "PF_3", "PF_4", "PF_5"]

    d = 3
    r_t = 1
    cur_expand_mat = coilset._opt_currents_expand_mat
    a_mat = coil_harmonic_amplitude_matrix(coilset, d, r_t, sh_coil_names)
    b_vec = np.array([1e-1, 1e-2, 1e-18])
    test_vector = cur_expand_mat @ np.array([1, 1, 1])
    test_result = a_mat @ test_vector
    test_constraint = SphericalHarmonicConstraintFunction(a_mat, b_vec, 0.0, 1)

    test_f_constraint = test_constraint.f_constraint(test_vector)

    for fc, res in zip(
        test_f_constraint,
        (test_result - b_vec),
        strict=False,
    ):
        assert fc == res


class TestRegressionSH:
    @classmethod
    def setup_class(cls):
        cls.eq = Equilibrium.from_eqdsk(
            Path(TEST_PATH, "SH_test_file.json").as_posix(),
            from_cocos=3,
            qpsi_sign=Sign.NEGATIVE,
            )
        cls.sh_coil_names, cls.bdry_r = coils_outside_lcfs_sphere(cls.eq)
        cls.test_colocation = collocation_points(
            plasma_boundary=cls.eq.get_LCFS(),
            point_type=PointType.ARC,
            n_points=18,
            )
        cls.test_v_psi = np.zeros(np.shape(cls.eq.grid.x))
        for n in cls.sh_coil_names:
            cls.test_v_psi = np.sum(
                [cls.test_v_psi, cls.eq.coilset[n].psi(cls.eq.grid.x, cls.eq.grid.z)],
                axis=0
            )

    def test_coils_outside_sphere_vacuum_psi(self):
        assert len(self.sh_coil_names) == 16
        assert self.bdry_r == pytest.approx(1.366, abs=0.005)

        test_p_psi = self.eq.plasma.psi()
        test_v_psi = np.zeros(np.shape(self.eq.grid.x))
        for n in self.sh_coil_names:
            test_v_psi = np.sum(
                [test_v_psi, self.eq.coilset[n].psi(self.eq.grid.x, self.eq.grid.z)],
                axis=0
            )
        non_cc_psi = self.eq.coilset.psi(self.eq.grid.x, self.eq.grid.z) - test_v_psi

        test_total = (test_v_psi + test_p_psi + non_cc_psi) - self.eq.psi()
        grid_zeros = test_total * 0.0

        assert test_total == pytest.approx(grid_zeros, abs=0.005)

    def test_get_psi_harmonic_amplitudes(self):
        test_sh_amps = get_psi_harmonic_amplitudes(
            self.test_v_psi,
            self.eq.grid,
            self.test_colocation,
            1.2
            )

        sh_amps = np.array([
            2.93e-01, 5.74e-01, -5.76e-04, 3.17e-01,
            5.92e-04, 2.06e-01, -5.19e-04, 1.05e-01,
            -3.37e-04, 3.90e-02, -1.79e-04, 9.48e-03
        ])

        assert test_sh_amps == pytest.approx(sh_amps, abs=0.005)

    def test_spherical_harmonic_approximation(self):
        (
            _,
            test_harmonic_amps,
            test_degree,
            test_fit_metric,
            _,
            test_r_t,
            test_sh_coilset_current,
            ) = spherical_harmonic_approximation(
                self.eq,
                n_points=10,
                point_type=PointType.GRID_POINTS,
                acceptable_fit_metric=0.02,
                )

        ref_harmonics = np.array([
            0.047, 0.116, -0.001, -0.038, 0.001,
            -0.013, -0.001, -0.000
        ])

        sh_coilset_current = np.array([
            7629.11, -3050.73, 35211.63,
            127127.43, 51966.15, 27968.25      ,
            -173896.19, -119241.84, 59982.21,
            12076.32, -8612.46, 32388.96,
            128677.73, 60747.42, 13703.61,
            -159090.69, -138734.02, 64852.55
        ])

        assert test_sh_coilset_current == pytest.approx(sh_coilset_current, abs=0.005)
        assert test_r_t == pytest.approx(1.3661, abs=0.0001)
        assert test_harmonic_amps == pytest.approx(ref_harmonics, abs=0.0005)
        assert test_degree == 8
        assert test_fit_metric == pytest.approx(0.0098, abs=0.0001)

    def test_SphericalHarmonicConstraint(self):
        r_t = 1.37
        ref_harmonics = get_psi_harmonic_amplitudes(
            self.test_v_psi,
            self.eq.grid,
            self.test_colocation,
            r_t
            )

        test_constraint_class = SphericalHarmonicConstraint(
            ref_harmonics=ref_harmonics,
            r_t=r_t,
            sh_coil_names=self.sh_coil_names,
        )

        assert test_constraint_class.constraint_type == "equality"
        assert test_constraint_class.max_degree == len(ref_harmonics)

        for test_tol, ref_tol in zip(
            test_constraint_class.tolerance,
                np.array([
                0.0001, 0.0001, 1e-06,
                0.0001, 1e-06, 0.0001,
                1e-06, 0.0001, 1e-06,
                0.0001, 1e-06, 1e-05
            ]),
            strict=False,
        ):
            assert test_tol == ref_tol

        tolerance = 0.0
        test_constraint_class = SphericalHarmonicConstraint(
            ref_harmonics=ref_harmonics,
            r_t=r_t,
            sh_coil_names=self.sh_coil_names,
            tolerance=tolerance,
        )

        assert len(test_constraint_class.tolerance) == len(ref_harmonics)
        for test_name, ref_name in zip(
            test_constraint_class.control_coil_names, self.sh_coil_names, strict=False
        ):
            assert test_name == ref_name

        test_eval = test_constraint_class.evaluate(self.eq)

        assert all(test_eval == 0)
        assert len(test_eval) == 12
