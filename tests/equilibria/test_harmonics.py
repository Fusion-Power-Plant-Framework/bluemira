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

from bluemira.base.constants import EPS
from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.coils import Coil, CoilSet, SymmetricCircuit
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.harmonics.harmonics_approx_functions import (
    PointType,
    coil_harmonic_amplitude_matrix,
    coils_outside_fs_sphere,
    collocation_points,
    fs_fit_metric,
    get_psi_harmonic_amplitudes,
    harmonic_amplitude_marix,
    spherical_harmonic_approximation,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_constraint_functions import (
    SphericalHarmonicConstraintFunction,
    ToroidalHarmonicConstraintFunction,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_constraints import (
    SphericalHarmonicConstraint,
    ToroidalHarmonicConstraint,
)
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (  # noqa: E501
    TauLimit,
    ToroidalHarmonicsSelectionResult,
    _approximation_direct_from_currents,
    _approximation_from_psi_fitting,
    _get_plasma_mask,
    _separate_psi_contributions,
    _set_n_degrees_of_freedom,
    coil_toroidal_harmonic_amplitude_matrix,
    f_hypergeometric,
    legendre_p,
    legendre_q,
    plot_toroidal_harmonic_approximation,
    toroidal_harmonic_approximate_psi,
    toroidal_harmonic_approximation,
    toroidal_harmonic_grid_and_coil_setup,
    toroidal_harmonics_to_positions,
)
from bluemira.geometry.coordinates import Coordinates, in_polygon
from bluemira.optimisation._tools import approx_derivative

TEST_PATH = get_bluemira_path("equilibria/test_data", subfolder="tests")


def test_fs_fit_metric():
    xa = [1, 2, 2, 1, 1]
    xb = [3, 4, 4, 3, 3]
    xc = [1.5, 2.5, 2.5, 1.5, 1.5]
    za = [1, 1, 2, 2, 1]
    zc = [1.5, 1.5, 2.5, 2.5, 1.5]

    poly1 = Coordinates({"x": xa, "z": za})
    poly2 = Coordinates({"x": xb, "z": za})
    poly3 = Coordinates({"x": xc, "z": zc})
    poly4 = Coordinates({"x": xc, "z": za})

    assert fs_fit_metric(poly1, poly1) == 0
    assert fs_fit_metric(poly1, poly2) == 1
    assert fs_fit_metric(poly1, poly3) == pytest.approx(0.75, rel=0, abs=EPS)
    assert fs_fit_metric(poly1, poly4) == pytest.approx(0.5, rel=0, abs=EPS)


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
            qpsi_positive=False,
        )
        cls.sh_coil_names, cls.bdry_r = coils_outside_fs_sphere(cls.eq)
        cls.test_colocation = collocation_points(
            plasma_boundary=cls.eq.get_LCFS(),
            point_type=PointType.GRID_POINTS,
            n_points=10,
        )
        cls.test_v_psi = np.zeros(np.shape(cls.eq.grid.x))
        for n in cls.sh_coil_names:
            cls.test_v_psi = np.sum(
                [cls.test_v_psi, cls.eq.coilset[n].psi(cls.eq.grid.x, cls.eq.grid.z)],
                axis=0,
            )

    def test_coils_outside_sphere_vacuum_psi(self):
        assert len(self.sh_coil_names) == 16
        assert self.bdry_r == pytest.approx(1.366, abs=0.005)

        test_p_psi = self.eq.plasma.psi()
        test_v_psi = np.zeros(np.shape(self.eq.grid.x))
        for n in self.sh_coil_names:
            test_v_psi = np.sum(
                [test_v_psi, self.eq.coilset[n].psi(self.eq.grid.x, self.eq.grid.z)],
                axis=0,
            )
        non_cc_psi = self.eq.coilset.psi(self.eq.grid.x, self.eq.grid.z) - test_v_psi

        test_total = (test_v_psi + test_p_psi + non_cc_psi) - self.eq.psi()
        grid_zeros = test_total * 0.0

        assert test_total == pytest.approx(grid_zeros, abs=0.005)

    def test_get_psi_harmonic_amplitudes(self):
        test_sh_amps = get_psi_harmonic_amplitudes(
            self.test_v_psi, self.eq.grid, self.test_colocation, 1.3661
        )

        sh_amps = np.array([
            2.7949789e-05,
            1.1597561e-01,
            -6.9713936e-04,
            -3.7900780e-02,
            1.0315314e-03,
            -1.2796998e-02,
            -1.1939407e-03,
            -2.0413978e-04,
            -1.3979516e-03,
            4.3234701e-03,
            -8.3544534e-04,
            6.4929064e-03,
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

        ref_harmonics = get_psi_harmonic_amplitudes(
            self.test_v_psi,
            self.eq.grid,
            self.test_colocation,
            test_r_t,
        )

        ref_harmonics = ref_harmonics[:test_degree]

        sh_coilset_current = np.array([
            7629.11,
            -9301.684,
            31443.84,
            131204.70,
            49954.62,
            32081.58,
            -174210.2,
            -127567.9,
            64428.81,
            12076.32,
            -10068.75,
            32112.56,
            133528.9,
            57743.38,
            14104.47,
            -162735.4,
            -131556.2,
            68837.19,
        ])

        assert test_sh_coilset_current == pytest.approx(sh_coilset_current, rel=1e-3)
        assert test_r_t == pytest.approx(1.3661, abs=0.0001)
        # Even numbered harmonics zero'd out
        assert test_harmonic_amps[1::2] == pytest.approx(ref_harmonics[1::2], rel=1e-3)
        assert test_degree == 8
        assert test_fit_metric == pytest.approx(0.01048, rel=1e-3)

    def test_SphericalHarmonicConstraint(self):
        r_t = 1.37
        ref_harmonics = get_psi_harmonic_amplitudes(
            self.test_v_psi, self.eq.grid, self.test_colocation, r_t
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
                1e-06,
                0.0001,
                1e-06,
                1e-05,
                1e-06,
                1e-05,
                1e-06,
                1e-06,
                1e-06,
                1e-06,
                1e-06,
                1e-06,
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


def test_hypergeometric_function():
    zs = np.linspace(0, 1, 10, endpoint=False)
    expected_hypergeometric_values = [
        1.0,
        1.1111111111111112,
        1.2499999999999976,
        1.4285714285564857,
        1.6666666593365895,
        1.9999990463256836,
        2.4999451576234,
        3.3314715137863895,
        4.953883139815727,
        8.905810108684877,
    ]
    test_hypergeometric_values = [f_hypergeometric(1, 1, 1, z) for z in zs]

    np.testing.assert_almost_equal(
        test_hypergeometric_values, expected_hypergeometric_values
    )


def test_legendre_p_function():
    # test edge case x=1
    assert (legendre_p(1 / 2, 1, 1)) == 0

    # test on float input for x
    expected_leg_p_values = [
        0.5465954438007155,
        0.6763192198914646,
        1.1915501725459268,
        2.6271654951241987,
        6.609207914698652,
        18.034312803906946,
    ]
    tau_c = 1.317059523987338
    test_leg_p_values = [legendre_p(m - 1 / 2, 1, np.cosh(tau_c)) for m in range(6)]
    np.testing.assert_almost_equal(test_leg_p_values, expected_leg_p_values)

    # test on array input for x
    tau = [
        [
            0.0,
            0.08372955,
            0.16399749,
            0.23765702,
            0.30217625,
            0.35588229,
            0.39808616,
            0.42904197,
            0.44975039,
            0.4616742,
        ],
        [
            0.0,
            0.12133246,
            0.23644833,
            0.33959315,
            0.4260977,
            0.49309235,
            0.53997706,
            0.56831107,
            0.581127,
            0.58202564,
        ],
        [
            0.0,
            0.18447222,
            0.35851489,
            0.51119464,
            0.63214509,
            0.7146438,
            0.75862372,
            0.77049285,
            0.75971917,
            0.73523295,
        ],
        [
            0.0,
            0.28770908,
            0.56575751,
            0.81555949,
            1.00499185,
            1.10298173,
            1.11064294,
            1.06120413,
            0.98863453,
            0.91246107,
        ],
        [
            0.0,
            0.41454837,
            0.8554559,
            1.34411363,
            1.82028299,
            1.92922411,
            1.67339115,
            1.41273933,
            1.2137264,
            1.06314908,
        ],
        [
            0.0,
            0.44040904,
            0.92353766,
            1.51526075,
            2.29399827,
            2.40455168,
            1.8496136,
            1.4901662,
            1.25492087,
            1.08800009,
        ],
        [
            0.0,
            0.32647169,
            0.64856608,
            0.94808807,
            1.17801893,
            1.28002018,
            1.25564261,
            1.16635152,
            1.06229698,
            0.9644803,
        ],
        [
            0.0,
            0.21072502,
            0.40995364,
            0.58446862,
            0.72026324,
            0.8075756,
            0.84662664,
            0.84744186,
            0.82369049,
            0.7870265,
        ],
        [
            0.0,
            0.13692432,
            0.26647453,
            0.3816919,
            0.47680051,
            0.54827125,
            0.59560302,
            0.62118671,
            0.62924839,
            0.62451388,
        ],
        [
            0.0,
            0.09310907,
            0.18209437,
            0.26320399,
            0.33343162,
            0.39084136,
            0.43474237,
            0.46563115,
            0.48491198,
            0.494502,
        ],
    ]
    expected_leg_p_array_values = [
        [
            0.0,
            0.04186784,
            0.08202192,
            0.11889972,
            0.15123614,
            0.17818564,
            0.19938864,
            0.21495697,
            0.22538017,
            0.23138509,
        ],
        [
            0.0,
            0.06067558,
            0.11829428,
            0.17000826,
            0.2134756,
            0.24721977,
            0.27088564,
            0.2852108,
            0.29169647,
            0.29215139,
        ],
        [
            0.0,
            0.09226917,
            0.17950754,
            0.25635186,
            0.31755608,
            0.35952973,
            0.38199711,
            0.38807258,
            0.38255762,
            0.37003928,
        ],
        [
            0.0,
            0.14398195,
            0.28391901,
            0.41119106,
            0.50939919,
            0.56100979,
            0.56507249,
            0.53892802,
            0.50084333,
            0.46119982,
        ],
        [
            0.0,
            0.20766601,
            0.43172826,
            0.69118788,
            0.96779818,
            1.03602741,
            0.87903472,
            0.72924805,
            0.62017331,
            0.53995338,
        ],
        [
            0.0,
            0.2206775,
            0.46694506,
            0.78710863,
            1.28247932,
            1.36364808,
            0.98595564,
            0.77282818,
            0.64243886,
            0.55307723,
        ],
        [
            0.0,
            0.16342341,
            0.32589439,
            0.47970191,
            0.60099105,
            0.65607995,
            0.64283031,
            0.59474588,
            0.53950413,
            0.48823783,
        ],
        [
            0.0,
            0.10541196,
            0.20535533,
            0.2933882,
            0.36239668,
            0.40708945,
            0.42717724,
            0.4275973,
            0.41537099,
            0.39654467,
        ],
        [
            0.0,
            0.06847561,
            0.13333812,
            0.19114936,
            0.23900644,
            0.27507714,
            0.2990271,
            0.31199578,
            0.31608597,
            0.31368365,
        ],
        [
            0.0,
            0.04655875,
            0.09107898,
            0.13169913,
            0.16691592,
            0.19574715,
            0.21782546,
            0.23337839,
            0.24309511,
            0.24793067,
        ],
    ]
    test_leg_p_array_values = legendre_p(1 - 1 / 2, 1, np.cosh(tau))
    np.testing.assert_array_almost_equal(
        test_leg_p_array_values, expected_leg_p_array_values
    )


def test_legendre_q_function():
    # test edge case x=1
    assert (legendre_q(1 / 2, 1, 1)) == np.inf

    # test on float input for x
    expected_leg_q_values = [
        1.0307750675502585,
        0.277742192957337,
        0.038136379987310565,
        0.0035080783837524734,
        0.0002425055233220778,
        1.3424505697649924e-05,
    ]
    tau_c = 1.2824746787307681
    test_leg_q_values = [legendre_q(m - 1 / 2, 1, np.cosh(tau_c)) for m in range(6)]
    np.testing.assert_array_almost_equal(test_leg_q_values, expected_leg_q_values)

    # test on array input for x
    expected_leg_q_array_values = [
        [
            np.inf,
            1.23904515,
            1.92372849,
            2.04343658,
            1.89355154,
            1.69645186,
            1.53574682,
            1.4249278,
            1.35676829,
            1.32131784,
        ],
        [
            np.inf,
            1.64513686,
            2.0426168,
            1.7448037,
            1.41237249,
            1.19371218,
            1.06787662,
            1.0034538,
            0.97965134,
            0.98350464,
        ],
        [
            np.inf,
            2.00767348,
            1.65743101,
            1.12066779,
            0.84139608,
            0.70633603,
            0.65024658,
            0.64160748,
            0.6629882,
            0.70390679,
        ],
        [
            np.inf,
            1.92592547,
            0.9764955,
            0.56041541,
            0.38635565,
            0.32581474,
            0.32916628,
            0.36912197,
            0.4296457,
            0.50176621,
        ],
        [
            np.inf,
            1.48799023,
            0.54590827,
            0.21698133,
            0.08495667,
            0.07183359,
            0.1285417,
            0.20972024,
            0.2994839,
            0.39255699,
        ],
        [
            np.inf,
            1.48799023,
            0.54590827,
            0.21698133,
            0.08495667,
            0.07183359,
            0.1285417,
            0.20972024,
            0.2994839,
            0.39255699,
        ],
        [
            np.inf,
            1.92592547,
            0.9764955,
            0.56041541,
            0.38635565,
            0.32581474,
            0.32916628,
            0.36912197,
            0.4296457,
            0.50176621,
        ],
        [
            np.inf,
            2.00767348,
            1.65743101,
            1.12066779,
            0.84139608,
            0.70633603,
            0.65024658,
            0.64160748,
            0.6629882,
            0.70390679,
        ],
        [
            np.inf,
            1.64513686,
            2.0426168,
            1.7448037,
            1.41237249,
            1.19371218,
            1.06787662,
            1.0034538,
            0.97965134,
            0.98350464,
        ],
        [
            np.inf,
            1.23904515,
            1.92372849,
            2.04343658,
            1.89355154,
            1.69645186,
            1.53574682,
            1.4249278,
            1.35676829,
            1.32131784,
        ],
    ]

    tau = [
        [
            0.0,
            0.08824793,
            0.17271863,
            0.24997798,
            0.31726988,
            0.37279683,
            0.41586667,
            0.44684381,
            0.46691519,
            0.47775572,
        ],
        [
            0.0,
            0.12882178,
            0.25086733,
            0.35981143,
            0.45047258,
            0.51968076,
            0.56687984,
            0.59400448,
            0.60463245,
            0.60288812,
        ],
        [
            0.0,
            0.1971082,
            0.38319882,
            0.54623865,
            0.67422941,
            0.75913876,
            0.8010357,
            0.80790876,
            0.79111267,
            0.76086156,
        ],
        [
            0.0,
            0.30679727,
            0.60606806,
            0.87892896,
            1.08645363,
            1.18650918,
            1.180427,
            1.11297593,
            1.02555129,
            0.93885095,
        ],
        [
            0.0,
            0.42901091,
            0.89299788,
            1.43438632,
            2.03536735,
            2.14522972,
            1.76656246,
            1.4556128,
            1.23691967,
            1.07724722,
        ],
        [
            0.0,
            0.42901091,
            0.89299788,
            1.43438632,
            2.03536735,
            2.14522972,
            1.76656246,
            1.4556128,
            1.23691967,
            1.07724722,
        ],
        [
            0.0,
            0.30679727,
            0.60606806,
            0.87892896,
            1.08645363,
            1.18650918,
            1.180427,
            1.11297593,
            1.02555129,
            0.93885095,
        ],
        [
            0.0,
            0.1971082,
            0.38319882,
            0.54623865,
            0.67422941,
            0.75913876,
            0.8010357,
            0.80790876,
            0.79111267,
            0.76086156,
        ],
        [
            0.0,
            0.12882178,
            0.25086733,
            0.35981143,
            0.45047258,
            0.51968076,
            0.56687984,
            0.59400448,
            0.60463245,
            0.60288812,
        ],
        [
            0.0,
            0.08824793,
            0.17271863,
            0.24997798,
            0.31726988,
            0.37279683,
            0.41586667,
            0.44684381,
            0.46691519,
            0.47775572,
        ],
    ]
    test_leg_q_array_values = legendre_q(1 - 1 / 2, 1, np.cosh(tau))
    np.testing.assert_array_almost_equal(
        test_leg_q_array_values, expected_leg_q_array_values
    )

    # test the different cases in the if block are calculated as expected
    # testing legQ is float and x==1, so expect legQ to be set to inf
    test_result = legendre_q(2, 2, 1)
    assert test_result == np.inf

    # testing legQ is float and x!=1, so returned legQ should not be inf
    test_result = legendre_q(2, 2, 2)
    assert test_result != np.inf

    # one mode, multiple x, check that the entry where x==1 is inf and the
    # entry where x==2 is not inf
    test_result = legendre_q(2, 2, np.array([1, 2]))
    assert test_result[0] == np.inf
    assert test_result[1] != np.inf

    # multiple modes, one x, x==1, check legQ is all np.inf
    test_result = legendre_q(np.array([1, 2]), 2, 1)
    assert all(test_result == np.inf)

    # multiple modes, one x, x!=1, check legQ is not all np.inf
    test_result = legendre_q(np.array([1, 2]), 2, 2)
    assert all(test_result != np.inf)

    # one mode, multiple x
    test_result = legendre_q(1, 2, np.array([1, 2]))
    assert test_result[0] == np.inf
    assert test_result[1] != np.inf

    # multiple modes, multiple x
    # grid coordinates
    test_result = legendre_q(
        np.array([1, 2])[:, None, None], 2, np.array([[1, 2], [3, 4]])
    )
    assert all(test_result[:, 0, 0] == np.inf)
    assert all(test_result[:, 1, 0] != np.inf)

    # array coordinates
    test_result = legendre_q(np.array([1, 2])[:, None], 2, np.array([1, 2]))
    assert all(test_result[:, 0] == np.inf)
    assert all(test_result[:, 1] != np.inf)


class TestRegressionTH:
    @staticmethod
    def _read_json(file_path: str) -> dict:
        with open(file_path) as f:
            return json.load(f)

    @classmethod
    def setup_class(cls):
        cls.eq = Equilibrium.from_eqdsk(
            Path(TEST_PATH, "eqref_OOB.json").as_posix(),
            from_cocos=7,
        )
        cls.param_dict = cls._read_json(
            Path(TEST_PATH, "toroidal_harmonics_test_data.json")
        )

        cls.eq_cc_set = deepcopy(cls.eq)
        cls.eq_cc_set.coilset.control = cls.eq.coilset.get_coiltype("PF").name
        cls.R_0, cls.Z_0 = cls.eq.effective_centre()
        cls.test_th_params = toroidal_harmonic_grid_and_coil_setup(
            eq=cls.eq, R_0=cls.R_0, Z_0=cls.Z_0
        )
        cls.test_cc_th_params = toroidal_harmonic_grid_and_coil_setup(
            eq=cls.eq_cc_set, R_0=cls.R_0, Z_0=cls.Z_0
        )
        cls.cos_m = np.array([0, 1, 2, 3])
        cls.sin_m = np.array([2, 4])
        cls.psi_norm = 0.95
        cls.n_degrees_of_freedom = 6
        cls.max_harmonic_mode = 5
        cls.test_approx_result = toroidal_harmonic_approximation(
            eq=cls.eq,
            th_params=cls.test_th_params,
            psi_norm=cls.psi_norm,
            n_degrees_of_freedom=cls.n_degrees_of_freedom,
            max_harmonic_mode=cls.max_harmonic_mode,
            plasma_mask=True,
        )
        cls.collocation = collocation_points(
            cls.eq.get_LCFS(),
            PointType.GRID_POINTS,
        )

    @pytest.mark.parametrize("cc_all", [True, False])
    def test_toroidal_harmonic_grid_and_coil_setup(self, cc_all):
        # Testing with the default TauLimit=LCFS argument
        expected_th_number_of_coils = 11 if cc_all else 6
        expected_shape = (150, 200)
        eq = self.eq if cc_all else self.eq_cc_set
        th_params = self.test_th_params if cc_all else self.test_cc_th_params

        assert len(th_params.sigma_c) == expected_th_number_of_coils
        assert len(th_params.th_coil_names) == expected_th_number_of_coils

        assert th_params.R_0 == self.R_0
        assert th_params.Z_0 == self.Z_0

        assert np.shape(th_params.R) == expected_shape
        assert np.shape(th_params.Z) == expected_shape

        assert len(eq.coilset.get_control_coils().x) == expected_th_number_of_coils
        assert eq.coilset.get_control_coils().x == pytest.approx(th_params.R_coils)

        assert len(eq.coilset.get_control_coils().z) == expected_th_number_of_coils
        assert eq.coilset.get_control_coils().z == pytest.approx(th_params.Z_coils)

        # fmt: off
        expected_tau = self.param_dict["test_toroidal_harmonic_grid_and_coil_setup"]["expected_tau"]  # noqa: E501
        # fmt: on

        np.testing.assert_array_almost_equal(expected_tau, th_params.tau[0])
        assert np.shape(th_params.tau) == expected_shape

        # fmt: off
        expected_sigma = self.param_dict["test_toroidal_harmonic_grid_and_coil_setup"]["expected_sigma"]  # noqa: E501

        # fmt: on
        np.testing.assert_array_almost_equal(expected_sigma, th_params.sigma[:, 0])
        assert np.shape(th_params.sigma) == expected_shape

        expected_tau_c = (
            [
                0.54743133,
                0.91207134,
                1.02497178,
                1.05937582,
                0.87854201,
                0.49935848,
                0.29062942,
                0.41637489,
                0.5559156,
                0.41855603,
                0.29235689,
            ]
            if cc_all
            else [
                0.54743133,
                0.91207134,
                1.02497178,
                1.05937582,
                0.87854201,
                0.49935848,
            ]
        )

        np.testing.assert_array_almost_equal(expected_tau_c, th_params.tau_c)
        assert len(th_params.tau_c) == expected_th_number_of_coils

        expected_sigma_c = (
            [
                1.31418168,
                0.7621819,
                0.23862408,
                -0.2691331,
                -0.80828933,
                -1.28074456,
                1.6158092,
                2.08653842,
                3.13527895,
                -2.09555349,
                -1.62195238,
            ]
            if cc_all
            else [
                1.31418168,
                0.7621819,
                0.23862408,
                -0.2691331,
                -0.80828933,
                -1.28074456,
            ]
        )

        np.testing.assert_array_almost_equal(expected_sigma_c, th_params.sigma_c)
        assert len(th_params.sigma_c) == expected_th_number_of_coils

        # Test with excluded coils by using the TauLimit.Manual and min_tau_value
        # arguments
        # fmt: off
        expected_tau_c = np.array([0.2885049, 0.51950757, 0.15212496, 0.22407844]) if cc_all else np.array([0.2885049, 0.51950757])  # noqa: E501

        expected_sigma_c = np.array([0.86775446, 0.62098295, 1.01835347, 1.26097536]) if cc_all else np.array([0.86775446, 0.62098295])  # noqa: E501
        # fmt: on
        lcfs = self.eq.get_LCFS()
        arg = np.argmin(lcfs.z)
        excluded_coils_R_0 = lcfs.x[arg]  # noqa: N806
        excluded_coils_Z_0 = lcfs.z[arg]  # noqa: N806

        excluded_coil_th_params = toroidal_harmonic_grid_and_coil_setup(
            eq=eq,
            R_0=excluded_coils_R_0,
            Z_0=excluded_coils_Z_0,
            tau_limit=TauLimit.MANUAL,
            min_tau_value=0.6,
        )
        np.testing.assert_array_almost_equal(
            expected_tau_c, excluded_coil_th_params.tau_c
        )
        np.testing.assert_array_almost_equal(
            expected_sigma_c, excluded_coil_th_params.sigma_c
        )

        # Test TauLimit.Coil
        expected_tau_c = (
            np.array([
                0.54743133,
                0.91207134,
                1.02497178,
                1.05937582,
                0.87854201,
                0.49935848,
                0.29062942,
                0.41637489,
                0.5559156,
                0.41855603,
                0.29235689,
            ])
            if cc_all
            else np.array([
                0.54743133,
                0.91207134,
                1.02497178,
                1.05937582,
                0.87854201,
                0.49935848,
            ])
        )

        expected_sigma_c = (
            np.array([
                1.31418168,
                0.7621819,
                0.23862408,
                -0.2691331,
                -0.80828933,
                -1.28074456,
                1.6158092,
                2.08653842,
                3.13527895,
                -2.09555349,
                -1.62195238,
            ])
            if cc_all
            else np.array([
                1.31418168,
                0.7621819,
                0.23862408,
                -0.2691331,
                -0.80828933,
                -1.28074456,
            ])
        )

        tau_limit_coil_th_params = toroidal_harmonic_grid_and_coil_setup(
            eq=eq, R_0=self.R_0, Z_0=self.Z_0, tau_limit=TauLimit.COIL
        )

        np.testing.assert_array_almost_equal(
            expected_tau_c, tau_limit_coil_th_params.tau_c
        )
        np.testing.assert_array_almost_equal(
            expected_sigma_c, tau_limit_coil_th_params.sigma_c
        )

    @pytest.mark.parametrize("cc_all", [True, False])
    def test_coil_toroidal_harmonic_amplitude_matrix(self, cc_all):
        eq = self.eq if cc_all else self.eq_cc_set
        th_params = self.test_th_params if cc_all else self.test_cc_th_params
        test_Am_cos, test_Am_sin = coil_toroidal_harmonic_amplitude_matrix(  # noqa: N806
            input_coils=eq.coilset,
            th_params=th_params,
            cos_m_chosen=self.cos_m,
            sin_m_chosen=self.sin_m,
        )
        expected_cos_shape = (len(self.cos_m), 11 if cc_all else 6)
        expected_sin_shape = (len(self.sin_m), 11 if cc_all else 6)

        # fmt: off
        expected_Am_cos = np.array(self.param_dict["test_coil_toroidal_harmonic_amplitude_matrix"]["expected_Am_cos"])  # noqa: N806, E501
        if not cc_all:
            expected_Am_cos = expected_Am_cos[:, :6]  # noqa: N806
        # fmt: on

        np.testing.assert_array_almost_equal(test_Am_cos, expected_Am_cos)
        assert np.shape(test_Am_cos) == expected_cos_shape

        # fmt: off
        expected_Am_sin = np.array(self.param_dict["test_coil_toroidal_harmonic_amplitude_matrix"]["expected_Am_sin"])  # noqa: E501, N806
        if not cc_all:
            expected_Am_sin = expected_Am_sin[:, :6]  # noqa: N806
        # fmt: on

        np.testing.assert_array_almost_equal(test_Am_sin, expected_Am_sin)
        assert np.shape(test_Am_sin) == expected_sin_shape

    @pytest.mark.parametrize("cc_all", [True, False])
    def test_toroidal_harmonic_approximate_psi(self, cc_all):
        eq = self.eq if cc_all else self.eq_cc_set
        th_params = self.test_th_params if cc_all else self.test_cc_th_params
        test_approx_coilset_psi, test_Am_cos, test_Am_sin = (  # noqa: N806
            toroidal_harmonic_approximate_psi(
                eq=eq,
                th_params=th_params,
                cos_m_chosen=self.cos_m,
                sin_m_chosen=self.sin_m,
            )
        )
        expected_psi_shape = (150, 200)
        mask = _get_plasma_mask(
            eq=eq, th_params=th_params, psi_norm=0.90, plasma_mask=True
        )

        # fmt: off
        # Large array so test subset within LCFS is as expected
        expected_coilset_psi = self.param_dict["test_toroidal_harmonic_approximate_psi"]["expected_coilset_psi"]  # noqa: E501
        # fmt: on

        # Note R and Z positions at [149, :] chosen and then masked to get a
        # subset of grid values to test that are within the LCFS
        # (i.e., where we are trying to get a good approximation).
        if not cc_all:
            non_cc = self.eq.coilset.get_coiltype("CS").psi(th_params.R, th_params.Z)
            expected_coilset_psi -= non_cc[149, :]

        np.testing.assert_allclose(
            test_approx_coilset_psi[149, :] * mask[149, :],
            expected_coilset_psi * mask[149, :],
            rtol=2e-2,
        )
        assert np.shape(test_approx_coilset_psi) == expected_psi_shape

        expected_Am_cos = (  # noqa: N806
            [-4.23864252, -3.58288708, -10.51447447, -11.673279]
            if cc_all
            else [-2.67549703, -5.30120682, -9.69272712, -12.50976612]
        )

        np.testing.assert_array_almost_equal(test_Am_cos, expected_Am_cos)
        assert np.shape(test_Am_cos) == (len(self.cos_m),)

        expected_Am_sin = (  # noqa: N806
            [0.19437508, 12.28756463] if cc_all else [0.27944301, 11.93858154]
        )

        np.testing.assert_array_almost_equal(test_Am_sin, expected_Am_sin)
        assert np.shape(test_Am_sin) == (len(self.sin_m),)

    def test_toroidal_harmonic_approximation(self):
        expected_cos_modes = np.array([0, 1, 2, 3, 4])
        expected_sin_modes = np.array([0])
        expected_cos_amplitudes = np.array([
            -4.2389231,
            -3.58792624,
            -10.58312314,
            -12.03133123,
            -15.05828234,
        ])
        expected_sin_amplitudes = np.array([-19.33699313])

        result = toroidal_harmonic_approximation(
            eq=self.eq,
            th_params=self.test_th_params,
            psi_norm=self.psi_norm,
            n_degrees_of_freedom=self.n_degrees_of_freedom,
            max_harmonic_mode=self.max_harmonic_mode,
            plasma_mask=True,
        )
        mask = _get_plasma_mask(
            eq=self.eq,
            th_params=self.test_th_params,
            psi_norm=self.psi_norm,
            plasma_mask=True,
        )

        assert result.error < 10
        assert len(result.cos_m) + len(result.sin_m) == self.n_degrees_of_freedom
        np.testing.assert_array_almost_equal(result.cos_m, expected_cos_modes)
        np.testing.assert_array_almost_equal(result.sin_m, expected_sin_modes)
        assert np.max(np.append(result.cos_m, result.sin_m)) <= self.max_harmonic_mode

        np.testing.assert_almost_equal(result.cos_amplitudes, expected_cos_amplitudes)
        np.testing.assert_almost_equal(result.sin_amplitudes, expected_sin_amplitudes)

        np.testing.assert_allclose(
            self.eq.plasma.psi(self.test_th_params.R, self.test_th_params.Z),
            result.fixed_psi,
        )
        np.testing.assert_allclose(
            self.eq.coilset.psi(self.test_th_params.R, self.test_th_params.Z) * mask,
            result.coilset_psi * mask,
            rtol=0.1,
        )

    def test_ToroidalHarmonicConstraint(self):
        test_cos_modes = np.array([0, 1, 2, 3, 4])
        test_sin_modes = np.array([3])
        test_cos_amplitudes = np.array([
            -4.23864252,
            -3.58288708,
            -10.51447447,
            -11.673279,
            -14.26727472,
        ])
        test_sin_amplitudes = np.array([3.15627377])
        test_result = ToroidalHarmonicsSelectionResult(
            cos_m=test_cos_modes,
            sin_m=test_sin_modes,
            cos_amplitudes=test_cos_amplitudes,
            sin_amplitudes=test_sin_amplitudes,
            error=0.0,
            coilset_psi=np.zeros(10),
            fixed_psi=np.zeros(10),
            true_unfixed_psi=np.zeros(10),
            th_params=self.test_th_params,
        )
        test_constraint_class_equality = ToroidalHarmonicConstraint(
            th_result=test_result,
            relative_tolerance_cos=1e-3,
            relative_tolerance_sin=1e-3,
            constraint_type="equality",
        )

        assert test_constraint_class_equality.constraint_type == "equality"
        assert len(test_constraint_class_equality.tolerance) == len(
            test_cos_modes
        ) + len(test_sin_modes)
        for test_tol, ref_tol in zip(
            test_constraint_class_equality.tolerance,
            np.abs(
                np.array([
                    test_cos_amplitudes[0] * 1e-3,
                    test_cos_amplitudes[1] * 1e-3,
                    test_cos_amplitudes[2] * 1e-3,
                    test_cos_amplitudes[3] * 1e-3,
                    test_cos_amplitudes[4] * 1e-3,
                    test_sin_amplitudes[0] * 1e-3,
                ])
            ),
            strict=False,
        ):
            assert test_tol == ref_tol

        test_constraint_class_inequality = ToroidalHarmonicConstraint(
            th_result=test_result,
            relative_tolerance_cos=1e-3,
            relative_tolerance_sin=1e-3,
            constraint_type="inequality",
        )

        # Multiply by 2 because inequality constraint is equivalent to 2 equality
        # constraints combined
        assert len(test_constraint_class_inequality.tolerance) == 2 * (
            len(test_cos_modes) + len(test_sin_modes)
        )

        for test_name, ref_name in zip(
            test_constraint_class_inequality.control_coil_names,
            self.test_th_params.th_coil_names,
            strict=False,
        ):
            assert test_name == ref_name

        test_eval_fn = test_constraint_class_inequality.evaluate(self.eq)

        assert all(test_eval_fn == 0)
        assert len(test_eval_fn) == 2 * (len(test_cos_modes) + len(test_sin_modes))

    # This test currently does not pass:
    def test_ToroidalHarmonicConstraintFunction(self):
        cos_modes = np.array([0, 1, 2, 3, 4])
        sin_modes = np.array([3])
        cos_amplitudes = np.array([
            -4.23864252,
            -3.58288708,
            -10.51447447,
            -11.673279,
            -14.26727472,
        ])
        sin_amplitudes = np.array([3.15627377])

        test_result = ToroidalHarmonicsSelectionResult(
            cos_m=cos_modes,
            sin_m=sin_modes,
            cos_amplitudes=cos_amplitudes,
            sin_amplitudes=sin_amplitudes,
            error=0.0,
            coilset_psi=np.zeros(10),
            fixed_psi=np.zeros(10),
            true_unfixed_psi=np.zeros(10),
            th_params=self.test_th_params,
        )

        # Vector of currents in MA for arg in constraint function
        vector = self.eq.coilset.current * 1e-6

        constraint_class = ToroidalHarmonicConstraint(
            th_result=test_result,
            relative_tolerance_cos=1e-3,
            relative_tolerance_sin=1e-3,
            constraint_type="equality",
        )
        constraint_class.prepare(self.eq)

        result = constraint_class._args["a_mat"] @ self.eq.coilset.current
        ref_function_result = result - constraint_class._args["b_vec"]

        test_constraint_function = ToroidalHarmonicConstraintFunction(
            a_mat=constraint_class._args["a_mat"],
            b_vec=constraint_class._args["b_vec"],
            scale=constraint_class._args["scale"],
            value=constraint_class._args["value"],
        )

        test_result = test_constraint_function.f_constraint(vector)

        for fc, res in zip(
            test_result,
            ref_function_result,
            strict=False,
        ):
            assert fc == pytest.approx(res)

        assert test_constraint_function.df_constraint(vector) == pytest.approx(
            approx_derivative(test_constraint_function.f_constraint, vector)
        )

    @pytest.mark.parametrize(
        ("n_dof", "max_harmonic_mode", "max_n_dof", "expected_dof"),
        [
            # Case where max_n_dof is hit
            (5, 5, 4, 4),
            # Case where 2 * max_harmonic_mode is hit
            (5, 2, 5, 4),
            # Case where everything OK
            (5, 5, 10, 5),
            # Case where max_n_dof is exceed and still > 2 * max_harmonic_mode
            (10, 4, 9, 8),
            # Case where n_dof is not specified and defaults to max
            (None, 5, 9, 9),
            # Case where n_dof is not specified and defaults to 2 * max_harmonic_mode
            (None, 4, 9, 8),
        ],
    )
    def test_th_n_dof_limits(
        self,
        n_dof: int | None,
        max_harmonic_mode: int,
        max_n_dof: int,
        expected_dof: int,
    ):
        n_dof = _set_n_degrees_of_freedom(n_dof, max_harmonic_mode, max_n_dof)
        assert n_dof == expected_dof

    def test_separate_psi_contributions(self):
        # fmt: off
        expected_true_coilset_psi = self.param_dict["test_separate_psi_contributions"]["expected_true_coilset_psi_collocation"]  # noqa: E501

        expected_fixed_psi = self.param_dict["test_separate_psi_contributions"]["expected_fixed_psi_collocation"]  # noqa: E501

        expected_collocation_psi = self.param_dict["test_separate_psi_contributions"]["expected_collocation_psi_collocation"]  # noqa: E501
        # fmt: on
        # Test with collocation points
        true_coilset_psi, fixed_psi, collocation_psi = _separate_psi_contributions(
            self.eq, self.test_th_params, self.collocation
        )

        np.testing.assert_almost_equal(true_coilset_psi[0], expected_true_coilset_psi)
        np.testing.assert_almost_equal(fixed_psi[0], expected_fixed_psi)
        np.testing.assert_almost_equal(collocation_psi, expected_collocation_psi)

        # Test without collocation points
        true_coilset_psi, fixed_psi, collocation_psi = _separate_psi_contributions(
            self.eq, self.test_th_params, collocation=None
        )
        np.testing.assert_almost_equal(true_coilset_psi[0], expected_true_coilset_psi)
        np.testing.assert_almost_equal(fixed_psi[0], expected_fixed_psi)
        assert collocation_psi is None

        # Test with excluded coils - using a larger approx region so only 4 coils are
        # included in the approximation

        lcfs = self.eq.get_LCFS()
        arg = np.argmin(lcfs.z)
        R_0 = lcfs.x[arg]
        Z_0 = lcfs.z[arg]
        excluded_coil_th_params = toroidal_harmonic_grid_and_coil_setup(
            eq=self.eq, R_0=R_0, Z_0=Z_0, tau_limit=TauLimit.MANUAL, min_tau_value=0.6
        )
        # fmt: off
        expected_true_coilset_psi = self.param_dict["test_separate_psi_contributions"]["expected_true_coilset_psi_excluded_coils"]  # noqa: E501

        expected_fixed_psi = self.param_dict["test_separate_psi_contributions"]["expected_fixed_psi_excluded_coils"]  # noqa: E501

        expected_collocation_psi = self.param_dict["test_separate_psi_contributions"]["expected_collocation_psi_excluded_coils"]  # noqa: E501
        # fmt: on

        true_coilset_psi, fixed_psi, collocation_psi = _separate_psi_contributions(
            self.eq, excluded_coil_th_params, collocation=self.collocation
        )

        # Check that 7 coils are excluded and 4 are included
        assert len(excluded_coil_th_params.th_coil_names) == (
            len(self.eq.coilset.name) - 7
        )

        np.testing.assert_almost_equal(true_coilset_psi[0], expected_true_coilset_psi)
        np.testing.assert_almost_equal(fixed_psi[0], expected_fixed_psi)
        np.testing.assert_almost_equal(collocation_psi, expected_collocation_psi)

    def test_get_plasma_mask(self):
        # fmt: off
        expected_mask_true = self.param_dict["test_get_plasma_mask"]["expected_mask_true"]  # noqa: E501
        # fmt: on
        expected_mask_false = 1
        mask = _get_plasma_mask(
            self.eq, self.test_th_params, self.psi_norm, plasma_mask=True
        )
        np.testing.assert_almost_equal(mask[0], expected_mask_true)

        mask = _get_plasma_mask(
            self.eq, self.test_th_params, self.psi_norm, plasma_mask=False
        )
        assert mask == expected_mask_false

    def test_toroidal_harmonics_to_positions(self):
        # test with collocation

        # fmt: off
        expected_collocation_cos = self.param_dict["test_toroidal_harmonics_to_positions"]["expected_collocation_cos"]  # noqa: E501
        expected_collocation_sin = self.param_dict["test_toroidal_harmonics_to_positions"]["expected_collocation_sin"]  # noqa: E501
        # fmt: on

        harmonics_to_collocation_cos, harmonics_to_collocation_sin = (
            toroidal_harmonics_to_positions(
                th_params=self.test_th_params,
                n_allowed=self.n_degrees_of_freedom,
                collocation=self.collocation,
            )
        )

        np.testing.assert_array_almost_equal(
            expected_collocation_cos, harmonics_to_collocation_cos
        )
        np.testing.assert_array_almost_equal(
            expected_collocation_sin, harmonics_to_collocation_sin
        )

        # test without collocation

        # fmt:off
        # test index [0][0]
        expected_cos = self.param_dict["test_toroidal_harmonics_to_positions"]["expected_cos"]  # noqa: E501
        # test index [1][1]
        expected_sin = self.param_dict["test_toroidal_harmonics_to_positions"]["expected_sin"]  # noqa: E501

        # fmt: on
        harmonics_to_collocation_cos, harmonics_to_collocation_sin = (
            toroidal_harmonics_to_positions(
                th_params=self.test_th_params,
                n_allowed=self.n_degrees_of_freedom,
                collocation=None,
            )
        )

        np.testing.assert_array_almost_equal(
            expected_cos, harmonics_to_collocation_cos[0][0]
        )
        np.testing.assert_array_almost_equal(
            expected_sin, harmonics_to_collocation_sin[1][1]
        )

    def test_approximation_from_psi_fitting(self):
        # fmt: off
        expected_error = 8.52104445428635

        expected_psi_fit = self.param_dict["test_approximation_from_psi_fitting"]["expected_psi_fit"]  # noqa: E501

        expected_cos_amps = np.array([-4.2389231, -3.58792624, -10.58312314,
                                      -12.03133123, -15.05828234])

        expected_sin_amps = np.array([-19.33699313])
        # fmt: on

        mask = _get_plasma_mask(
            eq=self.eq,
            th_params=self.test_th_params,
            psi_norm=self.psi_norm,
            plasma_mask=True,
        )
        true_coilset_psi, _, collocation_psi = _separate_psi_contributions(
            self.eq, self.test_th_params, self.collocation
        )
        modes = np.array([0, 1, 2, 3, 4, 5])
        error, psi_from_fit_to_collocation_points, cos_amps, sin_amps = (
            _approximation_from_psi_fitting(
                th_params=self.test_th_params,
                n_degrees_of_freedom=6,
                collocation=self.collocation,
                mode_id=modes,
                max_harmonic_mode=5,
                collocation_psi=collocation_psi,
                mask=mask,
                true_coilset_psi=true_coilset_psi,
            )
        )

        np.testing.assert_almost_equal(error, expected_error)
        np.testing.assert_array_almost_equal(
            psi_from_fit_to_collocation_points[0], expected_psi_fit
        )
        np.testing.assert_array_almost_equal(cos_amps, expected_cos_amps)
        np.testing.assert_array_almost_equal(sin_amps, expected_sin_amps)

    def test_approximation_direct_from_currents(self):
        # fmt: off
        expected_error = 9.41443590779587

        expected_approx_coilset_psi = self.param_dict["test_approximation_direct_from_currents"]["expected_approx_coilset_psi"]  # noqa: E501

        expected_cos_amps = np.array([-4.23864252, -3.58288708, -10.51447447, -11.673279,
            -14.26727472])

        expected_sin_amps = np.array([0.])
        # fmt: on

        mask = _get_plasma_mask(
            eq=self.eq,
            th_params=self.test_th_params,
            psi_norm=self.psi_norm,
            plasma_mask=True,
        )
        true_coilset_psi, _, _ = _separate_psi_contributions(
            self.eq, self.test_th_params, self.collocation
        )
        cos_m = np.array([0, 1, 2, 3, 4])
        sin_m = np.array([0])
        error, approximate_coilset_psi, cos_amps, sin_amps = (
            _approximation_direct_from_currents(
                eq=self.eq,
                th_params=self.test_th_params,
                cos_m_chosen=cos_m,
                sin_m_chosen=sin_m,
                true_coilset_psi=true_coilset_psi,
                mask=mask,
            )
        )

        np.testing.assert_almost_equal(error, expected_error)
        np.testing.assert_array_almost_equal(
            approximate_coilset_psi[0], expected_approx_coilset_psi
        )
        np.testing.assert_array_almost_equal(cos_amps, expected_cos_amps)
        np.testing.assert_array_almost_equal(sin_amps, expected_sin_amps)

    def test_plot_toroidal_harmonic_approximation(self):
        # Call the plotting function and check there are no exceptions raised
        plot_toroidal_harmonic_approximation(
            self.eq, self.test_th_params, self.test_approx_result
        )
