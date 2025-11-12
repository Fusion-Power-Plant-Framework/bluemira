# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

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
    @classmethod
    def setup_class(cls):
        cls.eq = Equilibrium.from_eqdsk(
            Path(TEST_PATH, "eqref_OOB.json").as_posix(),
            from_cocos=7,
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
        expected_tau = [1.15773453, 1.18171853, 1.20570252, 1.22968651, 1.2536705,
       1.27765449, 1.30163848, 1.32562247, 1.34960646, 1.37359045,
       1.39757444, 1.42155843, 1.44554242, 1.46952641, 1.4935104,
       1.51749439, 1.54147838, 1.56546237, 1.58944636, 1.61343036,
       1.63741435, 1.66139834, 1.68538233, 1.70936632, 1.73335031,
       1.7573343, 1.78131829, 1.80530228, 1.82928627, 1.85327026,
       1.87725425, 1.90123824, 1.92522223, 1.94920622, 1.97319021,
       1.9971742, 2.02115819, 2.04514219, 2.06912618, 2.09311017,
       2.11709416, 2.14107815, 2.16506214, 2.18904613, 2.21303012,
       2.23701411, 2.2609981, 2.28498209, 2.30896608, 2.33295007,
       2.35693406, 2.38091805, 2.40490204, 2.42888603, 2.45287003,
       2.47685402, 2.50083801, 2.524822, 2.54880599, 2.57278998,
       2.59677397, 2.62075796, 2.64474195, 2.66872594, 2.69270993,
       2.71669392, 2.74067791, 2.7646619, 2.78864589, 2.81262988,
       2.83661387, 2.86059786, 2.88458186, 2.90856585, 2.93254984,
       2.95653383, 2.98051782, 3.00450181, 3.0284858, 3.05246979,
       3.07645378, 3.10043777, 3.12442176, 3.14840575, 3.17238974,
       3.19637373, 3.22035772, 3.24434171, 3.2683257, 3.2923097,
       3.31629369, 3.34027768, 3.36426167, 3.38824566, 3.41222965,
       3.43621364, 3.46019763, 3.48418162, 3.50816561, 3.5321496,
       3.55613359, 3.58011758, 3.60410157, 3.62808556, 3.65206955,
       3.67605354, 3.70003753, 3.72402153, 3.74800552, 3.77198951,
       3.7959735, 3.81995749, 3.84394148, 3.86792547, 3.89190946,
       3.91589345, 3.93987744, 3.96386143, 3.98784542, 4.01182941,
       4.0358134, 4.05979739, 4.08378138, 4.10776537, 4.13174936,
       4.15573336, 4.17971735, 4.20370134, 4.22768533, 4.25166932,
       4.27565331, 4.2996373, 4.32362129, 4.34760528, 4.37158927,
       4.39557326, 4.41955725, 4.44354124, 4.46752523, 4.49150922,
       4.51549321, 4.5394772, 4.5634612, 4.58744519, 4.61142918,
       4.63541317, 4.65939716, 4.68338115, 4.70736514, 4.73134913,
       4.75533312, 4.77931711, 4.8033011, 4.82728509, 4.85126908,
       4.87525307, 4.89923706, 4.92322105, 4.94720504, 4.97118903,
       4.99517303, 5.01915702, 5.04314101, 5.067125, 5.09110899,
       5.11509298, 5.13907697, 5.16306096, 5.18704495, 5.21102894,
       5.23501293, 5.25899692, 5.28298091, 5.3069649, 5.33094889,
       5.35493288, 5.37891687, 5.40290087, 5.42688486, 5.45086885,
       5.47485284, 5.49883683, 5.52282082, 5.54680481, 5.5707888,
       5.59477279, 5.61875678, 5.64274077, 5.66672476, 5.69070875,
       5.71469274, 5.73867673, 5.76266072, 5.78664471, 5.8106287,
       5.8346127, 5.85859669, 5.88258068, 5.90656467, 5.93054866]
        # fmt: on

        np.testing.assert_array_almost_equal(expected_tau, th_params.tau[0])
        assert np.shape(th_params.tau) == expected_shape

        # fmt: off
        expected_sigma = [-3.14159265, -3.09942362, -3.0572546, -3.01508557, -2.97291654,
       -2.93074751, -2.88857848, -2.84640945, -2.80424042, -2.76207139,
       -2.71990236, -2.67773334, -2.63556431, -2.59339528, -2.55122625,
       -2.50905722, -2.46688819, -2.42471916, -2.38255013, -2.3403811,
       -2.29821208, -2.25604305, -2.21387402, -2.17170499, -2.12953596,
       -2.08736693, -2.0451979, -2.00302887, -1.96085984, -1.91869082,
       -1.87652179, -1.83435276, -1.79218373, -1.7500147, -1.70784567,
       -1.66567664, -1.62350761, -1.58133858, -1.53916956, -1.49700053,
       -1.4548315, -1.41266247, -1.37049344, -1.32832441, -1.28615538,
       -1.24398635, -1.20181732, -1.15964829, -1.11747927, -1.07531024,
       -1.03314121, -0.99097218, -0.94880315, -0.90663412, -0.86446509,
       -0.82229606, -0.78012703, -0.73795801, -0.69578898, -0.65361995,
       -0.61145092, -0.56928189, -0.52711286, -0.48494383, -0.4427748,
       -0.40060577, -0.35843675, -0.31626772, -0.27409869, -0.23192966,
       -0.18976063, -0.1475916, -0.10542257, -0.06325354, -0.02108451,
        0.02108451, 0.06325354, 0.10542257, 0.1475916, 0.18976063,
        0.23192966, 0.27409869, 0.31626772, 0.35843675, 0.40060577,
        0.4427748, 0.48494383, 0.52711286, 0.56928189, 0.61145092,
        0.65361995, 0.69578898, 0.73795801, 0.78012703, 0.82229606,
        0.86446509, 0.90663412, 0.94880315, 0.99097218, 1.03314121,
        1.07531024, 1.11747927, 1.15964829, 1.20181732, 1.24398635,
        1.28615538, 1.32832441, 1.37049344, 1.41266247, 1.4548315,
        1.49700053, 1.53916956, 1.58133858, 1.62350761, 1.66567664,
        1.70784567, 1.7500147, 1.79218373, 1.83435276, 1.87652179,
        1.91869082, 1.96085984, 2.00302887, 2.0451979, 2.08736693,
        2.12953596, 2.17170499, 2.21387402, 2.25604305, 2.29821208,
        2.3403811, 2.38255013, 2.42471916, 2.46688819, 2.50905722,
        2.55122625, 2.59339528, 2.63556431, 2.67773334, 2.71990236,
        2.76207139, 2.80424042, 2.84640945, 2.88857848, 2.93074751,
        2.97291654, 3.01508557, 3.0572546, 3.09942362, 3.14159265]

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
        expected_Am_cos = np.array([[3.56396716e-08, 1.13470959e-07, 1.58806895e-07,  # noqa: N806
         1.63547008e-07, 1.05077598e-07, 3.05896343e-08,
         9.03590421e-09, 1.54525917e-08, 2.37410318e-08,
         1.55733032e-08, 9.11630279e-09],
       [1.40846784e-08, 1.36481306e-07, 2.63553188e-07,
         2.71646853e-07, 1.19803799e-07, 1.35374182e-08,
        -6.16359691e-10, -1.16813502e-08, -3.70087674e-08,
        -1.19620390e-08, -7.06729931e-10],
       [-6.74135465e-08, 1.46354017e-08, 4.31862082e-07,
         4.43549836e-07, -1.30068041e-08, -5.42238507e-08,
        -1.75970745e-08, -1.62092957e-08, 5.17758157e-08,
        -1.58552348e-08, -1.77414406e-08],
       [-7.47994051e-08, -3.72996690e-07, 7.26703422e-07,
         7.28802461e-07, -3.75991626e-07, -6.69497567e-08,
         2.92230837e-09, 4.08398396e-08, -7.22623580e-08,
         4.12497200e-08, 3.35139782e-09]])
        if not cc_all:
            expected_Am_cos = expected_Am_cos[:, :6]  # noqa: N806
        # fmt: on

        np.testing.assert_array_almost_equal(test_Am_cos, expected_Am_cos)
        assert np.shape(test_Am_cos) == expected_cos_shape

        # fmt: off
        expected_Am_sin = np.array([[3.79946851e-08, 3.14970654e-07, 2.23323725e-07,  # noqa: N806
        -2.64832640e-07, -2.83902429e-07, -3.55337811e-08,
        -1.58848333e-09, -2.70834275e-08, -6.53828975e-10,
         2.76095960e-08, 1.82152436e-09],
       [-1.30124437e-07, 1.01506708e-07, 1.68074189e-06,
        -2.04990408e-06, 8.51665739e-08, 1.09780324e-07,
         4.69448814e-09, 4.65830266e-08, -2.59963682e-09,
        -4.61605764e-08, -5.38443604e-09]])
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
        expected_coilset_psi = np.array([
            -24.65091628, -24.8034873, -24.95459247, -25.10455675,
            -25.25366744, -25.40217676, -25.55030437, -25.69823978,
            -25.84614462, -25.99415484, -26.14238278, -26.29091918,
            -26.43983498, -26.58918317, -26.73900039, -26.8893085,
            -27.04011609, -27.19141979, -27.34320564, -27.49545018,
            -27.64812167, -27.80118106, -27.95458298, -28.10827661,
            -28.26220653, -28.41631345, -28.57053491, -28.72480593,
            -28.87905958, -29.03322754, -29.18724052, -29.34102877,
            -29.49452243, -29.64765192, -29.80034821, -29.95254317,
            -30.10416975, -30.25516227, -30.40545659, -30.55499024,
            -30.70370264, -30.85153517, -30.99843129, -31.14433662,
            -31.28919904, -31.43296869, -31.57559805, -31.71704196,
            -31.85725762, -31.99620462, -32.13384488, -32.27014273,
            -32.40506479, -32.53858, -32.67065959, -32.801277,
            -32.93040785, -33.05802994, -33.18412313, -33.30866931,
            -33.43165238, -33.55305813, -33.67287422, -33.79109012,
            -33.90769701, -34.02268778, -34.1360569, -34.24780039,
            -34.35791576, -34.46640195, -34.57325924, -34.67848921,
            -34.7820947, -34.88407968, -34.98444928, -35.08320967,
            -35.18036802, -35.27593247, -35.36991203, -35.46231656,
            -35.55315672, -35.6424439, -35.7301902, -35.81640835,
            -35.9011117, -35.98431415, -36.06603011, -36.1462745,
            -36.22506265, -36.3024103, -36.37833357, -36.45284889,
            -36.52597301, -36.59772293, -36.66811591, -36.7371694,
            -36.80490105, -36.87132865, -36.93647012, -37.00034351,
            -37.06296693, -37.12435857, -37.18453665, -37.24351943,
            -37.30132516, -37.35797209, -37.41347844, -37.46786239,
            -37.52114205, -37.57333548, -37.62446065, -37.67453542,
            -37.72357757, -37.77160474, -37.81863447, -37.86468413,
            -37.90977098, -37.95391212, -37.99712449, -38.03942486,
            -38.08082984, -38.12135586, -38.16101916, -38.19983582,
            -38.23782171, -38.27499251, -38.31136372, -38.34695061,
            -38.38176829, -38.41583163, -38.44915531, -38.48175381,
            -38.5136414, -38.54483213, -38.57533986, -38.60517822,
            -38.63436063, -38.66290033, -38.69081032, -38.71810341,
            -38.74479218, -38.77088903, -38.79640613, -38.82135546,
            -38.84574878, -38.86959766, -38.89291347, -38.91570737,
            -38.93799033, -38.95977312, -38.98106631, -39.00188029,
            -39.02222524, -39.04211118, -39.06154791, -39.08054507,
            -39.09911212, -39.11725831, -39.13499275, -39.15232434,
            -39.16926184, -39.18581382, -39.20198868, -39.21779466,
            -39.23323983, -39.24833212, -39.26307928, -39.27748891,
            -39.29156846, -39.30532522, -39.31876634, -39.33189881,
            -39.34472951, -39.35726514, -39.36951228, -39.38147737,
            -39.39316671, -39.40458647, -39.41574271, -39.42664134,
            -39.43728814, -39.44768879, -39.45784883, -39.4677737,
            -39.4774687, -39.48693904, -39.49618981, -39.50522597,
            -39.51405242, -39.5226739, -39.53109507, -39.53932051,
            -39.54735468, -39.55520192, -39.56286652, -39.57035264,
            -39.57766437, -39.58480571, -39.59178055, -39.59859271
        ])
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
        expected_true_coilset_psi = np.array([-26.79455257, -26.75073328, -26.72400013,
                                              -26.71282144,
            -26.71583298, -26.73181438, -26.75966931, -26.79840896,
            -26.84713811, -26.90504329, -26.9713828, -27.04547808,
            -27.12670646, -27.21449479, -27.30831406, -27.40767468,
            -27.5121224, -27.62123482, -27.73461825, -27.85190506,
            -27.97275131, -28.09683465, -28.22385254, -28.35352065,
            -28.48557136, -28.61975261, -28.75582667, -28.89356923,
            -29.03276843, -29.17322412, -29.3147471, -29.45715846,
            -29.60028903, -29.74397879, -29.88807641, -30.03243882,
            -30.17693073, -30.32142432, -30.46579882, -30.60994025,
            -30.75374105, -30.89709983, -31.0399211, -31.18211497,
            -31.32359698, -31.46428782, -31.60411313, -31.74300327,
            -31.88089318, -32.01772214, -32.15343358, -32.28797496,
            -32.42129756, -32.55335632, -32.68410973, -32.8135196,
            -32.941551, -33.06817206, -33.19335386, -33.3170703,
            -33.43929798, -33.56001606, -33.67920615, -33.79685219,
            -33.91294036, -34.02745897, -34.14039832, -34.25175064,
            -34.36150999, -34.46967215, -34.57623455, -34.68119615,
            -34.7845574, -34.88632013, -34.98648748, -35.08506383,
            -35.18205473, -35.2774668, -35.37130772, -35.46358611,
            -35.5543115, -35.64349427, -35.73114557, -35.8172773,
            -35.90190203, -35.98503295, -36.06668386, -36.14686907,
            -36.22560338, -36.30290207, -36.37878079, -36.4532556,
            -36.52634287, -36.59805928, -36.66842178, -36.73744755,
            -36.80515399, -36.87155866, -36.93667928, -37.00053371,
            -37.06313988, -37.12451584, -37.18467966, -37.24364948,
            -37.30144342, -37.35807963, -37.41357623, -37.46795131,
            -37.52122292, -37.57340902, -37.62452753, -37.67459624,
            -37.72363288, -37.77165505, -37.81868022, -37.86472574,
            -37.90980884, -37.95394656, -37.99715581, -38.03945336,
            -38.08085577, -38.12137945, -38.16104063, -38.19985536,
            -38.23783949, -38.2750087, -38.31137846, -38.34696403,
            -38.38178051, -38.41584276, -38.44916545, -38.48176306,
            -38.51364983, -38.54483982, -38.57534686, -38.60518461,
            -38.63436647, -38.66290566, -38.69081519, -38.71810786,
            -38.74479625, -38.77089275, -38.79640953, -38.82135857,
            -38.84575164, -38.86960028, -38.89291588, -38.91570958,
            -38.93799237, -38.95977499, -38.98106804, -39.00188188,
            -39.02222671, -39.04211254, -39.06154918, -39.08054625,
            -39.09911321, -39.11725933, -39.1349937, -39.15232523,
            -39.16926267, -39.1858146, -39.20198941, -39.21779535,
            -39.23324049, -39.24833274, -39.26307987, -39.27748947,
            -39.29156899, -39.30532572, -39.31876682, -39.33189928,
            -39.34472996, -39.35726557, -39.3695127, -39.38147777,
            -39.3931671, -39.40458685, -39.41574308, -39.4266417,
            -39.43728849, -39.44768913, -39.45784917, -39.46777403,
            -39.47746903, -39.48693936, -39.49619012, -39.50522629,
            -39.51405273, -39.5226742, -39.53109538, -39.53932082,
            -39.54735497, -39.55520222, -39.56286681, -39.57035293,
            -39.57766466, -39.584806, -39.59178083, -39.59859299])

        expected_fixed_psi = np.array([19.14389966, 19.55596624, 19.98873562,
                                       20.44047406, 20.9096357,
            21.39480107, 21.89469645, 22.40814183, 22.93406787, 23.47147609,
            24.01945099, 24.57713545, 25.14374466, 25.71852692, 26.30076533,
            26.88979809, 27.48508057, 28.08605147, 28.6919427, 29.30194076,
            29.91518042, 30.53078303, 31.14787646, 31.76557373, 32.38302837,
            32.99939244, 33.61385717, 34.22564823, 34.83401391, 35.43825714,
            36.03771263, 36.63175743, 37.21982117, 37.8013696, 38.37591554,
            38.94301996, 39.50228282, 40.05334772, 40.59590109, 41.12966786,
            41.65441061, 42.16992793, 42.67605352, 43.17265324, 43.65962179,
            44.13688209, 44.60438539, 45.06210594, 45.51003864, 45.94820007,
            46.37662733, 46.79537305, 47.20450374, 47.60410088, 47.99426033,
            48.37508789, 48.74669805, 49.10921367, 49.46276743, 49.80749839,
            50.14355034, 50.47107114, 50.79021287, 51.10113236, 51.40398871,
            51.69894248, 51.98615523, 52.26578926, 52.53800826, 52.80297618,
            53.06085634, 53.31181114, 53.5560018, 53.79358826, 54.02472943,
            54.24958272, 54.4683035, 54.68104498, 54.8879581, 55.08919141,
            55.28489105, 55.47520084, 55.66026209, 55.84021342, 56.0151907,
            56.18532702, 56.35075272, 56.5115953, 56.66797949, 56.82002725,
            56.96785781, 57.11158759, 57.25133024, 57.38719665, 57.51929497,
            57.64773067, 57.77260655, 57.89402277, 58.01207691, 58.12686399,
            58.23847654, 58.3470046, 58.45253581, 58.55515543, 58.65494638,
            58.75198932, 58.84636265, 58.93814262, 59.02740332, 59.11421676,
            59.1986529, 59.28077974, 59.36066331, 59.43836774, 59.51395532,
            59.58748654, 59.65902016, 59.7286132, 59.79632105, 59.86219747,
            59.92629465, 59.98866325, 60.04935247, 60.10841001, 60.16588222,
            60.22181406, 60.27624916, 60.32922988, 60.38079732, 60.43099135,
            60.47985069, 60.5274129, 60.57371441, 60.6187906, 60.66267579,
            60.70540329, 60.74700543, 60.78751358, 60.82695819, 60.86536881,
            60.90277412, 60.93920196, 60.97467934, 61.0092325, 61.04288688,
            61.0756672, 61.10759744, 61.1387009, 61.16900017, 61.19851721,
            61.22727333, 61.25528921, 61.28258494, 61.30918005, 61.33509346,
            61.36034358, 61.38494828, 61.40892491, 61.43229033, 61.45506091,
            61.47725256, 61.49888073, 61.51996043, 61.54050626, 61.56053237,
            61.58005254, 61.59908015, 61.61762821, 61.63570935, 61.65333587,
            61.67051969, 61.68727242, 61.70360535, 61.71952945, 61.73505538,
            61.75019352, 61.76495393, 61.77934643, 61.79338055, 61.80706556,
            61.82041049, 61.8334241, 61.84611493, 61.85849127, 61.8705612,
            61.88233258, 61.89381304, 61.90501003, 61.91593078, 61.92658233,
            61.93697154, 61.94710506, 61.9569894, 61.96663088, 61.97603563,
            61.98520964, 61.99415875, 62.00288862, 62.01140478, 62.0197126])

        expected_collocation_psi = np.array([-27.32528973, -28.57093524, -28.80612948,
                                             -27.87742605,
            -26.08253257, -24.56706749, -27.39521052, -29.34033433,
            -30.48539594, -30.69546152, -29.92011915, -28.32941347,
            -26.11144874, -26.93625281, -29.84002054, -31.78892574,
            -32.88094138, -33.08667607, -32.41329928, -30.97595268,
            -28.90386863, -29.9056778, -32.79457432, -34.71832629,
            -35.76923131, -35.97151872, -35.36662833, -34.04948228,
            -32.12545463, -33.47047635, -36.27884632, -38.14719996,
            -39.14969105, -39.33688854, -38.77264195, -37.54858829,
            -35.7651372, -40.31282299, -42.09840138, -43.03436815,
            -43.18592426, -42.63455628, -41.47696648, -39.81861952,
            -44.90576979, -46.58600433, -47.43033284, -47.51816924,
            -46.94891914, -45.82813299, -51.62336924, -52.34398837,
            -52.33226638, -51.71004913, -50.59201032])
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

        # Test with excluded coils - using a larger approx region so one coil is included
        # in the region

        lcfs = self.eq.get_LCFS()
        arg = np.argmin(lcfs.z)
        R_0 = lcfs.x[arg]
        Z_0 = lcfs.z[arg]
        excluded_coil_th_params = toroidal_harmonic_grid_and_coil_setup(
            eq=self.eq, R_0=R_0, Z_0=Z_0, tau_limit=TauLimit.MANUAL, min_tau_value=1.2
        )

        # fmt: off
        expected_true_coilset_psi = np.array([-17.7226812, -17.56639411, -17.41746507,
                                              -17.27555945,
            -17.14038513, -17.01168529, -16.88923222, -16.77282203,
            -16.66227016, -16.55740755, -16.4580775, -16.36413299,
            -16.27543449, -16.19184823, -16.11324464, -16.03949729,
            -15.97048192, -15.90607573, -15.84615686, -15.79060402,
            -15.73929618, -15.69211244, -15.64893192, -15.60963377,
            -15.57409719, -15.5422015, -15.51382626, -15.4888514,
            -15.46715738, -15.44862532, -15.43313717, -15.42057592,
            -15.4108257, -15.403772, -15.39930177, -15.39730362,
            -15.39766792, -15.40028693, -15.40505495, -15.41186837,
            -15.42062583, -15.43122824, -15.44357891, -15.45758356,
            -15.47315042, -15.49019023, -15.50861629, -15.52834449,
            -15.54929331, -15.57138382, -15.59453973, -15.61868729,
            -15.64375536, -15.66967536, -15.69638122, -15.72380939,
            -15.75189877, -15.78059069, -15.80982889, -15.83955941,
            -15.86973062, -15.9002931, -15.93119965, -15.96240519,
            -15.99386671, -16.02554327, -16.05739584, -16.08938737,
            -16.12148261, -16.15364814, -16.18585228, -16.21806502,
            -16.25025799, -16.28240439, -16.31447895, -16.34645783,
            -16.37831863, -16.41004029, -16.44160307, -16.47298845,
            -16.50417916, -16.53515903, -16.56591306, -16.59642725,
            -16.62668865, -16.65668528, -16.68640608, -16.71584088,
            -16.74498036, -16.77381602, -16.8023401, -16.83054561,
            -16.85842624, -16.88597634, -16.91319092, -16.94006556,
            -16.96659642, -16.99278023, -17.01861418, -17.04409598,
            -17.06922381, -17.09399624, -17.11841231, -17.14247138,
            -17.16617324, -17.18951797, -17.212506, -17.23513806,
            -17.25741516, -17.27933859, -17.30090987, -17.32213076,
            -17.34300325, -17.3635295, -17.3837119, -17.40355299,
            -17.42305547, -17.4422222, -17.46105618, -17.47956051,
            -17.49773844, -17.5155933, -17.53312854, -17.55034767,
            -17.5672543, -17.5838521, -17.6001448, -17.61613619,
            -17.63183011, -17.64723043, -17.66234107, -17.67716597,
            -17.6917091, -17.70597443, -17.71996597, -17.73368772,
            -17.74714369, -17.7603379, -17.77327435, -17.78595704,
            -17.79838995, -17.81057708, -17.82252237, -17.83422977,
            -17.8457032, -17.85694655, -17.86796368, -17.87875844,
            -17.88933463, -17.89969604, -17.9098464, -17.91978942,
            -17.92952877, -17.9390681, -17.94841098, -17.95756098,
            -17.96652161, -17.97529635, -17.98388861, -17.9923018,
            -18.00053925, -18.00860426, -18.01650009, -18.02422995,
            -18.031797, -18.03920438, -18.04645514, -18.05355232,
            -18.06049892, -18.06729786, -18.07395204, -18.0804643,
            -18.08683747, -18.09307428, -18.09917746, -18.10514968,
            -18.11099356, -18.11671168, -18.12230658, -18.12778076,
            -18.13313666, -18.13837669, -18.14350323, -18.14851859,
            -18.15342506, -18.15822488, -18.16292026, -18.16751335,
            -18.17200627, -18.17640112, -18.18069993, -18.18490471,
            -18.18901743, -18.19304001, -18.19697437, -18.20082235,
            -18.20458577, -18.20826643, -18.21186607, -18.21538643])

        expected_fixed_psi = np.array([5.69891238, 5.86703493, 6.03876936, 6.21353406,
                                       6.39077841,
                6.56998629, 6.75071374, 6.93256773, 7.11519152, 7.29826848,
                7.48150617, 7.66464867, 7.84746147, 8.0297329, 8.21127418,
                8.39191061, 8.57148722, 8.74986274, 8.92690785, 9.10250775,
                9.27655701, 9.44896021, 9.61963237, 9.78849584, 9.95548091,
            10.12052571, 10.28357443, 10.44457742, 10.60349114, 10.76027718,
            10.91490203, 11.06733693, 11.21755749, 11.36554339, 11.51127803,
            11.65474838, 11.79594487, 11.93486113, 12.07149348, 12.20584107,
            12.3379058, 12.46769205, 12.59520612, 12.72045654, 12.84345401,
            12.96421114, 13.08274197, 13.19906208, 13.31318876, 13.42514078,
            13.53493811, 13.64260158, 13.74815321, 13.85161616, 13.95301458,
            14.05237333, 14.14971775, 14.24507391, 14.33846859, 14.42992917,
            14.5194835, 14.60715961, 14.69298579, 14.77699069, 14.85920323,
            14.9396526, 15.01836811, 15.09537905, 15.1707146, 15.24440403,
            15.3164766, 15.38696154, 15.45588803, 15.52328511, 15.5891816,
            15.65360602, 15.71658673, 15.77815187, 15.83832937, 15.89714688,
            15.95463182, 16.01081128, 16.06571203, 16.11936043, 16.1717825,
            16.22300392, 16.27305004, 16.32194584, 16.36971595, 16.41638462,
            16.46197571, 16.50651271, 16.55001872, 16.59251639, 16.63402797,
            16.67457535, 16.71418, 16.75286301, 16.79064509, 16.82754655,
            16.86358733, 16.89878696, 16.9331646, 16.96673902, 16.99952862,
            17.03155143, 17.06282507, 17.0933668, 17.12319354, 17.15232182,
            17.18076786, 17.20854749, 17.23567623, 17.26216924, 17.28804137,
            17.31330713, 17.33798072, 17.362076, 17.38560653, 17.40858559,
            17.43102613, 17.45294081, 17.47434201, 17.49524182, 17.51565206,
            17.53558427, 17.55504972, 17.57405942, 17.59262414, 17.61075438,
            17.6284604, 17.64575222, 17.66263964, 17.6791322, 17.69523924,
            17.71096987, 17.72633299, 17.74133729, 17.75599124, 17.77030312,
            17.78428102, 17.79793282, 17.81126624, 17.8242888, 17.83700782,
            17.84943049, 17.86156379, 17.87341456, 17.88498945, 17.89629499,
            17.90733751, 17.91812322, 17.92865817, 17.93894826, 17.94899927,
            17.95881682, 17.96840641, 17.97777339, 17.98692301, 17.99586037,
            18.00459045, 18.01311814, 18.02144818, 18.02958521, 18.03753375,
            18.04529823, 18.05288296, 18.06029216, 18.06752993, 18.07460029,
            18.08150716, 18.08825436, 18.09484562, 18.10128459, 18.10757483,
            18.11371981, 18.11972293, 18.12558749, 18.13131672, 18.13691378,
            18.14238176, 18.14772365, 18.1529424, 18.15804088, 18.16302188,
            18.16788813, 18.1726423, 18.17728701, 18.18182478, 18.18625812,
            18.19058943, 18.1948211, 18.19895543, 18.20299468, 18.20694106,
            18.21079671, 18.21456375, 18.21824422, 18.22184013, 18.22535344])

        expected_collocation_psi = np.array([-25.05956201, -26.59577416, -27.08792222,
                                             -26.3831561,
            -24.78213267, -21.04424749, -24.30353995, -26.64525801,
            -28.14381627, -28.66315282, -28.15523809, -26.79475734,
            -24.77418745, -22.77137909, -26.20927716, -28.63944915,
            -30.15387908, -30.72507088, -30.36519732, -29.19623998,
            -27.35339692, -25.05484559, -28.59629416, -31.09487737,
            -32.64224008, -33.26919984, -33.02575682, -32.01639017,
            -30.35431267, -27.90210982, -31.49565404, -34.03934364,
            -35.61565871, -36.28831045, -36.13418473, -35.25764426,
            -33.76902927, -34.93848104, -37.50406301, -39.09241611,
            -39.79023938, -39.69729618, -38.92650068, -37.59546596,
            -38.94867755, -41.5134034, -43.0869864, -43.7800074,
            -43.71580323, -43.01971654, -46.0917751, -47.61342706,
            -48.26179151, -48.18817379, -47.53041753])
        # fmt: on

        true_coilset_psi, fixed_psi, collocation_psi = _separate_psi_contributions(
            self.eq, excluded_coil_th_params, collocation=self.collocation
        )

        assert (
            len(excluded_coil_th_params.th_coil_names) == len(self.eq.coilset.name) - 1
        )

        np.testing.assert_almost_equal(true_coilset_psi[0], expected_true_coilset_psi)
        np.testing.assert_almost_equal(fixed_psi[0], expected_fixed_psi)
        np.testing.assert_almost_equal(collocation_psi, expected_collocation_psi)

    def test_get_plasma_mask(self):
        # fmt: off
        expected_mask_true = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                                       0., 0., 0., 0., 0.,
            0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])
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
        expected_collocation_cos = np.array([[8.22894230e+00, 8.14816202e+00,
                                              8.12908305e+00,
            8.17200494e+00, 8.27618817e+00, 8.85072896e+00,
            8.65893839e+00, 8.52298914e+00, 8.44515085e+00,
            8.42677366e+00, 8.46812050e+00, 8.56853604e+00,
            8.72628982e+00, 9.14159996e+00, 8.95629639e+00,
            8.82508922e+00, 8.75002150e+00, 8.73230443e+00,
            8.77216931e+00, 8.86903379e+00, 9.02134315e+00,
            9.44072879e+00, 9.26157786e+00, 9.13485329e+00,
            9.06239867e+00, 9.04530356e+00, 9.08377179e+00,
            9.17728464e+00, 9.32444147e+00, 9.74641100e+00,
            9.57307622e+00, 9.45057635e+00, 9.38057953e+00,
            9.36406888e+00, 9.40122439e+00, 9.49158273e+00,
            9.63387839e+00, 9.89060282e+00, 9.77209245e+00,
            9.70441241e+00, 9.68845224e+00, 9.72437112e+00,
            9.81175422e+00, 9.94945501e+00, 1.02135886e+01,
            1.00988478e+01, 1.00333532e+01, 1.00179120e+01,
            1.00526649e+01, 1.01372399e+01, 1.04303602e+01,
            1.03669290e+01, 1.03519773e+01, 1.03856301e+01,
            1.04675526e+01],
        [-2.36833140e+00, -2.60437457e+00, -2.66043174e+00,
            -2.53448621e+00, -2.23124127e+00, -8.31591277e-01,
            -1.36338696e+00, -1.74662839e+00, -1.96848298e+00,
            -2.02112478e+00, -1.90282800e+00, -1.61763700e+00,
            -1.17547386e+00, -2.56016939e-01, -7.58931988e-01,
            -1.12046040e+00, -1.32939299e+00, -1.37893022e+00,
            -1.26758933e+00, -9.98862699e-01, -5.81387948e-01,
            3.27704854e-01, -1.48582889e-01, -4.90203209e-01,
            -6.87331861e-01, -7.34037917e-01, -6.29042906e-01,
            -3.75374119e-01, 1.94207697e-02, 9.16332996e-01,
            4.64538705e-01, 1.41140105e-01, -4.52202930e-02,
            -8.93474963e-02, 9.86508913e-03, 2.49782519e-01,
            6.23782752e-01, 1.08039174e+00, 7.73724470e-01,
            5.97222151e-01, 5.55452626e-01, 6.49376997e-01,
            8.76693296e-01, 1.23156999e+00, 1.69818885e+00,
            1.40691025e+00, 1.23945059e+00, 1.19984108e+00,
            1.28891916e+00, 1.50466664e+00, 2.04019715e+00,
            1.88104881e+00, 1.84342245e+00, 1.92804992e+00,
            2.13315591e+00],
        [-1.15550729e-02, 3.46698053e-01, 4.34171251e-01,
            2.38933757e-01, -2.12226319e-01, -1.25187976e+00,
            -6.31168504e-01, -1.38003750e-01, 1.65547039e-01,
            2.39550046e-01, 7.43166394e-02, -3.08367195e-01,
            -8.58824764e-01, -1.16179292e+00, -6.32671223e-01,
            -2.13917975e-01, 4.31310350e-02, 1.05718023e-01,
            -3.40683464e-02, -3.58415030e-01, -8.26463794e-01,
            -1.04808971e+00, -5.98319271e-01, -2.43439342e-01,
            -2.60664987e-02, 2.68065293e-02, -9.13127678e-02,
            -3.65788859e-01, -7.62873620e-01, -9.11738004e-01,
            -5.30553029e-01, -2.30424406e-01, -4.68785494e-02,
            -2.26736591e-03, -1.01947879e-01, -3.33831962e-01,
            -6.69913640e-01, -4.31682635e-01, -1.78649769e-01,
            -2.40608895e-02, 1.34932281e-02, -7.04291622e-02,
            -2.65796695e-01, -5.49270702e-01, -3.03986351e-01,
            -9.15387441e-02, 3.82042528e-02, 6.97150132e-02,
            -7.06538706e-04, -1.64698283e-01, 2.77634084e-02,
            1.36139937e-01, 1.62462772e-01, 1.03635886e-01,
            -3.33401282e-02],
        [1.05560742e-01, -3.24778365e-02, -7.02825409e-02,
            1.19156770e-02, 1.70622296e-01, 1.84608177e-01,
            1.68801955e-01, 8.01084010e-02, -4.19353746e-03,
            -2.79631039e-02, 2.33990826e-02, 1.17751624e-01,
            1.87841171e-01, 4.99225331e-02, 8.42036075e-02,
            5.01348247e-02, 5.56462966e-03, -7.83665775e-03,
            2.07404403e-02, 6.74912991e-02, 8.23645898e-02,
            -5.57650059e-02, 1.45663901e-02, 2.03984788e-02,
            4.95121418e-03, -8.45444854e-04, 1.10217025e-02,
            2.29076917e-02, -2.41460233e-03, -1.34872447e-01,
            -3.95287222e-02, -5.19776473e-03, 3.39768609e-04,
            3.65203923e-05, -1.60745337e-04, -1.33397899e-02,
            -6.71659903e-02, -7.79054784e-02, -2.37415849e-02,
            -3.35622498e-03, 2.47379626e-04, -8.52034327e-03,
            -3.93559346e-02, -1.12762150e-01, -1.00369603e-01,
            -3.29785270e-02, -2.39593533e-03, 3.91545328e-03,
            -1.07804098e-02, -5.36477346e-02, -3.12345774e-02,
            6.02985930e-03, 1.41444140e-02, -4.48519912e-03,
            -5.51267678e-02],
        [-3.03289053e-02, -1.09920905e-04, 1.12680267e-02,
            -1.19484688e-02, -3.60464100e-02, 7.93540248e-02,
            4.52127404e-03, -1.24969818e-02, -1.93160712e-03,
            3.20880163e-03, -6.86897938e-03, -1.12481426e-02,
            2.55724596e-02, 7.48488491e-02, 1.78031515e-02,
            -2.04818243e-03, -1.24211518e-03, 5.58157509e-04,
            -2.58460655e-03, 1.96331693e-03, 3.49829877e-02,
            5.82199264e-02, 1.94253038e-02, 2.51690409e-03,
            -2.63334288e-04, 2.00297708e-05, -8.48375050e-05,
            6.71782063e-03, 3.17293952e-02, 3.48139580e-02,
            1.35748808e-02, 2.77100654e-03, 1.16159655e-04,
            2.92649018e-08, 5.53857686e-04, 5.67707144e-03,
            2.06793396e-02, 3.55926176e-03, 3.17948487e-04,
            -1.31243573e-04, 1.39776863e-06, -1.56825533e-04,
            1.16147496e-03, 5.56038517e-03, -7.99419316e-03,
            -3.30279813e-03, -5.16247275e-04, 2.11245536e-04,
            -1.38024318e-03, -4.82956073e-03, -6.64699286e-03,
            -2.60380970e-04, 1.21515396e-03, -2.11677991e-03,
            -1.05644077e-02],
        [4.28262991e-03, 1.02951800e-03, -1.78726502e-03,
            3.34072138e-03, 1.23439298e-03, -2.20339988e-02,
            -9.17953266e-03, 2.13909512e-04, 5.26913882e-04,
            -3.61133920e-04, 1.05921520e-03, -2.15408210e-03,
            -1.47848248e-02, -5.43125109e-03, -4.86664024e-03,
            -7.60966292e-04, 1.38460898e-04, -3.77684032e-05,
            1.16215971e-04, -2.03915824e-03, -6.32530315e-03,
            5.27194399e-03, -7.94525458e-04, -4.39023081e-04,
            -1.41450309e-06, -9.29175756e-08, -7.62502010e-05,
            -7.56984206e-04, 1.67662200e-04, 1.04074652e-02,
            1.82253093e-03, 1.05858316e-04, -1.41432480e-06,
            -6.36360869e-09, 1.45602724e-06, 3.91479984e-04,
            3.87518989e-03, 2.58780388e-03, 3.18530739e-04,
            2.44991273e-06, -1.54321470e-07, 3.95141475e-05,
            8.00667468e-04, 4.74449490e-03, 1.58554308e-03,
            1.11673628e-05, -5.01212401e-05, 1.08248871e-05,
            -8.79854104e-05, 3.16677338e-04, -7.45324551e-04,
            -1.07589073e-04, 1.02872199e-04, -3.39226509e-04,
            -9.18706577e-04]])
        expected_collocation_sin = np.array([[-0.00000000e+00, -0.00000000e+00,
                                              0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, -0.00000000e+00,
            -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
            -0.00000000e+00, -0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
            -0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, -0.00000000e+00,
            -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
            -0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, -0.00000000e+00,
            -0.00000000e+00, -0.00000000e+00, 0.00000000e+00,
            0.00000000e+00, 0.00000000e+00, -0.00000000e+00,
            -0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
            0.00000000e+00],
        [-2.40772141e+00, -1.08434518e+00, 2.59601510e-01,
            1.59744231e+00, 2.90640928e+00, -4.73578237e+00,
            -3.55723773e+00, -2.32073373e+00, -1.04453581e+00,
            2.50034539e-01, 1.53907311e+00, 2.80238196e+00,
            4.01853411e+00, -4.57845900e+00, -3.43474167e+00,
            -2.23873833e+00, -1.00708091e+00, 2.41037389e-01,
            1.48412571e+00, 2.70421941e+00, 3.88189383e+00,
            -4.42895529e+00, -3.31878418e+00, -2.16133946e+00,
            -9.71784268e-01, 2.32562031e-01, 1.43231905e+00,
            2.61147085e+00, 3.75236891e+00, -4.28738210e+00,
            -3.20935939e+00, -2.08848638e+00, -9.38609896e-01,
            2.24599082e-01, 1.38360594e+00, 2.52409462e+00,
            3.62998984e+00, -3.10590827e+00, -2.01976759e+00,
            -9.07359712e-01, 2.17100375e-01, 1.33770020e+00,
            2.44161331e+00, 3.51416276e+00, -3.00805982e+00,
            -1.95490432e+00, -8.77898184e-01, 2.10032884e-01,
            1.29440657e+00, 2.36370552e+00, -1.89363133e+00,
            -8.50097549e-01, 2.03365539e-01, 1.25354051e+00,
            2.29006371e+00],
        [7.00480658e-01, 3.49240621e-01, -8.55462835e-02,
            -4.99700616e-01, -7.93516803e-01, 4.53641564e-01,
            5.67126597e-01, 4.79117781e-01, 2.44544238e-01,
            -6.01909358e-02, -3.47670319e-01, -5.33897103e-01,
            -5.49447899e-01, 1.30337134e-01, 2.93937612e-01,
            2.85687782e-01, 1.53355765e-01, -3.81239633e-02,
            -2.15127522e-01, -3.06609321e-01, -2.53237892e-01,
            -1.55953171e-01, 5.36814376e-02, 1.16415126e-01,
            7.37840749e-02, -1.88812642e-02, -9.93716299e-02,
            -1.07375874e-01, 7.89687267e-03, -4.08382364e-01,
            -1.56876528e-01, -3.12870962e-02, 4.52753297e-03,
            -2.14310699e-03, 1.45384754e-03, 6.67249565e-02,
            2.37243556e-01, -3.41663285e-01, -1.60413041e-01,
            -5.58839227e-02, 1.24496201e-02, 8.94604469e-02,
            2.19126795e-01, 4.38900106e-01, -5.03794177e-01,
            -2.73323658e-01, -1.08605400e-01, 2.51788560e-02,
            1.66309044e-01, 3.52542649e-01, -3.72067926e-01,
            -1.54632672e-01, 3.62872714e-02, 2.33434219e-01,
            4.69336544e-01],
        [-1.00462353e-01, -7.96400832e-02, 2.11120006e-02,
            1.01931653e-01, 7.08136923e-02, 3.21324134e-01,
            8.63387796e-02, -3.05863322e-02, -3.89520797e-02,
            1.08219196e-02, 4.61193370e-02, -3.01582011e-05,
            -1.63779632e-01, 2.95111275e-01, 1.10216605e-01,
            9.05485895e-03, -1.41721796e-02, 4.47818902e-03,
            1.27106325e-02, -3.76911088e-02, -1.72266894e-01,
            2.47548344e-01, 1.07872763e-01, 2.57952546e-02,
            -1.40235855e-03, 1.11135782e-03, -3.76711254e-03,
            -5.01755987e-02, -1.55500728e-01, 1.84329898e-01,
            8.59090973e-02, 2.53248412e-02, 2.33622776e-03,
            -1.69682435e-05, -7.51397340e-03, -4.37563500e-02,
            -1.19925281e-01, 4.95533702e-02, 1.21586431e-02,
            -5.95325246e-04, 5.08206847e-04, -1.86057532e-03,
            -2.34105774e-02, -7.06247709e-02, 2.90840001e-03,
            -1.02247740e-02, -8.39464861e-03, 2.24123732e-03,
            1.06438820e-02, 7.00471972e-03, -3.91285380e-02,
            -1.96718660e-02, 4.83889958e-03, 2.80305157e-02,
            4.44897679e-02],
        [-1.00087874e-03, 1.50432917e-02, -4.61970470e-03,
            -1.48131607e-02, 2.07666813e-02, -6.62040799e-02,
            -4.21782077e-02, -7.85051280e-03, 4.82764603e-03,
            -1.72119729e-03, -3.07717315e-03, 1.94977281e-02,
            5.53935233e-02, -1.70080754e-02, -2.10962565e-02,
            -6.98181539e-03, 7.58698859e-04, -4.62742767e-04,
            8.39672808e-04, 1.25249731e-02, 2.36597324e-02,
            1.77182585e-02, -3.51397826e-03, -3.12092562e-03,
            -2.12595370e-04, -5.59968570e-05, 1.00189663e-03,
            4.31587525e-03, -6.56962933e-04, 3.90150316e-02,
            8.79688262e-03, 7.66630042e-04, -2.26486723e-05,
            5.18955128e-07, -1.58000036e-05, -2.36385620e-03,
            -1.67471337e-02, 1.50814517e-02, 2.94714708e-03,
            1.38731391e-04, 1.73453166e-05, -6.49444444e-04,
            -5.97826026e-03, -2.45811152e-02, 1.51707128e-02,
            2.49176595e-03, -4.14492378e-04, 1.75480535e-04,
            -1.17277186e-05, -5.77227887e-03, -9.97540819e-04,
            -2.03878486e-03, 5.71330232e-04, 2.34094403e-03,
            -1.50853411e-03],
        [4.65123630e-03, -2.42265113e-03, 9.44951554e-04,
            1.14296599e-03, -9.29256377e-03, -1.86219659e-02,
            2.43424513e-03, 2.53197002e-03, -4.45926951e-04,
            2.55331207e-04, -2.80561925e-04, -3.72730438e-03,
            2.20315803e-03, -1.89378066e-02, -2.55518577e-03,
            7.08048462e-04, 1.38901452e-05, 4.43396259e-05,
            -2.80529642e-04, -4.09835829e-04, 6.88133090e-03,
            -1.36211835e-02, -3.49226935e-03, -2.15132372e-04,
            2.22146984e-05, 2.53120551e-06, -4.14868586e-05,
            8.73933248e-04, 6.47755585e-03, -5.93126598e-03,
            -2.08329902e-03, -3.01762462e-04, -5.76187594e-06,
            -2.12497325e-09, 4.08257549e-05, 7.28343821e-04,
            3.39811630e-03, 2.67481877e-04, 8.41783849e-05,
            1.04159118e-05, 5.13046620e-07, -3.25377717e-05,
            -1.23330478e-04, -5.46217948e-04, 2.46346066e-03,
            4.95728486e-04, -3.02253286e-06, 1.27395875e-05,
            -8.98744970e-05, -9.98188416e-04, 5.09023926e-04,
            -1.74900109e-04, 6.29832273e-05, 8.99346903e-05,
            -1.31978976e-03]])
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
        expected_collocation_cos = np.array([7.3373371, 7.37071929, 7.40388035,
                                             7.43680764, 7.46948926,
            7.50191398, 7.53407127, 7.56595123, 7.59754463, 7.62884283,
            7.6598378, 7.69052209, 7.72088881, 7.75093163, 7.78064472,
            7.81002277, 7.83906096, 7.86775494, 7.89610083, 7.92409517,
            7.95173494, 7.97901752, 8.00594067, 8.03250255, 8.05870165,
            8.08453685, 8.11000731, 8.13511254, 8.15985235, 8.18422682,
            8.20823632, 8.23188148, 8.25516319, 8.27808255, 8.30064091,
            8.32283983, 8.34468106, 8.36616655, 8.38729842, 8.40807898,
            8.42851068, 8.44859613, 8.46833806, 8.48773937, 8.50680304,
            8.5255322, 8.54393005, 8.56199991, 8.57974518, 8.59716935,
            8.61427598, 8.6310687, 8.64755119, 8.66372721, 8.67960054,
            8.69517502, 8.71045453, 8.72544298, 8.7401443, 8.75456244,
            8.76870139, 8.78256513, 8.79615766, 8.80948299, 8.82254513,
            8.83534807, 8.84789582, 8.86019237, 8.87224171, 8.88404778,
            8.89561456, 8.90694596, 8.9180459, 8.92891826, 8.9395669,
            8.94999564, 8.9602083, 8.97020865, 8.98000041, 8.9895873,
            8.99897297, 9.00816107, 9.01715518, 9.02595886, 9.03457562,
            9.04300894, 9.05126224, 9.05933892, 9.06724232, 9.07497575,
            9.08254246, 9.08994568, 9.09718856, 9.10427425, 9.11120582,
            9.1179863, 9.12461869, 9.13110594, 9.13745094, 9.14365656,
            9.1497256, 9.15566082, 9.16146496, 9.16714068, 9.17269061,
            9.17811735, 9.18342344, 9.18861138, 9.19368362, 9.19864258,
            9.20349063, 9.2082301, 9.21286328, 9.21739241, 9.22181969,
            9.22614729, 9.23037734, 9.23451191, 9.23855306, 9.24250279,
            9.24636307, 9.25013582, 9.25382295, 9.2574263, 9.26094771,
            9.26438894, 9.26775176, 9.27103788, 9.27424897, 9.27738669,
            9.28045265, 9.28344843, 9.28637558, 9.28923562, 9.29203003,
            9.29476027, 9.29742777, 9.30003393, 9.3025801, 9.30506764,
            9.30749784, 9.30987201, 9.31219138, 9.3144572, 9.31667066,
            9.31883296, 9.32094523, 9.32300861, 9.3250242, 9.32699308,
            9.32891632, 9.33079495, 9.33262997, 9.33442238, 9.33617316,
            9.33788324, 9.33955356, 9.34118501, 9.3427785, 9.34433489,
            9.34585502, 9.34733972, 9.34878982, 9.35020609, 9.35158932,
            9.35294027, 9.35425967, 9.35554826, 9.35680674, 9.35803581,
            9.35923614, 9.3604084, 9.36155324, 9.36267128, 9.36376315,
            9.36482945, 9.36587078, 9.36688771, 9.36788081, 9.36885063,
            9.3697977, 9.37072257, 9.37162574, 9.37250771, 9.37336899,
            9.37421004, 9.37503135, 9.37583336, 9.37661654, 9.37738131,
            9.37812811, 9.37885736, 9.37956946, 9.38026482, 9.38094382,
            9.38160685, 9.38225428, 9.38288647, 9.38350379, 9.38410657])
        # test index [1][1]
        expected_collocation_sin = np.array([-0.18732243, -0.18403523, -0.18078176,
                                             -0.17756278, -0.17437897,
            -0.17123096, -0.1681193, -0.16504451, -0.16200702, -0.15900724,
            -0.15604551, -0.15312213, -0.15023734, -0.14739137, -0.14458437,
            -0.14181647, -0.13908777, -0.13639831, -0.13374812, -0.13113718,
            -0.12856546, -0.12603288, -0.12353934, -0.12108472, -0.11866887,
            -0.11629161, -0.11395276, -0.11165209, -0.10938937, -0.10716435,
            -0.10497677, -0.10282633, -0.10071273, -0.09863566, -0.0965948,
            -0.0945898, -0.09262032, -0.09068599, -0.08878645, -0.08692132,
            -0.08509021, -0.08329273, -0.08152848, -0.07979706, -0.07809806,
            -0.07643106, -0.07479564, -0.07319139, -0.07161789, -0.07007469,
            -0.06856138, -0.06707754, -0.06562272, -0.06419649, -0.06279844,
            -0.06142813, -0.06008512, -0.05876899, -0.05747932, -0.05621568,
            -0.05497764, -0.05376479, -0.0525767, -0.05141296, -0.05027316,
            -0.04915688, -0.04806372, -0.04699328, -0.04594515, -0.04491894,
            -0.04391426, -0.04293071, -0.04196792, -0.0410255, -0.04010307,
            -0.03920028, -0.03831674, -0.0374521, -0.03660599, -0.03577808,
            -0.03496799, -0.03417541, -0.03339997, -0.03264136, -0.03189923,
            -0.03117327, -0.03046315, -0.02976857, -0.02908921, -0.02842476,
            -0.02777492, -0.02713941, -0.02651792, -0.02591017, -0.02531589,
            -0.02473478, -0.02416659, -0.02361105, -0.02306788, -0.02253684,
            -0.02201767, -0.02151012, -0.02101395, -0.02052892, -0.02005478,
            -0.01959131, -0.01913829, -0.01869549, -0.01826268, -0.01783966,
            -0.01742622, -0.01702215, -0.01662724, -0.0162413, -0.01586414,
            -0.01549555, -0.01513537, -0.01478339, -0.01443944, -0.01410336,
            -0.01377495, -0.01345406, -0.01314052, -0.01283416, -0.01253483,
            -0.01224237, -0.01195663, -0.01167745, -0.0114047, -0.01113823,
            -0.0108779, -0.01062357, -0.01037511, -0.01013238, -0.00989526,
            -0.00966362, -0.00943733, -0.00921629, -0.00900035, -0.00878943,
            -0.00858339, -0.00838213, -0.00818553, -0.00799351, -0.00780594,
            -0.00762273, -0.00744378, -0.00726899, -0.00709827, -0.00693152,
            -0.00676865, -0.00660958, -0.00645422, -0.00630248, -0.00615428,
            -0.00600954, -0.00586818, -0.00573011, -0.00559528, -0.00546359,
            -0.00533498, -0.00520938, -0.00508672, -0.00496692, -0.00484994,
            -0.00473568, -0.00462411, -0.00451515, -0.00440874, -0.00430483,
            -0.00420335, -0.00410425, -0.00400748, -0.00391297, -0.00382069,
            -0.00373057, -0.00364257, -0.00355663, -0.00347271, -0.00339076,
            -0.00331074, -0.0032326, -0.0031563, -0.00308179, -0.00300903,
            -0.00293799, -0.00286861, -0.00280087, -0.00273472, -0.00267013,
            -0.00260706, -0.00254547, -0.00248534, -0.00242662, -0.00236928,
            -0.0023133, -0.00225863, -0.00220525, -0.00215314, -0.00210224])

        # fmt: on
        harmonics_to_collocation_cos, harmonics_to_collocation_sin = (
            toroidal_harmonics_to_positions(
                th_params=self.test_th_params,
                n_allowed=self.n_degrees_of_freedom,
                collocation=None,
            )
        )

        np.testing.assert_array_almost_equal(
            expected_collocation_cos, harmonics_to_collocation_cos[0][0]
        )
        np.testing.assert_array_almost_equal(
            expected_collocation_sin, harmonics_to_collocation_sin[1][1]
        )

    def test_approximation_from_psi_fitting(self):
        # fmt: off
        expected_error = 8.52104445428635

        expected_psi_fit = np.array([-26.59850849, -26.58559396, -26.58499145,
                                     -26.59591397,
            -26.61762205, -26.64942076, -26.69065616, -26.74071272,
            -26.79901044, -26.86500239, -26.93817239, -27.01803293,
            -27.10412331, -27.19600765, -27.29327345, -27.39553005,
            -27.50240715, -27.6135536, -27.72863631, -27.84733904,
            -27.96936157, -28.0944186, -28.22223914, -28.35256558,
            -28.48515301, -28.61976863, -28.75619106, -28.89420984,
            -29.03362487, -29.17424597, -29.31589234, -29.45839223,
            -29.6015825, -29.74530822, -29.8894224, -30.03378556,
            -30.17826552, -30.32273701, -30.46708145, -30.61118667,
            -30.75494665, -30.89826129, -31.04103614, -31.18318222,
            -31.32461582, -31.46525822, -31.60503556, -31.74387865,
            -31.88172274, -32.01850737, -32.15417621, -32.28867688,
            -32.4219608, -32.553983, -32.68470203, -32.81407977,
            -32.9420813, -33.06867476, -33.19383125, -33.31752464,
            -33.4397315, -33.56043096, -33.67960459, -33.7972363,
            -33.91331221, -34.02782057, -34.14075164, -34.25209759,
            -34.36185241, -34.47001183, -34.5765732, -34.68153544,
            -34.78489892, -34.88666542, -34.986838, -35.085421,
            -35.18241988, -35.27784122, -35.37169263, -35.46398267,
            -35.55472082, -35.64391739, -35.73158348, -35.81773094,
            -35.90237227, -35.98552062, -36.06718973, -36.14739387,
            -36.22614779, -36.3034667, -36.37936624, -36.4538624,
            -36.52697151, -36.59871022, -36.66909543, -36.73814429,
            -36.80587414, -36.87230253, -36.93744714, -37.00132578,
            -37.06395638, -37.12535692, -37.18554548, -37.24454013,
            -37.302359, -37.3590202, -37.41454183, -37.46894195,
            -37.52223859, -37.57444969, -37.62559315, -37.67568675,
            -37.72474819, -37.77279505, -37.8198448, -37.86591478,
            -37.91102217, -37.95518404, -37.99841727, -38.0407386,
            -38.0821646, -38.12271167, -38.16239604, -38.20123373,
            -38.23924059, -38.2764323, -38.3128243, -38.34843189,
            -38.38327012, -38.41735386, -38.45069779, -38.48331636,
            -38.51522382, -38.54643423, -38.57696143, -38.60681905,
            -38.6360205, -38.664579, -38.69250756, -38.71981898,
            -38.74652584, -38.77264052, -38.79817521, -38.82314187,
            -38.84755227, -38.87141797, -38.89475035, -38.91756055,
            -38.93985956, -38.96165813, -38.98296685, -39.0037961,
            -39.02415607, -39.04405677, -39.063508, -39.08251942,
            -39.10110046, -39.11926041, -39.13700834, -39.1543532,
            -39.17130371, -39.18786846, -39.20405585, -39.21987413,
            -39.23533138, -39.25043551, -39.26519428, -39.2796153,
            -39.29370601, -39.30747372, -39.32092557, -39.33406857,
            -39.34690958, -39.35945531, -39.37171235, -39.38368714,
            -39.39538599, -39.40681507, -39.41798043, -39.42888799,
            -39.43954354, -39.44995276, -39.4601212, -39.47005428,
            -39.47975733, -39.48923555, -39.49849402, -39.50753774,
            -39.51637157, -39.52500028, -39.53342853, -39.5416609,
            -39.54970184, -39.55755571, -39.5652268, -39.57271927,
            -39.58003721, -39.58718462, -39.5941654, -39.60098338])

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

        expected_approx_coilset_psi = np.array([-26.57176664, -26.56162327,
                                                -26.5634907, -26.57661325,
            -26.60027995, -26.63382147, -26.67660713, -26.72804222,
            -26.78756551, -26.85464693, -26.92878546, -27.00950721,
            -27.09636355, -27.18892956, -27.28680243, -27.38960012,
            -27.49696007, -27.60853801, -27.7240069, -27.84305594,
            -27.96538963, -28.09072697, -28.21880064, -28.34935632,
            -28.482152, -28.6169574, -28.75355337, -28.89173139,
            -29.03129304, -29.17204961, -29.31382158, -29.45643832,
            -29.59973762, -29.74356541, -29.88777537, -30.03222868,
            -30.17679362, -30.32134542, -30.46576585, -30.60994307,
            -30.75377133, -30.89715076, -31.03998711, -31.18219158,
            -31.32368057, -31.46437549, -31.60420261, -31.74309278,
            -31.88098136, -32.01780794, -32.15351626, -32.28805398,
            -32.42137256, -32.5534271, -32.68417618, -32.81358172,
            -32.94160884, -33.06822573, -33.19340351, -33.31711611,
            -33.43934013, -33.56005476, -33.6792416, -33.79688461,
            -33.91296996, -34.02748594, -34.14042286, -34.25177294,
            -34.36153023, -34.46969049, -34.57625114, -34.68121115,
            -34.78457095, -34.88633235, -34.98649849, -35.08507374,
            -35.18206364, -35.27747481, -35.3713149, -35.46359255,
            -35.55431727, -35.64349942, -35.73115018, -35.81728141,
            -35.90190569, -35.98503621, -36.06668676, -36.14687164,
            -36.22560566, -36.30290408, -36.37878257, -36.45325716,
            -36.52634424, -36.59806048, -36.66842282, -36.73744845,
            -36.80515477, -36.87155932, -36.93667985, -37.00053418,
            -37.06314028, -37.12451617, -37.18467993, -37.24364968,
            -37.30144357, -37.35807974, -37.4135763, -37.46795134,
            -37.52122292, -37.57340899, -37.62452747, -37.67459616,
            -37.72363278, -37.77165493, -37.81868008, -37.86472559,
            -37.90980867, -37.95394638, -37.99715563, -38.03945316,
            -38.08085556, -38.12137924, -38.16104041, -38.19985514,
            -38.23783927, -38.27500847, -38.31137822, -38.34696379,
            -38.38178027, -38.41584251, -38.4491652, -38.48176281,
            -38.51364957, -38.54483956, -38.57534661, -38.60518435,
            -38.63436621, -38.6629054, -38.69081493, -38.71810759,
            -38.74479599, -38.77089249, -38.79640927, -38.82135831,
            -38.84575137, -38.86960002, -38.89291561, -38.91570932,
            -38.9379921, -38.95977472, -38.98106777, -39.00188161,
            -39.02222645, -39.04211227, -39.06154891, -39.08054598,
            -39.09911294, -39.11725906, -39.13499343, -39.15232496,
            -39.1692624, -39.18581433, -39.20198914, -39.21779508,
            -39.23324021, -39.24833247, -39.2630796, -39.2774892,
            -39.29156872, -39.30532545, -39.31876655, -39.33189901,
            -39.34472969, -39.3572653, -39.36951243, -39.3814775,
            -39.39316683, -39.40458658, -39.41574281, -39.42664143,
            -39.43728822, -39.44768886, -39.4578489, -39.46777376,
            -39.47746876, -39.48693909, -39.49618985, -39.50522602,
            -39.51405245, -39.52267393, -39.53109511, -39.53932054,
            -39.5473547, -39.55520194, -39.56286654, -39.57035266,
            -39.57766439, -39.58480572, -39.59178056, -39.59859272])

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
