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
    _get_plasma_mask,
    brute_force_toroidal_harmonic_approximation,
    coil_toroidal_harmonic_amplitude_matrix,
    f_hypergeometric,
    legendre_p,
    legendre_q,
    toroidal_harmonic_approximate_psi,
    toroidal_harmonic_grid_and_coil_setup,
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

    assert test_hypergeometric_values == expected_hypergeometric_values


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
    assert test_leg_p_values == expected_leg_p_values

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
    assert test_leg_q_values == expected_leg_q_values

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
        # (i.e., where we are tring to get a good approximation).
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
        n_degrees_of_freedom = 6
        max_harmonic_mode = 5

        expected_cos_degrees = np.array([0, 1, 2, 3, 4])
        expected_sin_degrees = np.array([3])
        expected_cos_amplitudes = np.array([
            -4.23864252,
            -3.58288708,
            -10.51447447,
            -11.673279,
            -14.26727472,
        ])
        expected_sin_amplitudes = np.array([3.15627377])

        result = brute_force_toroidal_harmonic_approximation(
            eq=self.eq,
            th_params=self.test_th_params,
            psi_norm=self.psi_norm,
            n_degrees_of_freedom=n_degrees_of_freedom,
            max_harmonic_mode=max_harmonic_mode,
            plasma_mask=True,
            cl=False,
        )
        mask = _get_plasma_mask(
            eq=self.eq,
            th_params=self.test_th_params,
            psi_norm=self.psi_norm,
            plasma_mask=True,
        )

        assert result.error < 10
        assert len(result.cos_m) + len(result.sin_m) == n_degrees_of_freedom
        np.testing.assert_array_almost_equal(result.cos_m, expected_cos_degrees)
        np.testing.assert_array_almost_equal(result.sin_m, expected_sin_degrees)
        assert np.max(np.append(result.cos_m, result.sin_m)) <= max_harmonic_mode

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
        test_cos_degrees = np.array([0, 1, 2, 3, 4])
        test_sin_degrees = np.array([3])
        test_cos_amplitudes = np.array([
            -4.23864252,
            -3.58288708,
            -10.51447447,
            -11.673279,
            -14.26727472,
        ])
        test_sin_amplitudes = np.array([3.15627377])

        test_constraint_class_equality = ToroidalHarmonicConstraint(
            ref_harmonics_cos=test_cos_degrees,
            ref_harmonics_sin=test_sin_degrees,
            ref_harmonics_cos_amplitudes=test_cos_amplitudes,
            ref_harmonics_sin_amplitudes=test_sin_amplitudes,
            th_params=self.test_th_params,
            tolerance=None,
            constraint_type="equality",
        )

        assert test_constraint_class_equality.constraint_type == "equality"
        assert len(test_constraint_class_equality.tolerance) == len(
            test_cos_degrees
        ) + len(test_sin_degrees)
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
            ref_harmonics_cos=test_cos_degrees,
            ref_harmonics_sin=test_sin_degrees,
            ref_harmonics_cos_amplitudes=test_cos_amplitudes,
            ref_harmonics_sin_amplitudes=test_sin_amplitudes,
            th_params=self.test_th_params,
            tolerance=None,
            constraint_type="inequality",
        )

        # Multiply by 2 because inequality constraint is equivalent to 2 equality
        # constraints combined
        assert len(test_constraint_class_inequality.tolerance) == 2 * (
            len(test_cos_degrees) + len(test_sin_degrees)
        )

        for test_name, ref_name in zip(
            test_constraint_class_inequality.control_coil_names,
            self.test_th_params.th_coil_names,
            strict=False,
        ):
            assert test_name == ref_name

        test_eval_cos, test_eval_sin = test_constraint_class_inequality.evaluate(self.eq)

        assert all(test_eval_cos == 0)
        assert all(test_eval_sin == 0)
        assert len(test_eval_cos) == len(test_cos_amplitudes) * 2
        assert len(test_eval_sin) == len(test_sin_amplitudes) * 2

    # This test currently does not pass:
    def test_ToroidalHarmonicConstraintFunction(self):
        cos_degrees = np.array([0, 1, 2, 3, 4])
        sin_degrees = np.array([3])
        cos_amplitudes = np.array([
            -4.23864252,
            -3.58288708,
            -10.51447447,
            -11.673279,
            -14.26727472,
        ])
        sin_amplitudes = np.array([3.15627377])

        # Vector of currents in MA for arg in constraint function
        vector = self.eq.coilset.current * 1e-6

        constraint_class = ToroidalHarmonicConstraint(
            ref_harmonics_cos=cos_degrees,
            ref_harmonics_sin=sin_degrees,
            ref_harmonics_cos_amplitudes=cos_amplitudes,
            ref_harmonics_sin_amplitudes=sin_amplitudes,
            th_params=self.test_th_params,
            tolerance=None,
            constraint_type="equality",
        )
        constraint_class.prepare(self.eq)

        result_cos = constraint_class._args["a_mat_cos"] @ self.eq.coilset.current
        result_sin = constraint_class._args["a_mat_sin"] @ self.eq.coilset.current
        ref_function_result = np.append(
            result_cos - constraint_class._args["b_vec_cos"],
            result_sin - constraint_class._args["b_vec_sin"],
            axis=0,
        )

        test_constraint_function = ToroidalHarmonicConstraintFunction(
            a_mat_cos=constraint_class._args["a_mat_cos"],
            a_mat_sin=constraint_class._args["a_mat_sin"],
            b_vec_cos=constraint_class._args["b_vec_cos"],
            b_vec_sin=constraint_class._args["b_vec_sin"],
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
