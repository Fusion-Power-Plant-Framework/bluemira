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
    coil_toroidal_harmonic_amplitude_matrix,
    f_hypergeometric,
    legendre_p,
    legendre_q,
    toroidal_harmonic_approximate_psi,
    toroidal_harmonic_approximation,
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

    for fc, res in zip(test_f_constraint, (test_result - b_vec), strict=False):
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
            self.test_v_psi, self.eq.grid, self.test_colocation, test_r_t
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
            ref_harmonics=ref_harmonics, r_t=r_t, sh_coil_names=self.sh_coil_names
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
            Path(TEST_PATH, "eqref_OOB.json").as_posix(), from_cocos=7
        )
        cls.eq.get_OX_points()
        cls.R_0, cls.Z_0 = cls.eq._o_points[0].x, cls.eq._o_points[0].z
        cls.test_th_params = toroidal_harmonic_grid_and_coil_setup(
            eq=cls.eq, R_0=cls.R_0, Z_0=cls.Z_0
        )
        cls.max_degree = 6

    def test_toroidal_harmonic_grid_and_coil_setup(self):
        expected_th_number_of_coils = len(self.eq.coilset.name)
        expected_shape = (150, 200)
        assert self.test_th_params.R_0 == self.R_0
        assert self.test_th_params.Z_0 == self.Z_0

        assert np.shape(self.test_th_params.R) == expected_shape
        assert np.shape(self.test_th_params.Z) == expected_shape

        assert len(self.eq.coilset.x) == expected_th_number_of_coils
        assert self.eq.coilset.x == pytest.approx(self.test_th_params.R_coils)

        assert len(self.eq.coilset.z) == expected_th_number_of_coils
        assert self.eq.coilset.z == pytest.approx(self.test_th_params.Z_coils)

        # fmt: off
        expected_tau = [1.15189302, 1.17597984, 1.20006666, 1.22415348, 1.24824029,
                    1.27232711, 1.29641393, 1.32050075, 1.34458756, 1.36867438,
                    1.3927612, 1.41684802, 1.44093484, 1.46502165, 1.48910847,
                    1.51319529, 1.53728211, 1.56136892, 1.58545574, 1.60954256,
                    1.63362938, 1.65771619, 1.68180301, 1.70588983, 1.72997665,
                    1.75406346, 1.77815028, 1.8022371, 1.82632392, 1.85041073,
                    1.87449755, 1.89858437, 1.92267119, 1.946758, 1.97084482,
                    1.99493164, 2.01901846, 2.04310527, 2.06719209, 2.09127891,
                    2.11536573, 2.13945254, 2.16353936, 2.18762618, 2.211713,
                    2.23579981, 2.25988663, 2.28397345, 2.30806027, 2.33214709,
                    2.3562339, 2.38032072, 2.40440754, 2.42849436, 2.45258117,
                    2.47666799, 2.50075481, 2.52484163, 2.54892844, 2.57301526,
                    2.59710208, 2.6211889, 2.64527571, 2.66936253, 2.69344935,
                    2.71753617, 2.74162298, 2.7657098, 2.78979662, 2.81388344,
                    2.83797025, 2.86205707, 2.88614389, 2.91023071, 2.93431752,
                    2.95840434, 2.98249116, 3.00657798, 3.03066479, 3.05475161,
                    3.07883843, 3.10292525, 3.12701206, 3.15109888, 3.1751857,
                    3.19927252, 3.22335934, 3.24744615, 3.27153297, 3.29561979,
                    3.31970661, 3.34379342, 3.36788024, 3.39196706, 3.41605388,
                    3.44014069, 3.46422751, 3.48831433, 3.51240115, 3.53648796,
                    3.56057478, 3.5846616, 3.60874842, 3.63283523, 3.65692205,
                    3.68100887, 3.70509569, 3.7291825, 3.75326932, 3.77735614,
                    3.80144296, 3.82552977, 3.84961659, 3.87370341, 3.89779023,
                    3.92187704, 3.94596386, 3.97005068, 3.9941375, 4.01822431,
                    4.04231113, 4.06639795, 4.09048477, 4.11457159, 4.1386584,
                    4.16274522, 4.18683204, 4.21091886, 4.23500567, 4.25909249,
                    4.28317931, 4.30726613, 4.33135294, 4.35543976, 4.37952658,
                    4.4036134, 4.42770021, 4.45178703, 4.47587385, 4.49996067,
                    4.52404748, 4.5481343, 4.57222112, 4.59630794, 4.62039475,
                    4.64448157, 4.66856839, 4.69265521, 4.71674202, 4.74082884,
                    4.76491566, 4.78900248, 4.81308929, 4.83717611, 4.86126293,
                    4.88534975, 4.90943656, 4.93352338, 4.9576102, 4.98169702,
                    5.00578384, 5.02987065, 5.05395747, 5.07804429, 5.10213111,
                    5.12621792, 5.15030474, 5.17439156, 5.19847838, 5.22256519,
                    5.24665201, 5.27073883, 5.29482565, 5.31891246, 5.34299928,
                    5.3670861, 5.39117292, 5.41525973, 5.43934655, 5.46343337,
                    5.48752019, 5.511607, 5.53569382, 5.55978064, 5.58386746,
                    5.60795427, 5.63204109, 5.65612791, 5.68021473, 5.70430154,
                    5.72838836, 5.75247518, 5.776562, 5.80064881, 5.82473563,
                    5.84882245, 5.87290927, 5.89699609, 5.9210829, 5.94516972]
        # fmt: on

        np.testing.assert_array_almost_equal(expected_tau, self.test_th_params.tau[0])
        assert np.shape(self.test_th_params.tau) == expected_shape

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
                    0.02108451, .06325354, 0.10542257, 0.1475916, 0.18976063,
                    0.23192966, .27409869, 0.31626772, 0.35843675, 0.40060577,
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
        np.testing.assert_array_almost_equal(
            expected_sigma, self.test_th_params.sigma[:, 0]
        )
        assert np.shape(self.test_th_params.sigma) == expected_shape

        expected_tau_c = [
            0.55217896,
            0.92665869,
            1.0438244,
            1.07576558,
            0.88507866,
            0.49909628,
            0.29180372,
            0.41500436,
            0.54743742,
            0.41363068,
            0.29070009,
        ]
        np.testing.assert_array_almost_equal(expected_tau_c, self.test_th_params.tau_c)
        assert len(self.test_th_params.tau_c) == expected_th_number_of_coils
        expected_sigma_c = [
            1.3339956,
            0.77675472,
            0.24155139,
            -0.27826736,
            -0.82320578,
            -1.2932888,
            1.63609574,
            2.10769457,
            -3.13756135,
            -2.10187947,
            -1.63210505,
        ]
        np.testing.assert_array_almost_equal(
            expected_sigma_c, self.test_th_params.sigma_c
        )
        assert len(self.test_th_params.sigma_c) == expected_th_number_of_coils

        assert len(self.test_th_params.th_coil_names) == expected_th_number_of_coils

    def test_coil_toroidal_harmonic_amplitude_matrix(self):
        test_Am_cos, test_Am_sin = coil_toroidal_harmonic_amplitude_matrix(  # noqa: N806
            input_coils=self.eq.coilset,
            th_params=self.test_th_params,
            max_degree=self.max_degree,
        )
        expected_shape = (self.max_degree, len(self.eq.coilset.name))

        # fmt: off
        expected_Am_cos = [[3.58356905e-08, 1.15270127e-07, 1.61845433e-07,  # noqa: N806
         1.65978945e-07, 1.05443910e-07, 3.03431499e-08,
         9.02431806e-09, 1.52648924e-08, 2.30375063e-08,
         1.51896928e-08, 8.97334827e-09],
       [1.30985354e-08, 1.37152666e-07, 2.69684814e-07,
         2.76155841e-07, 1.18498620e-07, 1.28625018e-08,
        -8.92739284e-10, -1.19657715e-08, -3.58708003e-08,
        -1.17886268e-08, -8.33449853e-10],
       [-6.94181320e-08, 5.60381039e-09, 4.46322183e-07,
         4.51962783e-07, -2.16566742e-08, -5.46473060e-08,
        -1.75016096e-08, -1.48581070e-08, 5.00191787e-08,
        -1.50925314e-08, -1.74149183e-08],
       [-7.08454745e-08, -4.07218134e-07, 7.59738827e-07,
         7.39611785e-07, -3.95738896e-07, -6.42385304e-08,
         4.22284740e-09, 4.02740957e-08, -6.94620196e-08,
         4.00490422e-08, 3.94260565e-09],
       [9.01083333e-08, -1.15356013e-06, 1.25555387e-06,
         1.09231220e-06, -9.39252722e-07, 5.27838385e-08,
         2.53256814e-08, -2.84161796e-08, 9.83114485e-08,
        -2.72015977e-08, 2.52560315e-08],
       [2.07811059e-07, -1.74331419e-06, 1.80043667e-06,
         1.04474345e-06, -1.05083380e-06, 1.62471356e-07,
        -1.01153734e-08, -3.00039039e-08, -1.42382481e-07,
        -3.15115206e-08, -9.44604128e-09]]
        # fmt: on

        np.testing.assert_array_almost_equal(test_Am_cos, expected_Am_cos)
        assert np.shape(test_Am_cos) == expected_shape

        # fmt: off
        expected_Am_sin = [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00,  # noqa: N806
        -0.00000000e+00, -0.00000000e+00, -0.00000000e+00,
         0.00000000e+00, 0.00000000e+00, -0.00000000e+00,
        -0.00000000e+00, -0.00000000e+00],
       [5.42767923e-08, 1.34801984e-07, 6.64399885e-08,
        -7.88920217e-08, -1.27815615e-07, -4.51541340e-08,
         1.36520369e-08, 2.01030678e-08, -1.44607017e-10,
        -2.00700891e-08, -1.35772748e-08],
       [3.55771358e-08, 3.24133254e-07, 2.34123200e-07,
        -2.81180791e-07, -2.85860142e-07, -3.38827912e-08,
        -2.29877400e-09, -2.73925931e-08, 4.03294140e-10,
         2.70688191e-08, 2.14613927e-09],
       [-8.23566065e-08, 4.28903905e-07, 6.72624195e-07,
        -8.16555144e-07, -3.14800143e-07, 5.84525047e-08,
        -2.12798731e-08, 1.60772532e-09, -8.40109285e-10,
        -8.99376134e-10, 2.11935438e-08],
       [-1.25263918e-07, 3.98988066e-08, 1.81727939e-06,
        -2.21734870e-06, 1.43136516e-07, 1.06332537e-07,
         6.76966531e-09, 4.36769061e-08, 1.58543222e-09,
        -4.40168617e-08, -6.32088380e-09],
       [8.46436539e-08, -1.59878791e-06, 4.73952733e-06,
        -5.75897973e-06, 1.54806095e-06, -3.01121415e-08,
         2.98727287e-08, -6.10029039e-08, -2.87032670e-09,
         5.96258697e-08, -2.98433653e-08]]
        # fmt: on

        np.testing.assert_array_almost_equal(test_Am_sin, expected_Am_sin)
        assert np.shape(test_Am_sin) == expected_shape

    def test_toroidal_harmonic_approximate_psi(self):
        test_approx_coilset_psi, test_Am_cos, test_Am_sin = (  # noqa: N806
            toroidal_harmonic_approximate_psi(
                eq=self.eq, th_params=self.test_th_params, max_degree=self.max_degree
            )
        )
        expected_shape = (150, 200)

        # fmt: off
        expected_coilset_psi = [-26.45646563, -26.46489139, -26.48441793, -26.51438706,
       -26.55417502, -26.60319035, -26.66087182, -26.72668654,
       -26.80012823, -26.88071556, -26.96799067, -27.06151777,
       -27.16088183, -27.26568745, -27.37555771, -27.49013318,
       -27.60907098, -27.73204392, -27.85873972, -27.98886024,
       -28.12212081, -28.25824961, -28.39698708, -28.53808535,
       -28.68130776, -28.82642838, -28.97323155, -29.12151148,
       -29.27107185, -29.42172546, -29.57329384, -29.72560698,
       -29.87850296, -30.03182771, -30.18543466, -30.33918455,
       -30.49294509, -30.64659077, -30.80000259, -30.95306786,
       -31.10567992, -31.257738, -31.40914696, -31.5598171,
       -31.70966398, -31.8586082, -32.00657525, -32.15349532,
       -32.2993031, -32.44393765, -32.5873422, -32.72946403,
       -32.87025426, -33.00966775, -33.14766292, -33.2842016,
       -33.41924893, -33.55277318, -33.68474562, -33.81514043,
       -33.94393453, -34.07110749, -34.19664139, -34.32052069,
       -34.44273218, -34.56326479, -34.68210956, -34.79925947,
       -34.91470942, -35.02845604, -35.14049767, -35.25083426,
       -35.35946724, -35.4663995, -35.57163526, -35.67518001,
       -35.77704043, -35.87722435, -35.97574062, -36.0725991,
       -36.16781057, -36.26138666, -36.35333982, -36.44368324,
       -36.53243079, -36.61959699, -36.70519696, -36.78924636,
       -36.87176132, -36.95275845, -37.03225478, -37.11026769,
       -37.1868149, -37.26191443, -37.33558458, -37.40784385,
       -37.47871098, -37.54820484, -37.61634448, -37.68314905,
       -37.74863779, -37.81283001, -37.87574507, -37.93740237,
       -37.99782128, -38.05702118, -38.11502142, -38.1718413,
       -38.22750005, -38.28201683, -38.3354107, -38.38770063,
       -38.43890545, -38.48904389, -38.53813453, -38.5861958,
       -38.63324599, -38.67930319, -38.72438537, -38.76851028,
       -38.81169551, -38.85395845, -38.89531629, -38.93578604,
       -38.97538447, -39.01412819, -39.05203356, -39.08911675,
       -39.1253937, -39.16088013, -39.19559155, -39.22954325,
       -39.26275029, -39.2952275, -39.3269895, -39.35805068,
       -39.38842521, -39.41812703, -39.44716986, -39.47556718,
       -39.50333227, -39.53047819, -39.55701776, -39.58296359,
       -39.60832807, -39.63312339, -39.6573615, -39.68105416,
       -39.70421289, -39.72684904, -39.74897372, -39.77059786,
       -39.79173216, -39.81238714, -39.83257311, -39.8523002,
       -39.87157834, -39.89041726, -39.90882651, -39.92681545,
       -39.94439327, -39.96156895, -39.97835132, -39.99474903,
       -40.01077055, -40.02642417, -40.04171804, -40.05666012,
       -40.07125822, -40.08551999, -40.09945291, -40.11306432,
       -40.12636141, -40.13935121, -40.1520406, -40.16443634,
       -40.17654501, -40.18837309, -40.19992689, -40.21121261,
       -40.22223632, -40.23300393, -40.24352125, -40.25379397,
       -40.26382763, -40.27362767, -40.28319942, -40.29254807,
       -40.30167872, -40.31059634, -40.31930581, -40.32781188,
       -40.33611923, -40.34423239, -40.35215584, -40.35989393,
       -40.36745092, -40.37483097, -40.38203817, -40.38907649]
        # fmt: on

        np.testing.assert_array_almost_equal(
            test_approx_coilset_psi[0], expected_coilset_psi
        )
        assert np.shape(test_approx_coilset_psi) == expected_shape

        expected_Am_cos = [  # noqa: N806
            -4.26075939,
            -3.73344285,
            -10.65579929,
            -11.78881006,
            -13.85350128,
            -1.92190518,
        ]
        np.testing.assert_array_almost_equal(test_Am_cos, expected_Am_cos)
        assert np.shape(test_Am_cos) == (self.max_degree,)

        expected_Am_sin = [  # noqa: N806
            0.0,
            0.07724163,
            0.25389907,
            3.51033947,
            13.08808431,
            32.23240491,
        ]
        np.testing.assert_array_almost_equal(test_Am_sin, expected_Am_sin)
        assert np.shape(test_Am_sin) == (self.max_degree,)

    def test_toroidal_harmonic_approximation(self):
        (
            test_th_parameters,
            test_Am_cos,  # noqa: N806
            test_Am_sin,  # noqa: N806
            test_degree,
            test_fit_metric_value,
            test_approx_total_psi,
            test_approx_coilset_psi,
        ) = toroidal_harmonic_approximation(
            eq=self.eq, th_params=self.test_th_params, psi_norm=0.95
        )

        assert test_th_parameters == self.test_th_params

        expected_Am_cos = [  # noqa: N806
            -4.26075939,
            -3.73344285,
            -10.65579929,
            -11.78881006,
            -13.85350128,
        ]
        np.testing.assert_array_almost_equal(test_Am_cos, expected_Am_cos)
        assert np.shape(test_Am_cos) == (test_degree,)

        expected_Am_sin = [  # noqa: N806
            0.0,
            0.07724163,
            0.25389907,
            3.51033947,
            13.08808431,
        ]
        np.testing.assert_array_almost_equal(test_Am_sin, expected_Am_sin)
        assert np.shape(test_Am_sin) == (test_degree,)

        assert test_degree == 5

        assert np.isclose(test_fit_metric_value, 0.01014672263679008)

        expected_shape = (150, 200)

        # fmt: off
        expected_total_psi = [
            -7.13555756, -6.69620581, -6.24790205, -5.79161244, -5.32818423,
            -4.85834828, -4.38276182, -3.90199265, -3.41656176, -2.92692642,
            -2.4335088, -1.93667635, -1.4367845, -0.93417512, -0.42914352,
            0.07812142, 0.58735529, 1.09803937, 1.60960317, 2.12138883,
            2.63273163, 3.14293076, 3.65128222, 4.15709321, 4.65966528,
            5.15834095, 5.65246809, 6.14143542, 6.62466683, 7.10161406,
            7.57178027, 8.03470259, 8.48996059, 8.93718153, 9.37603008,
            9.806214, 10.2274835, 10.63962678, 11.04246967, 11.4358742,
            11.81973771, 12.19398856, 12.5585837, 12.91351029, 13.25878115,
            13.59443039, 13.92051409, 14.23711068, 14.54431438, 14.84223328,
            15.13099143, 15.41072692, 15.68158691, 15.94372654, 16.19731057,
            16.44251215, 16.67950888, 16.90848188, 17.12961598, 17.34310079,
            17.54912747, 17.74788764, 17.93957291, 18.12437528, 18.30248732,
            18.47410023, 18.63940329, 18.79858364, 18.95182614, 19.09931397,
            19.24122757, 19.37774423, 19.50903792, 19.63527913, 19.75663492,
            19.87326911, 19.98534191, 20.09300974, 20.19642517, 20.29573688,
            20.39108972, 20.48262465, 20.57047893, 20.654786, 20.73567546,
            20.81327311, 20.88770102, 20.95907753, 21.02751735, 21.09313156,
            21.15602777, 21.21631007, 21.27407918, 21.32943247, 21.38246405,
            21.43326486, 21.48192269, 21.52852232, 21.57314556, 21.61587131,
            21.65677567, 21.69593197, 21.73341088, 21.76928051, 21.80360646,
            21.83645188, 21.86787759, 21.89794209, 21.92670171, 21.95421058,
            21.9805208, 22.00568243, 22.02974362, 22.05275059, 22.07474777,
            22.09577782, 22.11588172, 22.13509881, 22.15346688, 22.17102217,
            22.1877995, 22.20383224, 22.21915243, 22.23379079, 22.24777679,
            22.26113867, 22.27390352, 22.2860973, 22.2977449, 22.30887017,
            22.31949594, 22.3296441, 22.33933562, 22.34859057, 22.35742817,
            22.36586684, 22.37392418, 22.38161707, 22.38896168, 22.39597344,
            22.40266716, 22.40905698, 22.41515646, 22.42097856, 22.42653566,
            22.43183963, 22.4369018, 22.44173302, 22.44634367, 22.45074366,
            22.45494247, 22.45894916, 22.46277241, 22.46642049, 22.46990133,
            22.47322249, 22.47639119, 22.47941436, 22.4822986, 22.48505021,
            22.48767522, 22.49017939, 22.49256822, 22.49484698, 22.49702067,
            22.49909409, 22.50107182, 22.50295823, 22.5047575, 22.50647362,
            22.50811039, 22.50967146, 22.51116028, 22.51258017, 22.51393431,
            22.5152257, 22.51645724, 22.51763167, 22.51875163, 22.51981961,
            22.52083802, 22.52180914, 22.52273516, 22.52361814, 22.52446008,
            22.52526287, 22.52602833, 22.52675817, 22.52745404, 22.52811752,
            22.52875011, 22.52935323, 22.52992824, 22.53047646, 22.53099913,
            22.53149742, 22.53197247, 22.53242536, 22.53285712, 22.53326872
        ]
        # fmt: on

        np.testing.assert_array_almost_equal(
            test_approx_total_psi[0], expected_total_psi
        )
        assert np.shape(test_approx_total_psi) == expected_shape

        # fmt: off
        expected_coilset_psi = [
            -26.5410245, -26.54043045, -26.55188576, -26.57463433,
            -26.60796437, -26.65120525, -26.70372461, -26.76492566,
            -26.83424474, -26.91114902, -26.99513444, -27.08572377,
            -27.18246485, -27.28492896, -27.39270935, -27.50541988,
            -27.62269372, -27.74418228, -27.8695541, -27.99849387,
            -28.13070161, -28.26589177, -28.40379251, -28.54414501,
            -28.68670281, -28.83123121, -28.97750674, -29.12531663,
            -29.27445831, -29.42473902, -29.57597533, -29.72799277,
            -29.88062548, -30.03371585, -30.18711417, -30.34067836,
            -30.49427363, -30.64777224, -30.80105319, -30.95400201,
            -31.10651048, -31.2584764, -31.40980339, -31.56040062,
            -31.71018265, -31.8590692, -32.00698497, -32.15385943,
            -32.29962667, -32.44422517, -32.58759769, -32.72969103,
            -32.87045594, -33.00984693, -33.14782209, -33.284343,
            -33.41937454, -33.55288474, -33.68484471, -33.81522844,
            -33.9440127, -34.07117691, -34.19670303, -34.32057543,
            -34.44278079, -34.56330795, -34.68214788, -34.7992935,
            -34.91473963, -35.02848286, -35.14052148, -35.2508554,
            -35.35948601, -35.46641616, -35.57165005, -35.67519313,
            -35.77705208, -35.87723469, -35.9757498, -36.07260725,
            -36.1678178, -36.26139308, -36.35334552, -36.44368829,
            -36.53243527, -36.61960097, -36.7052005, -36.78924949,
            -36.8717641, -36.95276092, -37.03225697, -37.11026963,
            -37.18681662, -37.26191596, -37.33558594, -37.40784506,
            -37.47871205, -37.54820579, -37.61634532, -37.6831498,
            -37.74863845, -37.8128306, -37.8757456, -37.93740283,
            -37.99782169, -38.05702155, -38.11502175, -38.17184159,
            -38.22750031, -38.28201706, -38.3354109, -38.3877008,
            -38.43890561, -38.48904403, -38.53813466, -38.58619591,
            -38.63324608, -38.67930328, -38.72438545, -38.76851035,
            -38.81169557, -38.8539585, -38.89531634, -38.93578608,
            -38.97538451, -39.01412823, -39.05203359, -39.08911678,
            -39.12539372, -39.16088015, -39.19559157, -39.22954327,
            -39.2627503, -39.29522751, -39.32698951, -39.35805069,
            -39.38842522, -39.41812704, -39.44716986, -39.47556718,
            -39.50333228, -39.53047819, -39.55701776, -39.58296359,
            -39.60832807, -39.63312339, -39.6573615, -39.68105416,
            -39.7042129, -39.72684904, -39.74897373, -39.77059786,
            -39.79173216, -39.81238714, -39.83257311, -39.85230021,
            -39.87157834, -39.89041726, -39.90882651, -39.92681546,
            -39.94439327, -39.96156895, -39.97835132, -39.99474903,
            -40.01077055, -40.02642417, -40.04171804, -40.05666012,
            -40.07125822, -40.08551999, -40.09945291, -40.11306432,
            -40.12636141, -40.13935121, -40.1520406, -40.16443634,
            -40.17654501, -40.18837309, -40.19992689, -40.21121261,
            -40.22223632, -40.23300393, -40.24352125, -40.25379397,
            -40.26382763, -40.27362767, -40.28319942, -40.29254807,
            -40.30167872, -40.31059634, -40.31930581, -40.32781188,
            -40.33611923, -40.34423239, -40.35215584, -40.35989393,
            -40.36745092, -40.37483097, -40.38203817, -40.38907649
        ]
        # fmt: on

        np.testing.assert_array_almost_equal(
            test_approx_coilset_psi[0], expected_coilset_psi
        )
        assert np.shape(test_approx_coilset_psi) == expected_shape

    def test_ToroidalHarmonicConstraint(self):
        (th_params, ref_harmonics_cos, ref_harmonics_sin, degree, _, _, _) = (
            toroidal_harmonic_approximation(
                eq=self.eq, psi_norm=0.95, acceptable_fit_metric=0.01
            )
        )

        test_constraint_class = ToroidalHarmonicConstraint(
            ref_harmonics_cos=ref_harmonics_cos,
            ref_harmonics_sin=ref_harmonics_sin,
            th_params=th_params,
            tolerance=None,
            constraint_type="equality",
        )

        assert test_constraint_class.constraint_type == "equality"
        assert test_constraint_class.max_degree == degree

        for test_tol, ref_tol in zip(
            test_constraint_class.tolerance,
            np.abs(
                np.array([
                    ref_harmonics_cos[0] * 1e-3,
                    ref_harmonics_cos[1] * 1e-3,
                    ref_harmonics_cos[2] * 1e-3,
                    ref_harmonics_cos[3] * 1e-3,
                    ref_harmonics_cos[4] * 1e-3,
                    ref_harmonics_sin[0] * 1e-3,
                    ref_harmonics_sin[1] * 1e-3,
                    ref_harmonics_sin[2] * 1e-3,
                    ref_harmonics_sin[3] * 1e-3,
                    ref_harmonics_sin[4] * 1e-3,
                ])
            ),
            strict=False,
        ):
            assert test_tol == ref_tol

        tolerance = 0.0
        test_constraint_class = ToroidalHarmonicConstraint(
            ref_harmonics_cos=ref_harmonics_cos,
            ref_harmonics_sin=ref_harmonics_sin,
            th_params=th_params,
            tolerance=tolerance,
            constraint_type="equality",
        )
        assert len(test_constraint_class.tolerance) == len(ref_harmonics_cos) + len(
            ref_harmonics_sin
        )
        for test_name, ref_name in zip(
            test_constraint_class.control_coil_names,
            self.test_th_params.th_coil_names,
            strict=False,
        ):
            assert test_name == ref_name

        test_eval = test_constraint_class.evaluate(self.eq)

        assert all(test_eval == 0)
        assert len(test_eval) == len(ref_harmonics_cos)

    def test_ToroidalHarmonicConstraintFunction(self):
        cur_expand_mat = self.eq.coilset._opt_currents_expand_mat
        a_mat_cos, a_mat_sin = coil_toroidal_harmonic_amplitude_matrix(
            input_coils=self.eq.coilset, th_params=self.test_th_params, max_degree=5
        )
        b_vec_cos = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
        b_vec_sin = np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
        b_vec = np.append(b_vec_cos, b_vec_sin, axis=0)

        test_vector = cur_expand_mat @ np.ones(len(self.eq.coilset.name))
        test_result_cos = a_mat_cos @ test_vector
        test_result_sin = a_mat_sin @ test_vector
        test_result = np.append(test_result_cos, test_result_sin, axis=0)
        test_constraint = ToroidalHarmonicConstraintFunction(
            a_mat_cos=a_mat_cos,
            a_mat_sin=a_mat_sin,
            b_vec_cos=b_vec_cos,
            b_vec_sin=b_vec_sin,
            value=0.0,
            scale=1,
        )

        test_f_constraint = test_constraint.f_constraint(test_vector)

        for fc, res in zip(test_f_constraint, (test_result - b_vec), strict=False):
            assert fc == res

        vector = self.eq.coilset.current
        assert test_constraint.df_constraint(vector) == pytest.approx(
            approx_derivative(test_constraint.f_constraint, vector)
        )
