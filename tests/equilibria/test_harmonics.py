# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import json
import pathlib
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
    ToroidalHarmonicsParams,
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
    tau = np.array([
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
    ])
    expected_leg_p_array_values = np.array([
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
    ])
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
    expected_leg_q_array_values = np.array([
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
    ])

    tau = np.array([
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
    ])
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


@pytest.mark.parametrize(
    ("n_dof", "max_harmonic_mode", "max_n_dof", "expected_dof"),
    [
        # Case where max_n_dof is hit
        (5, 5, 4, 4),
        # Case where 2 * max_harmonic_mode is hit
        (5, 2, 5, 3),
        # Case where everything OK
        (5, 5, 10, 5),
        # Case where max_n_dof is exceed and still > 2 * max_harmonic_mode - 1
        (10, 4, 9, 7),
        # Case where n_dof is not specified and defaults to max
        (None, 5, 9, 9),
        # Case where n_dof is not specified and defaults to 2 * max_harmonic_mode - 1
        (None, 4, 9, 7),
    ],
)
def test_th_n_dof_limits(
    n_dof: int | None, max_harmonic_mode: int, max_n_dof: int, expected_dof: int
):
    n_dof = _set_n_degrees_of_freedom(n_dof, max_harmonic_mode, max_n_dof)
    assert n_dof == expected_dof


@pytest.mark.parametrize(
    ("cos_m_chosen", "sin_m_chosen", "expected_Am_cos", "expected_Am_sin"),
    [
        # Case where cos_m_chosen empty
        (
            np.array([]),
            np.array([1]),
            None,
            np.array([[2.78003029e-09, -2.78003029e-09]]),
        ),
        # Case where sin_m_chosen empty
        (
            np.array([0]),
            np.array([]),
            np.array([[4.13778177e-09, 4.13778177e-09]]),
            None,
        ),
        # Case where both empty
        (np.array([]), np.array([]), None, None),
        # Case where neither empty
        (
            np.array([1]),
            np.array([2]),
            np.array([[5.56006057e-09, 5.56006057e-09]]),
            np.array([[6.24538906e-09, -6.24538906e-09]]),
        ),
        # Case with arrays chosen for each
        (
            np.array([2, 3, 4]),
            np.array([2, 3, 4]),
            np.array([
                [4.68404177e-09, 4.68404177e-09],
                [1.64194750e-09, 1.64194750e-09],
                [-2.92277587e-09, -2.92277587e-09],
            ]),
            np.array([
                [6.24538906e-09, -6.24538906e-09],
                [9.03071139e-09, -9.03071139e-09],
                [1.00209457e-08, -1.00209457e-08],
            ]),
        ),
    ],
)
def test_coil_toroidal_harmonic_amplitude_matrix_unit(
    cos_m_chosen: np.ndarray,
    sin_m_chosen: np.ndarray,
    expected_Am_cos: np.ndarray,  # noqa: N803
    expected_Am_sin: np.ndarray,  # noqa: N803
):
    # Make a coilset for use in the test
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
            name="PF_1",
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
            name="PF_2",
        ),
    )

    coilset = CoilSet(circuit)

    th_params = ToroidalHarmonicsParams(
        R_0=1.5,
        Z_0=0.0,
        min_tau=0.0,
        R=np.array([0.0, 0.5, 1.0, 1.5, 2.0]),
        Z=np.array([-10.0, -5.0, 0.0, 5.0, 10.0]),
        R_coils=np.array([1.5, 1.5]),
        Z_coils=np.array([6.0, -6.0]),
        tau=np.array([0.0, 0.05459965, 1.60943791, 0.15374235, 0.05653073]),
        sigma=np.array([-0.2977799, -0.57790194, 0.0, 0.5404195, 0.28671642]),
        tau_c=np.array([0.11157178, 0.11157178]),
        sigma_c=np.array([0.46364761, -0.46364761]),
        th_coil_names=["PF_1", "PF_2"],
    )
    Am_cos, Am_sin = coil_toroidal_harmonic_amplitude_matrix(  # noqa: N806
        input_coils=coilset,
        th_params=th_params,
        cos_m_chosen=cos_m_chosen,
        sin_m_chosen=sin_m_chosen,
    )
    # Can't use assert_array_almost_equal if Am_cos or Am_sin is None
    if Am_cos is None:
        assert Am_cos == expected_Am_cos
    else:
        np.testing.assert_array_almost_equal(Am_cos, expected_Am_cos)

    if Am_sin is None:
        assert Am_sin == expected_Am_sin
    else:
        np.testing.assert_array_almost_equal(Am_sin, expected_Am_sin)


@pytest.mark.parametrize(
    (
        "n_allowed",
        "expected_cos",
        "expected_sin",
        "expected_cos_collocation",
        "expected_sin_collocation",
    ),
    [
        # Testing certain entries instead of the whole arrays as they are large
        (1, 1.73529216e-03, 0.0, 1.57065987, 0.0),
        (2, 2.13803459e-03, -6.56107179e-04, 0.32913602, -1.45709709),
        (3, 1.37513594e-03, -9.31729349e-04, -0.68552459, -0.3263508),
        (4, 8.07476967e-04, -1.00356911e-03, -2.43033551e-01, 3.08997293e-01),
    ],
)
def test_toroidal_harmonics_to_positions(
    n_allowed,
    expected_cos,
    expected_sin,
    expected_cos_collocation,
    expected_sin_collocation,
):
    # Dummy args
    th_params = ToroidalHarmonicsParams(
        R_0=1.5,
        Z_0=0.0,
        min_tau=0.0,
        R=np.array([
            [0.1, 0.6, 1.1, 1.6],
            [0.2, 0.7, 1.2, 1.7],
            [0.3, 0.7, 1.3, 1.8],
            [0.4, 0.8, 1.4, 1.9],
        ]),
        Z=np.array([
            [-10.0, -10.0, -10.0, -10.0],
            [-5.0, -5.0, -5.0, -5.0],
            [0.1, 0.1, 0.1, 0.1],
            [5.0, 5.0, 5.0, 5.0],
        ]),
        R_coils=np.array([1.5, 1.5]),
        Z_coils=np.array([6.0, -6.0]),
        tau=np.array([
            [0.0, 0.01463519, 0.02906387, 0.04308885],
            [0.01100558, 0.06528637, 0.1164761, 0.16243346],
        ]),
        sigma=np.array([
            [-0.2977799, -0.29706421, -0.29493706, -0.29145679],
            [-0.58271165, -0.57572093, -0.55934928, -0.53499839],
        ]),
        tau_c=np.array([0.11157178, 0.11157178]),
        sigma_c=np.array([0.46364761, -0.46364761]),
        th_coil_names=["PF_1", "PF_2"],
    )
    x = [1, 1.5, 2, 2.1, 2, 1.5, 1, 0.9, 1]
    z = [-1.8, -1.9, -1.8, 0, 1.8, 1.9, 1.8, 0, -1.8]
    plasma_boundary = Coordinates({"x": x, "z": z})
    # Use GRID_POINTS as this is what we use in toroidal_harmonic_approximation
    point_type = PointType.GRID_POINTS
    colloc = collocation_points(plasma_boundary, point_type)

    # Test without collocation points
    cos, sin = toroidal_harmonics_to_positions(th_params=th_params, n_allowed=n_allowed)
    assert len(cos) == n_allowed
    assert len(sin) == n_allowed
    np.testing.assert_array_almost_equal(cos[n_allowed - 1][0][0], expected_cos)
    np.testing.assert_array_almost_equal(sin[n_allowed - 1][0][0], expected_sin)

    # Test with collocation points
    cos, sin = toroidal_harmonics_to_positions(
        th_params=th_params, n_allowed=n_allowed, collocation=colloc
    )
    assert len(cos) == n_allowed
    assert len(sin) == n_allowed
    np.testing.assert_almost_equal(cos[n_allowed - 1][0], expected_cos_collocation)
    np.testing.assert_almost_equal(sin[n_allowed - 1][0], expected_sin_collocation)


@pytest.mark.parametrize(
    (
        "n_degrees_of_freedom",
        "cos_m_chosen",
        "sin_m_chosen",
        "expected_error_mask_true",
        "expected_error_mask_false",
        "expected_psi",
        "expected_cos",
        "expected_sin",
    ),
    [  # expected_psi is first entry of the psi array
        (
            2,
            np.array([0]),
            np.array([1]),
            25.678786,
            29.537489,
            3.40593137e-03,
            np.array([3.34186462]),
            np.array([3.64754447]),
        ),
        (
            3,
            np.array([0, 2]),
            np.array([1, 2]),
            93.980878,
            108.248855,
            -0.06319918,
            np.array([3.37203713, -45.6617406]),
            np.array([5.58472399, 2.78554199]),
        ),
        (
            4,
            np.array([0, 2, 3]),
            np.array([1, 4]),
            921.87294,
            1127.94384,
            -0.76721926,
            np.array([3.47958164, -33.03489877, -152.56876336]),
            np.array([5.48849864, 636.94804332]),
        ),
        (
            5,
            np.array([4, 5]),
            np.array([1, 2, 3]),
            33282.38242,
            41134.52516,
            -3.74251913e00,
            np.array([-4207.47659676, -25672.60714349]),
            np.array([-13.66718836, -69.74142636, 532.28407125]),
        ),
        # Test with no sin components selected
        (
            2,
            np.array([0, 1]),
            np.array([], dtype="int64"),
            17.203162,
            21.472632,
            1.27045259e-02,
            np.array([3.53601634, 3.07221615]),
            np.array([]),
        ),
        # Test with no cos components selected
        (
            3,
            np.array([], dtype="int64"),
            np.array([1, 2, 3]),
            307.22005,
            372.52797,
            -1.36538890e-01,
            np.array([]),
            np.array([5.1084459, 2.3062025, 130.5724179]),
        ),
    ],
)
def test_approximation_from_psi_fitting(
    n_degrees_of_freedom,
    cos_m_chosen,
    sin_m_chosen,
    expected_error_mask_true,
    expected_error_mask_false,
    expected_psi,
    expected_cos,
    expected_sin,
):
    th_params = ToroidalHarmonicsParams(
        R_0=1.5,
        Z_0=0.0,
        min_tau=0.0,
        R=np.array([
            [0.1, 0.6, 1.1, 1.6],
            [0.2, 0.7, 1.2, 1.7],
            [0.3, 0.7, 1.3, 1.8],
            [0.4, 0.8, 1.4, 1.9],
        ]),
        Z=np.array([
            [-10.0, -10.0, -10.0, -10.0],
            [-5.0, -5.0, -5.0, -5.0],
            [0.1, 0.1, 0.1, 0.1],
            [5.0, 5.0, 5.0, 5.0],
        ]),
        R_coils=np.array([1.5, 1.5]),
        Z_coils=np.array([6.0, -6.0]),
        tau=np.array([
            [0.0, 0.01463519, 0.02906387, 0.04308885],
            [0.01100558, 0.06528637, 0.1164761, 0.16243346],
        ]),
        sigma=np.array([
            [-0.2977799, -0.29706421, -0.29493706, -0.29145679],
            [-0.58271165, -0.57572093, -0.55934928, -0.53499839],
        ]),
        tau_c=np.array([0.11157178, 0.11157178]),
        sigma_c=np.array([0.46364761, -0.46364761]),
        th_coil_names=["PF_1", "PF_2"],
    )
    x = [1, 1.5, 2, 2.1, 2, 1.5, 1, 0.9, 1]
    z = [-1.8, -1.9, -1.8, 0, 1.8, 1.9, 1.8, 0, -1.8]
    plasma_boundary = Coordinates({"x": x, "z": z})
    colloc = collocation_points(plasma_boundary, PointType.RANDOM, n_points=11)
    coilset_psi = np.array([
        [0.0, 5.0, 10.0, 15.0],
        [1.0, 6.0, 11.0, 16.0],
        [2.0, 7.0, 12.0, 17.0],
        [3.0, 8.0, 13.0, 18.0],
    ])
    collocation_psi = np.arange(11)
    mask_true = np.array([[0, 0, 1, 1], [0, 0, 1, 1], [1, 0, 1, 0], [0, 1, 1, 0]])
    mask_false = 1
    error, psi, cos, sin = _approximation_from_psi_fitting(
        th_params=th_params,
        n_deg_of_freedom=n_degrees_of_freedom,
        collocation=colloc,
        cos_m_chosen=cos_m_chosen,
        sin_m_chosen=sin_m_chosen,
        collocation_psi=collocation_psi,
        mask=mask_true,
        true_coilset_psi=coilset_psi,
    )
    np.testing.assert_almost_equal(error, expected_error_mask_true, decimal=5)
    np.testing.assert_almost_equal(psi[0][0], expected_psi)
    np.testing.assert_almost_equal(cos, expected_cos)
    np.testing.assert_almost_equal(sin, expected_sin)

    error, psi, cos, sin = _approximation_from_psi_fitting(
        th_params=th_params,
        n_deg_of_freedom=n_degrees_of_freedom,
        collocation=colloc,
        cos_m_chosen=cos_m_chosen,
        sin_m_chosen=sin_m_chosen,
        collocation_psi=collocation_psi,
        mask=mask_false,
        true_coilset_psi=coilset_psi,
    )

    # Only the error should be different when the mask is different,
    # but test them all to be sure
    np.testing.assert_almost_equal(error, expected_error_mask_false, decimal=5)
    np.testing.assert_almost_equal(psi[0][0], expected_psi)
    np.testing.assert_almost_equal(cos, expected_cos)
    np.testing.assert_almost_equal(sin, expected_sin)


class TestRegressionTH:
    @staticmethod
    def _read_json(file_path: str | pathlib.PosixPath) -> dict:
        with open(file_path) as f:
            return json.load(f)

    @classmethod
    def setup_class(cls):
        cls.eq = Equilibrium.from_eqdsk(
            Path(TEST_PATH, "eqref_OOB.json").as_posix(), from_cocos=7
        )
        file_path = Path(TEST_PATH, "toroidal_harmonics_test_data.json")
        cls.param_dict = cls._read_json(file_path)

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
            cls.eq.get_LCFS(), PointType.RANDOM, n_points=11
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
            else [0.54743133, 0.91207134, 1.02497178, 1.05937582, 0.87854201, 0.49935848]
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
        expected_sin_modes = np.array([3])
        expected_cos_amplitudes = np.array([
            -4.23455112,
            -3.52365388,
            -10.05836255,
            -7.74944276,
            -6.82548272,
        ])
        expected_sin_amplitudes = np.array([2.47437837])

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

        # Check sin mode 0 not selected
        assert 0 not in result.sin_m

        # Check error is suitably small
        assert result.error < 20
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

        for fc, res in zip(test_result, ref_function_result, strict=False):
            assert fc == pytest.approx(res)

        assert test_constraint_function.df_constraint(vector) == pytest.approx(
            approx_derivative(test_constraint_function.f_constraint, vector)
        )

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
