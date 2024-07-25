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
from bluemira.utilities.tools import (
    cylindrical_to_toroidal,
    toroidal_to_cylindrical,
)

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


def test_coils_outside_sphere_vacuum_psi():
    eq = Equilibrium.from_eqdsk(Path(TEST_PATH, "SH_test_file.json").as_posix())

    sh_coil_names, bdry_r = coils_outside_lcfs_sphere(eq)
    assert len(sh_coil_names) == 16
    assert bdry_r == pytest.approx(1.366, abs=0.005)

    test_p_psi = eq.plasma.psi()
    test_v_psi = np.zeros(np.shape(eq.grid.x))
    for n in sh_coil_names:
        test_v_psi = np.sum(
            [test_v_psi, eq.coilset[n].psi(eq.grid.x, eq.grid.z)], axis=0
        )
    non_cc_psi = eq.coilset.psi(eq.grid.x, eq.grid.z) - test_v_psi

    test_total = (test_v_psi + test_p_psi + non_cc_psi) - eq.psi()
    grid_zeros = test_total * 0.0

    assert test_total == pytest.approx(grid_zeros, abs=0.005)


def test_get_psi_harmonic_amplitudes():
    eq = Equilibrium.from_eqdsk(Path(TEST_PATH, "SH_test_file.json").as_posix())

    test_colocation = collocation_points(
        plasma_boundary=eq.get_LCFS(),
        point_type=PointType.ARC,
        n_points=18,
    )

    sh_coil_names, _ = coils_outside_lcfs_sphere(eq)
    test_v_psi = np.zeros(np.shape(eq.grid.x))
    for n in sh_coil_names:
        test_v_psi = np.sum(
            [test_v_psi, eq.coilset[n].psi(eq.grid.x, eq.grid.z)], axis=0
        )

    test_sh_amps = get_psi_harmonic_amplitudes(test_v_psi, eq.grid, test_colocation, 1.2)

    sh_amps = np.array([
        6.27021137e-03,
        1.13305430e-01,
        -7.10644651e-04,
        -1.21121072e-02,
        2.26606652e-04,
        8.62564002e-03,
        -1.09407173e-03,
        1.40751192e-02,
        -9.94184734e-04,
        1.18311187e-02,
        -7.55837162e-04,
        7.67252084e-03,
        -3.87118182e-04,
        2.66066774e-03,
        -1.87114805e-04,
        1.09257867e-03,
        -5.05761500e-05,
    ])

    assert test_sh_amps == pytest.approx(sh_amps)


def test_spherical_harmonic_approximation():
    eq = Equilibrium.from_eqdsk(Path(TEST_PATH, "SH_test_file.json").as_posix())

    (
        _,
        test_harmonic_amps,
        test_degree,
        test_fit_metric,
        _,
        test_r_t,
        test_sh_coilset_current,
    ) = spherical_harmonic_approximation(
        eq,
        n_points=10,
        point_type=PointType.GRID_POINTS,
        acceptable_fit_metric=0.01,
    )

    sh_coilset_current = np.array([
        7629.10582467,
        70698.33769641,
        67490.69283484,
        31121.21496069,
        74960.89700748,
        -15405.55822637,
        -107155.40127941,
        -119761.54614875,
        -22836.61530337,
        12076.3164052,
        74352.13320461,
        71459.62798936,
        33758.08895758,
        81867.51397822,
        -26684.21663108,
        -107953.36722597,
        -127015.41959899,
        -21793.83849537,
    ])

    harmonic_amps = np.array([
        0.11582153,
        -0.00059338,
        -0.03868344,
        0.0014262,
        -0.01530302,
    ])

    assert test_sh_coilset_current == pytest.approx(sh_coilset_current, abs=0.0005)
    assert test_r_t == pytest.approx(1.3661, abs=0.0001)
    assert test_harmonic_amps == pytest.approx(harmonic_amps, abs=0.0005)
    assert test_degree == 6
    assert test_fit_metric == pytest.approx(0.0025, abs=0.0001)


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
    b_vec = np.array([1e-2, 1e-18])
    test_vector = cur_expand_mat @ np.array([1, 1, 1])
    test_result = a_mat[1:,] @ test_vector
    test_constraint = SphericalHarmonicConstraintFunction(a_mat, b_vec, 0.0, 1)

    test_f_constraint = test_constraint.f_constraint(test_vector)

    for fc, res in zip(
        test_f_constraint,
        (test_result - b_vec),
        strict=False,
    ):
        assert fc == res


def test_SphericalHarmonicConstraint():
    eq = Equilibrium.from_eqdsk(Path(TEST_PATH, "SH_test_file.json").as_posix())

    sh_coil_names, _ = coils_outside_lcfs_sphere(eq)
    ref_harmonics = np.array([
        0.1165182,
        -0.00254487,
        -0.03455892,
        -0.00585685,
        -0.00397113,
        -0.01681114,
        0.01649549,
        -0.02803212,
        0.03035956,
        -0.03828872,
        0.04051739,
        -0.04283815,
    ])
    r_t = 1.37

    test_constraint_class = SphericalHarmonicConstraint(
        ref_harmonics=ref_harmonics,
        r_t=r_t,
        sh_coil_names=sh_coil_names,
    )

    assert test_constraint_class.constraint_type == "equality"
    assert test_constraint_class.max_degree == len(ref_harmonics) + 1

    for test_tol, ref_tol in zip(
        test_constraint_class.tolerance,
        np.array([
            1e-4,
            1e-6,
            1e-5,
            1e-6,
            1e-6,
            1e-5,
            1e-5,
            1e-5,
            1e-5,
            1e-5,
            1e-5,
            1e-5,
        ]),
        strict=False,
    ):
        assert test_tol == ref_tol

    tolerance = 0.0
    test_constraint_class = SphericalHarmonicConstraint(
        ref_harmonics=ref_harmonics,
        r_t=r_t,
        sh_coil_names=sh_coil_names,
        tolerance=tolerance,
    )

    assert len(test_constraint_class.tolerance) == len(ref_harmonics)
    for test_name, ref_name in zip(
        test_constraint_class.control_coil_names, sh_coil_names, strict=False
    ):
        assert test_name == ref_name

    test_eval = test_constraint_class.evaluate(eq)
    assert all(test_eval == 0)
    assert len(test_eval) == 12


def test_toroidal_coordinate_transform():
    # set values to be used in the tests
    R_0_test = 1.0  # noqa: N806
    z_0_test = 0.0
    # set tau and sigma isosurfaces to be used in the tests and the min & max r & z
    # points for each isosurface
    # fmt: off
    tau_test_isosurface_rz_points = np.array([
        [
            0.24491866, 0.24497605, 0.24514832, 0.24543577, 0.24583895, 0.24635857, 0.24699559, 
            0.24775117, 0.24862671, 0.24962383, 0.25074437, 0.25199043, 0.25336434, 0.25486872, 
            0.25650642, 0.2582806, 0.2601947, 0.26225246, 0.26445795, 0.26681557, 0.26933009, 
            0.27200665, 0.27485079, 0.27786846, 0.28106608, 0.28445053, 0.28802921, 0.29181007, 
            0.29580161, 0.30001299, 0.30445402, 0.30913522, 0.31406787, 0.3192641, 0.32473692, 
            0.33050029, 0.33656921, 0.3429598, 0.34968939, 0.35677662, 0.36424155, 0.37210577, 
            0.38039257, 0.38912702, 0.39833621, 0.40804935, 0.41829804, 0.42911641, 0.44054143, 
            0.45261309, 0.46537477, 0.47887352, 0.4931604, 0.50829089, 0.52432531, 0.54132932, 
            0.55937441, 0.57853847, 0.5989065, 0.62057122, 0.64363392, 0.66820523, 0.69440612, 
            0.72236886, 0.75223812, 0.78417213, 0.81834398, 0.85494285, 0.89417545, 0.93626739, 
            0.9814645, 1.03003417, 1.08226639, 1.1384746, 1.19899595, 1.26419094, 1.33444201, 
            1.41015062, 1.49173242, 1.57960973, 1.67420047, 1.77590249, 1.88507207, 2.00199523, 
            2.12685026, 2.2596603, 2.40023496, 2.54810089, 2.70242316, 2.86192128, 3.02478749, 
            3.18861912, 3.3503814, 3.50642039, 3.65254676, 3.78420667, 3.89674489, 3.9857473, 
            4.0474279, 4.07900554, 4.07900554, 4.0474279, 3.9857473, 3.89674489, 3.78420667, 
            3.65254676, 3.50642039, 3.3503814, 3.18861912, 3.02478749, 2.86192128, 2.70242316, 
            2.54810089, 2.40023496, 2.2596603, 2.12685026, 2.00199523, 1.88507207, 1.77590249, 
            1.67420047, 1.57960973, 1.49173242, 1.41015062, 1.33444201, 1.26419094, 1.19899595, 
            1.1384746, 1.08226639, 1.03003417, 0.9814645, 0.93626739, 0.89417545, 0.85494285, 
            0.81834398, 0.78417213, 0.75223812, 0.72236886, 0.69440612, 0.66820523, 0.64363392, 
            0.62057122, 0.5989065, 0.57853847, 0.55937441, 0.54132932, 0.52432531, 0.50829089, 
            0.4931604, 0.47887352, 0.46537477, 0.45261309, 0.44054143, 0.42911641, 0.41829804, 
            0.40804935, 0.39833621, 0.38912702, 0.38039257, 0.37210577, 0.36424155, 0.35677662, 
            0.34968939, 0.3429598, 0.33656921, 0.33050029, 0.32473692, 0.3192641, 0.31406787, 
            0.30913522, 0.30445402, 0.30001299, 0.29580161, 0.29181007, 0.28802921, 0.28445053, 
            0.28106608, 0.27786846, 0.27485079, 0.27200665, 0.26933009, 0.26681557, 0.26445795, 
            0.26225246, 0.2601947, 0.2582806, 0.25650642, 0.25486872, 0.25336434, 0.25199043, 
            0.25074437, 0.24962383, 0.24862671, 0.24775117, 0.24699559, 0.24635857, 0.24583895, 
            0.24543577, 0.24514832, 0.24497605, 0.24491866,
        ],
    
    # fmt: on
        [
            -5.75593088e-17,
            -1.48409294e-02,
            -2.96879267e-02,
            -4.45470685e-02,
            -5.94244485e-02,
            -7.43261856e-02,
            -8.92584334e-02,
            -1.04227388e-01,
            -1.19239298e-01,
            -1.34300472e-01,
            -1.49417289e-01,
            -1.64596205e-01,
            -1.79843766e-01,
            -1.95166616e-01,
            -2.10571504e-01,
            -2.26065300e-01,
            -2.41654997e-01,
            -2.57347730e-01,
            -2.73150780e-01,
            -2.89071589e-01,
            -3.05117768e-01,
            -3.21297113e-01,
            -3.37617611e-01,
            -3.54087457e-01,
            -3.70715065e-01,
            -3.87509079e-01,
            -4.04478388e-01,
            -4.21632140e-01,
            -4.38979752e-01,
            -4.56530928e-01,
            -4.74295671e-01,
            -4.92284297e-01,
            -5.10507448e-01,
            -5.28976111e-01,
            -5.47701626e-01,
            -5.66695703e-01,
            -5.85970433e-01,
            -6.05538306e-01,
            -6.25412214e-01,
            -6.45605469e-01,
            -6.66131806e-01,
            -6.87005394e-01,
            -7.08240836e-01,
            -7.29853170e-01,
            -7.51857868e-01,
            -7.74270822e-01,
            -7.97108333e-01,
            -8.20387082e-01,
            -8.44124101e-01,
            -8.68336728e-01,
            -8.93042543e-01,
            -9.18259297e-01,
            -9.44004810e-01,
            -9.70296851e-01,
            -9.97152984e-01,
            -1.02459038e00,
            -1.05262557e00,
            -1.08127418e00,
            -1.11055056e00,
            -1.14046736e00,
            -1.17103501e00,
            -1.20226110e00,
            -1.23414956e00,
            -1.26669979e00,
            -1.29990549e00,
            -1.33375335e00,
            -1.36822134e00,
            -1.40327678e00,
            -1.43887397e00,
            -1.47495132e00,
            -1.51142792e00,
            -1.54819945e00,
            -1.58513330e00,
            -1.62206275e00,
            -1.65878011e00,
            -1.69502858e00,
            -1.73049277e00,
            -1.76478773e00,
            -1.79744633e00,
            -1.82790504e00,
            -1.85548819e00,
            -1.87939109e00,
            -1.89866258e00,
            -1.91218826e00,
            -1.91867604e00,
            -1.91664670e00,
            -1.90443309e00,
            -1.88019283e00,
            -1.84194047e00,
            -1.78760601e00,
            -1.71512655e00,
            -1.62257652e00,
            -1.50833783e00,
            -1.37130478e00,
            -1.21110868e00,
            -1.02833543e00,
            -8.24698777e-01,
            -6.03126370e-01,
            -3.67720331e-01,
            -1.23570809e-01,
            1.23570809e-01,
            3.67720331e-01,
            6.03126370e-01,
            8.24698777e-01,
            1.02833543e00,
            1.21110868e00,
            1.37130478e00,
            1.50833783e00,
            1.62257652e00,
            1.71512655e00,
            1.78760601e00,
            1.84194047e00,
            1.88019283e00,
            1.90443309e00,
            1.91664670e00,
            1.91867604e00,
            1.91218826e00,
            1.89866258e00,
            1.87939109e00,
            1.85548819e00,
            1.82790504e00,
            1.79744633e00,
            1.76478773e00,
            1.73049277e00,
            1.69502858e00,
            1.65878011e00,
            1.62206275e00,
            1.58513330e00,
            1.54819945e00,
            1.51142792e00,
            1.47495132e00,
            1.43887397e00,
            1.40327678e00,
            1.36822134e00,
            1.33375335e00,
            1.29990549e00,
            1.26669979e00,
            1.23414956e00,
            1.20226110e00,
            1.17103501e00,
            1.14046736e00,
            1.11055056e00,
            1.08127418e00,
            1.05262557e00,
            1.02459038e00,
            9.97152984e-01,
            9.70296851e-01,
            9.44004810e-01,
            9.18259297e-01,
            8.93042543e-01,
            8.68336728e-01,
            8.44124101e-01,
            8.20387082e-01,
            7.97108333e-01,
            7.74270822e-01,
            7.51857868e-01,
            7.29853170e-01,
            7.08240836e-01,
            6.87005394e-01,
            6.66131806e-01,
            6.45605469e-01,
            6.25412214e-01,
            6.05538306e-01,
            5.85970433e-01,
            5.66695703e-01,
            5.47701626e-01,
            5.28976111e-01,
            5.10507448e-01,
            4.92284297e-01,
            4.74295671e-01,
            4.56530928e-01,
            4.38979752e-01,
            4.21632140e-01,
            4.04478388e-01,
            3.87509079e-01,
            3.70715065e-01,
            3.54087457e-01,
            3.37617611e-01,
            3.21297113e-01,
            3.05117768e-01,
            2.89071589e-01,
            2.73150780e-01,
            2.57347730e-01,
            2.41654997e-01,
            2.26065300e-01,
            2.10571504e-01,
            1.95166616e-01,
            1.79843766e-01,
            1.64596205e-01,
            1.49417289e-01,
            1.34300472e-01,
            1.19239298e-01,
            1.04227388e-01,
            8.92584334e-02,
            7.43261856e-02,
            5.94244485e-02,
            4.45470685e-02,
            2.96879267e-02,
            1.48409294e-02,
            5.75593088e-17,
        ],
    ])
    tau_test_r_max = np.max(tau_test_isosurface_rz_points[0])
    tau_test_r_min = np.min(tau_test_isosurface_rz_points[0])
    tau_test_z_max = np.max(tau_test_isosurface_rz_points[1])
    tau_test_z_min = np.min(tau_test_isosurface_rz_points[1])

    sigma_test_isosurface_rz_points = np.array([
        [
            0.0,
            0.20473915,
            0.40647062,
            0.60233523,
            0.78975568,
            0.96654224,
            1.13096272,
            1.28177399,
            1.41821761,
            1.53998554,
            1.64716442,
            1.74016738,
            1.81966123,
            1.88649575,
            1.94163941,
            1.98612427,
            2.02100119,
            2.04730509,
            2.06602971,
            2.07811025,
            2.08441281,
            2.08572888,
            2.08277395,
            2.07618884,
            2.06654303,
            2.05433923,
            2.04001863,
            2.02396646,
            2.00651753,
            1.98796162,
            1.96854858,
            1.948493,
            1.92797854,
            1.90716178,
            1.88617571,
            1.86513282,
            1.84412781,
            1.82324002,
            1.8025355,
            1.78206886,
            1.76188489,
            1.7420199,
            1.72250299,
            1.70335705,
            1.68459969,
            1.666244,
            1.64829924,
            1.63077143,
            1.61366383,
            1.5969774,
            1.58071114,
            1.56486244,
            1.54942733,
            1.53440075,
            1.51977671,
            1.5055485,
            1.49170882,
            1.47824992,
            1.46516368,
            1.45244174,
            1.44007556,
            1.42805649,
            1.4163758,
            1.4050248,
            1.39399478,
            1.38327713,
            1.37286332,
            1.36274493,
            1.35291368,
            1.34336144,
            1.33408024,
            1.32506227,
            1.31629991,
            1.30778571,
            1.29951241,
            1.29147293,
            1.28366039,
            1.27606809,
            1.26868951,
            1.26151832,
            1.25454838,
            1.24777371,
            1.24118853,
            1.23478721,
            1.2285643,
            1.22251453,
            1.21663276,
            1.21091403,
            1.20535352,
            1.19994657,
            1.19468867,
            1.18957542,
            1.18460259,
            1.17976607,
            1.17506188,
            1.17048615,
            1.16603515,
            1.16170526,
            1.15749296,
            1.15339487,
            1.14940769,
            1.14552822,
            1.14175338,
            1.13808018,
            1.13450569,
            1.13102713,
            1.12764175,
            1.12434691,
            1.12114006,
            1.1180187,
            1.11498044,
            1.11202293,
            1.10914391,
            1.10634119,
            1.10361264,
            1.10095619,
            1.09836984,
            1.09585165,
            1.09339973,
            1.09101225,
            1.08868744,
            1.08642357,
            1.08421898,
            1.08207204,
            1.07998118,
            1.07794487,
            1.07596162,
            1.07403,
            1.0721486,
            1.07031607,
            1.06853108,
            1.06679236,
            1.06509866,
            1.06344877,
            1.06184152,
            1.06027577,
            1.0587504,
            1.05726434,
            1.05581654,
            1.05440599,
            1.0530317,
            1.05169269,
            1.05038805,
            1.04911686,
            1.04787824,
            1.04667134,
            1.0454953,
            1.04434933,
            1.04323264,
            1.04214445,
            1.04108402,
            1.04005063,
            1.03904356,
            1.03806214,
            1.03710568,
            1.03617355,
            1.03526511,
            1.03437975,
            1.03351686,
            1.03267587,
            1.0318562,
            1.0310573,
            1.03027865,
            1.02951971,
            1.02877997,
            1.02805895,
            1.02735616,
            1.02667113,
            1.0260034,
            1.02535253,
            1.02471809,
            1.02409966,
            1.02349682,
            1.02290918,
            1.02233635,
            1.02177794,
            1.0212336,
            1.02070295,
            1.02018566,
            1.01968139,
            1.01918979,
            1.01871055,
            1.01824335,
            1.01778789,
            1.01734387,
            1.01691099,
            1.01648898,
            1.01607756,
            1.01567646,
            1.01528542,
            1.01490418,
            1.0145325,
            1.01417013,
            1.01381684,
            1.0134724,
            1.01313659,
            1.01280918,
            1.01248997,
            1.01217875,
            1.01187531,
        ],
        [
            3.91631736,
            3.90624473,
            3.8763291,
            3.82745481,
            3.76102424,
            3.67886035,
            3.58308826,
            3.47600914,
            3.35997917,
            3.23730322,
            3.11014965,
            2.98048893,
            2.85005542,
            2.72032988,
            2.5925387,
            2.46766561,
            2.34647174,
            2.22952052,
            2.11720458,
            2.00977261,
            1.90735476,
            1.80998574,
            1.71762512,
            1.63017495,
            1.54749441,
            1.46941206,
            1.39573575,
            1.32626053,
            1.26077489,
            1.1990656,
            1.14092134,
            1.08613543,
            1.03450778,
            0.98584622,
            0.93996737,
            0.89669713,
            0.85587092,
            0.81733368,
            0.78093969,
            0.74655236,
            0.71404384,
            0.68329464,
            0.65419322,
            0.62663555,
            0.60052464,
            0.57577013,
            0.55228785,
            0.52999941,
            0.50883181,
            0.48871707,
            0.46959184,
            0.45139713,
            0.43407795,
            0.41758304,
            0.40186459,
            0.38687801,
            0.37258166,
            0.35893668,
            0.34590674,
            0.3334579,
            0.32155841,
            0.31017855,
            0.29929052,
            0.28886824,
            0.27888731,
            0.2693248,
            0.26015922,
            0.25137039,
            0.24293934,
            0.23484825,
            0.22708035,
            0.21961986,
            0.21245192,
            0.20556254,
            0.19893852,
            0.1925674,
            0.18643745,
            0.18053756,
            0.17485726,
            0.16938663,
            0.16411631,
            0.15903743,
            0.15414159,
            0.14942084,
            0.14486764,
            0.14047484,
            0.13623567,
            0.13214367,
            0.12819274,
            0.12437708,
            0.12069116,
            0.11712973,
            0.11368779,
            0.1103606,
            0.10714362,
            0.10403255,
            0.10102326,
            0.09811184,
            0.09529455,
            0.09256782,
            0.08992825,
            0.08737258,
            0.0848977,
            0.08250065,
            0.08017857,
            0.07792875,
            0.0757486,
            0.07363562,
            0.07158742,
            0.06960173,
            0.06767635,
            0.06580919,
            0.06399824,
            0.06224157,
            0.06053732,
            0.05888371,
            0.05727905,
            0.05572169,
            0.05421006,
            0.05274264,
            0.05131798,
            0.04993468,
            0.0485914,
            0.04728684,
            0.04601975,
            0.04478893,
            0.04359324,
            0.04243155,
            0.0413028,
            0.04020595,
            0.03914001,
            0.03810401,
            0.03709704,
            0.03611819,
            0.03516661,
            0.03424147,
            0.03334196,
            0.03246731,
            0.03161676,
            0.03078961,
            0.02998514,
            0.02920269,
            0.0284416,
            0.02770124,
            0.026981,
            0.0262803,
            0.02559856,
            0.02493524,
            0.02428979,
            0.02366171,
            0.02305049,
            0.02245564,
            0.02187671,
            0.02131324,
            0.02076479,
            0.02023093,
            0.01971125,
            0.01920535,
            0.01871285,
            0.01823337,
            0.01776654,
            0.01731202,
            0.01686946,
            0.01643853,
            0.01601891,
            0.01561029,
            0.01521236,
            0.01482484,
            0.01444744,
            0.01407987,
            0.01372188,
            0.0133732,
            0.01303358,
            0.01270278,
            0.01238055,
            0.01206667,
            0.0117609,
            0.01146304,
            0.01117288,
            0.01089019,
            0.01061479,
            0.01034649,
            0.01008508,
            0.00983039,
            0.00958224,
            0.00934046,
            0.00910488,
            0.00887533,
            0.00865166,
            0.00843371,
            0.00822133,
            0.00801437,
            0.0078127,
            0.00761617,
            0.00742464,
            0.007238,
            0.00705611,
            0.00687884,
            0.00670608,
            0.00653771,
        ],
    ])

    sigma_test_r_max = np.max(sigma_test_isosurface_rz_points[0])
    sigma_test_r_min = np.min(sigma_test_isosurface_rz_points[0])
    sigma_test_z_max = np.max(sigma_test_isosurface_rz_points[1])
    sigma_test_z_min = np.min(sigma_test_isosurface_rz_points[1])

    # test that the coordinate transform functions are inverses of each other
    # use the test tau isosurface r,z points for this
    toroidal_conversion = cylindrical_to_toroidal(
        R_0=R_0_test,
        z_0=z_0_test,
        R=tau_test_isosurface_rz_points[0],
        Z=tau_test_isosurface_rz_points[1],
    )
    cylindrical_conversion = toroidal_to_cylindrical(
        R_0=R_0_test,
        z_0=z_0_test,
        tau=toroidal_conversion[0],
        sigma=toroidal_conversion[1],
    )
    # assert that the converted coordinates match the original coordinates
    np.testing.assert_almost_equal(cylindrical_conversion, tau_test_isosurface_rz_points)

    # generate tau and sigma isosurfaces and test that the max and min r&z points are as
    # expected
    # tau isosurface test:
    tau_input = [0.5]
    sigma_input = np.linspace(-np.pi, np.pi, 200)
    rzlist = toroidal_to_cylindrical(
        R_0=R_0_test, z_0=z_0_test, sigma=sigma_input, tau=tau_input
    )
    rs = rzlist[0]
    zs = rzlist[1]
    rs_max = np.max(rs)
    rs_min = np.min(rs)
    zs_max = np.max(zs)
    zs_min = np.min(zs)
    np.testing.assert_almost_equal(rs_max, tau_test_r_max)
    np.testing.assert_almost_equal(rs_min, tau_test_r_min)
    np.testing.assert_almost_equal(zs_max, tau_test_z_max)
    np.testing.assert_almost_equal(zs_min, tau_test_z_min)

    # sigma isosurface test
    sigma_input = [0.5]
    tau_input = np.linspace(0, 5, 200)
    rzlist = toroidal_to_cylindrical(
        R_0=R_0_test, z_0=z_0_test, sigma=sigma_input, tau=tau_input
    )
    rs = rzlist[0]
    zs = rzlist[1]
    rs_max = np.max(rs)
    rs_min = np.min(rs)
    zs_max = np.max(zs)
    zs_min = np.min(zs)
    np.testing.assert_almost_equal(rs_max, sigma_test_r_max)
    np.testing.assert_almost_equal(rs_min, sigma_test_r_min)
    np.testing.assert_almost_equal(zs_max, sigma_test_z_max)
    np.testing.assert_almost_equal(zs_min, sigma_test_z_min)
