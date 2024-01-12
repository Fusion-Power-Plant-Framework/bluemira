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

    x = [1, 1.5, 2, 2.1, 2, 1.5, 1, 0.9, 1]
    z = [-1.8, -1.9, -1.8, 0, 1.8, 1.9, 1.8, 0, -1.8]
    plasma_boundary = Coordinates({"x": x, "z": z})

    point_type_1 = "arc"
    point_type_2 = "arc_plus_extrema"
    point_type_3 = "random"
    point_type_4 = "random_plus_extrema"

    colloc1 = collocation_points(n_points, plasma_boundary, point_type_1)
    colloc2 = collocation_points(n_points, plasma_boundary, point_type_2)
    colloc3 = collocation_points(n_points, plasma_boundary, point_type_3)
    colloc4 = collocation_points(n_points, plasma_boundary, point_type_4)

    assert colloc1.r.shape[0] == 8
    assert colloc2.r.shape[0] == 12
    assert colloc3.r.shape[0] == 8
    assert colloc4.r.shape[0] == 12

    for x, z in zip(colloc2.x, colloc2.z):
        assert in_polygon(x, z, plasma_boundary.xz.T, include_edges=True)

    for x, z in zip(colloc4.x, colloc4.z):
        assert in_polygon(x, z, plasma_boundary.xz.T, include_edges=True)


def test_coils_outside_sphere_vacuum_psi():
    path = get_bluemira_path("equilibria/test_data", subfolder="tests")
    eq = Equilibrium.from_eqdsk(Path(path, "SH_test_file.json"))

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
    path = get_bluemira_path("equilibria/test_data", subfolder="tests")
    eq = Equilibrium.from_eqdsk(Path(path, "SH_test_file.json"))

    test_colocation = collocation_points(
        n_points=18, plasma_boundary=eq.get_LCFS(), point_type="arc"
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
    path = get_bluemira_path("equilibria/test_data", subfolder="tests")
    eq = Equilibrium.from_eqdsk(Path(path, "SH_test_file.json"))

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
        n_points=20,
        point_type="arc_plus_extrema",
        acceptable_fit_metric=0.05,
    )

    sh_coilset_current = np.array([
        7.62910582e03,
        1.43520553e05,
        -1.39959367e05,
        1.38307695e05,
        -3.03734221e05,
        -7.65433152e04,
        -6.29515063e05,
        -1.95322559e05,
        2.07073399e06,
        1.20763164e04,
        -7.36786063e06,
        8.49088150e06,
        3.62769044e06,
        -2.38451828e06,
        2.04882463e06,
        -2.60376386e05,
        -7.47634692e04,
        -3.23836517e06,
    ])

    harmonic_amps = np.array([
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

    assert test_sh_coilset_current == pytest.approx(sh_coilset_current)
    assert test_r_t == pytest.approx(1.3661669578370919)
    assert test_harmonic_amps == pytest.approx(harmonic_amps)
    assert test_degree == 13
    assert test_fit_metric == pytest.approx(0.031, abs=0.0006)


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

    sh_coil_names = ["PF_2", "PF_3", "PF_4", "PF_5"]

    d = 3
    r_t = 1
    a_mat = coil_harmonic_amplitude_matrix(coilset, d, r_t, sh_coil_names)
    b_vec = np.array([1e-2, 1e-18])
    test_vector = np.array([1, 1, 1, 1])
    test_result = a_mat[1:,] @ test_vector

    test_constraint = SphericalHarmonicConstraintFunction(a_mat, b_vec, 0.0, 1)

    test_f_constraint = test_constraint.f_constraint(test_vector)

    assert all(test_f_constraint - (test_result - b_vec)) == 0.0
