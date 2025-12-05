# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
This is a complex constrained optimisation problem from Colville
Found in chapter 4 here: https://apps.dtic.mil/sti/tr/pdf/AD0679037.pdf
Note that there is a typo in one of the equality constraints.
See https://courses.mai.liu.se/GU/TAOP04/process-optimization.pdf
"""

import numpy as np
import pytest

from bluemira.optimisation import Algorithm, optimise


class AlkylationData:
    c1 = 0.063
    c2 = 5.04
    c3 = 0.035
    c4 = 10.0
    c5 = 3.36
    d4l = 99.0 / 100.0
    d4u = 100.0 / 99.0
    d7l = 99.0 / 100.0
    d7u = 100.0 / 99.0
    d9l = 9.0 / 10.0
    d9u = 10.0 / 9.0
    d10l = 99.0 / 100.0
    d10u = 100.0 / 99.0
    dimension = 10
    lower_bounds = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 85.0, 90.0, 3.0, 1.2, 145.0])
    upper_bounds = np.array([
        2000.0,
        16000.0,
        120.0,
        5000.0,
        2000.0,
        93.0,
        95.0,
        12.0,
        4.0,
        162.0,
    ])
    suggested_x0 = np.array([
        1745.0,
        12000.0,
        110.0,
        3048.0,
        1974.0,
        89.2,
        92.8,
        8.0,
        3.6,
        145.0,
    ])
    # Given to 1DP
    true_x = np.array([
        1698.0,
        15818.0,
        54.1,
        3031.0,
        2000.0,
        90.1,
        95.0,
        10.5,
        1.6,
        154.0,
    ])
    # Given to 0DP
    true_f_x = 1769.0


ALKYLATION_DATA = AlkylationData()


def f_objective(x):
    """
    This is a maximisation objective, so we flip the sign.
    """
    return (
        -ALKYLATION_DATA.c1 * x[3] * x[6]
        + ALKYLATION_DATA.c2 * x[0]
        + ALKYLATION_DATA.c3 * x[1]
        + ALKYLATION_DATA.c4 * x[2]
        + ALKYLATION_DATA.c5 * x[4]
    )


def df_objective(x):
    grad = np.zeros(ALKYLATION_DATA.dimension)
    grad[0] = ALKYLATION_DATA.c2
    grad[1] = ALKYLATION_DATA.c3
    grad[2] = ALKYLATION_DATA.c4
    grad[3] = -ALKYLATION_DATA.c1 * x[6]
    grad[4] = ALKYLATION_DATA.c5
    grad[6] = -ALKYLATION_DATA.c1 * x[3]
    return grad


def f_equality1(x):
    return 1.22 * x[3] - x[0] - x[4]


def df_equality1(x):
    grad = np.zeros(ALKYLATION_DATA.dimension)
    grad[0] = -1.0
    grad[3] = 1.22
    grad[4] = -1.0
    return grad


def f_equality2(x):
    # There is a typo where I found this orginally...
    return 98_000.0 * x[2] / (x[3] * x[8] + 1000.0 * x[2]) - x[5]


def df_equality2(x):
    grad = np.zeros(ALKYLATION_DATA.dimension)
    grad[2] = (98_000.0 * x[3] * x[8]) / (x[3] * x[8] + 1000.0 * x[2]) ** 2
    grad[3] = -98_000.0 * x[2] * x[8] / (x[8] * x[3] + 1000.0 * x[2]) ** 2
    grad[5] = -1.0
    grad[8] = -98_000.0 * x[2] * x[3] / (x[8] * x[3] + 1000.0 * x[2]) ** 2
    return grad


def f_equality3(x):
    return (x[1] + x[4]) / x[0] - x[7]


def df_equality3(x):
    grad = np.zeros(ALKYLATION_DATA.dimension)
    grad[0] = -(x[1] + x[4]) / x[0] ** 2
    grad[1] = 1.0 / x[0]
    grad[4] = 1.0 / x[0]
    grad[7] = -1.0
    return grad


def f_inequality1(x):
    a = 1.12 + 0.13167 * x[7] - 0.00667 * x[7] ** 2
    return -x[0] * a + ALKYLATION_DATA.d4l * x[3]


def df_inequality1(x):
    grad = np.zeros(ALKYLATION_DATA.dimension)
    a = 1.12 + 0.13167 * x[7] - 0.00667 * x[7] ** 2
    grad[0] = -a
    grad[3] = ALKYLATION_DATA.d4l
    grad[7] = -x[0] * (0.13167 - 2.0 * 0.00667 * x[7])
    return grad


def f_inequality2(x):
    a = 1.12 + 0.13167 * x[7] - 0.00667 * x[7] ** 2
    return x[0] * a - ALKYLATION_DATA.d4u * x[3]


def df_inequality2(x):
    grad = np.zeros(ALKYLATION_DATA.dimension)
    a = 1.12 + 0.13167 * x[7] - 0.00667 * x[7] ** 2
    grad[0] = a
    grad[3] = -ALKYLATION_DATA.d4u
    grad[7] = x[0] * (0.13167 - 2.0 * 0.00667 * x[7])
    return grad


def f_inequality3(x):
    return (
        -(86.35 + 1.098 * x[7] - 0.038 * x[7] ** 2 + 0.325 * (x[5] - 89.0))
        + ALKYLATION_DATA.d7l * x[6]
    )


def df_inequality3(x):
    grad = np.zeros(ALKYLATION_DATA.dimension)
    grad[5] = -0.325
    grad[6] = ALKYLATION_DATA.d7l
    grad[7] = -1.098 + 2.0 * 0.038 * x[7]
    return grad


def f_inequality4(x):
    return (
        86.35 + 1.098 * x[7] - 0.038 * x[7] ** 2 + 0.325 * (x[5] - 89.0)
    ) - ALKYLATION_DATA.d7u * x[6]


def df_inequality4(x):
    grad = np.zeros(ALKYLATION_DATA.dimension)
    grad[5] = 0.325
    grad[6] = -ALKYLATION_DATA.d7u
    grad[7] = 1.098 - 2.0 * 0.038 * x[7]
    return grad


def f_inequality5(x):
    return -(35.82 - 0.222 * x[9]) + ALKYLATION_DATA.d9l * x[8]


def df_inequality5(x):
    grad = np.zeros(ALKYLATION_DATA.dimension)
    grad[8] = ALKYLATION_DATA.d9l
    grad[9] = 0.222
    return grad


def f_inequality6(x):
    return (35.82 - 0.222 * x[9]) - ALKYLATION_DATA.d9u * x[8]


def df_inequality6(x):
    grad = np.zeros(ALKYLATION_DATA.dimension)
    grad[8] = -ALKYLATION_DATA.d9u
    grad[9] = -0.222
    return grad


def f_inequality7(x):
    return -(-133.0 + 3.0 * x[6]) + ALKYLATION_DATA.d10l * x[9]


def df_inequality7(x):
    grad = np.zeros(ALKYLATION_DATA.dimension)
    grad[6] = -3.0
    grad[9] = ALKYLATION_DATA.d10l
    return grad


def f_inequality8(x):
    return (-133.0 + 3.0 * x[6]) - ALKYLATION_DATA.d10u * x[9]


def df_inequality8(x):
    grad = np.zeros(ALKYLATION_DATA.dimension)
    grad[6] = 3.0
    grad[9] = -ALKYLATION_DATA.d10u
    return grad

@pytest.mark.parametrize("algorithm", [Algorithm.SLSQP, Algorithm.SLSQP_SCIPY])
def test_alkylation_problem(algorithm):
    result = optimise(
        f_objective,
        x0=ALKYLATION_DATA.suggested_x0,
        df_objective=df_objective,
        eq_constraints=[
            {
                "f_constraint": f_equality1,
                "df_constraint": df_equality1,
                "tolerance": np.array([1e-6]),
            },
            {
                "f_constraint": f_equality2,
                "df_constraint": df_equality2,
                "tolerance": np.array([1e-6]),
            },
            {
                "f_constraint": f_equality3,
                "df_constraint": df_equality3,
                "tolerance": np.array([1e-6]),
            },
        ],
        ineq_constraints=[
            {
                "f_constraint": f_inequality1,
                "df_constraint": df_inequality1,
                "tolerance": np.array([1e-6]),
            },
            {
                "f_constraint": f_inequality2,
                "df_constraint": df_inequality2,
                "tolerance": np.array([1e-6]),
            },
            {
                "f_constraint": f_inequality3,
                "df_constraint": df_inequality3,
                "tolerance": np.array([1e-6]),
            },
            {
                "f_constraint": f_inequality4,
                "df_constraint": df_inequality4,
                "tolerance": np.array([1e-6]),
            },
            {
                "f_constraint": f_inequality5,
                "df_constraint": df_inequality5,
                "tolerance": np.array([1e-6]),
            },
            {
                "f_constraint": f_inequality6,
                "df_constraint": df_inequality6,
                "tolerance": np.array([1e-6]),
            },
            {
                "f_constraint": f_inequality7,
                "df_constraint": df_inequality7,
                "tolerance": np.array([1e-6]),
            },
            {
                "f_constraint": f_inequality8,
                "df_constraint": df_inequality8,
                "tolerance": np.array([1e-6]),
            },
        ],
        bounds=(ALKYLATION_DATA.lower_bounds, ALKYLATION_DATA.upper_bounds),
        algorithm=algorithm,
        opt_conditions={"max_eval": 5000, "ftol_rel": 1e-9},
    )

    assert np.round(abs(result.f_x)) == ALKYLATION_DATA.true_f_x
    np.testing.assert_allclose(
        np.round(result.x, decimals=1),
        ALKYLATION_DATA.true_x,
        atol=0.0,
        rtol=0.0033,
    )
