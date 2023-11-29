# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import numpy as np


def rosenbrock(x: np.ndarray, a: float = 1, b: float = 100) -> float:
    return (a - x[0]) ** 2 + b * (x[1] - x[0] ** 2) ** 2


def d_rosenbrock(x: np.ndarray, a: float = 1, b: float = 100) -> np.ndarray:
    grad = np.zeros(2)
    grad[0] = -2 * a + 4 * b * x[0] ** 3 - 4 * b * x[0] * x[1] + 2 * x[0]
    grad[1] = 2 * b * (x[1] - x[0] ** 2)
    return grad
