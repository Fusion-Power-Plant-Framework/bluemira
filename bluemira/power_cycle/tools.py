# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Utility functions for the power cycle model.
"""

import json
from itertools import pairwise
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.constants import EPS

DEFAULT_EPSILON = 1e-10


def read_json(file_path) -> dict[str, Any]:
    """
    Returns the contents of a 'json' file.
    """
    with open(file_path) as json_file:
        return json.load(json_file)


def create_axes(ax=None):
    """
    Create axes object.

    If 'None', creates a new 'axes' instance.
    """
    if ax is None:
        _, ax = plt.subplots()
    return ax


def unique_domain(
    x: np.ndarray,
    y: np.ndarray,
    epsilon: np.number = DEFAULT_EPSILON,
):
    """
    Ensure x has only unique values to make (Domain: x -> Image: y) a function.

    Epsilon must be small enough so that consecutive values can be considered
    equal within the context/scale in which the function is defined.

    Nudge forward each non-unique element in x by (N * epsilon), given N times,
    after the first appearance, that value has appeared in x before.
    """
    n_points = len(x)
    if len(y) != n_points:
        # pad x or y depending on another argument
        raise ValueError("x and y must have the same number of elements.")
    new_y = y.copy()

    new_x = [x[0]]
    if n_points > 1:
        nudge = 0
        for x_last, x_this in pairwise(x):
            if np.isclose(x_last, x_this, rtol=EPS):
                nudge += epsilon
                new_x_this = x_last + nudge
            else:
                new_x_this = x_this
                nudge = 0
            new_x.append(new_x_this)

    return np.asarray(new_x), np.asarray(new_y)


def match_domains(
    x_set: list[np.ndarray],
    y_set: list[np.ndarray],
    epsilon: np.number = DEFAULT_EPSILON,
):
    """
    Match the domains of multiple functions, each represented by 2 vectors.

    First, for each pair of vectors (x,y) that define a function, the domain
    x is ensured to be unique with 'unique_domain' (otherwise, calling 'unique'
    on 'matched_x' later can neglect step functions).

    Then, for each pair (x,y), values in every 'y' vector are interpolated,
    to ensure that all of them have one element associated to every element
    of the union of all distinct 'x' vectors. Values defined before and after
    the original domain 'x' of the image 'y' are set to zero.
    """
    n_vectors = len(x_set)
    for v in range(n_vectors):
        x_set[v], y_set[v] = unique_domain(
            x_set[v],
            y_set[v],
            epsilon,
        )

    matched_x = np.concatenate(x_set)
    matched_x = np.unique(matched_x)

    for v in range(n_vectors):
        x = x_set[v]
        y = y_set[v]

        y = np.interp(matched_x, x, y, left=0, right=0)
        y_set[v] = y

    return matched_x, y_set
