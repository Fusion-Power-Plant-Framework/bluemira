# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Utility functions for the power cycle model."""

import json
import pprint
from copy import deepcopy
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


def read_json(file_path) -> Dict[str, Any]:
    """Return the contents of a 'json' file."""
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


def recursive_value_types_in_dict(dictionary):
    """Recursively display value types in a dictionary."""
    types_dict = dictionary.copy()
    for key, value in types_dict.items():
        if isinstance(types_dict[key], dict):
            types_dict[key] = recursive_value_types_in_dict(types_dict[key])
        else:
            types_dict[key] = type(value)
    return types_dict


def pp(obj, types_only=False):
    """Prety Printer compatible with dataclasses."""
    kwargs = {"indent": 4}
    target = deepcopy(obj)
    if is_dataclass(target):
        kwargs["sort_dicts"] = False
        target = asdict(target)
    if types_only and isinstance(target, dict):
        target = recursive_value_types_in_dict(target)
    return pprint.pp(target, **kwargs)


def symmetrical_subplot_distribution(n_plots, direction="row"):
    """Create a symmetrical (squared) distribution for subplots."""
    n_primary = np.ceil(np.sqrt(n_plots))
    n_secondary = np.ceil(n_plots / n_primary)

    valid_row_args = {"row", "rows", "r", "R"}
    valid_col_args = {"col", "cols", "c", "C"}
    if direction in valid_row_args:
        n_rows = int(n_primary)
        n_cols = int(n_secondary)
    elif direction in valid_col_args:
        n_rows = int(n_secondary)
        n_cols = int(n_primary)
    else:
        raise ValueError(
            f"Invalid argument: '{direction}'. The parameter"
            "'direction' can only assume one of the following values:"
            "'row' or 'col'."
        )
    return n_rows, n_cols


def match_domains(
    x_set: List[np.ndarray],
    y_set: List[np.ndarray],
):
    """
    Match the domains of multiple functions, each represented by 2 vectors.

    First, for each pair of vectors (x,y) that define a function, the domain
    x is ensured to be unique with 'unique_domain' (otherwise, calling 'unique'
    on 'matched_x' later can neglect step functions).

    Then, for each pair (x,y), values in every 'y' vector are interpolated,
    to ensure that all of them have one element associated to every element
    of the union of all distinct 'x' vectors.
    """
    n_vectors = len(x_set)
    for v in range(n_vectors):
        x_set[v], y_set[v] = unique_domain(x_set[v], y_set[v])

    matched_x = np.concatenate(x_set)
    matched_x = np.unique(matched_x)

    for v in range(n_vectors):
        x = x_set[v]
        y = y_set[v]

        y = np.interp(matched_x, x, y)
        y_set[v] = y

    return matched_x, y_set


def unique_domain(x, y, epslon=1e-10):
    """
    Ensure x has only unique values to make (Domain: x -> Image: y) a function.

    Nudge forward each non-unique element in x by (N * epslon), given N times,
    after the first appearance, that value has appeared in x before.

    Epslon must be small enough so that consecutive values can be considered
    equal within the context/scale in which the function is defined.
    """
    n_points = len(x)
    if len(y) != n_points:
        raise ValueError("x and y must have the same number of elements.")
    new_x = [x[0]]
    new_y = [y[0]]

    nudge = 0
    for p in range(1, n_points):
        x_last = x[p - 1]
        x_this = x[p]
        y_this = y[p]
        if x_last == x_this:
            nudge += epslon
            x_this = x_last + nudge
        else:
            nudge = 0
        new_x.append(x_this)
        new_y.append(y_this)
    return new_x, new_y
