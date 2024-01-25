# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Utility functions for the power cycle model.
"""

import json
import pprint
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


def pp(obj):
    """Prety Printer compatible with dataclasses."""
    if is_dataclass(obj):
        return pprint.pp(asdict(obj), sort_dicts=False, indent=4)
    return pprint.pp(obj, indent=4)


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

    For each pair of vectors (x,y), interpolate values in every 'y' vector,
    to ensure that all of them have one element associated to every element
    of the union of all distinct 'x' vectors.
    """
    n_vectors = len(x_set)

    matched_x = np.concatenate(x_set)
    matched_x = np.unique(matched_x)

    for v in range(n_vectors):
        x = x_set[v]
        y = y_set[v]

        y = np.interp(matched_x, x, y)
        y_set[v] = y

    return matched_x, y_set
