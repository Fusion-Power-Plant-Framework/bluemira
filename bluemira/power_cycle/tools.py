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
from typing import Any, Dict

import matplotlib.pyplot as plt


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
