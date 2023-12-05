# COPYRIGHT PLACEHOLDER

"""
Utility functions for the power cycle model.
"""

import json
from typing import Any, Dict

import matplotlib.pyplot as plt


def read_json(file_path) -> Dict[str, Any]:
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
