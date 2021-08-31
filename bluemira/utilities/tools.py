# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
A collection of miscellaneous tools.
"""

import numpy as np
from json import JSONEncoder


class NumpyJSONEncoder(JSONEncoder):
    """
    A JSON encoder that can handle numpy arrays.
    """

    def default(self, obj):
        """
        Override the JSONEncoder default object handling behaviour for np.arrays.
        """
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def is_num(thing):
    """
    Determine whether or not the input is a number.

    Parameters
    ----------
    thing: unknown type
        The input which we need to determine is a number or not

    Returns
    -------
    num: bool
        Whether or not the input is a number
    """
    if thing is True or thing is False:
        return False
    if thing is np.nan:
        return False
    try:
        float(thing)
        return True
    except (ValueError, TypeError):
        return False


def abs_rel_difference(v2, v1_ref):
    """
    Calculate the absolute relative difference between a new value and an old
    reference value.

    Parameters
    ----------
    v2: float
        The new value to compare to the old
    v1_ref: float
        The old reference value

    Returns
    -------
    delta: float
        The absolute relative difference between v2 and v1ref
    """
    return abs((v2 - v1_ref) / v1_ref)
