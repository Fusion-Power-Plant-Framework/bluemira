# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Code to deal with any floating point calculation inaccuracies in freecad."""

import numpy as np


def next_freecad_float(number: float):
    """
    Find the neighbouring float larger than the supplied number.

    Parameters
    ----------
    number:
        the reference number that we want to compare against.

    Returns
    -------
    :
        The number immediately larger than the reference number.
        No float can exist between the reference number and this number due to the finite
        precision of the computer.
    """
    return np.nextafter(np.float32(number), number * 1.1)


def prev_freecad_float(number: float):
    """
    Find the neighbouring float smaller than the supplied number.

    Parameters
    ----------
    number:
        the reference number that we want to compare against.

    Returns
    -------
    :
        The number immediately smaller than the reference number.
        No float can exist between the reference number and this number due to the finite
        precision of the computer.
    """
    return np.nextafter(np.float32(number), number * 0.9)
