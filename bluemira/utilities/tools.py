# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
import string
from bluemira.base.constants import E_I, E_IJ, E_IJK


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


# =====================================================
# Einsum utilities
# =====================================================
def asciistr(length):
    """
    Get a string of characters of desired length.

    Current max is 52 characters

    Parameters
    ----------
    length: int
        number of characters to return

    Returns
    -------
    str of length specified

    """
    if length > 52:
        raise ValueError("Unsupported string length")

    return string.ascii_letters[:length]


class EinsumWrapper:
    """
    Preallocator for einsum versions of dot, cross and norm.
    """

    def __init__(self):

        norm_a0 = "ij, ij -> j"
        norm_a1 = "ij, ij -> i"

        self.norm_strs = [norm_a0, norm_a1]

        # Not fool proof for huge no's of dims
        self.dot_1x1 = "i, i -> ..."
        self.dot_1x2 = "i, ik -> k"
        self.dot_2x1 = "ij, j -> i"
        self.dot_2x2 = "ij, jk -> ik"
        self.dot_1xn = "y, {}yz -> {}z"
        self.dot_nx1 = "{}z, z -> {}"
        self.dot_nxn = "{}y, {}yz -> {}z"

        cross_2x1 = "i, i, i -> i"
        cross_2x2 = "xy, ix, iy -> i"
        cross_2x3 = "xyz, ix, iy -> iz"

        self.cross_strs = [cross_2x1, cross_2x2, cross_2x3]
        self.cross_lcts = [E_I, E_IJ, E_IJK]

    def norm(self, ix, axis=0):
        """
        Emulates some of the functionality of np.linalg.norm for 2D arrays.

        Specifically:
        np.linalg.norm(ix, axis=0)
        np.linalg.norm(ix, axis=1)

        For optimum speed and customisation use np.einsum modified for your use case.

        Parameters
        ----------
        ix: np.array
            Array to perform norm on
        axis: int
            axis for the norm to occur on

        Returns
        -------
        np.array

        """
        try:
            return np.sqrt(np.einsum(self.norm_strs[axis], ix, ix))
        except IndexError:
            raise ValueError("matrices dimensions >2d Unsupported")

    def dot(self, ix, iy, out=None):
        """
        A dot product emulation using np.einsum.

        For optimum speed and customisation use np.einsum modified for your use case.

        Should follow the same mechanics as np.dot, a few examples:

        ein_str = 'i, i -> ...'
        ein_str = 'ij, jk -> ik' # Classic dot product
        ein_str = 'ij, j -> i'
        ein_str = 'i, ik -> k'
        ein_str = 'aij, ajk -> aik' # for loop needed with np.dot

        Parameters
        ----------
        ix: np.array
            First array
        iy: np.array
            Second array
        out: np.array
            output array for inplace dot product

        Returns
        -------
        np.array

        """
        # Ordered hopefully by most used
        if ix.ndim == 2 and iy.ndim == 2:
            out_str = self.dot_2x2
        elif ix.ndim > 2 and iy.ndim > 2:
            ix_str = asciistr(ix.ndim - 1)
            iy_str = asciistr(iy.ndim - 2)
            out_str = self.dot_nxn.format(ix_str, iy_str, ix_str)
        elif ix.ndim < 2 and iy.ndim == 2:
            out_str = self.dot_1x2
        elif ix.ndim >= 2 and iy.ndim < 2:
            ix_str = asciistr(ix.ndim - 1)
            out_str = self.dot_nx1.format(ix_str, ix_str)
        elif iy.ndim >= 2 or ix.ndim == 2:
            raise ValueError(
                f"Undefined behaviour ix.shape:{ix.shape}, iy.shape:{iy.shape}"
            )
        else:
            out_str = self.dot_1x1

        return np.einsum(out_str, ix, iy, out=out)

    def cross(self, ix, iy, out=None):
        """
        A row-wise cross product of a 2D matrices of vectors.

        This function mirrors the properties of np.cross
        such as vectors of 2 or 3 elements. 1D is also accepted
        but just do x * y.
        Only 7D has similar orthogonal properties above 3D.

        For optimum speed and customisation use np.einsum modified for your use case.

        Parameters
        ----------
        ix: np.array
            1st array to cross
        iy: np.array
            2nd array to cross
        out: np.array
            output array for inplace cross product

        Returns
        -------
        np.array (ix.shape)

        Raises
        ------
        ValueError
            If the dimensions of the cross product are > 3

        """
        dim = ix.shape[-1] - 1 if ix.ndim > 1 else 0

        try:
            return np.einsum(self.cross_strs[dim], self.cross_lcts[dim], ix, iy, out=out)
        except IndexError:
            raise ValueError("Incompatible dimension for cross product")


wrap = EinsumWrapper()

norm = wrap.norm
dot = wrap.dot
cross = wrap.cross
