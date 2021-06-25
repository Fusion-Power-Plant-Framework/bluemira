#  bluemira is an integrated inter-disciplinary design tool for future fusion
#  reactors. It incorporates several modules, some of which rely on other
#  codes, to carry out a range of typical conceptual fusion reactor design
#  activities.
#  #
#  Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                     J. Morris, D. Short
#  #
#  bluemira is free software; you can redistribute it and/or
#  modify it under the terms of the GNU Lesser General Public
#  License as published by the Free Software Foundation; either
#  version 2.1 of the License, or (at your option) any later version.
#  #
#  bluemira is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#  Lesser General Public License for more details.
#  #
#  You should have received a copy of the GNU Lesser General Public
#  License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
Useful functions for bluemira geometries.
"""
# import from freecadapi
from . import _freecadapi

# import bluemira geometries
from .wire import BluemiraWire

# import mathematical modules
import numpy

# import typing
from typing import Union

###################################
# Geometry creation
###################################


def make_polygon(points: Union[list, numpy.ndarray], label: str = "", closed: bool =
                 False) -> BluemiraWire:
    """Make a polygon from a set of points.

    Args:
        points (Union[list, numpy.ndarray]): list of points. It can be given
            as a list of 3D tuples, a 3D numpy array, or similar.
        label (str): a label string.
        closed (bool, optional): if True, the first and last points will be
            connected in order to form a closed polygon. Defaults to False.

    Returns:
        BluemiraWire: a bluemira wire that contains the polygon
    """

    return BluemiraWire(_freecadapi.make_polygon(points, closed), label=label)


###################################
# Save functions
###################################

def save_as_STEP(shapes, filename="test", scale=1):
    """
    Saves a series of Shape objects as a STEP assembly

    Parameters
    ----------
    shapes: (Shape, ..)
        Iterable of shape objects to be saved
    filename: str
        Full path filename of the STP assembly
    scale: float (default 1)
        The scale in which to save the Shape objects
    """

    if not filename.endswith(".STP"):
        filename += ".STP"

    if not isinstance(shapes, list):
        shapes = [shapes]

    freecad_shapes = [s._shape for s in shapes]
    _freecadapi.save_as_STEP(freecad_shapes, filename, scale)

