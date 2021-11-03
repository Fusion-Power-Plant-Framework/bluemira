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
api for plotting using freecad
"""
from __future__ import annotations

# import typing
from typing import List, Optional, Union

# import errors
from .error import DisplayError

# import visualisation
from ..geometry import _freecadapi

import bluemira.geometry as geo

from . import display
import copy

DEFAULT = {
    "rgb": (0.5, 0.5, 0.5),
    "transparency": 0.0,
}


# =======================================================================================
# Visualisation
# =======================================================================================
class FreeCADPlotOptions(display.PlotCADOptions):
    """
    The options that are available for displaying objects in 3D
    Parameters
    ----------
    rgb: Tuple[float, float, float]
        The RBG colour to display the object, by default (0.5, 0.5, 0.5).
    transparency: float
        The transparency to display the object, by default 0.0.
    """

    def __init__(self, **kwargs):
        self._options = copy.deepcopy(DEFAULT)
        if kwargs:
            for k in kwargs:
                if k in self.options:
                    self.options[k] = kwargs[k]
        # TODO: in this way class attributes are not seen till runtime. Not sure if
        #  this should be changed manually declaring all the attributes.
        for k in self._options:
            setattr(self, k, self._options[k])

    def as_dict(self):
        return self._options


def plotcad(
    parts: Union[geo.base.BluemiraGeo, List[geo.base.BluemiraGeo]],
    options: Optional[Union[FreeCADPlotOptions, List[FreeCADPlotOptions]]] = None,
):
    """
    The implementation of the display API for FreeCAD parts.

    Parameters
    ----------
    parts: Union[Part.Shape, List[Part.Shape]]
        The parts to display.
    options: Optional[Union[FreeCADPlotOptions, List[FreeCADPlotOptions]]]
        The options to use to display the parts.
    """
    if not isinstance(parts, list):
        parts = [parts]

    if options is None:
        options = [FreeCADPlotOptions()] * len(parts)
    elif not isinstance(options, list):
        options = [options] * len(parts)

    if len(options) != len(parts):
        raise DisplayError(
            "If options for display are provided then there must be as many options as "
            "there are parts to display."
        )

    shapes = [part._shape for part in parts]
    freecad_options = [o.as_dict() for o in options]

    _freecadapi.plotcad(shapes, freecad_options)

