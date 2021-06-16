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
Base classes and functionality for the bluemira geometry module.
"""

from __future__ import annotations

# import from freecad
import freecad
import Part
from FreeCAD import Base

# import typing
from typing import Union

# import for abstract class
from abc import ABC, abstractmethod

# import for logging
import logging
module_logger = logging.getLogger(__name__)


class BluemiraGeo(ABC):
    """Base abstract class for geometry"""
    def __init__(
            self,
            boundary,
            label: str = "",
            lcar: Union[float, [float]] = 0.1,
            boundary_classes: [cls] = None
    ):

        self._boundary_classes = boundary_classes
        self.boundary = boundary
        self.label = label
        self.lcar = lcar

    @abstractmethod
    def _check_boundary(self, objs):
        """Check if objects objs can be used as boundaries"""
        if not hasattr(objs, '__len__'):
            objs = [objs]
        check = False
        for c in self._boundary_classes:
            check = check or (all(isinstance(o, c) for o in objs))
            if check:
                return objs
        raise TypeError("Only {} objects can be used for {}".format(
            self._boundary_classes))

    @property
    def boundary(self):
        return self._boundary

    @boundary.setter
    @abstractmethod
    def boundary(self, objs):
        self._boundary = self._check_boundary(objs)

    @property
    @abstractmethod
    def shape(self):
        """Representative shape of the object"""
        pass

    @property
    def Length(self):
        """float: total length of the shape."""

        # Note:the method is recursively implemented considering that
        # the used FreeCAD object in self.shape has a similar Length property.

        return self.shape.Length

    @property
    def Area(self):
        """float: total area of the shape."""

        # Note:the method is recursively implemented considering that
        # the used FreeCAD object in self.shape has a similar Area property.

        return self.shape.Area

    @property
    def Volume(self):
        """float: total volume of the shape."""

        # Note:the method is recursively implemented considering that
        # the used FreeCAD object in self.shape has a similar Volume property.

        return self.shape.Volume

    def search(self, label: str):
        """Search for a shape with the specified label

        Parameters
        ----------
        label : str :
            shape label.

        Returns
        -------
        output : [BluemiraGeo]
            list of shapes that have the specified label.

        """
        output = []
        if self.label == label:
            output.append(self)
        for o in self.boundary:
            if isinstance(o, BluemiraGeo):
                output += o.search(label)
        return output

    def __repr__(self):
        new = []
        new.append("([{}] = Label: {}".format(type(self).__name__, self.label))
        new.append(" Length: {}".format(self.Length))
        new.append(" Area: {}".format(self.Area))
        new.append(" Volume: {}".format(self.Volume))
        new.append(")")
        return ", ".join(new)