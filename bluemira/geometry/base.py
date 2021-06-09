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
            lcar: Union[float, [float]] = 0.1
    ):

        self.__boundary_classes = []
        self.boundary = boundary
        self.label = label
        self.lcar = lcar

    def __check_objs(self, objs):
        """Check if objects in objs are of the correct type for this class"""
        if not hasattr(objs, '__len__'):
            objs = [objs]
        check = False
        for c in self.__boundary_classes:
            check = check or (all(isinstance(o, c) for o in objs))
            if check:
                return check
        return check

    @property
    def boundary(self):
        return self._boundary

    @boundary.setter
    def boundary(self, objs):
        self._boundary = None
        # if objs is not a list, it is converted into a list.
        if self.__check_objs(objs):
            self._boundary = objs
        else:
            raise ValueError("Only {} objects can be used for {}".format(
                self.__boundary_classes, self.__class__()))

    @property
    @abstractmethod
    def Length(self):
        """float: total length of the shape."""
        pass

    @property
    @abstractmethod
    def Area(self):
        """float: total area of the shape."""
        pass

    @property
    @abstractmethod
    def Volume(self):
        """float: total volume of the shape."""
        pass

    def search(self, label: str):
        """Search for a shape with the specified label

        Parameters
        ----------
        label : str :
            shape label.

        Returns
        -------
        output : list(Shape
            list of shapes that have the specified label.

        """
        output = []
        if self.label == label:
            output.append(self)
        for o in self.boundary:
            if isinstance(o, BluemiraBase):
                output += o.search(label)
        return output
