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
Base classes and functionality for the bluemira geometry module.
"""

from __future__ import annotations

# import for abstract class
from abc import ABC, abstractmethod
from typing import List, Optional, Union

# import freecad api
from . import _freecadapi

# import bluemira base functionality
from bluemira.base.display import DisplayOptions, Displayer
from bluemira.base.error import DisplayError


class BluemiraGeo(ABC):
    """Base abstract class for geometry

    Parameters
    ----------
    boundary:
        shape's boundary
    label: str
        identification label for the shape
    boundary_classes:
        list of allowed class types for shape's boundary
    """

    # # Obsolete
    # # a set of property and methods that are inherited from FreeCAD objects
    # props = {
    #     'length': 'Length',
    #     'area': 'Area',
    #     'volume': 'Volume',
    #     'center_of_mass': 'CenterOfMass'
    # }
    # metds = {
    #     'is_null': 'isNull',
    #     'is_closed': 'isClosed'
    # }
    # attrs = {**props, **metds}

    def __init__(
        self,
        boundary,
        label: str = "",
        boundary_classes=None,
    ):
        self._boundary_classes = boundary_classes
        self.boundary = boundary
        self.label = label

    @staticmethod
    def _converter(func):
        """Function used in __getattr__ to modify the added functions"""
        return func

    # Obsolete.
    # It was used to getattr from the primitive object, but it was replaced
    # with specific implementation into the respective api. However it could be still
    # useful (for this reason is just commented).
    # def __getattr__(self, key):
    #     """
    #     Transfer the key getattr to shape object.
    #     """
    #     if key in type(self).attrs:
    #         output = getattr(self._shape, type(self).attrs[key])
    #         if callable(output):
    #             return self.__class__._converter(output)
    #         else:
    #             return output
    #     else:
    #         raise AttributeError("'{}' has no attribute '{}'".format(str(type(
    #             self).__name__), key))

    def _check_boundary(self, objs):
        """Check if objects objs can be used as boundaries"""
        if not hasattr(objs, "__len__"):
            objs = [objs]
        check = False
        for c in self._boundary_classes:
            check = check or (all(isinstance(o, c) for o in objs))
            if check:
                return objs
        raise TypeError(
            f"Only {self._boundary_classes} objects can be used for {self.__class__}"
        )

    @property
    def boundary(self):
        """Shape's boundary"""
        return self._boundary

    @boundary.setter
    def boundary(self, objs):
        self._boundary = self._check_boundary(objs)

    @property
    @abstractmethod
    def _shape(self):
        """Primitive shape of the object"""
        # Note: this is the "hidden" connection with primitive shapes
        pass

    @property
    def length(self):
        """Shape's length"""
        return _freecadapi.length(self._shape)

    @property
    def area(self):
        """Shape's area"""
        return _freecadapi.area(self._shape)

    @property
    def volume(self):
        """Shape's volume"""
        return _freecadapi.volume(self._shape)

    @property
    def center_of_mass(self):
        """Shape's center of mass"""
        return _freecadapi.center_of_mass(self._shape)

    @property
    def bounding_box(self):
        """Checks if the shape is closed"""
        return _freecadapi.bounding_box(self._shape)

    def is_null(self):
        """Checks if the shape is null."""
        return _freecadapi.is_null(self._shape)

    def is_closed(self):
        """Checks if the shape is closed"""
        return _freecadapi.is_closed(self._shape)

    def search(self, label: str):
        """Search for a shape with the specified label

        Parameters
        ----------
        label : str
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

    def scale(self, factor) -> None:
        """Apply scaling with factor to this object. This function modifies the self
        object.
        """
        for o in self.boundary:
            o.scale(factor)

    def translate(self, vector) -> None:
        """Translate this shape with the vector. This function modifies the self
        object.
        """
        for o in self.boundary:
            o.translate(vector)

    def __repr__(self):  # noqa D105
        new = []
        new.append(f"([{type(self).__name__}] = Label: {self.label}")
        new.append(f" length: {self.length}")
        new.append(f" area: {self.area}")
        new.append(f" volume: {self.volume}")
        new.append(")")
        return ", ".join(new)


class GeometryDisplayer(Displayer):
    """
    A Displayer class for displaying BluemiraGeo objects in 3D.
    """

    def display(
        self,
        geos: Union[BluemiraGeo, List[BluemiraGeo]],
        options: Optional[Union[DisplayOptions, List[DisplayOptions]]] = None,
    ) -> None:
        """
        Display a BluemiraGeo object using the underlying shape.

        Parameters
        ----------
        geo: Union[BluemiraGeo, List[BluemiraGeo]]
            The geometry to be displayed.
        options: Optional[Union[DisplayOptions, List[DisplayOptions]]]
            The options to use to display the geometry.
            By default None, in which case the display_options assigned to the
            BluemiraGeo object will be used.
        """
        if options is None:
            options = DisplayOptions()

        if not isinstance(geos, list):
            geos = [geos]
        if not isinstance(options, list):
            options = [options]

        if len(options) == 1 and len(geos) > 1:
            options *= len(geos)

        if len(options) != len(geos):
            raise DisplayError(
                "Either a single display option or the same number of display options "
                "geometries must be provided."
            )

        shapes = []
        for geo in geos:
            shapes += [geo._shape]
        super().display(shapes, options)
