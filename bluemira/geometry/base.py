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

import copy
import enum

# import for abstract class
from abc import ABC, abstractmethod

import bluemira.mesh.meshing as meshing

# import freecad api
from bluemira.codes import _freecadapi as cadapi
from bluemira.geometry.bound_box import BoundingBox


class GeoMeshable(meshing.Meshable):
    """
    Extended Meshable class for BluemiraGeo objects.
    """

    def remove_mesh_options(self, recursive=False):
        """
        Remove mesh options for this object.
        """
        super().remove_mesh_options()
        if hasattr(self, "boundary"):
            for obj in self.boundary:
                if isinstance(obj, GeoMeshable):
                    obj.remove_mesh_options(recursive=True)

    def print_mesh_options(self, recursive=True):
        """
        Print the mesh options for this object.
        """
        # TODO: improve the output of this function
        output = []
        output.append(self.mesh_options)
        if hasattr(self, "boundary"):
            for obj in self.boundary:
                if isinstance(obj, GeoMeshable):
                    output.append(obj.print_mesh_options(True))
        return output


class _Orientation(enum.Enum):
    FORWARD = "Forward"
    REVERSED = "Reversed"


class BluemiraGeo(ABC, GeoMeshable):
    """
    Abstract base class for geometry.

    Parameters
    ----------
    boundary:
        shape's boundary
    label: str
        identification label for the shape
    boundary_classes:
        list of allowed class types for shape's boundary
    """

    def __init__(
        self,
        boundary,
        label: str = "",
        boundary_classes=None,
    ):
        super().__init__()
        self._boundary_classes = boundary_classes
        self.boundary = boundary
        self.label = label
        self.__orientation = _Orientation("Forward")

    @property
    def _orientation(self):
        return self.__orientation

    @_orientation.setter
    def _orientation(self, value):
        self.__orientation = _Orientation(value)

    def _check_reverse(self, obj):
        if self._orientation != _Orientation(obj.Orientation):
            obj.reverse()
            self._orientation = _Orientation(obj.Orientation)
        return obj

    @staticmethod
    def _converter(func):
        """
        Function used in __getattr__ to modify the added functions.
        """
        return func

    def _check_boundary(self, objs):
        """
        Check if objects objs can be used as boundaries.
        """
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
        """
        The shape's boundary.
        """
        return self._boundary

    @boundary.setter
    def boundary(self, objs):
        self._boundary = self._check_boundary(objs)

    @property
    @abstractmethod
    def _shape(self):
        """
        The primitive shape of the object.
        """
        # Note: this is the "hidden" connection with primitive shapes
        pass

    @property
    def length(self):
        """
        The shape's length.
        """
        return cadapi.length(self._shape)

    @property
    def area(self):
        """
        The shape's area.
        """
        return cadapi.area(self._shape)

    @property
    def volume(self):
        """
        The shape's volume.
        """
        return cadapi.volume(self._shape)

    @property
    def center_of_mass(self):
        """
        The shape's center of mass.
        """
        return cadapi.center_of_mass(self._shape)

    @property
    def bounding_box(self):
        """
        The bounding box of the shape."""
        x_min, y_min, z_min, x_max, y_max, z_max = cadapi.bounding_box(self._shape)
        return BoundingBox(x_min, x_max, y_min, y_max, z_min, z_max)

    def is_null(self):
        """
        Check if the shape is null.
        """
        return cadapi.is_null(self._shape)

    def is_closed(self):
        """
        Check if the shape is closed.
        """
        return cadapi.is_closed(self._shape)

    def is_valid(self):
        """
        Check if the shape is valid.
        """
        return cadapi.is_valid(self._shape)

    def search(self, label: str):
        """
        Search for a shape with the specified label

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
        """
        Apply scaling with factor to this object. This function modifies the self
        object.
        """
        for o in self.boundary:
            o.scale(factor)

    def translate(self, vector) -> None:
        """
        Translate this shape with the vector. This function modifies the self
        object.
        """
        for o in self.boundary:
            o.translate(vector)

    def rotate(self, base, direction, degree) -> None:
        """
        Rotate this shape.

        Parameters
        ----------
        base: tuple (x,y,z)
            Origin location of the rotation
        direction: tuple (x,y,z)
            The direction vector
        degree: float
            rotation angle
        """
        for o in self.boundary:
            o.rotate(base, direction, degree)

    def change_placement(self, placement) -> None:
        """
        Change the placement of self
        """
        for o in self.boundary:
            o.change_placement(placement)

    def __repr__(self):  # noqa D105
        new = []
        new.append(f"([{type(self).__name__}] = Label: {self.label}")
        new.append(f" length: {self.length}")
        new.append(f" area: {self.area}")
        new.append(f" volume: {self.volume}")
        new.append(")")
        return ", ".join(new)

    def copy(self, label=None):
        """
        Make a copy of the BluemiraGeo.
        """
        geo_copy = copy.copy(self)
        if label is not None:
            geo_copy.label = label
        else:
            geo_copy.label = self.label
        return geo_copy

    def deepcopy(self, label=None):
        """
        Make a deepcopy of the BluemiraGeo.
        """
        geo_copy = copy.deepcopy(self)
        if label is not None:
            geo_copy.label = label
        else:
            geo_copy.label = self.label
        return geo_copy

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
