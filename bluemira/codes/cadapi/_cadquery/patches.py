# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Side-effect-only module that augments CadQuery shape classes with
FreeCAD-compatible attributes (``.Orientation`` property, ``.reverse()``
method, ``.Wires``/``.Faces``/etc. as both property and callable).

Imported once from ``_cadquery/__init__.py`` — must run after ``core``
because we reuse ``_cq_area_prop`` from there for the ``cq.Face.Area``
property.
"""

from __future__ import annotations

from collections import UserList

import cadquery as cq
from OCP.TopAbs import TopAbs_REVERSED

from bluemira.codes.cadapi._cadquery.core import _cq_area_prop


def _cq_orientation(self) -> str:
    """Return 'Forward' or 'Reversed' mirroring FreeCAD's Orientation property."""
    return "Reversed" if self.wrapped.Orientation() == TopAbs_REVERSED else "Forward"


def _cq_reverse(self) -> None:
    """Reverse the shape orientation in-place (mirrors FreeCAD's reverse() method)."""
    reversed_shape = self.wrapped.Reversed()
    # Cast back to the concrete OCCT type so type-specific OCC functions still work.
    self.wrapped = cq.Shape.cast(reversed_shape).wrapped


for _cls in (cq.Wire, cq.Face, cq.Edge, cq.Shell, cq.Solid, cq.Compound):
    if not hasattr(_cls, "Orientation") or not isinstance(
        _cls.__dict__.get("Orientation"), property
    ):
        _cls.Orientation = property(_cq_orientation)
    if not hasattr(_cls, "reverse"):
        _cls.reverse = _cq_reverse

# Area as a property on Face (FreeCAD: face.Area property; CadQuery: face.Area() method)
if not isinstance(cq.Face.__dict__.get("Area"), property):
    cq.Face.Area = property(_cq_area_prop)


class _CallableList(UserList):
    """A list that is also callable (returns itself).

    Lets ``obj.Wires`` and ``obj.Wires()`` both yield the same value, which the
    FreeCAD-flavoured callsites in bluemira rely on.
    """

    def __call__(self):
        return list(self)


def _make_shape_collection_prop(method_name: str, orig_method):
    """Return a property that wraps the original method in a _CallableList."""

    def _prop(self):
        return _CallableList(orig_method(self))

    _prop.__name__ = method_name
    return property(_prop)


# Patch collection accessors on Solid and Shell so that both `shape.Wires` and
# `shape.Wires()` work. Only patch classes that are returned by cadapi geometry
# functions (not Compound, which CadQuery uses internally with `.Wires()` calls).
for _cls in (cq.Solid, cq.Shell):
    for _name in ("Wires", "Faces", "Edges", "Shells", "Solids", "Vertices"):
        _orig = getattr(_cls, _name, None)
        if _orig is not None and callable(_orig) and not isinstance(_orig, property):
            setattr(_cls, _name, _make_shape_collection_prop(_name, _orig))


__all__: list[str] = []
