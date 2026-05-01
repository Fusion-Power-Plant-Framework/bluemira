# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Working class for imprinting solids.
"""

from __future__ import annotations

from io import BytesIO

from bluemira.codes.python_occ._guard import occ_guard
from bluemira.geometry.solid import BluemiraSolid

try:
    from OCC.Core.BRepTools import breptools
    from OCC.Core.TopAbs import TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Solid, topods
    from OCC.Extend.TopologyUtils import TopologyExplorer
except ImportError:
    pass

try:
    import Part  # FreeCAD only
except ImportError:
    Part = None

try:
    import cadquery as _cq  # CadQuery only
except ImportError:
    _cq = None


def _bm_shape_to_occ_solid(bm_shape) -> TopoDS_Solid:
    """Convert a backend-native solid shape to an OCC.Core ``TopoDS_Solid``.

    Dispatches on the *active* geometry backend, not on mere import availability
    (both FreeCAD and CadQuery may be installed side-by-side).

    Returns
    -------
    TopoDS_Solid
        The pythonocc-core solid equivalent to *bm_shape*.

    Raises
    ------
    TypeError
        If *bm_shape* is neither a ``Part.Shape`` nor a ``cq.Shape``.
    RuntimeError
        If the CadQuery BRep round-trip yields a shape with no
        ``TopAbs_SOLID`` inside.
    """
    if Part is not None and isinstance(bm_shape, Part.Shape):
        return Part.__toPythonOCC__(bm_shape)
    if _cq is not None and isinstance(bm_shape, _cq.Shape):
        buf = BytesIO()
        bm_shape.exportBrep(buf)
        shape = breptools.ReadFromString(buf.getvalue().decode("ascii"))
        # ReadFromString may return a TopoDS_Compound wrapping the solid even
        # when the input was a cq.Solid; unchecked topods.Solid(...) then yields
        # an invalid pointer and segfaults in downstream OCC ops (e.g.
        # BOPAlgo_MakeConnected). Extract the first solid explicitly.
        if shape.ShapeType() == TopAbs_SOLID:
            return topods.Solid(shape)
        explorer = TopExp_Explorer(shape, TopAbs_SOLID)
        if explorer.More():
            return topods.Solid(explorer.Current())
        raise RuntimeError(
            f"BRep round-trip yielded no TopoDS_Solid (ShapeType={shape.ShapeType()})"
        )
    raise TypeError(f"Cannot convert {type(bm_shape)!r} to OCC.Core TopoDS_Solid")


def _occ_solid_to_bm_solid_shape(occ_solid: TopoDS_Solid):
    """Convert an OCC.Core ``TopoDS_Solid`` back to a backend-native solid shape.

    Dispatches on the active backend via ``cadapi.apiSolid`` so that a side-by-side
    FreeCAD install doesn't get preferred when CadQuery is selected.

    Returns
    -------
    apiSolid
        A ``cq.Solid`` under the CadQuery backend, or a ``Part.Solid`` under FreeCAD.

    Raises
    ------
    RuntimeError
        If no supported CAD backend is available to rehydrate *occ_solid*.
    """
    from bluemira.codes import _geometryapi as cadapi  # noqa: PLC0415

    if _cq is not None and cadapi.apiSolid is _cq.Solid:
        data = breptools.WriteToString(occ_solid)
        buf = BytesIO(data.encode("ascii"))
        cq_shape = _cq.Shape.importBrep(buf)
        return _cq.Solid(cq_shape.wrapped)
    if Part is not None and cadapi.apiSolid is Part.Solid:
        api_solid = Part.__fromPythonOCC__(occ_solid)
        return Part.Solid(api_solid)
    raise RuntimeError("No CAD backend available to rehydrate OCC.Core shape.")


class ImprintableSolid:
    """Represents a solid that can be imprinted."""

    @occ_guard
    def __init__(self, label: str, bm_solid: BluemiraSolid, occ_solid: TopoDS_Solid):
        self._label = label
        self._bm_solid = bm_solid
        self._imprinted_occ_solid = occ_solid
        self._has_imprinted = False
        self._imprinted_faces: set[TopoDS_Face] = set(
            TopologyExplorer(occ_solid).faces()
        )
        self._shadow_imprinted_faces: set[TopoDS_Face] = set()

    @classmethod
    def from_bluemira_solid(cls, label: str, bm_solid: BluemiraSolid):
        """
        Creates an ImprintableSolid from a BluemiraSolid.

        Parameters
        ----------
        label : str
            The label of the solid.
        bm_solid : BluemiraSolid
            The BluemiraSolid to imprint.

        Returns
        -------
        ImprintableSolid
            The ImprintableSolid.

        Raises
        ------
        TypeError
            If bm_solid is not a BluemiraSolid.
        """
        if not isinstance(bm_solid, BluemiraSolid):
            raise TypeError(f"bm_solid must be a BluemiraSolid, got: {type(bm_solid)}")
        return cls(label, bm_solid, _bm_shape_to_occ_solid(bm_solid.shape))

    @property
    def label(self) -> str:
        """Returns the label of the solid."""
        return self._label

    @property
    def occ_solid(self) -> TopoDS_Solid:
        """Returns the TopoDS_Solid of the solid."""
        return self._imprinted_occ_solid

    @property
    def imprinted_faces(self) -> set[TopoDS_Face]:
        """Returns the imprinted faces of the solid."""
        return self._imprinted_faces

    def bind_imprinted_face(self, face: TopoDS_Face):
        """
        Binds a face to the imprintable solid, adding it to the shadow set.
        The finalise_binding method must be called after binding all faces.
        """
        self._shadow_imprinted_faces.add(face)

    def finalise_binding(self):
        """
        Finalises the binding of the imprinted faces, moving them from the
        shadow set to the imprinted set.
        """
        self._imprinted_faces = self._shadow_imprinted_faces.copy()
        self._shadow_imprinted_faces.clear()

    def set_imprinted_solid(self, imprinted_occ_solid):
        """
        Sets the imprinted solid of the imprintable solid.
        This is used to set the solid after imprinting.
        """
        self._imprinted_occ_solid = imprinted_occ_solid
        self._has_imprinted = True

    def to_bluemira_solid(self) -> BluemiraSolid:
        """
        Returns the imprinted BluemiraSolid.
        If the solid has not been imprinted, it returns the original solid.

        Returns
        -------
        BluemiraSolid
            The imprinted BluemiraSolid.
        """
        if self._has_imprinted:
            return BluemiraSolid._create(
                _occ_solid_to_bm_solid_shape(self._imprinted_occ_solid), self._label
            )
        return self._bm_solid
