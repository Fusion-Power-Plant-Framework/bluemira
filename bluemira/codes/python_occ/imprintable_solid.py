# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Working class for imprinting solids.
"""

from bluemira.codes.python_occ._guard import occ_guard
from bluemira.geometry.solid import BluemiraSolid

try:
    from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Solid
    from OCC.Extend.TopologyUtils import TopologyExplorer

    import Part  # isort: skip
except ImportError:
    pass


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
        ImportError
            If OCC is not available.
        """
        return cls(label, bm_solid, Part.__toPythonOCC__(bm_solid.shape))

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
            api_solid = Part.__fromPythonOCC__(self._imprinted_occ_solid)
            return BluemiraSolid._create(Part.Solid(api_solid), self._label)
        return self._bm_solid
