# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Imprinting solids together using OpenCASCADE's OCP bindings.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.progress import track

from bluemira.base.look_and_feel import bluemira_print
from bluemira.geometry.overlap_checking import find_approx_overlapping_pairs
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.tools import make_compound

if TYPE_CHECKING:
    from collections.abc import Collection

    from OCP.TopoDS import TopoDS_Face, TopoDS_Shape, TopoDS_Solid

    from bluemira.geometry.compound import BluemiraCompound


try:
    from OCP.BOPAlgo import BOPAlgo_MakeConnected
    from OCP.TopAbs import TopAbs_FACE, TopAbs_SOLID
    from OCP.TopExp import TopExp, TopExp_Explorer
    from OCP.TopTools import TopTools_IndexedMapOfShape
    from OCP.TopoDS import TopoDS
except ImportError:
    OCP_AVAILABLE = False
else:
    OCP_AVAILABLE = True


def _require_cadquery_backend() -> None:
    """
    Imprinting may only be used with the CadQuery/OCP backend.

    Raises
    ------
    ImportError
        If CadQuery or OCP is not found.
    NotImplementedError
        If FreeCAD is found.
    """
    from bluemira.codes import _geometryapi as cadapi  # noqa: PLC0415

    try:  # check if CadQuery is active
        import cadquery as cq  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "Solid imprinting requires CadQuery and its OCP bindings but was not found."
        ) from exc

    if cadapi.apiSolid is not cq.Solid:
        raise NotImplementedError(
            "Solid imprinting is only supported with the CadQuery backend."
        )

    if not OCP_AVAILABLE:
        raise ImportError(
            "Solid imprinting requires OCP, which is installed with CadQuery."
        )


def _map_lookup(
    mapping: list[tuple[Any, ImprintableSolid]], key: TopoDS_Shape
) -> ImprintableSolid:
    """
    Find the imprintable associated with an original OCP shape.

    Returns
    -------
    ImprintableSolid
        The matching solid if it is found.

    Raises
    ------
    RuntimeError
        If the solid is not found.
    """
    for original_shape, imp in mapping:
        if key.IsSame(original_shape):
            return imp

    raise RuntimeError("Could not identify the original imprintable.")


def _get_faces(shape: TopoDS_Shape) -> list[TopoDS_Face]:
    """
    Return all faces belonging to an OCP solid.

    Returns
    -------
    list[TopoDS_Face]
        De-duplicated faces belonging to the solid.
    """
    face_map = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape, TopAbs_FACE, face_map)
    return [TopoDS.Face_s(face_map.FindKey(i)) for i in range(1, face_map.Extent() + 1)]


def _get_solids(shape: TopoDS_Shape) -> list[TopoDS_Solid]:
    """
    Extract solids from an OCP shape.

    Returns
    -------
    list[TopoDS_Solid]
        Solids from a shape.
    """
    if shape.ShapeType() == TopAbs_SOLID:
        return [TopoDS.Solid_s(shape)]

    explorer = TopExp_Explorer(shape, TopAbs_SOLID)
    solids = []

    while explorer.More():
        solids.append(TopoDS.Solid_s(explorer.Current()))
        explorer.Next()

    return solids


def _bm_shape_to_ocp_solid(bm_shape) -> TopoDS_Solid:
    """
    Get the OCP solid wrapped by a CadQuery shape.

    Returns
    -------
    TopoDS_Solid
        Underlying OCP solid.

    Raises
    ------
    TypeError
        If the BluemiraShape is not CadQuery.
    RuntimeError
        If the underlying OCP cannot be extracted from the CadQuery shape.
    ValueError
        If multiple solids are retrieved.
    """
    import cadquery as cq  # noqa: PLC0415

    if not isinstance(bm_shape, cq.Shape):
        raise TypeError(
            f"Solid imprinting requires a CadQuery shape, but received {type(bm_shape)}."
        )

    solids = _get_solids(bm_shape.wrapped)
    if not solids:
        raise RuntimeError(
            "CadQuery shape contains no TopoDS_Solid "
            f"(ShapeType={bm_shape.wrapped.ShapeType()})."
        )
    if len(solids) > 1:
        raise ValueError(
            f"Expected a single solid for imprinting, but received {len(solids)}."
        )
    return solids[0]


def _ocp_solid_to_bm_solid_shape(ocp_solid: TopoDS_Solid):
    """
    Wrap an OCP solid in a CadQuery solid.

    Returns
    -------
    cq.Solid
        CadQuery solid.
    """
    import cadquery as cq  # noqa: PLC0415

    return cq.Solid(ocp_solid)


class ImprintableSolid:
    """Represents a solid that can be imprinted."""

    def __init__(self, label: str, bm_solid: BluemiraSolid, ocp_solid: TopoDS_Solid):
        self._label = label
        self._bm_solid = bm_solid
        self._imprinted_ocp_solid = ocp_solid
        self._has_imprinted = False
        self._imprinted_faces: set[TopoDS_Face] = set(_get_faces(ocp_solid))
        self._shadow_imprinted_faces: set[TopoDS_Face] = set()

    @classmethod
    def from_bluemira_solid(
        cls, label: str, bm_solid: BluemiraSolid
    ) -> ImprintableSolid:
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
        return cls(label, bm_solid, _bm_shape_to_ocp_solid(bm_solid.shape))

    @property
    def label(self) -> str:
        """Returns the label of the solid."""
        return self._label

    @property
    def ocp_solid(self) -> TopoDS_Solid:
        """Returns the TopoDS_Solid of the solid."""
        return self._imprinted_ocp_solid

    @property
    def imprinted_faces(self) -> set[TopoDS_Face]:
        """Returns the imprinted faces of the solid."""
        return self._imprinted_faces

    def bind_imprinted_face(self, face: TopoDS_Face) -> None:
        """
        Binds a face to the imprintable solid, adding it to the shadow set.
        The finalise_binding method must be called after binding all faces.
        """
        self._shadow_imprinted_faces.add(face)

    def finalise_binding(self) -> None:
        """
        Finalises the binding of the imprinted faces, moving them from the
        shadow set to the imprinted set.
        """
        self._imprinted_faces = self._shadow_imprinted_faces.copy()
        self._shadow_imprinted_faces.clear()

    def set_imprinted_solid(self, imprinted_ocp_solid: TopoDS_Solid) -> None:
        """
        Sets the imprinted solid of the imprintable solid.
        This is used to set the solid after imprinting.
        """
        self._imprinted_ocp_solid = imprinted_ocp_solid
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
                _ocp_solid_to_bm_solid_shape(self._imprinted_ocp_solid), self._label
            )
        return self._bm_solid


class _Imprinter:
    """Imprints solids together using the BOPAlgo_MakeConnected algorithm."""

    def __init__(self, *, run_parallel=False, parallel_mode=False, use_obb=False):
        self._imprint_builder = BOPAlgo_MakeConnected()
        self._imprint_builder.SetRunParallel(run_parallel)
        self._imprint_builder.SetParallelMode_s(parallel_mode)
        self._imprint_builder.SetUseOBB(use_obb)

    def __call__(self, imprintables: list[ImprintableSolid]) -> int:
        """
        Imprints the solids together, internally mutating the ImprintableSolid.

        Parameters
        ----------
        imprintables : list[ImprintableSolid]
            The imprintables to imprint together.

        Returns
        -------
        int
            The number of imprints performed.

        Raises
        ------
        ValueError
            If the imprintables are not valid.
        """
        bldr = self._imprint_builder
        bldr.Clear()

        org_solid_to_imp_map: list[tuple[Any, ImprintableSolid]] = []
        org_face_to_imp_map: list[tuple[Any, ImprintableSolid]] = []

        for imp in imprintables:
            ocp_sld = imp.ocp_solid
            bldr.AddArgument(ocp_sld)
            org_solid_to_imp_map.append((ocp_sld, imp))

            for resulting_face in _get_faces(ocp_sld):
                org_face_to_imp_map.append((resulting_face, imp))  # noqa: PERF401

        bldr.Perform()
        result = bldr.Shape()

        # Update each original wrapper with its resulting imprinted solid.
        for resulting_solid in _get_solids(result):
            solid_origin = bldr.GetOrigins(resulting_solid)

            for i, s_origin in enumerate(solid_origin, start=1):
                if i > 1:
                    raise ValueError(
                        "Imprinter does not support overlapping solids yet."
                    )
                imp_from_solid = _map_lookup(org_solid_to_imp_map, s_origin)
                imp_from_solid.set_imprinted_solid(resulting_solid)

        imprints_performed = 0

        for resulting_face in _get_faces(result):
            face_origin = bldr.GetOrigins(resulting_face)
            original_faces = []
            j = 0

            for j, f_origin in enumerate(face_origin, start=1):
                if j > 2:  # noqa: PLR2004
                    raise ValueError(
                        "Somehow there are more than 2 faces that imprinted this face."
                    )
                original_faces.append(f_origin)

            if j == 0:
                original_faces.append(resulting_face)
            if j == 2:  # noqa: PLR2004
                imprints_performed += 1

            for org_f in original_faces:
                imp_from_face = _map_lookup(org_face_to_imp_map, org_f)
                imp_from_face.bind_imprinted_face(resulting_face)

        for imp in imprintables:
            imp.finalise_binding()

        return imprints_performed


class ImprintResult:
    """Result of imprinting solids together."""

    def __init__(self, imprintables: list[ImprintableSolid], total_imprints: int):
        self._imprintables = imprintables
        self.total_imprints = total_imprints

    @property
    def imprintables(self) -> list[ImprintableSolid]:
        """Returns the imprintables."""
        return self._imprintables

    @property
    def labels(self) -> list[str]:
        """Returns the labels of the imprintables."""
        return [imp.label for imp in self._imprintables]

    @property
    def solids(self) -> list[BluemiraSolid]:
        """Returns the imprinted BluemiraSolids."""
        return [imp.to_bluemira_solid() for imp in self._imprintables]

    @property
    def as_compound(self) -> BluemiraCompound:
        """Returns the imprinted BluemiraCompound."""
        # will throw if only one solid
        return make_compound(self.solids)

    @property
    def ocp_solids(self) -> list[TopoDS_Solid]:
        """Returns the imprinted TopoDS_Solids."""
        return [imp.ocp_solid for imp in self._imprintables]

    @property
    def ocp_faces(self) -> list[TopoDS_Face]:
        """Returns the imprinted TopoDS_Face."""
        return [face for imp in self._imprintables for face in imp.imprinted_faces]


def imprint_solids(
    solids: Collection[BluemiraSolid],
    labels: Collection[str] | str,
    *,
    use_cgal: bool = True,
) -> ImprintResult:
    """
    Imprints solids together.

    Imprinting is only supported with the CadQuery geometry backend.

    Parameters
    ----------
    solids:
        The solids to imprint together.
    labels:
        The labels to use for the solids. If None, the labels will be
        taken from the solids. Must be the same length as solids.
    use_cgal:
        Whether to use CGAL for improved overlap checking speed and precision.
        If True and CGAL is not available, a numpy based approach will be used
        as a fallback.
        If False, the numpy based approach will be used regardless of CGAL
        availability.

    Returns
    -------
    ImprintResult
        The imprintable solids.

    Raises
    ------
    NotImplementedError
        If the active geometry backend is FreeCAD.
    ImportError
        If CadQuery or OCP is unavailable.
    ValueError
        If the labels are not the same length as the solids.
    TypeError
        If the solids are not of type BluemiraSolid.
    """
    _require_cadquery_backend()

    if isinstance(labels, str):
        labels = [labels] * len(solids)
    if len(labels) != len(solids):
        raise ValueError(
            "Labels must be the same length as the solids iterable: "
            f"{len(labels)} vs. {len(solids)}"
        )
    for sld, lbl in zip(solids, labels, strict=True):
        if not isinstance(sld, BluemiraSolid):
            raise TypeError(f"Solids must be BluemiraSolid only - {lbl} is {type(sld)}")

    bluemira_print(f"Imprinting {set(labels)}")

    pairs = find_approx_overlapping_pairs(solids, use_cgal=use_cgal)
    imprintables = [
        ImprintableSolid.from_bluemira_solid(lbl, sld)
        for sld, lbl in zip(solids, labels, strict=True)
    ]

    bluemira_print(f"{len(pairs)} potential pairs found.")
    total_imprints = 0

    if pairs:
        imprinter = _Imprinter(parallel_mode=True, run_parallel=True)
        for a, b in track(pairs):
            total_imprints += imprinter([imprintables[a], imprintables[b]])

        bluemira_print(f"Total imprints performed: {total_imprints}")

    return ImprintResult(imprintables, total_imprints)
