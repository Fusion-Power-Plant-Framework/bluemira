# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
CAD file I/O and placement application for the CadQuery backend.

Covers ``CADFileType``, ``save_cad`` / ``save_as_STP`` / ``import_cad``
(STEP read+write, optional XCAF labelling), the metre-vs-mm unit handling
that bluemira-via-OCCT requires, and ``change_placement`` (apply a
``_CQPlacement`` to a shape's location).
"""

from __future__ import annotations

import contextlib
import enum
import math
from pathlib import Path
from typing import TYPE_CHECKING

import cadquery as cq
from OCP.BRep import BRep_Builder
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.IFSelect import IFSelect_RetDone
from OCP.Interface import Interface_Static
from OCP.Message import Message_ProgressRange
from OCP.RWGltf import RWGltf_CafWriter
from OCP.STEPCAFControl import STEPCAFControl_Writer
from OCP.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
from OCP.ShapeAnalysis import ShapeAnalysis_FreeBounds
from OCP.StlAPI import StlAPI_Writer
from OCP.TColStd import TColStd_IndexedDataMapOfStringString
from OCP.TCollection import TCollection_AsciiString, TCollection_ExtendedString
from OCP.TDataStd import TDataStd_Name
from OCP.TDocStd import TDocStd_Document
from OCP.TopLoc import TopLoc_Location
from OCP.TopTools import TopTools_HSequenceOfShape
from OCP.TopoDS import TopoDS_Compound
from OCP.XCAFApp import XCAFApp_Application
from OCP.XCAFDoc import XCAFDoc_DocumentTool
from OCP.gp import gp_Ax1, gp_Dir, gp_Pnt, gp_Trsf, gp_Vec

from bluemira.codes.cadapi._cadquery.aliases import (
    _ANGLE_PARALLEL_TOL,
    apiCompound,
    apiShape,
)
from bluemira.codes.error import FreeCADError

if TYPE_CHECKING:
    from collections.abc import Iterable

    from bluemira.codes.cadapi._cadquery.placement import _CQPlacement


class CADFileType(enum.Enum):
    """Minimal CAD file type enum (mirrors _freecad.api.CADFileType)."""

    STEP = "stp"
    STEP_ZIP = "stpz"
    IGES = "iges"
    BREP = "brep"
    FREECAD = "FCStd"
    STL = "stl"
    GLTRANSMISSION = "gltf"  # also handles .glb (binary variant)

    @property
    def ext(self) -> str:
        """File extension (without leading dot)."""
        return self.value

    @classmethod
    def _missing_(cls, value: str) -> CADFileType:
        # Allow "step" → STEP, "stp" → STEP, etc.
        _aliases = {
            "step": cls.STEP,
            "stp": cls.STEP,
            "iges": cls.IGES,
            "igs": cls.IGES,
            "brep": cls.BREP,
            "stl": cls.STL,
            "gltf": cls.GLTRANSMISSION,
            "glb": cls.GLTRANSMISSION,
        }
        return _aliases.get(str(value).lower())

    @classmethod
    def unitless_formats(cls) -> tuple[CADFileType, ...]:
        return (cls.BREP, cls.FREECAD, cls.STL, cls.GLTRANSMISSION)

    @classmethod
    def mesh_import_formats(cls) -> tuple[CADFileType, ...]:
        return ()

    @classmethod
    def not_importable_formats(cls) -> tuple[CADFileType, ...]:
        return (cls.STEP_ZIP, cls.FREECAD, cls.STL, cls.GLTRANSMISSION)

    @classmethod
    def manual_mesh_formats(cls) -> tuple[CADFileType, ...]:
        return ()


def make_compound(shapes: list[apiShape]) -> apiCompound:
    """Make a compound of multiple shapes."""
    comp = TopoDS_Compound()
    b = BRep_Builder()
    b.MakeCompound(comp)
    for s in shapes:
        b.Add(comp, s.wrapped)
    return cq.Shape.cast(comp)


@contextlib.contextmanager
def _step_write_settings():
    """Force the OCCT STEP writer into the same schema + unit as FreeCAD.

    FreeCAD writes STEP with ``write.step.unit = 'M'`` (metres) and
    schema ``AP242DIS`` (AP242 managed model-based 3D engineering). OCP
    defaults to ``AP214IS`` (``AUTOMOTIVE_DESIGN``) with the same metre
    unit, so only the schema needs overriding for byte-compatible output.

    Critical for fast_ctd: bluemira's native length unit is metres, so the
    STEP declaration must say ``SI_UNIT($,.METRE.)``. A mismatched
    ``.MILLI.`` prefix makes downstream consumers (fast_ctd's
    ``step_to_brep``) interpret a 1 m cube as a 1 mm cube, so its
    ``minimum_volume=1.0`` mm³ default silently filters every solid out
    of the BRep and ``merge_brep_geometries`` then dies with
    "no vertices in source shape".

    Both settings are writer-scoped globals that are only registered once a
    ``STEPControl_Writer`` has been instantiated (OCCT lazy-inits the
    parameter table), so the override must wrap the entire writer creation
    + transfer + write sequence. We instantiate a throw-away writer up front
    to force param registration before reading the originals.
    """
    STEPControl_Writer()
    keys = ("write.step.unit", "write.step.schema")
    targets = {"write.step.unit": "M", "write.step.schema": "AP242DIS"}
    originals = {k: Interface_Static.CVal_s(k) for k in keys}
    for k, v in targets.items():
        Interface_Static.SetCVal_s(k, v)
    try:
        yield
    finally:
        for k, v in originals.items():
            Interface_Static.SetCVal_s(k, v)


def save_as_STP(shapes: list[apiShape], filename: str = "test", **kwargs):
    """Save shapes as a STEP file (legacy single-file method)."""
    if not filename.lower().endswith((".stp", ".step")):
        filename += ".stp"

    if not isinstance(shapes, list):
        shapes = [shapes]

    with _step_write_settings():
        writer = STEPControl_Writer()
        for s in shapes:
            writer.Transfer(s.wrapped, STEPControl_AsIs)
        status = writer.Write(filename)

    if status != IFSelect_RetDone:
        raise FreeCADError(f"Failed to write STEP file: {filename}")


def _write_labeled_step(shapes, labels, filename):
    """Write a STEP file as an XCAF assembly with one named PRODUCT per shape.

    Mirrors FreeCAD's ``save_cad(..., labels=...)`` behaviour: downstream
    tools (``fast_ctd``, DAGMC converters) look up solids by name, so each
    input shape must appear as a distinct named entity in the STEP file.
    """
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    for s, name in zip(shapes, labels, strict=True):
        lbl = shape_tool.AddShape(s.wrapped, False)
        TDataStd_Name.Set_s(lbl, TCollection_ExtendedString(str(name)))
    writer = STEPCAFControl_Writer()
    writer.Transfer(doc, STEPControl_AsIs)
    return writer.Write(str(filename))


# Mesh deflection defaults — bluemira is metre-native, so 1 mm linear / ~28°
# angular gives smooth curves on reactor-scale geometry without exploding the
# triangle count. Tunable via _write_stl / _write_gltf kwargs.
_DEFAULT_LIN_DEFLECTION = 1e-3
_DEFAULT_ANG_DEFLECTION = 0.5


def _ensure_meshed(
    shape: apiShape,
    lin_deflection: float = _DEFAULT_LIN_DEFLECTION,
    ang_deflection: float = _DEFAULT_ANG_DEFLECTION,
) -> None:
    """Run BRepMesh_IncrementalMesh in place (idempotent if already meshed)."""
    BRepMesh_IncrementalMesh(shape.wrapped, lin_deflection, False, ang_deflection, True)


def _write_stl(
    shapes: list[apiShape],
    filename: str,
    *,
    binary: bool = True,
    lin_deflection: float = _DEFAULT_LIN_DEFLECTION,
    ang_deflection: float = _DEFAULT_ANG_DEFLECTION,
) -> None:
    """Write shapes as a single STL file. Labels are not preserved.

    Defaults to binary STL (smaller and faster for downstream consumers).
    """
    target = shapes[0] if len(shapes) == 1 else make_compound(shapes)
    _ensure_meshed(target, lin_deflection, ang_deflection)
    writer = StlAPI_Writer()
    writer.ASCIIMode = not binary
    if not writer.Write(target.wrapped, str(filename)):
        raise FreeCADError(f"Failed to write STL file: {filename}")


def _write_gltf(
    shapes: list[apiShape],
    labels: list[str] | None,
    filename: str,
    *,
    is_binary: bool,
    lin_deflection: float = _DEFAULT_LIN_DEFLECTION,
    ang_deflection: float = _DEFAULT_ANG_DEFLECTION,
) -> None:
    """Write shapes as a glTF (text) or GLB (binary) file via XCAF.

    Triangulation must be precomputed on each shape before ``Perform`` —
    ``RWGltf_CafWriter`` does not auto-tessellate.
    """
    app = XCAFApp_Application.GetApplication_s()
    doc = TDocStd_Document(TCollection_ExtendedString("MDTV-XCAF"))
    app.NewDocument(TCollection_ExtendedString("MDTV-XCAF"), doc)
    shape_tool = XCAFDoc_DocumentTool.ShapeTool_s(doc.Main())
    used_labels = (
        labels if labels is not None else [f"shape_{i}" for i in range(len(shapes))]
    )
    for s, name in zip(shapes, used_labels, strict=True):
        _ensure_meshed(s, lin_deflection, ang_deflection)
        lbl = shape_tool.AddShape(s.wrapped, False)
        TDataStd_Name.Set_s(lbl, TCollection_ExtendedString(str(name)))

    writer = RWGltf_CafWriter(TCollection_AsciiString(str(filename)), is_binary)
    file_info = TColStd_IndexedDataMapOfStringString()
    progress = Message_ProgressRange()
    if not writer.Perform(doc, file_info, progress):
        raise FreeCADError(f"Failed to write glTF file: {filename}")


def save_cad(
    shapes: Iterable[apiShape],
    filename: str,
    cad_format: str | CADFileType = "stp",
    labels: Iterable[str] | None = None,
    **kwargs,
):
    """Save CAD shapes to a file."""
    if not isinstance(shapes, list):
        shapes = list(shapes)
    labels_list = list(labels) if labels is not None else None

    # Capture the user-requested extension before resolving to the enum, so
    # GLB-vs-GLTF (both → CADFileType.GLTRANSMISSION) survives the round-trip.
    requested_ext = cad_format.lower() if isinstance(cad_format, str) else cad_format.ext
    cad_format = (
        CADFileType(cad_format)
        if not isinstance(cad_format, CADFileType)
        else cad_format
    )
    p = Path(filename)
    current_ext = p.suffix.lower().lstrip(".")
    valid_exts = {cad_format.ext.lower()}
    if cad_format == CADFileType.STEP:
        valid_exts.add("step")
    if cad_format == CADFileType.GLTRANSMISSION:
        valid_exts.update({"gltf", "glb"})
    if current_ext not in valid_exts:
        # Prefer the user-requested extension if it is one of the valid ones
        # (covers `cad_format="glb"` → append ".glb", not the enum default ".gltf").
        append_ext = requested_ext if requested_ext in valid_exts else cad_format.ext
        filename = str(p) + f".{append_ext}"

    if cad_format == CADFileType.STEP:
        with _step_write_settings():
            if labels_list:
                status = _write_labeled_step(shapes, labels_list, filename)
            else:
                writer = STEPControl_Writer()
                for s in shapes:
                    writer.Transfer(s.wrapped, STEPControl_AsIs)
                status = writer.Write(str(filename))
        if status != IFSelect_RetDone:
            raise FreeCADError(f"Failed to write STEP file: {filename}")
    elif cad_format == CADFileType.STL:
        _write_stl(shapes, filename)
    elif cad_format == CADFileType.GLTRANSMISSION:
        is_binary = str(filename).lower().endswith(".glb")
        _write_gltf(shapes, labels_list, filename, is_binary=is_binary)
    else:
        raise FreeCADError(f"CAD format not supported by CadQuery backend: {cad_format}")


_IMPORT_UNIT_SCALE_TO_METRES = {"m": 1.0, "mm": 1e-3, "cm": 1e-2, "km": 1e3}


def _scale_shape(shape: apiShape, factor: float) -> apiShape:
    trsf = gp_Trsf()
    trsf.SetScale(gp_Pnt(0.0, 0.0, 0.0), factor)
    moved = BRepBuilderAPI_Transform(shape.wrapped, trsf, True).Shape()
    return cq.Shape.cast(moved)


def import_cad(
    file,
    filetype=None,
    unit_scale: str = "m",
    **kwargs,
) -> list[tuple[apiShape, str]]:
    """Import CAD from file. Returns list of (shape, label) tuples."""
    file = Path(file)
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(file))
    if status != IFSelect_RetDone:
        raise FreeCADError(f"Failed to read STEP file: {file}")

    reader.TransferRoots()
    shape = reader.OneShape()
    result_shape = cq.Shape.cast(shape)

    # OCCT's STEP reader always normalises geometry to its internal unit
    # (millimetres). ``xstep.cascade.unit`` ostensibly overrides this, but
    # the setting latches on first-WorkSession creation in the process and
    # silently ignores subsequent ``Interface_Static.SetCVal_s`` calls, so
    # we cannot reliably swap the reader's target unit at runtime. We
    # therefore read with the default (mm) and scale explicitly here to
    # the caller's ``unit_scale`` target (default "m").
    _INTERNAL_METRES = 1e-3
    target_factor = _IMPORT_UNIT_SCALE_TO_METRES.get(unit_scale.lower(), 1.0)
    scale = _INTERNAL_METRES / target_factor
    if not math.isclose(scale, 1.0, rel_tol=1e-12):
        result_shape = _scale_shape(result_shape, scale)

    # STEP reader often returns a Compound of raw edges — try to upgrade to wires.
    if isinstance(result_shape, cq.Compound):
        edges = result_shape.Edges()
        wires = result_shape.Wires()
        shells = result_shape.Shells()
        solids = result_shape.Solids()
        if edges and not wires and not shells and not solids:
            try:
                edge_seq = TopTools_HSequenceOfShape()
                for e in edges:
                    edge_seq.Append(e.wrapped)
                result_wires = TopTools_HSequenceOfShape()
                ShapeAnalysis_FreeBounds.ConnectEdgesToWires_s(
                    edge_seq, 1e-6, False, result_wires
                )
                assembled = [
                    cq.Shape.cast(result_wires.Value(i))
                    for i in range(1, result_wires.Size() + 1)
                ]
                if len(assembled) == 1:
                    result_shape = assembled[0]
                elif assembled:
                    comp = TopoDS_Compound()
                    b = BRep_Builder()
                    b.MakeCompound(comp)
                    for w in assembled:
                        b.Add(comp, w.wrapped)
                    result_shape = cq.Shape.cast(comp)
            except Exception:  # noqa: BLE001, S110
                pass  # fall through with the original compound

    # CadQuery/OCC uses raw values without mm/m conversion — no scaling needed here.
    # (FreeCAD backend needs scaling because FreeCAD works in mm internally.)
    return [(result_shape, file.stem)]


def _placement_to_trsf(placement: _CQPlacement) -> gp_Trsf:
    """Build a gp_Trsf (rotation + translation) from a _CQPlacement."""
    trsf = gp_Trsf()
    axis = placement.Rotation.Axis
    angle = placement.Rotation.Angle
    if abs(angle) > _ANGLE_PARALLEL_TOL:
        ax1 = gp_Ax1(gp_Pnt(0.0, 0.0, 0.0), gp_Dir(axis.x, axis.y, axis.z))
        trsf.SetRotation(ax1, angle)
    base = placement.Base
    trsf.SetTranslationPart(gp_Vec(base.x, base.y, base.z))
    return trsf


def change_placement(geo: apiShape, placement: _CQPlacement) -> None:
    """Compose *placement* onto *geo*'s current location in place.

    FreeCAD's homonym does a somewhat idiosyncratic composition on
    ``geo.Placement``; here we instead apply the placement's rigid transform as
    a relative location update on the underlying ``TopoDS_Shape`` — the natural
    OCCT composition ``new = current * placement``. This matches the semantic
    intent ("move this shape by that placement") used by every caller we've
    seen, without trying to reproduce the FreeCAD base-vs-rotation asymmetry.
    """
    trsf = _placement_to_trsf(placement)
    geo.wrapped.Move(TopLoc_Location(trsf))


__all__ = [
    "CADFileType",
    "change_placement",
    "import_cad",
    "make_compound",
    "save_as_STP",
    "save_cad",
]
