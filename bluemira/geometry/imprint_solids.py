"""
Imprinting solids together.
"""

from collections.abc import Iterable
from typing import Any

from rich.progress import track

from bluemira.base.look_and_feel import bluemira_print
from bluemira.geometry.overlap_checking import find_approx_overlapping_pairs
from bluemira.geometry.solid import BluemiraSolid

try:
    from OCC.Core.BOPAlgo import BOPAlgo_MakeConnected
    from OCC.Core.TopTools import TopTools_ListIteratorOfListOfShape
    from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Solid
    from OCC.Extend.TopologyUtils import TopologyExplorer

    import Part  # isort: skip

    occ_available = True
except ImportError:
    occ_available = False


class ImprintableSolid:
    """Represents a solid that can be imprinted."""

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
        if not occ_available:
            raise ImportError("OCC is not available")

        return cls(label, bm_solid, Part.__toPythonOCC__(bm_solid.shape))

    @property
    def label(self) -> str:
        return self._label

    @property
    def occ_solid(self) -> TopoDS_Solid:
        return self._imprinted_occ_solid

    @property
    def imprinted_faces(self) -> set[TopoDS_Face]:
        return self._imprinted_faces

    def bind_imprinted_face(self, face: TopoDS_Face):
        """
        Binds a face to the imprintable solid, adding it to the shadow set.
        The finalise_binding method must be called after binding all faces.
        """
        self._shadow_imprinted_faces.add(face)

    def finalise_binding(self):
        self._imprinted_faces = self._shadow_imprinted_faces.copy()
        self._shadow_imprinted_faces.clear()

    def set_imprinted_solid(self, imprinted_occ_solid):
        self._imprinted_occ_solid = imprinted_occ_solid
        self._has_imprinted = True

    def to_bluemira_solid(self) -> BluemiraSolid:
        if self._has_imprinted:
            api_solid = Part.__fromPythonOCC__(self._imprinted_occ_solid)
            return BluemiraSolid._create(Part.Solid(api_solid), self._label)
        return self._bm_solid


class Imprinter:
    def __init__(
        self,
        *,
        run_parallel=False,
        parallel_mode=False,
        use_obb=False,
    ):
        self._imprint_builder = BOPAlgo_MakeConnected()
        self._imprint_builder.SetRunParallel(run_parallel)
        self._imprint_builder.SetParallelMode(parallel_mode)
        self._imprint_builder.SetUseOBB(use_obb)

    def __call__(self, imprintables: list[ImprintableSolid]) -> int:
        """Imprints the solids together, internally mutating the ImprintableSolid."""
        imprints_performed = 0
        bldr = self._imprint_builder
        bldr.Clear()

        org_solid_to_imp_map: dict[Any, ImprintableSolid] = {}
        org_face_to_imp_map: dict[Any, ImprintableSolid] = {}

        for imp in imprintables:
            occ_sld = imp.occ_solid
            bldr.AddArgument(occ_sld)
            org_solid_to_imp_map[occ_sld] = imp
            for resulting_face in TopologyExplorer(occ_sld).faces():
                org_face_to_imp_map[resulting_face] = imp

        bldr.Perform()
        res = bldr.Shape()
        ex = TopologyExplorer(res)

        # you have to update the occ_solid of the imprintable
        # with the new imprinted solid. Imagen if there are
        # two or more imprints on the same face, the final
        # shape needs to account for each.
        for resulting_solid in ex.solids():
            solid_origin = bldr.GetOrigins(resulting_solid)
            solid_origin_iter = TopTools_ListIteratorOfListOfShape(solid_origin)
            i = 0
            while solid_origin_iter.More():
                i += 1
                if i > 1:
                    raise ValueError(
                        "Imprinter does not support overlapping solids yet."
                    )
                # get the original solid
                org_solid = solid_origin_iter.Value()
                # use the map to set the new imprinted solid on
                # on the corresponding imprintable
                imp_from_solid = org_solid_to_imp_map[org_solid]
                imp_from_solid.set_imprinted_solid(resulting_solid)
                solid_origin_iter.Next()

        for resulting_face in ex.faces():
            face_origin = bldr.GetOrigins(resulting_face)
            face_origin_iter = TopTools_ListIteratorOfListOfShape(face_origin)
            original_faces = []
            i = 0
            while face_origin_iter.More():
                i += 1
                if i > 2:  # noqa: PLR2004
                    raise ValueError(
                        "Somehow there are more than 2 faces that imprinted this face."
                    )
                # get the original face
                original_faces.append(face_origin_iter.Value())
                face_origin_iter.Next()
            if i == 0:
                original_faces.append(resulting_face)

            if i == 2:
                imprints_performed += 1

            # bind faces
            for org_f in original_faces:
                imp_from_face = org_face_to_imp_map[org_f]
                imp_from_face.bind_imprinted_face(resulting_face)

        for imp in imprintables:
            imp.finalise_binding()

        return imprints_performed


def imprint_solids(solids: Iterable[BluemiraSolid]):
    """Imprints solids together."""
    imprintables = [
        ImprintableSolid.from_bluemira_solid(sld.label, sld) for sld in solids
    ]
    pairs = find_approx_overlapping_pairs(solids)

    imprinter = Imprinter(parallel_mode=True, run_parallel=True)

    total_imprints = 0
    bluemira_print(f"Imprinting solids together: {len(pairs)} potential pairs found.")
    for a, b in track(pairs):
        total_imprints += imprinter([imprintables[a], imprintables[b]])

    return imprintables
