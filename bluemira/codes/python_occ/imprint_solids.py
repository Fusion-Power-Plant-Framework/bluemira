# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Imprinting solids together.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from rich.progress import track

from bluemira.base.look_and_feel import bluemira_print
from bluemira.codes.python_occ._guard import occ_guard
from bluemira.codes.python_occ.imprintable_solid import ImprintableSolid
from bluemira.geometry.overlap_checking import find_approx_overlapping_pairs
from bluemira.geometry.solid import BluemiraSolid

if TYPE_CHECKING:
    from collections.abc import Iterable


try:
    from OCC.Core.BOPAlgo import BOPAlgo_MakeConnected
    from OCC.Core.TopTools import TopTools_ListIteratorOfListOfShape
    from OCC.Core.TopoDS import TopoDS_Face, TopoDS_Solid  # noqa: TC002
    from OCC.Extend.TopologyUtils import TopologyExplorer
except ImportError:
    pass


class _Imprinter:
    """Imprints solids together using the BOPAlgo_MakeConnected algorithm."""

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
        """Imprints the solids together, internally mutating the ImprintableSolid.

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

        imprints_performed = 0

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

            # i = 1 means the face was changed, but it's not the imprinted face

            if i == 2:  # noqa: PLR2004
                imprints_performed += 1

            # bind faces
            for org_f in original_faces:
                imp_from_face = org_face_to_imp_map[org_f]
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
    def occ_solids(self) -> list[TopoDS_Solid]:
        """Returns the imprinted TopoDS_Solids."""
        return [imp.occ_solid for imp in self._imprintables]

    @property
    def occ_faces(self) -> list[TopoDS_Face]:
        """Returns the imprinted TopoDS_Face."""
        return [face for imp in self._imprintables for face in imp.imprinted_faces]


@occ_guard
def imprint_solids(
    solids: Iterable[BluemiraSolid],
    labels: Iterable[str] | None = None,
    *,
    use_cgal=True,
) -> ImprintResult:
    """Imprints solids together.

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
        The imprintable solids.

    Raises
    ------
    ValueError
        If the labels are not the same length as the solids.
    TypeError
        If the solids are not of type BluemiraSolid.
    """
    if labels is None or len(labels) == 0:
        labels = [sld.label for sld in solids]
    if len(labels) != len(solids):
        raise ValueError(
            "Labels must be the same length as the solids iterable: "
            f"{len(labels)} vs. {len(solids)}"
        )
    for sld, lbl in zip(solids, labels, strict=True):
        if not isinstance(sld, BluemiraSolid):
            raise TypeError(f"solids must be BluemiraSolid only - {lbl} is {type(sld)}")

    pairs = find_approx_overlapping_pairs(solids, use_cgal=use_cgal)
    imprintables = [
        ImprintableSolid.from_bluemira_solid(lbl, sld)
        for sld, lbl in zip(solids, labels, strict=True)
    ]

    # pairs and imprintables have the same ordering
    # ie indexes match

    bluemira_print(f"{len(pairs)} potential pairs found.")
    total_imprints = 0

    if len(pairs) > 0:
        imprinter = _Imprinter(parallel_mode=True, run_parallel=True)
        for a, b in track(pairs):
            total_imprints += imprinter([imprintables[a], imprintables[b]])

        bluemira_print(f"Total imprints performed: {total_imprints}")

    return ImprintResult(imprintables, total_imprints)
