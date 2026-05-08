# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from __future__ import annotations

import pytest

from bluemira.geometry.error import DisjointedSolidError, GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.solid import BluemiraSolid
from bluemira.geometry.tools import extrude_shape, make_polygon


@pytest.fixture
def unit_box():
    face = BluemiraFace(
        make_polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], closed=True)
    )
    return extrude_shape(face, [0, 0, 1])


class TestBluemiraSolidCreate:
    def test_rejects_non_solid_input(self):
        with pytest.raises(TypeError, match=r"Only Part\.Solid objects"):
            BluemiraSolid._create("not a solid")

    def test_rejects_invalid_solid(self, unit_box, monkeypatch):
        monkeypatch.setattr(
            "bluemira.geometry.solid.cadapi.is_valid", lambda _obj: False
        )
        with pytest.raises(GeometryError, match=r"not\svalid"):
            BluemiraSolid._create(unit_box.shape)

    def test_rejects_multi_solid_input(self, unit_box, monkeypatch):
        """``_create`` rejects an apiSolid that contains >1 sub-solid."""
        monkeypatch.setattr(
            "bluemira.geometry.solid.cadapi.solids",
            lambda _obj: [unit_box.shape, unit_box.shape],
        )
        with pytest.raises(DisjointedSolidError):
            BluemiraSolid._create(unit_box.shape)

    def test_create_solid_disjointed_raises(self, unit_box, monkeypatch):
        """``_create_solid`` raises when ``boolean_cut`` yields >1 piece."""
        # Force the >1-boundary branch by giving the solid two boundary shells.
        shell0 = unit_box.boundary[0]
        unit_box._boundary = [shell0, shell0]
        monkeypatch.setattr(
            "bluemira.geometry.solid.cadapi.boolean_cut",
            lambda solid, _holes: [solid, solid],
        )
        with pytest.raises(DisjointedSolidError):
            unit_box._create_solid(check_reverse=False)
