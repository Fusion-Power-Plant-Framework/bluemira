# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path

import pytest

from bluemira.geometry import tools
from bluemira.geometry.face import BluemiraFace
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf


class TestMeshing:
    @pytest.mark.parametrize(("lcar", "nodes_num"), [(0.1, 40), (0.25, 16), (0.5, 8)])
    def test_mesh_poly(self, lcar, nodes_num, tmp_path):
        poly = tools.make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], closed=True, label="poly"
        )

        poly.mesh_options = {"lcar": lcar, "physical_group": "poly"}

        surf = BluemiraFace(poly, label="surf")
        surf.mesh_options = {"physical_group": "coil"}

        meshfiles = [
            Path(tmp_path, p).as_posix() for p in ["Mesh.geo_unrolled", "Mesh.msh"]
        ]
        m = meshing.Mesh(meshfile=meshfiles)
        m(surf)

        msh_to_xdmf("Mesh.msh", dimensions=(0, 1), directory=tmp_path)

        _, boundaries, _, labels = import_mesh(
            "Mesh",
            directory=tmp_path,
            subdomains=True,
        )

        arr = boundaries.array()
        assert (arr == labels["poly"]).sum() == nodes_num

    @pytest.mark.parametrize(("lcar", "nodes_num"), [(0.1, 40), (0.25, 16), (0.5, 8)])
    def test_override_lcar_surf(self, lcar, nodes_num, tmp_path):
        poly = tools.make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], closed=True, label="poly"
        )

        poly.mesh_options = {"lcar": lcar, "physical_group": "poly"}

        surf = BluemiraFace(poly, label="surf")
        surf.mesh_options = {"lcar": lcar / 2, "physical_group": "coil"}

        meshfiles = [
            Path(tmp_path, p).as_posix() for p in ["Mesh.geo_unrolled", "Mesh.msh"]
        ]
        m = meshing.Mesh(meshfile=meshfiles)
        m(surf)

        msh_to_xdmf("Mesh.msh", dimensions=(0, 1), directory=tmp_path)

        _, boundaries, _, labels = import_mesh(
            "Mesh",
            directory=tmp_path,
            subdomains=True,
        )

        arr = boundaries.array()
        assert (arr == labels["poly"]).sum() == nodes_num * 2
