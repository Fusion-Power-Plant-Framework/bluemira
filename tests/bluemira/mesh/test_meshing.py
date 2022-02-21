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

import os

import pytest

import bluemira.geometry.tools as tools
from bluemira.base.file import get_bluemira_root
from bluemira.geometry.face import BluemiraFace
from bluemira.mesh import meshing
from bluemira.utilities.tools import bluemira_warn

HAS_MSH2XDMF = False
try:
    from bluemira.utilities.tools import get_module

    msh2xdmf = get_module(
        os.path.join(get_bluemira_root(), "..", "msh2xdmf", "msh2xdmf.py")
    )

    HAS_MSH2XDMF = True
except ImportError as err:
    bluemira_warn(f"Unable to import msh2xdmf, dolfin examples will not run: {err}")


class TestMeshing:
    @pytest.mark.parametrize("lcar, nodes_num", ((0.1, 40), (0.25, 16), (0.5, 8)))
    def test_mesh_poly(self, lcar, nodes_num):
        poly = tools.make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], closed=True, label="poly"
        )

        poly.mesh_options = {"lcar": lcar, "physical_group": "poly"}

        surf = BluemiraFace(poly, label="surf")
        surf.mesh_options = {"physical_group": "coil"}

        m = meshing.Mesh()
        buffer = m(surf)

        msh2xdmf.msh2xdmf("Mesh.msh", dim=2, directory=".")

        mesh, boundaries, subdomains, labels = msh2xdmf.import_mesh(
            prefix="Mesh",
            dim=2,
            directory=".",
            subdomains=True,
        )

        arr = boundaries.array()
        assert (arr == labels["poly"]).sum() == nodes_num

    @pytest.mark.parametrize("lcar, nodes_num", ((0.1, 40), (0.25, 16), (0.5, 8)))
    def test_override_lcar_surf(self, lcar, nodes_num):
        poly = tools.make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], closed=True, label="poly"
        )

        poly.mesh_options = {"lcar": lcar, "physical_group": "poly"}

        surf = BluemiraFace(poly, label="surf")
        surf.mesh_options = {"lcar": lcar / 2, "physical_group": "coil"}

        m = meshing.Mesh()
        buffer = m(surf)

        msh2xdmf.msh2xdmf("Mesh.msh", dim=2, directory=".")

        mesh, boundaries, subdomains, labels = msh2xdmf.import_mesh(
            prefix="Mesh",
            dim=2,
            directory=".",
            subdomains=True,
        )

        arr = boundaries.array()
        assert (arr == labels["poly"]).sum() == nodes_num * 2
