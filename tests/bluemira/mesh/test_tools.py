# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
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

import filecmp
import os

from bluemira.base.file import get_bluemira_path
from bluemira.mesh.tools import msh_to_xdmf

DATA_PATH = get_bluemira_path("mesh/test_data", subfolder="tests/bluemira")


class TestMSHtoXDMF:
    def test_conversion_from_msh_to_xdm_new(self):
        """
        Regression test on example
        """
        msh_to_xdmf("GeneratedMesh.msh", dimension=2, directory=DATA_PATH)
        for filename in [
            "Mesh_linkfile.json",
            "Mesh_boundaries.h5",
            "Mesh_boundaries.xdmf",
            "Mesh_domain.h5",
            "Mesh_domain.xdmf",
        ]:
            generated = os.sep.join([DATA_PATH, "Generated" + filename])
            reference = os.sep.join([DATA_PATH, "Reference" + filename])
            filecmp.cmp(generated, reference)
            os.remove(generated)
