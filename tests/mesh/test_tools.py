# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import filecmp
from pathlib import Path

import pytest

from bluemira.base.file import get_bluemira_path
from bluemira.mesh.error import MeshConversionError
from bluemira.mesh.tools import import_mesh, msh_to_xdmf

DATA_PATH = get_bluemira_path("mesh/test_data", subfolder="tests")


class TestMSHtoXDMF:
    @pytest.mark.parametrize("dimensions", [2, (0, 1)])
    def test_conversion_from_msh_to_xdm_new(self, dimensions):
        """
        Regression test on x-y example. Also test that 2 defaults to x-y.
        """
        msh_to_xdmf("GeneratedMesh.msh", dimensions=dimensions, directory=DATA_PATH)
        for filename in [
            "Mesh_linkfile.json",
            "Mesh_boundaries.h5",
            "Mesh_boundaries.xdmf",
            "Mesh_domain.h5",
            "Mesh_domain.xdmf",
        ]:
            generated = Path(DATA_PATH, "Generated" + filename)
            reference = Path(DATA_PATH, "Reference" + filename)
            filecmp.cmp(generated, reference)
            generated.unlink()

    @pytest.mark.parametrize(
        "dimensions",
        [
            (0,),
            (0, 0),
            (1, 1),
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 2, 3),
            (0, 1, 1),
            (0, 2, 2),
        ],
    )
    def test_dimension_errors(self, dimensions):
        with pytest.raises(MeshConversionError):
            msh_to_xdmf("GeneratedMesh.msh", dimensions=dimensions, directory=DATA_PATH)

    def test_nofile(self):
        with pytest.raises(MeshConversionError):
            msh_to_xdmf("not_here.msh", dimensions=(0, 1), directory=DATA_PATH)

    def test_nofile_import(self):
        with pytest.raises(MeshConversionError):
            import_mesh("not_here.msh", directory=DATA_PATH)
