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

import pytest
import tempfile
import shutil
import os
import filecmp
import numpy as np

from pathlib import Path
from unittest.mock import patch

from BLUEPRINT.base.file import get_BP_path

TEST_DATA = os.sep.join([get_BP_path("codes", "tests"), "test_data"])

Path(TEST_DATA).mkdir(parents=True, exist_ok=True)


def _prominence_skip():
    return pytest.importorskip("prominence")


def _jetto_skip():
    import BLUEPRINT.codes.jetto as jto

    return jto


# Make temporary sub-directory for tests.
@pytest.fixture(scope="module")
def tempdir():
    tempdir = tempfile.mkdtemp(dir=TEST_DATA)
    yield tempdir
    shutil.rmtree(tempdir)


class TestSetup:
    @pytest.fixture(autouse=True, scope="class")
    def setup(self, tempdir):
        # this is because of a pytest bug
        # workaround https://github.com/pytest-dev/pytest/issues/3778#issuecomment-411899446
        cls = type(self)
        jto = _jetto_skip()
        Setup = jto.Setup
        cls.setup = Setup(tempdir)

    def test_boundar_contour_writer(self, tempdir):
        bndry_cntr = np.arange(10).reshape(2, 5)
        self.setup.boundary_contour_to_bnd_file(bndry_cntr)

        try:
            assert filecmp.cmp(
                os.path.join(tempdir, "jetto.bnd"),
                os.path.join(TEST_DATA, "jetto.bnd"),
                shallow=False,
            )
        except AssertionError as err:
            with open(os.path.join(tempdir, "jetto.bnd"), "r") as r:
                print(r.readlines())
            with open(os.path.join(TEST_DATA, "jetto.bnd"), "r") as r:
                print(r.readlines())
            raise err


class TestRun:
    @classmethod
    def setup_class(cls):
        jto = _jetto_skip()
        Run = jto.Run


class TestTeardown:
    @classmethod
    def setup_class(cls):
        jto = _jetto_skip()
        Teardown = jto.Teardown


class TestProminenceDownloader:
    @pytest.fixture(autouse=True, scope="class")
    def setup(self, tempdir):
        # this is because of a pytest bug
        # workaround https://github.com/pytest-dev/pytest/issues/3778#issuecomment-411899446
        cls = type(self)
        cls.jto = _jetto_skip()
        with patch.object(cls.jto.ProminenceDownloader, "_import_binary"):
            cls.downloader = cls.jto.ProminenceDownloader(jobid=555, save_dir=tempdir)

    def test_importer(self):
        prom_bin = pytest.importorskip(
            "self.jto.ProminenceDownloader._import_binary",
            reason="Can't find Prominence binary",
        )
        assert hasattr(prom_bin, "command_download")

    # @pytest.mark.skip(msg="waiting for logging merge")
    def test_captured_print(self):
        with patch("BLUEPRINT.base.lookandfeel.bluemira_print") as bp:
            with patch("BLUEPRINT.base.lookandfeel.bluemira_warn") as bw:
                with patch("builtins.print", new=self.downloader.captured_print):
                    print("hello")
                    print("Error hello")

        assert bp.assert_called_with("hello")
        assert bw.assert_called_with("Prominence Error hello")

    def test_captured_open(self):
        with patch("builtins.open", new=self.downloader.captured_open):
            with open("TESTFILE", "w") as testfile:
                testfile.write("test")

        with open(os.sep.join([self.downloader._save_dir, "TESTFILE"]), "r") as tf:
            assert tf.readlines() == ["test"]

    def test_instance_contents(self):
        # Attributes needed by prominence_bin.command_download
        for attr in ["id", "dir", "force"]:
            assert hasattr(self.downloader, attr)


if __name__ == "__main__":
    pytest.main([__file__])
