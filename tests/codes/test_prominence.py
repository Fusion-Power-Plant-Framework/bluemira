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

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import bluemira.codes._prominence as prom
from bluemira.base.file import get_bluemira_path

TEST_DATA = os.sep.join([get_bluemira_path("codes", "tests"), "test_generated_data"])

Path(TEST_DATA).mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="module")
def tempdir():
    # Make temporary sub-directory for tests.
    tempdir = tempfile.mkdtemp(dir=TEST_DATA)
    yield tempdir
    shutil.rmtree(tempdir)


class TestProminenceDownloader:
    @pytest.fixture(autouse=True, scope="class")
    def setup(self, tempdir):
        # this is because of a pytest wontfix bug, workaround:
        # https://github.com/pytest-dev/pytest/issues/3778#issuecomment-411899446
        cls = type(self)
        with patch.object(prom.ProminenceDownloader, "_load_binary"):
            cls.downloader = prom.ProminenceDownloader(jobid=555, save_dir=tempdir)

    def test_importer(self):
        pytest.importorskip(
            "prominence",
            reason="Can't find Prominence binary",
        )
        prom_bin = prom.ProminenceDownloader(jobid=555, save_dir=tempdir)._prom_bin
        assert hasattr(prom_bin, "command_download")

    def test_captured_print(self):
        with patch("bluemira.codes._prominence.bluemira_print") as bp:
            with patch("bluemira.codes._prominence.bluemira_warn") as bw:
                with patch("builtins.print", new=self.downloader.captured_print):
                    print("hello")
                    print("Error hello")

        bp.assert_called_with("hello")
        bw.assert_called_with("Prominence Error hello")

    def test_captured_open(self):
        with patch("builtins.open", new=self.downloader.captured_open):
            with open("TESTFILE", "w") as testfile:
                testfile.write("test")

        with open(os.sep.join([self.downloader._save_dir, "TESTFILE"]), "r") as tf:
            assert tf.readlines() == ["test"]
