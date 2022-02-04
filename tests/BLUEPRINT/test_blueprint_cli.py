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

import copy
import functools
import json
import os
import shutil
import tempfile
import traceback
from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from bluemira.base.file import BM_ROOT, get_bluemira_root
from BLUEPRINT.blueprint_cli import cli, get_reactor_class
from BLUEPRINT.reactor import ConfigurableReactor
from tests.BLUEPRINT.test_reactor import (
    REACTORNAME,
    SmokeTestSingleNullReactor,
    build_config,
    build_tweaks,
    config,
)

INDIR = os.path.join(get_bluemira_root(), "tests", "BLUEPRINT", "cli", "test_indir")
OUTDIR = os.path.join(get_bluemira_root(), "tests", "BLUEPRINT", "cli", "test_outdir")
NEWNAME = "CLI-TEST"


class DummyObjForReactor:
    Reactor = "test"


class TestGetReactor:
    def test_get_reactor(self):
        with patch(
            "BLUEPRINT.blueprint_cli.get_module", return_value=DummyObjForReactor
        ):
            assert get_reactor_class("/abc/def.py::Reactor") == "test"
            assert get_reactor_class("abc.def.Reactor") == "test"
            assert get_reactor_class("Reactor") == "test"
            with pytest.raises(ImportError):
                get_reactor_class("MyReactor")


# Make temporary sub-directory for tests.
@pytest.fixture
def tempdir():
    tempdir = tempfile.mkdtemp(dir=OUTDIR)
    yield tempdir
    shutil.rmtree(tempdir)


def temp_path_to_file(tempdir, reactorname, filename, subdir=None, use_prefix=True):
    """
    Accepts the current temporary directory, reactorname, filename suffix, and optional
    subdirectory. Returns the path to the given file within the temporary directory.
    """
    if use_prefix:
        filename = f"{reactorname}_{filename}"
    if not subdir:
        path_to_file = os.path.join(tempdir, "reactors", reactorname, filename)
    else:
        path_to_file = os.path.join(tempdir, "reactors", reactorname, subdir, filename)
    return path_to_file


# Patch in mock objects for reactor build and related functions.
def mock_mode(func):
    @patch.object(ConfigurableReactor, "save_CAD_model")
    @patch.object(ConfigurableReactor, "plot_xy")
    @patch.object(ConfigurableReactor, "plot_xz")
    @patch.object(ConfigurableReactor, "build")
    @patch("BLUEPRINT.blueprint_cli.get_reactor_class", return_value=ConfigurableReactor)
    @functools.wraps(func)
    def wrapper_grouped_decorator(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper_grouped_decorator


def assert_invoke(
    cli, indir, reactorname, outdir, expected_exit_code=0, extra_flags=None
):
    if extra_flags is None:
        extra_flags = []

    runner = CliRunner()

    # Set flags and run bluemira cli.
    flags = ["-i", indir, "-ri", reactorname, "-o", outdir] + extra_flags
    result = runner.invoke(cli, flags)
    assert result.exit_code == expected_exit_code, traceback.print_exception(
        *result.exc_info
    )


class TestCLI:
    @classmethod
    def setup_class(cls):
        Path(INDIR).mkdir(parents=True, exist_ok=True)
        Path(OUTDIR).mkdir(parents=True, exist_ok=True)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree(INDIR)
        shutil.rmtree(OUTDIR)

    def setup_method(self):
        self.reactor = SmokeTestSingleNullReactor(config, build_config, build_tweaks)
        self.reactor.config_to_json(INDIR)

    @mock_mode
    def test_cli_build(
        self,
        mock_rclass,
        mock_build,
        mock_plot_xz,
        mock_plot_xy,
        mock_save_CAD_model,
        tempdir,
    ):
        """
        Test that the CLI calls the reactor build function.
        """
        runner = CliRunner()

        # Set flags and run bluemira cli.
        flags = ["-i", INDIR, "-ri", REACTORNAME, "-o", tempdir]
        result = runner.invoke(cli, flags)
        assert result.exit_code == 0, traceback.print_exception(*result.exc_info)

        assert mock_build.call_count == 1

    @mock_mode
    def test_cli_copy_input_files(
        self,
        mock_rclass,
        mock_build,
        mock_plot_xz,
        mock_plot_xy,
        mock_save_CAD_model,
        tempdir,
    ):
        """
        Test that the CLI correctly makes copies of the input files to the output
        directory.
        """
        assert_invoke(cli, INDIR, REACTORNAME, tempdir)

        # Test input files are copied to output directory and not removed.
        for filename_suffix in (
            "template.json",
            "config.json",
            "build_config.json",
            "build_tweaks.json",
        ):
            path_to_file_in = os.path.join(INDIR, f"{REACTORNAME}_{filename_suffix}")
            path_to_file_out = temp_path_to_file(tempdir, REACTORNAME, filename_suffix)
            assert os.path.isfile(path_to_file_in)
            assert os.path.isfile(path_to_file_out)

    # Create list of output switches for the following test.
    switch_names = ["log", "data", "plots", "cad"]
    on_switches = [[f"--{switch}"] for switch in switch_names]
    off_switches = [[f"--no_{switch}"] for switch in switch_names]
    output_switches = on_switches + off_switches

    @pytest.mark.parametrize("switch_flag", output_switches)
    @mock_mode
    def test_cli_output_switches(
        self,
        mock_rclass,
        mock_build,
        mock_plot_xz,
        mock_plot_xy,
        mock_save_CAD_model,
        switch_flag,
        tempdir,
    ):
        """
        Test that the CLI returns the desired outputs and does not return others.
        """
        assert_invoke(cli, INDIR, REACTORNAME, tempdir, extra_flags=switch_flag)

        # Assign expected outputs from defaults and output switches.
        switch_dict = {
            "log": True,
            "data": True,
            "plots": True,
            "cad": False,
        }

        if switch_flag[0].startswith("--no_"):
            key = switch_flag[0].replace("--no_", "")
            switch = False
        elif switch_flag[0].startswith("--"):
            key = switch_flag[0].replace("--", "")
            switch = True
        switch_dict[key] = switch

        # Test correct output files are saved.
        if switch_dict["log"] is True:
            assert os.path.isfile(temp_path_to_file(tempdir, REACTORNAME, "output.txt"))
            assert os.path.isfile(temp_path_to_file(tempdir, REACTORNAME, "errors.txt"))
        else:
            assert not os.path.isfile(
                temp_path_to_file(tempdir, REACTORNAME, "output.txt")
            )
            assert not os.path.isfile(
                temp_path_to_file(tempdir, REACTORNAME, "errors.txt")
            )

        if switch_dict["data"] is True:
            assert os.path.isfile(temp_path_to_file(tempdir, REACTORNAME, "params.json"))
        else:
            assert not os.path.isfile(
                temp_path_to_file(tempdir, REACTORNAME, "params.json")
            )

        if switch_dict["plots"] is True:
            assert mock_plot_xz.call_count == 1
            assert mock_plot_xy.call_count == 1
        else:
            assert mock_plot_xz.call_count == 0
            assert mock_plot_xy.call_count == 0

        if switch_dict["cad"] is True:
            assert mock_save_CAD_model.call_count == 1
        else:
            assert mock_save_CAD_model.call_count == 0

    @pytest.mark.parametrize("tarball_flag", [["-t"], ["--tarball"]])
    @mock_mode
    def test_cli_tarball(
        self,
        mock_rclass,
        mock_build,
        mock_plot_xz,
        mock_plot_xy,
        mock_save_CAD_model,
        tarball_flag,
        tempdir,
    ):
        """
        Test that the tarball CLI option works correctly.
        """
        assert_invoke(cli, INDIR, REACTORNAME, tempdir, extra_flags=tarball_flag)

        # Test tarball flag successfully generates .tar file.
        path_to_file = temp_path_to_file(
            tempdir, REACTORNAME, f"{REACTORNAME}.tar.gz", use_prefix=False
        )
        assert os.path.isfile(path_to_file)

    @pytest.mark.parametrize("verbose_flag", [["-v"], ["--verbose"]])
    @mock_mode
    def test_cli_verbose(
        self,
        mock_rclass,
        mock_build,
        mock_plot_xz,
        mock_plot_xy,
        mock_save_CAD_model,
        verbose_flag,
        tempdir,
    ):
        """
        Test that the verbose CLI option works correctly.
        """
        assert_invoke(cli, INDIR, REACTORNAME, tempdir, extra_flags=verbose_flag)

        # Test verbose flag successfully activates verbose mode.
        path_to_file = temp_path_to_file(tempdir, REACTORNAME, "params.json")
        assert os.path.isfile(path_to_file)
        with open(path_to_file, "r") as fh:
            data = json.load(fh)
        assert isinstance(data, dict)
        assert isinstance(data["Name"], dict)

    @pytest.mark.parametrize(
        "name_flags",
        [["-ro", NEWNAME], ["--reactornameout", NEWNAME]],
    )
    @mock_mode
    def test_cli_reactornameout(
        self,
        mock_rclass,
        mock_build,
        mock_plot_xz,
        mock_plot_xy,
        mock_save_CAD_model,
        name_flags,
        tempdir,
    ):
        """
        Test that the reactornameout CLI option works correctly.
        """
        # Make copy of reference data directory in tempdir.
        # Note: this is done to avoid a FileExists error from previous tests.
        temp_reactor = copy.deepcopy(self.reactor)
        source = os.path.join(
            get_bluemira_root(),
            "tests",
            "BLUEPRINT",
            "test_data",
            "reactors",
            REACTORNAME,
        )
        destination = os.path.join(
            tempdir, "BLUEPRINT", "test_data", "reactors", REACTORNAME
        )
        shutil.copytree(source, destination)

        # Generate the input files for this test, using the new reference data root.
        temp_reactor.build_config["reference_data_root"] = os.path.join(
            tempdir, "BLUEPRINT", "test_data"
        )
        temp_indir = os.path.join(tempdir, "temp_indir")
        Path(temp_indir).mkdir(parents=True, exist_ok=True)
        temp_reactor.config_to_json(temp_indir)

        assert_invoke(cli, temp_indir, REACTORNAME, tempdir, extra_flags=name_flags)

        # Test output folder created with specified reactorname.
        assert os.path.isdir(os.path.join(tempdir, "reactors", NEWNAME))

        # Test reactor name overridden within config file values.
        path_to_config = temp_path_to_file(tempdir, NEWNAME, "config.json")
        with open(path_to_config, "r") as fh:
            data = json.load(fh)

        if isinstance(data["Name"], dict):
            data["Name"] = data["Name"]["value"]

        assert data["Name"] == NEWNAME

        # Test reactor name overridden within params file values.
        path_to_params = temp_path_to_file(tempdir, NEWNAME, "params.json")
        with open(path_to_params, "r") as fh:
            data = json.load(fh)

        if isinstance(data["Name"], dict):
            data["Name"] = data["Name"]["value"]

        assert data["Name"] == NEWNAME

        # Test reactor name overridden in filenames.
        # Note: CAD filenames covered by other tests and thus are not tested here.
        for filename_suffix in (
            "template.json",
            "config.json",
            "build_config.json",
            "build_tweaks.json",
            "params.json",
            "XZ.png",
            "XY.png",
        ):
            if filename_suffix == "XZ.png" or filename_suffix == "XY.png":
                subdir = "plots"
            else:
                subdir = None
            path_to_file = temp_path_to_file(
                tempdir,
                NEWNAME,
                filename_suffix,
                subdir=subdir,
            )
            assert os.path.isfile(path_to_file)

    @mock_mode
    def test_cli_bmroot_keyword(
        self, mock_build, mock_plot_xz, mock_plot_xy, mock_save_CAD_model, mock_rclass
    ):
        """
        Test that the CLI can handle keyword replacement for the bluemira root
        directory.
        """
        # Set temp outdir and ensure directory does not already exist.
        outdir_flag = os.path.join(BM_ROOT, "tests", "BLUEPRINT", "cli", "temp_outdir")
        outdir_path = outdir_flag.replace(BM_ROOT, get_bluemira_root())
        if os.path.exists(outdir_path):
            shutil.rmtree(outdir_path)

        assert_invoke(cli, INDIR, REACTORNAME, outdir_flag)

        # Test temp outdir was created correctly and clean up.
        assert os.path.isdir(outdir_path)
        shutil.rmtree(outdir_path)

    @mock_mode
    def test_cli_read_outdir_from_file(
        self,
        mock_rclass,
        mock_build,
        mock_plot_xz,
        mock_plot_xy,
        mock_save_CAD_model,
        tempdir,
    ):
        """
        Test that the output directory can be read from input files.
        """
        runner = CliRunner()

        # Generate the input files for this test, using the generated data root.
        temp_reactor = copy.deepcopy(self.reactor)
        temp_reactor.build_config["generated_data_root"] = tempdir
        temp_indir = os.path.join(tempdir, "temp_indir")
        Path(temp_indir).mkdir(parents=True, exist_ok=True)
        temp_reactor.config_to_json(temp_indir)

        # Set flags and run bluemira cli.
        flags = ["-i", temp_indir, "-ri", REACTORNAME]
        result = runner.invoke(cli, flags)
        assert result.exit_code == 0, traceback.print_exception(*result.exc_info)

        assert os.path.isdir(os.path.join(tempdir, "reactors", REACTORNAME))

    @pytest.mark.parametrize("rerun_flag", [["-f"], ["--force_rerun"]])
    @mock_mode
    def test_cli_force_rerun(
        self,
        mock_rclass,
        mock_build,
        mock_plot_xz,
        mock_plot_xy,
        mock_save_CAD_model,
        rerun_flag,
        tempdir,
    ):
        """
        Test that bluemira can be rerun from the CLI when the force flag is on.
        """
        runner = CliRunner()

        # Set flags and run bluemira cli.
        default_flags = ["-i", INDIR, "-ri", REACTORNAME, "-o", tempdir]
        flags = default_flags + rerun_flag
        runner.invoke(cli, default_flags)  # First run
        result = runner.invoke(cli, flags)  # Rerun with force flag on
        assert result.exit_code == 0, traceback.print_exception(*result.exc_info)

    @mock_mode
    def test_cli_avoid_rerun(
        self,
        mock_rclass,
        mock_build,
        mock_plot_xz,
        mock_plot_xy,
        mock_save_CAD_model,
        tempdir,
    ):
        """
        Test that bluemira can not be rerun from the CLI when the force flag is off.
        """
        runner = CliRunner()

        # Set flags and run bluemira cli.
        flags = ["-i", INDIR, "-ri", REACTORNAME, "-o", tempdir]
        runner.invoke(cli, flags)  # First run
        result = runner.invoke(cli, flags)  # Rerun with force flag off
        assert result.exit_code == 1, traceback.print_exception(*result.exc_info)

    @mock_mode
    def test_cli_invalid_flag(
        self,
        mock_rclass,
        mock_build,
        mock_plot_xz,
        mock_plot_xy,
        mock_save_CAD_model,
        tempdir,
    ):
        """
        Test that the bluemira CLI fails correctly when an invalid flag is passed.
        """
        invalid_flag = "--this_flag_does_not_exist"
        assert_invoke(
            cli,
            INDIR,
            REACTORNAME,
            tempdir,
            extra_flags=[invalid_flag],
            expected_exit_code=2,
        )

    def test_cli_invalid_inputs(self, tempdir):
        """
        Test that the bluemira CLI fails correctly when invalid inputs are passed.
        """
        invalid_inputs = [
            "this_file_does_not_exist.json",
            "this_file_does_not_exist.json",
            "this_file_does_not_exist.json",
            "this_file_does_not_exist.json",
        ]
        assert_invoke(
            cli,
            INDIR,
            REACTORNAME,
            tempdir,
            extra_flags=invalid_inputs,
            expected_exit_code=1,
        )

    @mock_mode
    def test_datadir(
        self,
        mock_rclass,
        mock_build,
        mock_plot_xz,
        mock_plot_xy,
        mock_save_CAD_model,
        tempdir,
    ):
        temp_datadir = tempfile.mkdtemp(dir=OUTDIR)
        shutil.copytree(
            os.sep.join([get_bluemira_root(), "data", "BLUEPRINT"]),
            temp_datadir,
            dirs_exist_ok=True,
        )

        indir = os.sep.join(
            [get_bluemira_root(), "examples", "BLUEPRINT", "cli", "indir"]
        )

        try:
            assert_invoke(
                cli,
                indir,
                REACTORNAME,
                tempdir,
                extra_flags=["-ri", "EU-DEMO", "-d", temp_datadir],
            )
        finally:
            shutil.rmtree(temp_datadir)
