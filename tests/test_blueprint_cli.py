# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.

from click.testing import CliRunner
import copy
import functools
import json
import os
from unittest.mock import patch
from pathlib import Path
import pytest
import shutil
import tempfile

from BLUEPRINT.base.file import get_BP_root
from BLUEPRINT import blueprint_cli
from BLUEPRINT.blueprint_cli import cli

REACTORNAME = "EU-DEMO"
INDIR = os.path.join(get_BP_root(), "examples", "cli", "indir")
OUTDIR = os.path.join(get_BP_root(), "tests", "cli")
Path(OUTDIR).mkdir(parents=True, exist_ok=True)


# Make temporary sub-directory for tests.
@pytest.fixture
def tmpdir():
    tmpdir = tempfile.mkdtemp(dir=OUTDIR)
    yield tmpdir
    shutil.rmtree(tmpdir)


# Patch in mock objects for reactor build and related functions.
def mock_mode(func):
    @patch("BLUEPRINT.reactor.ConfigurableReactor.save_CAD_model")
    @patch("BLUEPRINT.reactor.ConfigurableReactor.plot_xy")
    @patch("BLUEPRINT.reactor.ConfigurableReactor.plot_xz")
    @patch("BLUEPRINT.reactor.ConfigurableReactor.build")
    @functools.wraps(func)
    def wrapper_grouped_decorator(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper_grouped_decorator


# Test BLUEPRINT build is called by cli.
@patch("BLUEPRINT.reactor.ConfigurableReactor.build")
def test_cli_build(mock_build, tmpdir):
    runner = CliRunner()

    default_flags = ["-i", INDIR, "-ri", REACTORNAME, "-o", tmpdir]
    run_flags = default_flags + ["-m", "lite"]
    result = runner.invoke(cli, run_flags)
    assert result.exit_code == 0

    assert mock_build.call_count == 1


# Test output modes and output override switches.
output_flags = []
for val in dir(blueprint_cli.Output):
    if isinstance(getattr(blueprint_cli.Output, val), bool):
        output_flags.append(val)

outmodes = {}
for val in dir(blueprint_cli):
    attr = getattr(blueprint_cli, val)
    if isinstance(attr, blueprint_cli.Output):
        outmodes[attr.name] = copy.deepcopy(attr.__dict__)

param_list = []
for mode in outmodes:
    param_list.append(["-m", mode])  # output modes

on_switches = [f"--{flag}" for flag in output_flags]
off_switches = [f"--no_{flag}" for flag in output_flags]
for on, off in zip(on_switches, off_switches):
    param_list.append(["-m", "none", on])
    param_list.append(["-m", "full", off])


@pytest.mark.parametrize("flags", param_list)
@mock_mode
def test_cli_outmodes(
    mock_build, mock_plot_xz, mock_plot_xy, mock_save_CAD_model, flags, tmpdir
):
    runner = CliRunner()

    # Set run flags.
    default_flags = ["-i", INDIR, "-ri", REACTORNAME, "-o", tmpdir]
    run_flags = default_flags + flags

    # Run command line interface.
    result = runner.invoke(cli, run_flags)
    assert result.exit_code == 0

    def tmp_path_to_file(filename, subdir=None):
        if not subdir:
            path_to_file = os.path.join(tmpdir, REACTORNAME, filename)
        else:
            path_to_file = os.path.join(tmpdir, REACTORNAME, subdir, filename)
        return path_to_file

    # Test input files are returned to output directory and match the originals.
    for filename in (
        "template.json",
        "config.json",
        "build_config.json",
        "build_tweaks.json",
    ):
        path_to_file_in = os.path.join(INDIR, f"{REACTORNAME}_{filename}")
        path_to_file_out = tmp_path_to_file(f"{REACTORNAME}_{filename}")
        assert os.path.isfile(path_to_file_in)
        assert os.path.isfile(path_to_file_out)
        with open(path_to_file_in, "r") as f1:
            with open(path_to_file_out, "r") as f2:
                assert f1.read() == f2.read()

    # Reassign expected outputs from override switches.
    outmode = copy.copy(outmodes[flags[1]])
    if len(flags) > 2:
        for override_flag in flags[2:]:
            key = None
            switch = None
            if override_flag.startswith("--no_"):
                key = override_flag.replace("--no_", "")
                switch = False
            elif override_flag.startswith("--"):
                key = override_flag.replace("--", "")
                switch = True
            if key and key in outmode:
                outmode[key] = switch

    # Test correct output files are saved corresponding to CLI input options.
    if outmode["log"] is True:
        assert os.path.isfile(tmp_path_to_file("output.txt"))
        assert os.path.isfile(tmp_path_to_file("errors.txt"))
    else:
        assert not os.path.isfile(tmp_path_to_file("output.txt"))
        assert not os.path.isfile(tmp_path_to_file("errors.txt"))

    if outmode["data"] is True:
        assert os.path.isfile(tmp_path_to_file(f"{REACTORNAME}_params.json"))
    else:
        assert not os.path.isfile(tmp_path_to_file(f"{REACTORNAME}_params.json"))

    if outmode["plot_xz"] is True:
        assert mock_plot_xz.call_count == 1
    else:
        assert mock_plot_xz.call_count == 0

    if outmode["plot_xy"] is True:
        assert mock_plot_xy.call_count == 1
    else:
        assert mock_plot_xy.call_count == 0

    if outmode["cad"] is True:
        assert mock_save_CAD_model.call_count == 1
    else:
        assert mock_save_CAD_model.call_count == 0


# Test tarball flags.
@pytest.mark.parametrize("flags", [["-m", "none", "-t"], ["-m", "none", "--tarball"]])
@mock_mode
def test_cli_tarball(
    mock_build, mock_plot_xz, mock_plot_xy, mock_save_CAD_model, flags, tmpdir
):
    runner = CliRunner()

    # Set flags and run BLUEPRINT cli.
    default_flags = ["-i", INDIR, "-ri", REACTORNAME, "-o", tmpdir]
    run_flags = default_flags + flags
    result = runner.invoke(cli, run_flags)
    assert result.exit_code == 0

    # Test tarball flag successfully generates .tar file.
    filename = f"{REACTORNAME}.tar"
    path_to_file = os.path.join(tmpdir, REACTORNAME, filename)
    assert os.path.isfile(path_to_file)


# Test verbose flags.
@pytest.mark.parametrize("flags", [["-v"], ["--verbose"]])
@mock_mode
def test_cli_verbose(
    mock_build, mock_plot_xz, mock_plot_xy, mock_save_CAD_model, flags, tmpdir
):
    runner = CliRunner()

    # Set flags and run BLUEPRINT cli.
    default_flags = ["-i", INDIR, "-ri", REACTORNAME, "-o", tmpdir]
    run_flags = default_flags + flags
    result = runner.invoke(cli, run_flags)
    assert result.exit_code == 0

    # Test verbose flag successfully activates verbose mode.
    filename = f"{REACTORNAME}_params.json"
    path_to_file = os.path.join(tmpdir, REACTORNAME, filename)
    assert os.path.isfile(path_to_file)
    with open(path_to_file, "r") as fh:
        data = json.load(fh)
    assert isinstance(data, dict)
    assert isinstance(data["Name"], dict)


# Test output reactor name override option.
@pytest.mark.parametrize(
    "flags",
    [["-ro", "CLI-TEST"], ["--reactornameout", "CLI-TEST"]],
)
@mock_mode
def test_cli_reactornameout(
    mock_build, mock_plot_xz, mock_plot_xy, mock_save_CAD_model, flags, tmpdir
):
    runner = CliRunner()

    # Set flags and run BLUEPRINT cli.
    default_flags = ["-i", INDIR, "-ri", REACTORNAME, "-o", tmpdir]
    run_flags = default_flags + flags
    result = runner.invoke(cli, run_flags)
    assert result.exit_code == 0

    reactorname_override = flags[-1]

    # Test reactor name overriden within config file values.
    path_to_config = os.path.join(
        tmpdir, reactorname_override, f"{reactorname_override}_config.json"
    )
    with open(path_to_config, "r") as fh:
        data = json.load(fh)
    assert data["Name"] == reactorname_override

    # Test reactor name overriden within params file values.
    path_to_params = os.path.join(
        tmpdir, reactorname_override, f"{reactorname_override}_params.json"
    )
    with open(path_to_params, "r") as fh:
        data = json.load(fh)
    assert data["Name"] == reactorname_override

    # Define function for testing each output filename uses the override reactor name.
    # Note that the cad saving is covered by other tests and is not tested here.
    def check_reactorname_overriden(filename_suffix, subdir=False):
        path_to_dir_1 = os.path.join(tmpdir, REACTORNAME)
        path_to_dir_2 = os.path.join(tmpdir, reactorname_override)
        if subdir:
            path_to_dir_1 = os.path.join(path_to_dir_1, subdir)
            path_to_dir_2 = os.path.join(path_to_dir_2, subdir)
        filename_1 = f"{REACTORNAME}_{filename_suffix}"
        filename_2 = f"{reactorname_override}_{filename_suffix}"

        path_to_file_1 = os.path.join(path_to_dir_1, filename_1)
        path_to_file_2 = os.path.join(path_to_dir_1, filename_2)
        path_to_file_3 = os.path.join(path_to_dir_2, filename_1)
        path_to_file_4 = os.path.join(path_to_dir_2, filename_2)

        assert not os.path.isfile(path_to_file_1)
        assert not os.path.isfile(path_to_file_2)
        assert not os.path.isfile(path_to_file_3)
        assert os.path.isfile(path_to_file_4)

    # Test reactor name overriden in output filenames.
    check_reactorname_overriden("template.json")
    check_reactorname_overriden("config.json")
    check_reactorname_overriden("build_config.json")
    check_reactorname_overriden("build_tweaks.json")
    check_reactorname_overriden("params.json")
    check_reactorname_overriden("XZ.png", subdir="plots")
    check_reactorname_overriden("XY.png", subdir="plots")


@mock_mode
def test_cli_invalid_flag(
    mock_build, mock_plot_xz, mock_plot_xy, mock_save_CAD_model, tmpdir
):
    runner = CliRunner()

    # Set flags and run BLUEPRINT cli.
    default_flags = ["-i", INDIR, "-ri", REACTORNAME, "-o", tmpdir]
    run_flags = default_flags + ["--this_flag_does_not_exist"]
    result = runner.invoke(cli, run_flags)
    assert result.exit_code == 2


def test_cli_invalid_outmode(tmpdir):
    runner = CliRunner()

    # Set flags and run BLUEPRINT cli.
    default_flags = ["-i", INDIR, "-ri", REACTORNAME, "-o", tmpdir]
    run_flags = default_flags + [
        "-m",
        "this_mode_does_not_exist",
    ]
    result = runner.invoke(cli, run_flags)
    assert result.exit_code == 0
    assert "Invalid outmode." in result.stdout


def test_cli_invalid_inputs(tmpdir):
    runner = CliRunner()

    # Set flags and run BLUEPRINT cli.
    default_flags = ["-i", INDIR, "-ri", REACTORNAME, "-o", tmpdir]
    run_flags = default_flags + [
        "this_file_does_not_exist.json",
        "this_file_does_not_exist.json",
        "this_file_does_not_exist.json",
        "this_file_does_not_exist.json",
    ]
    result = runner.invoke(cli, run_flags)
    assert result.exit_code == 1


@mock_mode
def test_cli_avoid_rerun(
    mock_build, mock_plot_xz, mock_plot_xy, mock_save_CAD_model, tmpdir
):
    runner = CliRunner()
    # Set flags and run BLUEPRINT cli.
    default_flags = ["-i", INDIR, "-ri", REACTORNAME, "-o", tmpdir, "-m", "lite"]

    runner.invoke(cli, default_flags)
    result = runner.invoke(cli, default_flags)

    assert result.exit_code == 1
