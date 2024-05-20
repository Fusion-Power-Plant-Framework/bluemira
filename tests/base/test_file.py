# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import os
import tempfile
from pathlib import Path

import pytest

from bluemira.base.file import (
    SUB_DIRS,
    FileManager,
    force_file_extension,
    get_bluemira_path,
    try_get_bluemira_path,
    working_dir,
)

REACTOR_NAME = "TEST_REACTOR"
FILE_REF_PATH = Path("data").as_posix()


@pytest.mark.parametrize(
    ("path", "subfolder", "allow_missing", "expect_none"),
    [
        ("base", "tests", False, False),
        ("base", "tests", True, False),
        ("spam", "tests", True, True),
        ("spam", "ham", True, True),
    ],
)
def test_try_get_bluemira_path(path, subfolder, allow_missing, expect_none):
    output_path = try_get_bluemira_path(
        path, subfolder=subfolder, allow_missing=allow_missing
    )
    if expect_none:
        assert output_path is None
    else:
        assert output_path == get_bluemira_path(path, subfolder=subfolder)


@pytest.mark.parametrize(
    ("path", "subfolder"),
    [
        ("spam", "tests"),
        ("spam", "ham"),
    ],
)
def test_try_get_bluemira_path_raises(path, subfolder):
    with pytest.raises(ValueError):  # noqa: PT011
        try_get_bluemira_path(path, subfolder=subfolder, allow_missing=False)


@pytest.fixture(scope="session", autouse=True)
def file_manager_good(tmp_path_factory):
    """
    Create a FileManager to run some tests on.
    """
    file_gen_path = tmp_path_factory.mktemp("file_manager_gen")
    file_manager = FileManager(
        REACTOR_NAME,
        reference_data_root=FILE_REF_PATH,
        generated_data_root=file_gen_path.as_posix(),
    )

    yield file_manager

    # Make sure we clean up the directories after testing that they have been created.
    if Path(file_gen_path):
        remove_dir_and_subs(file_gen_path.as_posix())


@pytest.fixture(scope="session", autouse=True)
def file_manager_bad(tmp_path_factory):
    """
    Create a FileManager to run some tests on.
    """
    file_gen_path = tmp_path_factory.mktemp("file_manager_gen_bad")
    return FileManager(
        REACTOR_NAME,
        reference_data_root=FILE_REF_PATH + "bad",
        generated_data_root=file_gen_path.as_posix(),
    )


def assert_bluemira_path_exists(path: str):
    """
    Fails a test if the specified path does not exist.
    """
    try:
        get_bluemira_path(path, "")
    except ValueError as error:
        pytest.fail(f"{path} does not exist and has not been created: {error}")


def remove_dir_and_subs(path: str):
    """
    Removes the specified path and any empty sub-directories.
    """
    for root, dirs, _ in os.walk(get_bluemira_path(path, ""), topdown=False):
        for name in dirs:
            Path(root, name).rmdir()
    Path(get_bluemira_path(path, "")).rmdir()


def get_system_path(root: str, reactor_name: str, system: str) -> str:
    """
    Builds the path for a system given a root directory and a reactor
    """
    return Path(root, "reactors", reactor_name, system).as_posix()


def get_reactor_path(root: str, reactor_name: str) -> str:
    """
    Builds the path for a reactor given a root directory and a reactor
    """
    return Path(root, "reactors", reactor_name).as_posix()


def test_create_generated_data_root(file_manager_good: FileManager):
    """
    Tests that the generated data root directories are created.
    """
    file_manager_good.build_dirs()

    assert_bluemira_path_exists(file_manager_good.generated_data_root)


def test_set_reference_data_root(file_manager_good: FileManager):
    """
    Tests that the reference data root directories are set.
    """
    file_manager_good.build_dirs()

    assert_bluemira_path_exists(file_manager_good.reference_data_root)


def test_reference_data_dirs(file_manager_good: FileManager):
    """
    Tests that the reference data path dictionary is created and populated.
    """
    file_manager_good.set_reference_data_paths()

    target_data_dirs = sorted([*SUB_DIRS, "root"])

    # Check the keys match the expected list
    assert sorted(file_manager_good.reference_data_dirs.keys()) == target_data_dirs

    for key, value in file_manager_good.reference_data_dirs.items():
        if key == "root":
            assert value == get_reactor_path(
                file_manager_good.reference_data_root, file_manager_good.reactor_name
            )
        else:
            assert value == get_system_path(
                file_manager_good.reference_data_root,
                file_manager_good.reactor_name,
                key,
            )


def test_reference_data_dirs_error(file_manager_bad: FileManager):
    """
    Tests that a ValueError is raised if the data path does not exist.
    """
    with pytest.raises(ValueError):  # noqa: PT011
        file_manager_bad.set_reference_data_paths()


def test_create_reference_data_dirs(tmp_path):
    """
    Test that the reference data path dictionary is created and populated

    Creates the directory and removes it.
    """
    path = tmp_path / "create"
    file_manager = FileManager(
        REACTOR_NAME,
        reference_data_root=FILE_REF_PATH + "create",
        generated_data_root=path.as_posix(),
    )

    file_manager.create_reference_data_paths()

    target_data_dirs = sorted([*SUB_DIRS, "root"])

    # Check the keys match the expected list
    assert sorted(file_manager.reference_data_dirs.keys()) == target_data_dirs

    for key, value in file_manager.reference_data_dirs.items():
        if key == "root":
            assert value == get_reactor_path(
                file_manager.reference_data_root, file_manager.reactor_name
            )
        else:
            assert value == get_system_path(
                file_manager.reference_data_root,
                file_manager.reactor_name,
                key,
            )

    # Make sure we clean up the directories after testing that they have been created.
    if Path(path).exists():
        remove_dir_and_subs(path.as_posix())


def test_generated_data_dirs(file_manager_good: FileManager):
    """
    Tests that the generated data path dictionary is created and populated.
    """
    file_manager_good.create_generated_data_paths()

    target_data_dirs = sorted([*SUB_DIRS, "root"])

    # Check the keys match the expected list
    assert sorted(file_manager_good.generated_data_dirs.keys()) == target_data_dirs

    for key, value in file_manager_good.generated_data_dirs.items():
        if key == "root":
            assert value == get_reactor_path(
                file_manager_good.generated_data_root, file_manager_good.reactor_name
            )
        else:
            assert value == get_system_path(
                file_manager_good.generated_data_root,
                file_manager_good.reactor_name,
                key,
            )


def test_force_file_extension():
    file_path = "/path/to/a/file"

    assert force_file_extension(file_path, [".mf", ".mo"]) == file_path + ".mf"
    assert force_file_extension(file_path + ".mf", [".mf", ".mo"]) == file_path + ".mf"
    assert force_file_extension(file_path, ".mf") == file_path + ".mf"


def test_working_dir_context_manager():
    cwd = Path.cwd()
    with working_dir(tempfile.mkdtemp()):
        changed_cwd = Path.cwd()
    final_cwd = Path.cwd()
    assert cwd != changed_cwd
    assert cwd == final_cwd
