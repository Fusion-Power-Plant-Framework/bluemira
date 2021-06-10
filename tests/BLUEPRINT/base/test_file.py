# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
from BLUEPRINT.base.file import FileManager, get_BP_path, SUB_DIRS, try_get_BP_path

REACTOR_NAME = "TEST_REACTOR"
FILE_REF_PATH = os.path.join("data")
FILE_GEN_PATH = os.path.join("tests", "BLUEPRINT", "base", "file_manager_gen")


@pytest.fixture(scope="session", autouse=True)
def file_manager_good():
    """
    Create a FileManager to run some tests on.
    """
    file_manager = FileManager(
        REACTOR_NAME,
        reference_data_root=FILE_REF_PATH,
        generated_data_root=FILE_GEN_PATH,
    )

    yield file_manager

    # Make sure we clean up the directories after testing that they have been created.
    if os.path.exists(FILE_GEN_PATH):
        remove_dir_and_subs(FILE_GEN_PATH)


@pytest.fixture(scope="session", autouse=True)
def file_manager_bad():
    """
    Create a FileManager to run some tests on.
    """
    file_manager = FileManager(
        REACTOR_NAME,
        reference_data_root=FILE_REF_PATH + "bad",
        generated_data_root=FILE_GEN_PATH,
    )

    return file_manager


def assert_bp_path_exists(path):
    """
    Fails a test if the specified path does not exist.
    """
    try:
        get_BP_path(path, "")
    except ValueError as error:
        pytest.fail("{path} does not exist and has not been created: %s" % error)


def remove_dir_and_subs(path):
    """
    Removes the specified path and any empty sub-directories.
    """
    for root, dirs, _ in os.walk(get_BP_path(path, ""), topdown=False):
        for name in dirs:
            os.rmdir(os.path.join(root, name))
    os.rmdir(get_BP_path(path, ""))


def get_system_path(root, reactor_name, system):
    """
    Builds the path for a system given a root directory and a reactor
    """
    return os.path.join(root, "reactors", reactor_name, system)


def get_reactor_path(root, reactor_name):
    """
    Builds the path for a reactor given a root directory and a reactor
    """
    return os.path.join(root, "reactors", reactor_name)


def test_create_generated_data_root(file_manager_good):
    """
    Tests that the generated data root directories are created.
    """
    file_manager_good.build_dirs()

    assert_bp_path_exists(file_manager_good.generated_data_root)


def test_set_reference_data_root(file_manager_good):
    """
    Tests that the reference data root directories are set.
    """
    file_manager_good.build_dirs()

    assert_bp_path_exists(file_manager_good.reference_data_root)


def test_reference_data_dirs(file_manager_good):
    """
    Tests that the reference data path dictionary is created and populated.
    """
    file_manager_good.set_reference_data_paths()

    target_data_dirs = sorted(SUB_DIRS + ["root"])

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


def test_reference_data_dirs_error(file_manager_bad):
    """
    Tests that a ValueError is raised if the data path does not exist.
    """
    with pytest.raises(ValueError):
        file_manager_bad.set_reference_data_paths()


def test_create_reference_data_dirs():
    """
    Test that the reference data path dictionary is created and populated

    Creates the directory and removes it.
    """
    file_manager = FileManager(
        REACTOR_NAME,
        reference_data_root=FILE_REF_PATH + "create",
        generated_data_root=FILE_GEN_PATH,
    )

    file_manager.create_reference_data_paths()

    target_data_dirs = sorted(SUB_DIRS + ["root"])

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
    if os.path.exists(FILE_GEN_PATH):
        remove_dir_and_subs(FILE_GEN_PATH)


def test_generated_data_dirs(file_manager_good):
    """
    Tests that the generated data path dictionary is created and populated.
    """
    file_manager_good.create_generated_data_paths()

    target_data_dirs = sorted(SUB_DIRS + ["root"])

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


@pytest.mark.parametrize(
    "path,subfolder,allow_missing,expect_none",  # noqa(N802)
    [
        ("BLUEPRINT/geometry", "tests", False, False),
        ("BLUEPRINT/geometry", "tests", True, False),
        ("spam", "tests", True, True),
        ("spam", "ham", True, True),
    ],
)
def test_try_get_BP_path(path, subfolder, allow_missing, expect_none):
    output_path = try_get_BP_path(path, subfolder=subfolder, allow_missing=allow_missing)
    if expect_none:
        assert output_path is None
    else:
        assert output_path == get_BP_path(path, subfolder=subfolder)


@pytest.mark.parametrize(
    "path,subfolder",  # noqa(N802)
    [
        ("spam", "tests"),
        ("spam", "ham"),
    ],
)
def test_try_get_BP_path_raises(path, subfolder):
    with pytest.raises(ValueError):
        try_get_BP_path(path, subfolder=subfolder, allow_missing=False)
