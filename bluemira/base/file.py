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

"""
File I/O functions and some path operations
"""

import os
import pathlib
from typing import Dict, List, Union

BM_ROOT = "!BM_ROOT!"
SUB_DIRS = ["equilibria", "neutronics", "systems_code", "CAD", "plots", "geometry"]


def _get_relpath(folder: str, subfolder: str) -> str:
    path = os.sep.join([folder, subfolder])
    if os.path.isdir(path):
        return path
    else:
        raise ValueError(f"{path} Not a valid folder.")


def get_bluemira_root() -> str:
    """
    Get the bluemira root install folder.

    Returns
    -------
    root: str
        The full path to the bluemira root folder, e.g.:
            '/home/user/code/bluemira'
    """
    import bluemira

    path = list(bluemira.__path__)[0]
    root = os.path.split(path)[0]
    return root


def try_get_bluemira_private_data_root() -> str:
    """
    Get the bluemira-private-data root install folder.

    Returns
    -------
    private_root: Union[None, str]
        The full path to the bluemira root folder, e.g.:
            '/home/user/code/bluemira-private-data'

    Notes
    -----
    Normal users will not have access to bluemira-private-data; it will be used
    exclusively for tests which require private data and files.
    """
    root = get_bluemira_root()
    code_root = os.path.split(root)[0]
    try:
        return _get_relpath(code_root, "bluemira-private-data")
    except ValueError:
        return None


def get_bluemira_path(path: str = "", subfolder: str = "bluemira") -> str:
    """
    Get a bluemira path of a module subfolder. Defaults to root folder.

    Parameters
    ----------
    path: str
        The desired path from which to create a full path
    subfolder: str (default = 'bluemira')
        The subfolder (from the bluemira root) in which to create a path
        Defaults to the source code folder, but can be e.g. 'tests', or 'data'

    Returns
    -------
    path: str
        The full path to the desired `path` in the subfolder specified
    """
    root = get_bluemira_root()
    if "egg" in root:
        return f"/{subfolder}"

    path = path.replace("/", os.sep)
    bpath = _get_relpath(root, subfolder)
    return _get_relpath(bpath, path)


def try_get_bluemira_path(
    path: str = "", subfolder: str = "bluemira", allow_missing=True
):
    """
    Try to get the bluemira path of a module subfolder.

    If the path doesn't exist then optionally carry on regardless or raise an error.

    Parameters
    ----------
    path: str
        The desired path from which to create a full path
    subfolder: str (default = 'bluemira')
        The subfolder (from the bluemira root) in which to create a path
        Defaults to the source code folder, but can be e.g. 'tests', or 'data'
    allow_missing: bool
        Whether or not to raise an error if the path does not exist

    Returns
    -------
    path: Optional[str]
        The full path to the desired `path` in the subfolder specified, or None if the
        requested path doesn't exist.

    Raises
    ------
    ValueError
        If the requested path doesn't exist and the `allow_missing` flag is False.
    """
    try:
        return get_bluemira_path(path, subfolder)
    except ValueError as error:
        if allow_missing:
            return None
        else:
            raise error


def make_bluemira_path(path: str = "", subfolder: str = "bluemira"):
    """
    Create a new folder in the path, provided one does not already exist.
    """
    root = get_bluemira_root()
    if "egg" in root:
        root = "/"
    path = path.replace("/", os.sep)
    bpath = _get_relpath(root, subfolder)
    if bpath in path:
        path = path[len(bpath) :]  # Remove leading edge rootpath
    try:
        return _get_relpath(bpath, path)
    except ValueError:
        os.makedirs(os.sep.join([bpath, path]))
        return _get_relpath(bpath, path)


def force_file_extension(file_path: str, valid_extensions: Union[str, List[str]]) -> str:
    """
    If the file path does not have one of the valid extensions, append the first
    valid one

    Parameters
    ----------
    file_path: str
        path to file
    valid_extensions: Union[str, List[str]]
        collection of valid extensions

    Returns
    -------
    file_path: str

    """
    if isinstance(valid_extensions, str):
        valid_extensions = [valid_extensions]

    if not os.path.splitext(file_path)[1].casefold() in valid_extensions:
        file_path += valid_extensions[0]

    return file_path


def get_files_by_ext(folder: str, extension: str):
    """
    Get filenames of files in folder with the specified extension.

    Parameters
    ----------
    folder: str
        The full path directory in which to look for files
    extension: str
        The extension of the desired file-type

    Returns
    -------
    files: List[str]
        The list of full path filenames found in the folder
    """
    files = []
    for file in os.listdir(folder):
        if file.endswith(extension):
            files.append(file)
    if len(files) == 0:
        from bluemira.base.look_and_feel import bluemira_warn

        bluemira_warn(f"No files with extension {extension} found in folder {folder}")
    return files


def file_name_maker(filename: str, lowercase: bool = False) -> str:
    """
    Ensure the file name is acceptable.

    Parameters
    ----------
    filename: str
        Full filename or path
    lowercase: bool
        Whether or not to force lowercase filenames

    Returns
    -------
    filename: str
        Full filename or path, corrected
    """
    filename = filename.replace(" ", "_")
    if lowercase:
        split = filename.split(os.sep)
        filename = os.sep.join(split[:-1])
        filename = os.sep.join([filename, split[-1].lower()])
    return filename


class FileManager:
    """
    A class for managing file operations.
    """

    _reactor_name: str
    _reference_data_root: str
    _generated_data_root: str

    reference_data_dirs: Dict[str, str]
    generated_data_dirs: Dict[str, str]

    def __init__(
        self,
        reactor_name: str,
        reference_data_root: str = "data/bluemira",
        generated_data_root: str = "data/bluemira",
    ):
        self._reactor_name = reactor_name
        self._reference_data_root = reference_data_root
        self._generated_data_root = generated_data_root
        self.replace_bm_root()

    @property
    def reactor_name(self):
        """
        Gets the reactor name for this instance.
        """
        return self._reactor_name

    @property
    def generated_data_root(self) -> str:
        """
        Gets the generated data root directory for this instance.
        """
        return self._generated_data_root

    @property
    def reference_data_root(self) -> str:
        """
        Get the reference data root directory for this instance.
        """
        return self._reference_data_root

    def replace_bm_root(self, keyword=BM_ROOT):
        """
        Replace the keyword in input paths with path to local bluemira installation.
        """
        bm_root = get_bluemira_root()
        self._reference_data_root = self.reference_data_root.replace(keyword, bm_root)
        self._generated_data_root = self.generated_data_root.replace(keyword, bm_root)

    def _verify_reference_data_root(self):
        """
        Check that the reference data root defined in this instance is a valid
        directory.

        Raises
        ------
        ValueError
            If the reference data root for this instance is not a valid directory.
        """
        _get_relpath(self._reference_data_root, subfolder="")

    def make_reactor_folder(self, subfolder: str) -> Dict[str, str]:
        """
        Initialise a data storage folder tree.

        Parameters
        ----------
        subfolder: str
            The subfolder of the bluemira directory in which to add the data structure

        Returns
        -------
        mapping: dict
            The dictionary of subfolder names to full paths (useful shorthand)
        """
        root = os.path.join(subfolder, "reactors", self.reactor_name)
        pathlib.Path(root).mkdir(parents=True, exist_ok=True)

        mapping = {"root": root}
        for sub in SUB_DIRS:
            folder = os.sep.join([root, sub])
            pathlib.Path(folder).mkdir(exist_ok=True)

            mapping[sub] = folder

        return mapping

    def set_reference_data_paths(self):
        """
        Generate the reference data paths for this instance, based on the reactor name.
        """
        self._verify_reference_data_root()
        self.reference_data_dirs = self.make_reactor_folder(self._reference_data_root)

    def create_reference_data_paths(self):
        """
        Generate the reference data paths for this instance, based on the reactor name.

        Also builds the relevant directory structure.
        """
        pathlib.Path(self._reference_data_root).mkdir(parents=True, exist_ok=True)
        self.reference_data_dirs = self.make_reactor_folder(self._reference_data_root)

    def create_generated_data_paths(self):
        """
        Generate the generated data paths for this instance, based on the reactor name.

        Also builds the relevant directory structure.
        """
        pathlib.Path(self._generated_data_root).mkdir(parents=True, exist_ok=True)
        self.generated_data_dirs = self.make_reactor_folder(self._generated_data_root)

    def build_dirs(self, create_reference_data_paths: bool = False):
        """
        Create the directory structures for this instance and sets the path references.
        """
        if create_reference_data_paths:
            self.create_reference_data_paths()
        else:
            self.set_reference_data_paths()
        self.create_generated_data_paths()

    def get_path(self, sub_dir_name: str, path: str, make_dir: bool = False) -> str:
        """
        Get a path within the generated data sub-sdirectories.

        If the path does not exist then it will optionally be created as a directory.

        Parameters
        ----------
        sub_dir_name: str
            The name of the sub-directory to create the path under. Must be one of the
            names in bluemira.base.file.SUB_DIRS.
        path: str
            The path to create under the sub-directory.
        make_dir: bool
            Optionally create a directory at the path, by default False.

        Returns
        -------
        path: str
            The path within the data sub-directories.
        """
        path = os.sep.join(
            [self.generated_data_dirs[sub_dir_name], path.replace("/", os.sep)]
        )
        if make_dir:
            if not os.path.isdir(path):
                os.makedirs(path)
        return path
