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

"""
File I/O functions and some path operations
"""

import os


def _get_relpath(folder, subfolder):
    path = os.sep.join([folder, subfolder])
    if os.path.isdir(path):
        return path
    else:
        raise ValueError(f"{path} Not a valid folder.")


def get_bluemira_root():
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


def get_bluemira_path(path="", subfolder="bluemira"):
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


def try_get_bluemira_path(path="", subfolder="bluemira", allow_missing=True):
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


def make_bluemira_path(path="", subfolder="bluemira"):
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


def get_files_by_ext(folder, extension):
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
