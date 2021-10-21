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
from pathlib import Path

from bluemira.base.file import _get_relpath, get_bluemira_root

KEYWORD = "!BP_ROOT!"
SUB_DIRS = ["equilibria", "neutronics", "systems_code", "CAD", "plots", "geometry"]


def file_name_maker(filename, lowercase=False):
    """
    On s'assure ici que le nom du fichier que tu veux creer est acceptable.

    Selon Monsieur McIntosh, les noms de fichiers ne devrait avoir de " ".
    Il y'aura bientot d'autres restrictions, donc on laise cela ici et on vera
    ce que cela donne. O senhor Shimwell não gosta de letras capitulares

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


# =============================================================================
# File finders
# =============================================================================


def get_BP_path(path="", subfolder="BLUEPRINT"):
    """
    Returns a BLUEPRINT path of a module subfolder. Defaults to root folder

    Parameters
    ----------
    path: str
        The desired path from which to create a full path
    subfolder: str (default = 'BLUEPRINT')
        The subfolder (from the BLUEPRINT root) in which to create a path
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


def try_get_BP_path(path="", subfolder="BLUEPRINT", allow_missing=True):
    """
    Try to get the BLUEPRINT path of a module subfolder.

    If the path doesn't exist then optionally carry on regardless or raise an error.

    Parameters
    ----------
    path: str
        The desired path from which to create a full path
    subfolder: str (default = 'BLUEPRINT')
        The subfolder (from the BLUEPRINT root) in which to create a path
        Defaults to the source code folder, but can be e.g. 'tests', or 'data'

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
        return get_BP_path(path, subfolder)
    except ValueError as error:
        if allow_missing:
            return None
        else:
            raise error


def make_BP_path(path="", subfolder="BLUEPRINT"):
    """
    Creates a new folder in the path, provided one does not already exist
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
        # make_BP_path(path)  # Recursao nao funcione..
        return _get_relpath(bpath, path)


class FileManager:
    """
    A class for managing file operations.
    """

    _reactor_name: str
    _reference_data_root: str
    _generated_data_root: str

    reference_data_dirs: dict
    generated_data_dirs: dict

    def __init__(
        self,
        reactor_name,
        reference_data_root="data/BLUEPRINT",
        generated_data_root="data/BLUEPRINT",
    ):
        self._reactor_name = reactor_name
        self._reference_data_root = reference_data_root
        self._generated_data_root = generated_data_root
        self.replace_bp_root()

    @property
    def reactor_name(self):
        """
        Gets the reactor name for this instance.
        """
        return self._reactor_name

    @property
    def generated_data_root(self):
        """
        Gets the generated data root directory for this instance.
        """
        return self._generated_data_root

    @property
    def reference_data_root(self):
        """
        Gets the reference data root directory for this instance.
        """
        return self._reference_data_root

    def replace_bp_root(self, keyword=KEYWORD):
        """
        Replaces keyword in input paths with path to local BLUEPRINT installation.
        """
        bp_root = get_bluemira_root()
        self._reference_data_root = self.reference_data_root.replace(keyword, bp_root)
        self._generated_data_root = self.generated_data_root.replace(keyword, bp_root)

    def _verify_reference_data_root(self):
        """
        Checks that the reference data root defined in this instance is a valid
        directory.

        Raises
        ------
        ValueError
            If the reference data root for this instance is not a valid directory.
        """
        _get_relpath(self._reference_data_root, subfolder="")

    def make_reactor_folder(self, subfolder):
        """
        Initialises a data storage folder tree

        Parameters
        ----------
        subfolder: str
            The subfolder of the BLUEPRINT directory in which to add the data structure

        Returns
        -------
        mapping: dict
            The dictionary of subfolder names to full paths (useful shorthand)
        """
        root = os.path.join(subfolder, "reactors", self.reactor_name)
        Path(root).mkdir(parents=True, exist_ok=True)

        mapping = {"root": root}
        for sub in SUB_DIRS:
            folder = os.sep.join([root, sub])
            Path(folder).mkdir(exist_ok=True)

            mapping[sub] = folder

        return mapping

    def set_reference_data_paths(self):
        """
        Generates the reference data paths for this instance, based on the reactor name.
        """
        self._verify_reference_data_root()
        self.reference_data_dirs = self.make_reactor_folder(self._reference_data_root)

    def create_reference_data_paths(self):
        """
        Generates the reference data paths for this instance, based on the reactor name.
        Also builds the relevant directory structure.
        """
        Path(self._reference_data_root).mkdir(parents=True, exist_ok=True)
        self.reference_data_dirs = self.make_reactor_folder(self._reference_data_root)

    def create_generated_data_paths(self):
        """
        Generates the generated data paths for this instance, based on the reactor name.
        Also builds the relevant directory structure.
        """
        Path(self._generated_data_root).mkdir(parents=True, exist_ok=True)
        self.generated_data_dirs = self.make_reactor_folder(self._generated_data_root)

    def build_dirs(self, create_reference_data_paths=False):
        """
        Creates the directory structures for this instance and sets the path references.
        """
        if create_reference_data_paths:
            self.create_reference_data_paths()
        else:
            self.set_reference_data_paths()
        self.create_generated_data_paths()

    def get_path(self, sub_dir_name, path, make_dir=False):
        """
        Get a path within the generated data sub directories.

        If the path does not exist then it will optionally be created as a directory.

        Parameters
        ----------
        sub_dir_name: str
            The name of the sub-directory to create the path under. Must be one of the
            names in BLUEPRINT.base.file.SUB_DIRS.
        path: str
            The path to create under the sub-directory.
        make_dir: bool
            Optionally create a directory at the path, by default False.
        """
        path = os.sep.join(
            [self.generated_data_dirs[sub_dir_name], path.replace("/", os.sep)]
        )
        if make_dir:
            if not os.path.isdir(path):
                os.makedirs(path)
        return path


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
