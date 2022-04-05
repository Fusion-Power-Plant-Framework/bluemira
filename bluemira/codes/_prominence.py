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
Prominence API
"""

import sys
from pathlib import Path
from unittest.mock import patch

from bluemira.base.look_and_feel import bluemira_debug, bluemira_print, bluemira_warn
from bluemira.utilities.tools import get_module


class ProminenceDownloader:
    """
    Object to import prominence's binary file to enable job data downloading.

    the binary file has a lot of python code directly in it that is
    not accessible from the prominence module.

    Parameters
    ----------
    jobid: int
      prominence jobid
    save_dir: str
        directory to save the jetto output
    force: bool
        overwrite existing files

    Notes
    -----
    This is useful in the future for parameter scans are accessing remote data.
    We do not currently use any kind of stable API.
    Hopefully the prominence module will be updated in future to not have
    the download function only exist within the 'binary' file. That would allow
    us to use the module directly instead of this hack
    """

    BINARY = "prominence"

    def __init__(self, jobid, save_dir, force=False):
        self.id = jobid
        self.dir = False
        self.force = force

        self._save_dir = save_dir
        self._old_open = open

        self._prom_bin = self._load_binary()

    def __call__(self):
        """
        Download the data from a run.

        Temporarily changes directory so that saving happens where desired
        and not in working directory.

        """
        # Monkey patching in production code is obviously horrible
        # there doesnt seem a nice way currently to modify where
        # files are saved and printing verbosity and use in logging
        with patch("builtins.print", new=self.captured_print):
            with patch("builtins.open", new=self.captured_open):
                self._prom_bin.command_download(self)

    def _load_binary(self):
        """
        Import the prominence binary directly.

        There are a lot of python functions
        that do not exist in the main module
        """
        for path in self._get_binary_path():
            try:
                bluemira_debug("Loading {filepath}")
                return get_module(str(path))
            except ImportError:
                continue

    def _get_binary_path(self):
        """
        Find all paths to prominence binaries
        """
        # emulates `which -a prominence` in the python path(s)
        # shutil.which doesnt seem to have the all option
        for path in sys.path:
            filepath = Path(path, self.BINARY)
            if filepath.is_file():
                yield filepath

        raise ImportError("Valid prominence binary not found in sys.path")

    def captured_print(self, string, *args, **kwargs):
        """
        Capture prominence print statements to feed them into our logging system

        Parameters
        ----------
        string: str
            string to print`
        *args
           builtin print statement args
        *kwargs
           builtin print statement kwargs

        """
        if string.startswith("Error"):
            bluemira_warn(f"Prominence {string}")
        else:
            bluemira_print(string)

    def captured_open(self, filepath, *args, **kwargs):
        """
        Prepend save directory to filepath

        Parameters
        ----------
        filepath: str
            filepath
        *args
           builtin open statement args
        *kwargs
           builtin open statement kwargs

        Returns
        -------
        filehandle

        """
        filepath = Path(self._save_dir, filepath)
        return self._old_open(filepath, *args, **kwargs)
