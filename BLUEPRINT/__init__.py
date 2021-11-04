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
BLUEPRINT path, testing, and versioning essentials.
"""
from ._version import get_versions

__version__ = get_versions()["version"]
import pathlib  # noqa
import subprocess  # noqa (S404)
import sys  # noqa
from bluemira.base.file import get_bluemira_root  # noqa

__all__ = ["test", "__version__"]
del get_versions


def test(path=None, *, plotting=False):
    """
    Test utility function to run tests automatically from the source code file
    in IDEs. Will run the tests in the mirrored testing directory.

    Parameters
    ----------
    path: Union[str, None]
        The path to run the tests for. Will default to test_xfile.py for xfile.py
    plotting: bool
        Whether or not to plot the tests
    """
    if path is None:
        # Use the file we were called from
        frame = sys._getframe(1)
        path = frame.f_globals["__file__"]

    # Relative path from project root to the file as a tuple
    parts = pathlib.Path(path).absolute().relative_to(get_bluemira_root()).parts
    # We want the directory path with "tests/BLUEPRINT" instead of "BLUEPRINT" and
    # without the filename
    directory = ("tests/BLUEPRINT",) + parts[1:-1]
    test_file = "test_" + parts[-1]
    full_path = pathlib.Path(get_bluemira_root(), *directory, test_file)
    if not full_path.is_file():
        sys.stderr.write(f'Test file "{full_path}" not found.\n')
        sys.exit(1)

    args = ["pytest", str(full_path)]
    if plotting:
        args.append("--plotting-on")
    subprocess.run(args)  # noqa (S603, S607)


import_path = pathlib.Path(__file__).parent.parent.parent / "PROCESS" / "utilities"

if import_path.is_dir():
    if str(import_path) not in sys.path:
        sys.path.append(str(import_path))
