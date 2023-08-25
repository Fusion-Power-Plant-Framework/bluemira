# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
Initialise the bluemira package.
"""

import os
import sys

from . import _version  # noqa: ABS101

# this a change
if os.environ.get("BLUEMIRA_ENV", "") == "CONTAINER":
    sys.path.append("/usr/lib/freecad/Ext")
    sys.path.append("/usr/lib/freecad/Mod/Part")
    sys.path.append("/usr/lib/freecad/Mod/Draft")
    sys.path.append("/usr/lib/python3/dist-packages")

__version__ = _version.get_versions()["version"]
