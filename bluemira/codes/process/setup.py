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
PROCESS setup functions
"""

import os

# PROCESS imports
from bluemira.codes.process.api import DEFAULT_INDAT, InDat


class PROCESSInputWriter(InDat):
    """
    BLUEPRINT IN.DAT writer for PROCESS input.

    Parameters
    ----------
    template_indat: str
        Path to the IN.DAT file to use as the template for PROCESS parameters.
    """

    def __init__(self, template_indat=DEFAULT_INDAT):
        if os.path.isfile(template_indat):
            # InDat autoloads IN.DAT without checking for existence
            super().__init__(filename=template_indat)
        else:
            super().__init__(filename=None)
            self.filename = template_indat


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
