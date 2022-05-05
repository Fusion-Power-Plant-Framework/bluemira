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

from unittest.mock import patch

from bluemira.codes.process import api

PROCESS_OBS_VAR = {
    "ni": "ni wang",
    "ni wang": "ni peng",
    "garden": "shrubbery",
}


@patch("bluemira.codes.process.api.OBS_VARS", PROCESS_OBS_VAR)
def test_update_obsolete_vars():
    str1 = api.update_obsolete_vars("ni")
    str2 = api.update_obsolete_vars("garden")
    assert str1 == "ni peng" and str2 == "shrubbery"
