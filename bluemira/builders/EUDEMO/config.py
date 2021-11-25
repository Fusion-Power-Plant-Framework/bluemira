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
Configuration derivation and storage for EU-DEMO build
"""

import json

from bluemira.base.config import Configuration

params = {}
for param in Configuration.params:
    if param[4] is None and len(param) == 6:
        params[param[0]] = param[2]
    else:
        params[param[0]] = {}
        params[param[0]]["value"] = param[2]
        if param[4] is not None:
            params[param[0]]["description"] = param[4]
        if len(param) == 7:
            params[param[0]]["mapping"] = {
                key: value.to_dict() for key, value in param[6].items()
            }

params["Name"] = "EUDEMO"

with open("/home/dshort/code/bluemira/bluemira/builders/EUDEMO/params.json", "w") as fh:
    json.dump(params, fh, indent=2)

build_config = {
    "generated_data_root": "!BP_ROOT!/generated_data",
    "process_mode": "read",
}
