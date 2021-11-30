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
    params[param[0]] = {}
    params[param[0]]["value"] = param[2]
    params[param[0]]["source"] = param[5]
    if param[4] is not None:
        params[param[0]]["description"] = param[4]
    if len(param) == 7:
        params[param[0]]["mapping"] = {
            key: value.to_dict() for key, value in param[6].items()
        }

params = dict(sorted(params.items()))

params["Name"]["value"] = "EU-DEMO"
params["tau_flattop"]["value"] = 6900
params["n_TF"]["value"] = 18
params["fw_psi_n"]["value"] = 1.06
params["tk_tf_front_ib"]["value"] = 0.05
params["tk_bb_ib"]["value"] = 0.755
params["tk_bb_ob"]["value"] = 1.275
params["g_tf_pf"]["value"] = 0.05
params["C_Ejima"]["value"] = 0.3
params["eta_nb"]["value"] = 0.4
params["LPangle"]["value"] = -15
params["w_g_support"]["value"] = 1.5

with open("/home/dshort/code/bluemira/bluemira/builders/EUDEMO/params.json", "w") as fh:
    json.dump(params, fh, indent=2)

build_config = {
    "generated_data_root": "!BM_ROOT!/generated_data",
    "process_mode": "mock",
}
