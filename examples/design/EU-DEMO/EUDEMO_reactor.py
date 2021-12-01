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
Perform the EU-DEMO reactor design.
"""

import json

from bluemira.base.config import Configuration
from bluemira.base.file import get_bluemira_root
from bluemira.base.parameter import ParameterError

from bluemira.builders.EUDEMO.reactor import EUDEMO

from bluemira.codes import plot_PROCESS

# First define the configuration for the run.

params = {}
for param in Configuration.params:
    params[param[0]] = {}
    params[param[0]]["name"] = param[1]
    params[param[0]]["value"] = param[2]
    params[param[0]]["unit"] = param[3]
    params[param[0]]["source"] = param[5]
    if param[4] is not None:
        params[param[0]]["description"] = param[4]
    if len(param) == 7:
        params[param[0]]["mapping"] = {
            key: value.to_dict() for key, value in param[6].items()
        }

params = dict(sorted(params.items()))

with open(f"{get_bluemira_root()}/examples/design/EU-DEMO/template.json", "w") as fh:
    json.dump(params, fh, indent=2, ensure_ascii=False)

config = {
    "Name": "EU-DEMO",
    "tau_flattop": 6900,
    "n_TF": 18,
    "fw_psi_n": 1.06,
    "tk_tf_front_ib": 0.05,
    "tk_bb_ib": 0.755,
    "tk_bb_ob": 1.275,
    "g_tf_pf": 0.05,
    "C_Ejima": 0.3,
    "eta_nb": 0.4,
    "LPangle": -15,
    "w_g_support": 1.5,
}

for key, val in config.items():
    if isinstance(val, dict):
        for attr, attr_val in val.items():
            if attr in ["name", "unit"]:
                raise ParameterError(f"Cannot set {attr} in parameter configuration.")
            params[key][attr] = attr_val
    else:
        params[key]["value"] = val

with open(f"{get_bluemira_root()}/examples/design/EU-DEMO/params.json", "w") as fh:
    json.dump(config, fh, indent=2, ensure_ascii=False)

build_config = {
    "reference_data_root": "!BM_ROOT!/data",
    "generated_data_root": "!BM_ROOT!/generated_data",
    "process_mode": "mock",
}

with open(f"{get_bluemira_root()}/examples/design/EU-DEMO/build_config.json", "w") as fh:
    json.dump(build_config, fh, indent=2, ensure_ascii=False)

# If you have PROCESS installed then change these to enable a PROCESS run or to read
# an existing PROCESS output.

# build_config["process_mode"] = "run"
# build_config["process_mode"] = "read"

# Create the Reactor and run the design.

reactor = EUDEMO(params, build_config)
reactor.run()

# Display the PROCESS radial build.

if build_config["process_mode"] == "run":
    plot_PROCESS(reactor.file_manager.generated_data_dirs["systems_code"])
