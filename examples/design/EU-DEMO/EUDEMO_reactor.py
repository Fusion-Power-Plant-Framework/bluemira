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
import matplotlib.pyplot as plt
from bluemira.base.components import Component

from bluemira.base.config import Configuration
from bluemira.base.file import get_bluemira_root
from bluemira.base.logs import set_log_level
from bluemira.base.parameter import ParameterError

from bluemira.builders.EUDEMO.reactor import EUDEMOReactor
from bluemira.builders.EUDEMO.plasma import PlasmaComponent

from bluemira.equilibria.run import AbInitioEquilibriumProblem

from bluemira.codes import plot_PROCESS

# First define the configuration for the run.

set_log_level("DEBUG")

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
    json.dump(params, fh, indent=2, ensure_ascii=True)

config = {
    "Name": "EU-DEMO",
    "tau_flattop": 6900,
    "n_TF": 18,
    "fw_psi_n": 1.06,
    "tk_tf_front_ib": 0.04,
    "tk_tf_side": 0.1,
    "tk_tf_ins": 0.08,
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
    "PROCESS": {
        "runmode": "read",  # ["run", "read", "mock"]
    },
    "Plasma": {
        "runmode": "read",  # ["run", "read", "mock"]
    },
    "TF Coils": {
        "runmode": "run",  # ["run", "read", "mock"]
        "param_class": "TripleArc",
        "variables_map": {
            "x1": {
                "value": "r_tf_in_centre",
                "fixed": True,
            }
        },
    },
}

with open(f"{get_bluemira_root()}/examples/design/EU-DEMO/build_config.json", "w") as fh:
    json.dump(build_config, fh, indent=2, ensure_ascii=False)

# If you have PROCESS installed then change these to enable a PROCESS run or to read
# an existing PROCESS output.

# build_config["process_mode"] = "run"
# build_config["process_mode"] = "read"

# Create the Reactor and run the design.

# Uncomment this to mock the plasma run and use a parameterised boundary.
# (No equilibrium will be produced in this case)

# build_config["plasma_mode"] = "mock"

# Uncomment this to read the reference plasma equilibrium run from an existing file.

# build_config["plasma_mode"] = "read"

reactor = EUDEMOReactor(params, build_config)
component = reactor.run()

# Display the PROCESS radial build.

if build_config["PROCESS"]["runmode"] == "run":
    plot_PROCESS(reactor.file_manager.generated_data_dirs["systems_code"])

# Write out equilibrium, if it's been created.

directory = reactor.file_manager.generated_data_dirs["equilibria"]
plasma: PlasmaComponent = component.get_component("Plasma")
if plasma.equilibrium is not None:
    plasma.equilibrium.to_eqdsk(
        reactor.params["Name"] + "_eqref",
        directory=reactor.file_manager.generated_data_dirs["equilibria"],
    )

# Display the components.

# plasma.get_component("xz").plot_2d()
# plasma.get_component("xy").plot_2d()
# plasma.get_component("xyz").show_cad()

# Display the summary of the equilibrium design problem solved by the Plasma builder.

plasma_builder = reactor.get_builder("Plasma")
if plasma_builder.runmode == "run":
    eq_problem: AbInitioEquilibriumProblem = reactor.get_builder("Plasma").design_problem
    _, ax = plt.subplots()
    eq_problem.eq.plot(ax=ax)
    eq_problem.constraints.plot(ax=ax)
    eq_problem.coilset.plot(ax=ax)
    plt.show()

# Display the reference equilibrium

# if plasma.equilibrium is not None:
#     plasma.equilibrium.plot()
#     plt.show()


tf_coils = component.get_component("TF Coils")
xy = tf_coils.get_component("xy")
xy.plot_2d()
xz = tf_coils.get_component("xz")
xz.plot_2d()
xyz = tf_coils.get_component("xyz")
xyz.show_cad()

# Make some plots combining the various components

Component(
    "xy view", children=[tf_coils.get_component("xy"), plasma.get_component("xy")]
).plot_2d()
Component(
    "xz view", children=[tf_coils.get_component("xz"), plasma.get_component("xz")]
).plot_2d()
Component(
    "xyz view", children=[tf_coils.get_component("xyz"), plasma.get_component("xyz")]
).show_cad()

# Plot the TF coil design problem

builder = reactor.get_builder("TF Coils")
builder.design_problem.plot()
plt.show()
