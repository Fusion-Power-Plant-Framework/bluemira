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

# %%
import pprint as pprint

import matplotlib.pyplot as plt

from bluemira.base.components import Component
from bluemira.base.config import Configuration
from bluemira.base.error import ParameterError
from bluemira.base.file import get_bluemira_root
from bluemira.base.logs import set_log_level
from bluemira.base.parameter import ParameterEncoder
from bluemira.builders.cryostat import CryostatBuilder
from bluemira.builders.EUDEMO.blanket import BlanketBuilder
from bluemira.builders.EUDEMO.divertor import DivertorBuilder
from bluemira.builders.EUDEMO.pf_coils import PFCoilsBuilder
from bluemira.builders.EUDEMO.plasma import PlasmaBuilder, PlasmaComponent
from bluemira.builders.EUDEMO.reactor import EUDEMOReactor
from bluemira.builders.EUDEMO.tf_coils import TFCoilsBuilder
from bluemira.builders.radiation_shield import RadiationShieldBuilder
from bluemira.builders.tf_coils import RippleConstrainedLengthGOP
from bluemira.builders.thermal_shield import ThermalShieldBuilder
from bluemira.codes import plot_radial_build
from bluemira.codes.plasmod.mapping import (  # noqa: N812
    create_mapping as create_PLASMOD_mappings,
)
from bluemira.codes.process.mapping import mappings as PROCESS_mappings  # noqa: N812
from bluemira.display.displayer import ComponentDisplayer
from bluemira.equilibria._deprecated_run import AbInitioEquilibriumProblem
from bluemira.utilities.tools import json_writer

# %%[markdown]
# # Configuring and Running an EU-DEMO Design
#
# The bluemira design logic allows reactors to be designed with a variety of parameter
# settings and build configurations. This example shows how the EU-DEMO design can be
# set up, run, and how to analyse and visualise some of the resulting components.
#
# ## Setting up
#
# ### Logging
#
# First we'll set the global logging level to DEBUG so that we can see verbose output.
# For production runs you may want to skip this step to reduce the volume of logging
# output - the default logging level is INFO, and less verbose WARNING, ERROR, and
# CRITICAL levels are also available.

# %%
set_log_level("DEBUG")

# %%[markdown]
# ### Template Parameterisation
#
# Bluemira contains an example set of default parameter settings in the Configuration
# class. This acts as a template for the design parameterisation and allows most values
# to be left as defaults. Note, however, that in the future this functionality will move
# from a class-based configuration into configuration files that avoid the hardcoding of
# default values. To handle this, we can extract the parameters into a dictionary and
# write the template parameterisation to a json file.
#
# We also include the default PROCESS and PLASMOD parameter mappings so that these
# values can also be controlled in our template file.

# %%
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
    else:
        params[param[0]]["mapping"] = {}

        if PROCESS_mappings.get(param[0]) is not None:
            params[param[0]]["mapping"]["PROCESS"] = PROCESS_mappings[param[0]]

        PLASMOD_mappings = create_PLASMOD_mappings()
        if PLASMOD_mappings.get(param[0]) is not None:
            params[param[0]]["mapping"]["PLASMOD"] = PLASMOD_mappings[param[0]]

        if params[param[0]]["mapping"] == {}:
            params[param[0]].pop("mapping")

params = dict(sorted(params.items()))
pprint.pprint(params, indent=2, sort_dicts=False)

# %%[markdown]
# You can see the output of the next cell in template.json in the same directory as this
# notebook.

# %%
json_writer(
    params,
    f"{get_bluemira_root()}/examples/design/EU-DEMO/template.json",
    indent=2,
    cls=ParameterEncoder,
    ensure_ascii=True,
)

# %%[markdown]
# ### Parameter Configuration
#
# For each run of the bluemira design we will likely want to set different values for
# various parameters. So that we don't have to deal with the whole template.json file
# each time, it is convenient to create a parameter config file that sets the values for
# this specific run. This can be done by creating a dictionary that maps the parameter
# names to their new values. Any parameters not listed here will run with their default
# values. You can also map the name of the parameter to a dictionary if you would like
# to add a description to that setting or to change the mapping of a particular parameter
# to an external code.

# %%
config = {
    "Name": "EU-DEMO",
    "tau_flattop": 6900,
    "n_TF": 18,
    "fw_psi_n": 1.07,
    "tk_sol_ib": 0.225,
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

# %%[markdown]
# You can see the output of the next cell in params.json in the same directory as this
# notebook.

# %%
json_writer(
    config,
    f"{get_bluemira_root()}/examples/design/EU-DEMO/params.json",
    indent=2,
    cls=ParameterEncoder,
    ensure_ascii=False,
)

# %%[markdown]
# ### Build Configuration
#
# The bluemira parameters that we've seen so far control how the physical values of
# the target reactor are set. However, we can also change the way in which our bluemira
# design runs by setting values in the build config. These values control how each stage
# in our design is executed, as well as some general values like where reference data
# is read from (reference_data_root) and where data generated by our run will be written
# to (generated_data_root).
#
# Some design stages will have a runmode that allows the design problem being solved to
# be either executed from scratch (run), read from a previously generated file (read), or
# evaluated with some default (mock) values.
#
# For design stages that build based on a parameterised shape, it is possible to set
# what parameterisation class is used (param_class) and how the variables for that
# parameterisation are set (variables_map). The variables map can either set the
# initial value of the variable to be a number or to be derived from a parameter by
# specifying a parameter name. If the shape is being used in a design problem then the
# optimisation settings for the variable can also be changed, such as the lower and upper
# bounds (lower_bound and upper_bound), and whether the variable should be removed from
# the optimisation variables (fixed). The design problem can also be configured by
# setting the optimisation algorithm to be used (algorithm_name), various design
# problem-specific tweaking values (problem_settings), the optimisation conditions
# (opt_conditions) like the maximum number of evaluations, and the optimisation
# parameters (opt_parameters).
#
# Advanced users can also change the design problem being solved by specifying the name
# or module path of the class to use (problem_class). This allows plugin functionality
# to be loaded into bluemira from external packages. Similarly, the shape
# parameterisation (param_class) can be loaded from a module path.

# %%
build_config = {
    "reference_data_root": "!BM_ROOT!/data",
    "generated_data_root": "!BM_ROOT!/generated_data",
    "PROCESS": {
        "runmode": "mock",  # ["run", "read", "mock"]
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
            },
            "f1": {
                "value": 4,
                "lower_bound": 4,
            },
            "f2": {
                "value": 4,
                "lower_bound": 4,
            },
        },
        "algorithm_name": "COBYLA",
        "problem_settings": {
            "n_rip_points": 50,
            "nx": 1,
            "ny": 1,
        },
    },
    "PF Coils": {
        "runmode": "read",
    },
}

# %%[markdown]
# #### Modifying the Build Config
#
# At different stages of creating a design, we may want to use different build config
# settings for different stages. For example, if we want to quickly generate some
# geometry but do not need to solve a specific design problem, we could run the relevant
# stage in mock mode, if we're running in production, or preparing for a production run,
# we could run in run mode, or if we have a run that we want to reload then we could run
# in read mode.
#
# Note that running or reading certain stages may require additional third-party
# libraries to be installed. You may need to request the appropriate permissions and
# licenses to run those codes if they are not open source.
#
# If you have PROCESS installed then change these to enable a PROCESS run or to read
# an existing PROCESS output.

# %%
# build_config["PROCESS"]["runmode"] = "run"
# build_config["PROCESS"]["runmode"] = "read"

# %%[markdown]
# Uncomment one of the following and run the cell to mock the plasma design stage and use
# a parameterised boundary (no equilibrium will be produced in this case), or read the
# reference plasma equilibrium run from an existing file.

# %%
# build_config["Plasma"]["runmode"] = "mock"
# build_config["Plasma"]["runmode"] = "read"

# %%[markdown]
# You can see the output of the next cell in build_config.json in the same directory as
# this notebook.

# %%
json_writer(
    build_config,
    f"{get_bluemira_root()}/examples/design/EU-DEMO/build_config.json",
    indent=2,
)

# %%[markdown]
# ## Creating the Reactor and Running the Design
#
# Now that we have set up our design, we can create and run our Reactor Design object. We
# will be using the EUDEMOReactor, which performs an EU-DEMO like design with the
# following build stages:
#
# - Perform a 0-/1-D design using PROCESS (or load a mock PROCESS output)
# - Design and build an initial plasma shape, based on an equilibrium (or mock shape
#   parameterisation)
# - Design and build the TF coil system, based on a shape parameterisation that is
#   optimised with a minimum length based on a maximum ripple constraint defined by the
#   TF_ripple_limit parameter (or by generating the parameterised shape without solving
#   the design problem)
#
# The design run produces a Component object that contains the tree representation of all
# of the reactor systems that have been build by the design stages.

# %%
reactor = EUDEMOReactor(params, build_config)
component = reactor.run()

# %%[markdown]
# ## Extracting Results and Analyses
#
# Now that we have a designed reactor and the built components, we can visualise the
# results of the various design stages.
#
# ### Viewing the PROCESS Radial Build
#
# If you have performed your design with PROCESS in the "run" runmode, then we can take
# a look at the radial build by inspecting the generated data directory for our systems
# code.

# %%
if build_config["PROCESS"]["runmode"] == "run":
    plot_radial_build(reactor.file_manager.generated_data_dirs["systems_code"])
else:
    print(
        "The PROCESS design stage did not have the runmode set to run."
        "If you have PROCESS installed in your bluemira environment, then try rerunning "
        'after executing build_config["PROCESS"]["runmode"] = "run"'
    )

# %%[markdown]
# ### Accessing Components
#
# Our design has generated a tree of components, with various levels corresponding to
# components that have been built by the Builders invoked at the different design stages.
#
# We can inspect this top-level component to see the tree that we have generated.

# %%
print(component.tree())

# %%[markdown]
# We can also access the different components at various levels by searching through the
# tree.

# %%
plasma: PlasmaComponent = component.get_component("Plasma")
tf_coils = component.get_component("TF Coils")
pf_coils = component.get_component("PF Coils")

# %%[markdown]
# ### Saving the Equilibrium
#
# If we have built our plasma using either the run or read runmode then we can save the
# resulting equilibrium to a file. The Plasma component stores the equilibrium as one of
# its attributes (as it uses the PlasmaComponent type) so we can get the Plasma component
# from the result of our design run and save it using the `to_eqdsk` function. The result
# will be available in the reactor's equilibria folder. By default this will be at:
#
# generated_data/reactors/EU-DEMO/equilibria
#
# The preferred bluemira method for saving equilibria is to use json format as it is
# more descriptive than the traditional eqdsk format.

# %%
directory = reactor.file_manager.generated_data_dirs["equilibria"]
if plasma.equilibrium is not None:
    plasma.equilibrium.to_eqdsk(
        reactor.params["Name"] + "_eqref",
        directory=reactor.file_manager.generated_data_dirs["equilibria"],
    )

# %%[markdown]
# ### Viewing the Equilibrium
#
# If our design has generated an equilibrium, then we can also view it.

# %%
if plasma.equilibrium is not None:
    plasma.equilibrium.plot()
    plt.show()

# %%[markdown]
# ### Display the Plasma Components
#
# Now that we have retrieved our Plasma component, we can also get the various views of
# its underlying components and plot them (for 2D views) or show their CAD (for 3D
# views). Note that the 3D CAD will display in a separate pop-up window.

# %%
plasma.get_component("xz").plot_2d()

# %%
plasma.get_component("xy").plot_2d()

# %%
plasma.get_component("xyz").show_cad()

# %%[markdown]
# ### Viewing the Design Problem Solution
#
# We've seen how to access the results from the components that have been built by our
# design, but we can also interrogate the Builder that was used in our design stage. In
# particular, if we used the run runmode then we can inspect the `design_problem` and
# view the solution that was found.

# %%
plasma_builder = reactor.get_builder("Plasma")
if plasma_builder.runmode == "run":
    eq_problem: AbInitioEquilibriumProblem = reactor.get_builder("Plasma").design_problem
    _, ax = plt.subplots()
    eq_problem.eq.plot(ax=ax)
    eq_problem.constraints.plot(ax=ax)
    eq_problem.coilset.plot(ax=ax)
    plt.show()

# %%[markdown]
# ### Viewing the TF Coils
#
# In the same way that we viewed the resulting plasma, we can also take a look at the
# TF coils that we have built.

# %%
tf_coils.get_component("xy").plot_2d()

# %%
tf_coils.get_component("xz").plot_2d()

# %%
tf_coils.get_component("xyz").show_cad()

# %%[markdown]
# ### Saving the Geometry Parametrisation
#
# If we've run a design with a build stage in the "run" runmode then we may want to save
# the resulting geometry parameterisation to a file so that it can be read back in,
# saving time in future runs if we are only making downstream changes. This can be done
# by using the `save_shape` method on the `TFCoilsBuilder`.

# %%
tf_coils_builder: TFCoilsBuilder = reactor.get_builder("TF Coils")
if tf_coils_builder.runmode == "run":
    tf_coils_builder.save_shape()

# %%[markdown]
# ### Plotting the TF Coils Design Problem
#
# In the same way that we got the design problem from our Plasma Builder used in the
# corresponding design stage, we can also interrogate the design problem from our TF
# Coils build.

# %%
tf_coils_builder: TFCoilsBuilder = reactor.get_builder("TF Coils")
if tf_coils_builder.runmode == "run":
    design_problem: RippleConstrainedLengthGOP = tf_coils_builder.design_problem
    design_problem.plot()
    plt.show()

# %%[markdown]
# ### Visualising the Reactor
#
# Finally we can view the whole Reactor by plotting or showing the combined views for
# all of the components that we have built.

# %%
ax = tf_coils.get_component("xy").plot_2d(show=False)
plasma.get_component("xy").plot_2d(ax=ax, show=False)
blanket = component.get_component("Breeding Blanket")
blanket.get_component("xy").plot_2d(ax=ax, show=False)
pf_coils.get_component("xy").plot_2d(ax=ax)

# %%
ax = tf_coils.get_component("xz").plot_2d(show=False)
plasma.get_component("xz").plot_2d(ax=ax, show=False)

divertor = component.get_component("Divertor")
divertor.get_component("xz").plot_2d(ax=ax, show=False)
blanket.get_component("xz").plot_2d(ax=ax, show=False)
pf_coils.get_component("xz").plot_2d(ax=ax, show=False)

thermal_shield = component.get_component("Thermal Shield")
thermal_shield.get_component("xz").plot_2d(ax=ax, show=False)
cryostat = component.get_component("Cryostat")
cryostat.get_component("xz").plot_2d(ax=ax, show=False)
radiation_shield = component.get_component("Radiation Shield")
radiation_shield.get_component("xz").plot_2d(ax=ax)

# %%
ComponentDisplayer().show_cad(component.get_component("xyz", first=False))

# %%
sector = Component("Segment View")
plasma_builder: PlasmaBuilder = reactor.get_builder("Plasma")
divertor_builder: DivertorBuilder = reactor.get_builder("Divertor")
blanket_builder: BlanketBuilder = reactor.get_builder("Breeding Blanket")
tf_coils_builder: TFCoilsBuilder = reactor.get_builder("TF Coils")
pf_coils_builder: PFCoilsBuilder = reactor.get_builder("PF Coils")
thermal_shield_builder: ThermalShieldBuilder = reactor.get_builder("Thermal Shield")
cryostat_builder: CryostatBuilder = reactor.get_builder("Cryostat")
radiation_shield_builder: RadiationShieldBuilder = reactor.get_builder(
    "Radiation Shield"
)
sector.add_child(plasma_builder.build_xyz(degree=270))
sector.add_child(divertor_builder.build_xyz(degree=270))
sector.add_child(blanket_builder.build_xyz(degree=270))
sector.add_child(tf_coils_builder.build_xyz(degree=270))
sector.add_child(pf_coils_builder.build_xyz(degree=270))
sector.add_child(thermal_shield_builder.build_xyz(degree=270))
sector.add_child(cryostat_builder.build_xyz(degree=270))
sector.add_child(radiation_shield_builder.build_xyz(degree=270))
sector.show_cad()
