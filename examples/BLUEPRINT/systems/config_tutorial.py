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
Some examples on updating and extending Configurations
"""

# %%[markdown]
# # Configuration Tutorial
# This tutorial goes through different ways of modifying the BLUEPRINT configuration.
#
# The BLUEPRINT configuration controls the initial state of the physical parameters that
# setup the BLUEPRINT run for a given reactor design.

# %%
from bluemira.base.parameter import Parameter, ParameterFrame

from BLUEPRINT.base.config_schema import ConfigurationSchema
from BLUEPRINT.base.file import get_bluemira_root
from BLUEPRINT.reactor import Reactor
from BLUEPRINT.systems.config import Configuration

c = Configuration()
c.to_dict()

# %%[markdown]
# ## The Core BLUEPRINT Configuration
#
# BLUEPRINT has a default (core) configuration, which defines the physical parameters
# that control how the built-in functionality of BLUEPRINT works. It has default
# values provided for all of those parameters.

# %%
core_c = Configuration()

# %%[markdown]
# The configuration is based on the concept of a ParameterFrame. You can think of this as
# a table of parameters with their names and values, but also additional data that gives
# more description to the parameters without affecting their behaviour (that kind of data
# is called metadata). We can inspect the configuration parameters as a table simply by
# printing it out.

# %%
print(core_c)

# %%[markdown]
# We can easily explore the parameters and their metadata, either in your favourite IDE
# or in the python interpreter.

# %%
# You can explore the available parameters in your IDE by uncommenting the below
# line and adding a dot (core_c.)
# core_c

# %%
# Or we can get the parameter names as keys:
core_c.keys()

# %%
# We can see the value of a parameter by accessing it directly:
core_c.R_0

# %%
# Or we can get the underlying parameter by accessing via the get function:
param = core_c.get_param("R_0")

# %%
# And we can look at the metadata on the parameter, for example as a dictionary:
param.to_dict()

# %%[markdown]
# What if we want to change a configuration value though? Just being able to use the
# default values would make for some pretty boring analysis. If we're running an
# interactive analysis then we might just want to change a few values by hand.

# %%
core_c.R_0 = 8.5
core_c.Name = "My reactor"

# %%[markdown]
# That's great for exploring some settings, but not so good if we want to be able to
# look back on what we've done later. When you've settled on some values the it might
# be useful to create a custom class to codify your design parameters - you can see
# examples of that in the BLUEPRINT/systems/config.py file for DoubleNull and Spherical
# configurations, or below to see how the parameters we've changed above would be set
# in a configuration class.


# %%
class MyConfiguration(Configuration):
    """
    A configuration class that sets a couple of parameters to custom values.
    """

    new_params = {"P_el_net": 450, "Name": "My reactor"}

    def __init__(self, default_params=Configuration.params, custom_params=new_params):
        super().__init__(default_params, custom_params)


my_config = MyConfiguration()
print(my_config.P_el_net)
print(my_config.Name)

# %%[markdown]
# We can then define a reactor that uses our custom configuration to set its default
# parameters.


# %%
class MyReactor(Reactor):
    """
    My custom reactor class.
    """

    config: dict
    build_config: dict
    build_tweaks: dict
    default_params = MyConfiguration().to_records()


# %%[markdown]
# And we can set the reactor configuration values for each run.

# %%
config_run1 = {"A": 3.0}
config_run2 = {"A": 2.9}
config_run3 = {"A": 2.8}
config_run4 = {"A": 2.7}

# %%[markdown]
# The configuration of the physical parameters for the run are kept distinct from
# the configuration that controls how the run will be executed, know as build config.
# We'll need to define one of those to be able to define our reactor for each run.

# %%
build_config = {
    "reference_data_root": "data/BLUEPRINT",
    "generated_data_root": "generated_data/BLUEPRINT",
    "plot_flag": False,
    "process_mode": "read",
    "plasmod_mode": "read",
    "plasma_mode": "run",
    "tf_mode": "run",
    # TF coil config
    "TF_type": "S",
    "TF_objective": "L",
    # FW and VV config
    "VV_parameterisation": "S",
    "FW_parameterisation": "S",
    "BB_segmentation": "radial",
    "lifecycle_mode": "life",
    # Plasmod modes
    "eq_mode": "I_p",  # Or 'q_95'
    "HCD_method": "power",
}

# %%[markdown]
# If we want, we can also define some build tweak parameters, for example if needed to
# stabilise some of BLUEPRINT's optimisers

# %%
build_tweaks = {
    # TF coil optimisation tweakers (n ripple filaments)
    "nr": 1,
    "ny": 1,
    "nrippoints": 20,  # Number of points to check edge ripple on
}

# %%[markdown]
# Now we can define our reactors for each run. You could call the build() function if
# you'd like to see how the runs execute and the results that they produce, but we won't
# do that as part of this tutorial.

# %%
r_run1 = MyReactor(config_run1, build_config, build_tweaks)
r_run2 = MyReactor(config_run2, build_config, build_tweaks)
r_run3 = MyReactor(config_run3, build_config, build_tweaks)
r_run4 = MyReactor(config_run4, build_config, build_tweaks)

# %%[markdown]
# ## Extracting and Importing Configuration Parameters
#
# The method for defining our configuration that we've used so far is fine if we're
# writing our configuration in Python. However, it may be convenient to save and load
# configuration parameters from text files. This is supported in BLUEPRINT in JSON
# format.
#
# ### Extracting a BLUEPRINT configuration
#
# Let's first look at how to extract a configuration from BLUEPRINT. This can be done in
# two levels of verbosity: concise and verbose. The concise level just gives the
# parameter names and their values, while the verbose level expands each parameter with
# it's metadata. The default verbosity level is concise.

# %%
concise_json = core_c.to_json()
verbose_json = core_c.to_json(verbose=True)

# %%[markdown]
# It is also possible to write the configuration to JSON files.

# %%
core_c.to_json(
    output_path=f"{get_bluemira_root()}/examples/BLUEPRINT/systems/config_data/core_c_concise_out.json"
)
core_c.to_json(
    output_path=f"{get_bluemira_root()}/examples/BLUEPRINT/systems/config_data/core_c_verbose_out.json",
    verbose=True,
)

# %%[markdown]
# ### Loading a BLUEPRINT configuration
#
# We can also load configurations from JSON files. In this case the verbose and concise
# configuration formats perform two different roles.
#
# *  Verbose configurations allow the full configuration to be defined, along with
#    metadata;
# *  Concise configurations allow values to be set on defined paramters.
#
#
# Let's take a look at loading a verbose configuration first.


# %%
class MyVerboseReactor(Reactor):
    """
    My custom reactor class using a verbose configuration loaded from file.
    """

    config: dict
    build_config: dict
    build_tweaks: dict
    default_params = Configuration.from_json(
        f"{get_bluemira_root()}/examples/BLUEPRINT/systems/config_data/verbose_in.json"
    ).to_records()


# Make a reactor that just uses the default configuration values
r_verbose = MyVerboseReactor({}, build_config, build_tweaks)

# %%[markdown]
# Next let's load a concise configuration. Notice how in this case we need a default set
# of configuration values that have their values set using the concise configuration
# format. This allows the default metadata to be used, but the specific values in the
# configuration are set. In turn, this allows multiple analyses to be loaded from
# different files, allowing a record of those analyses to be held in JSON format.


# %%
class MyConciseReactor(Reactor):
    """
    My custom reactor class using a concise configuration loaded from file.
    """

    config_path: str
    config: dict
    build_config: dict
    build_tweaks: dict
    default_params = Configuration().to_records()

    def __init__(self, config_path, build_config, build_tweaks):
        config = Configuration().set_values_from_json(config_path).to_dict()
        super().__init__(config, build_config, build_tweaks)


# Make a reactor that loads all the configuration values from a file
r_concise = MyConciseReactor(
    f"{get_bluemira_root()}/examples/BLUEPRINT/systems/config_data/concise_in.json",
    build_config,
    build_tweaks,
)

# %%[markdown]
# In this case we've set all the values in our configuration, which is great for knowing
# exactly what we ran with, for example if the default values change as some point in
# the future. However, that may be overkill if we're making some exploratory runs, so
# it's also possible to a partial concise configuration that only sets the values that
# we want to change.

# %%
r_partial = MyConciseReactor(
    f"{get_bluemira_root()}/examples/BLUEPRINT/systems/config_data/partial_in.json",
    build_config,
    build_tweaks,
)

# %%[markdown]
# The configuration is used to set the params attribute on the reactor, so we can
# extract that back to JSON if we've setted on the values that we want to run with
# and want to extract the full set.

# %%
r_partial.params.to_json(
    f"{get_bluemira_root()}/examples/BLUEPRINT/systems/config_data/partial_full_out.json"
)

# %%[markdown]
# ## Extending the BLUEPRINT configuration
#
# It is possible to add new parameters or to define new values for parameters by
# extending the core BLUEPRINT configuration. This section gives some examples to
# show how you might go about doing that.
#
# ### Schema Definition
#
# The configuration schema tells BLUEPRINT what parameters are available for that run.
# The basic set of parameters are defined in the `ConfigurationSchema` class. However,
# it is possible to add new parameters with a custom schema class, which inherits from
# the basic `ConfigurationSchema`. This could be useful if you'd want to provide
# additional functionality in BLUEPRINT that needs your new parameter.
#
# ---
# **Note**
# The schema class should only contain parameters, it should not contain any functions.
# ---


# %%
class MyExtendedConfigurationSchema(ConfigurationSchema):
    """
    A custom schema that extends the core BLUEPRINT configuration with my_new_parameter.
    """

    my_new_parameter: Parameter


# %%[markdown]
# ### Configuration definition
#
# Once you've defined your configuration schema, you can then use it in a Configuration
# class. If you're extending the base configuration class, which will be most likely,
# then you can define the default parameters for your configuration by extending the
# default_params from the core Configuration class. You can also define custom values
# for your specific Configuration class.


# %%
class MyExtendedConfiguration(MyExtendedConfigurationSchema, ParameterFrame):
    """
    A custom configuration that extends the core BLUEPRINT configuration with
    my_new_parameter and sets a custom major radius value.
    """

    # Add our new parameter and default value to the list of default parameters
    params = Configuration.params + [["my_new_parameter", "Super Awesome Config!", 42]]

    new_params = {"R_0": 6}

    def __init__(self, default_params=params, custom_params=new_params):
        super().__init__(default_params)
        if custom_params is not None:
            self.add_parameters(custom_params)


# %%[markdown]
# If you want a new configuration that uses the same default parameters but sets
# some different custom parameter values then you can define a class that sets
# these.

# %%
class MyNewConfiguration(MyExtendedConfiguration):
    """
    A custom class that sets some different custom parameter values.
    """

    new_params = {
        **MyExtendedConfiguration.new_params,
        "Name": "Super Reactor!",
        "my_new_parameter": 999,
    }

    def __init__(
        self, default_params=MyExtendedConfiguration.params, custom_params=new_params
    ):
        super().__init__(default_params)
        if custom_params is not None:
            self.add_parameters(custom_params)


# %%[markdown]
# Or you can set the custom params when you create an instance of that class.


# %%
my_new_config = MyExtendedConfiguration(custom_params=MyNewConfiguration.new_params)

# %%[markdown]
# We can check the two methods are equivalent.

# %%
my_new_config.to_records() == MyNewConfiguration().to_records()
