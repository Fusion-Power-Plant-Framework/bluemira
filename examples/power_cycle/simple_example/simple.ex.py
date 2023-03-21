# %%
# COPYRIGHT PLACEHOLDER

"""
Time evolution of production and consumption power loads in simplified
power plant using the Power Cycle module.
"""

# Run with:
# python examples/power_cycle/simple_example/simple.ex.py

# %%
import os
from pprint import pprint

import matplotlib.pyplot as plt
from tabulate import tabulate

from bluemira.base.file import get_bluemira_root
from bluemira.power_cycle.net.manager import PowerCycleManager
from bluemira.power_cycle.time import ScenarioBuilder
from bluemira.power_cycle.tools import adjust_2d_graph_ranges, read_json, validate_axes

# %% [markdown]
# # Base inputs

# %%
# Path of BLUEMIRA project
project_path = get_bluemira_root()

# Directory elements of Power Cycle and Simple Example paths
examples_crumbs = (project_path, "examples", "power_cycle")
simple_example_crumbs = examples_crumbs + tuple(["simple_example"])

# Plot preparetion functions


def prepare_plot(figure_title):
    """Prepare axes for an example plot."""
    ax = validate_axes()
    plt.grid()
    plt.title(figure_title)
    return ax


def finalize_plot(ax):
    """Finalize axes and display example plot."""
    adjust_2d_graph_ranges(ax=ax)
    plt.show()
    return ax


# %% [markdown]
# # Build a scenario
#
# The simplified scenario is built from the `scenario_config.json` file,
# in which a single "standard pulse" (`std`) is applied a single time.
#
# Pulse types are found in the "pulse_library" JSON field. The `std`
# pulse is defined as being composed of 4 phases, each of which can be
# found in the "phase-library" JSON field.
#
# Phases are contructed either summing (logical `"&"`) or taking the
# largest (logical `"|"`) time periods defined in the "breakdown-library"
# JSON field.
#
# At the end of this secion, the total duration of the pulse, and of
# each of its phases, is displayed in tabulated format.
#

# %%
# Scenario inputs
scenario_config_filename = tuple(["scenario_config.json"])
scenario_config_crumbs = examples_crumbs + scenario_config_filename
scenario_config_path = os.path.join(*scenario_config_crumbs)

# Build a scenario
scenario_builder = ScenarioBuilder(scenario_config_path)

# Gather durations that make up phases in scenario
phase_library = scenario_builder.scenario.build_phase_library()
phase_durations = {p.name: p.duration for p in phase_library.values()}

# Gather durations that make up pulses in scenario
pulse_library = scenario_builder.scenario.build_pulse_library()
pulse_durations = {p.name: p.duration for p in pulse_library.values()}

# Visualization configuration
phase_headers = ["Phase", "duration (s)"]
pulse_headers = ["Pulse", "duration (s)"]
phase_table = tabulate(phase_durations.items(), headers=phase_headers)
pulse_table = tabulate(pulse_durations.items(), headers=pulse_headers)

# Visualize scenario configuration
print(phase_table)
print(pulse_table)

# %% [markdown]
# # Set up Power Cycle Manager to compute net power loads
#
# During its initialization, the `PowerCycleManager` class requires 2
# parameters: the path to the `scenario_config.json` file and the path
# to the `manager_config.json` file. The first is used to build the
# scenario, as presented before.
#
# The second is used to specify the paths of different input files used
# to characterize the different "groups" of systems in the power plant,
# each of which can have an arbitrary number of power loads associated
# with it.
# In this example we see three groups are initialized in the manager:
# - Heating & Current Drive (`loads_HCD.json` file);
# - Magnetic (`loads_MAG.json` file);
# - Balance-of-Plant (`loads_BOP.json` file).
#
# The "systems" field specifies which systems defined in the input files
# should be initialized.
#

# %%
# Power Cycle manager configuration file
manager_config_filename = tuple(["manager_config.json"])
manager_config_crumbs = simple_example_crumbs + manager_config_filename
manager_config_path = os.path.join(*manager_config_crumbs)

# Read manager configuration file
manager_config = read_json(manager_config_path)

# Visualize manager configuration
pprint(manager_config)

# %% [markdown]
# # Group configuration files
#
# Each group configuration file characterizes different systems in that
# group. In this example, the HCD file includes inputs for a single
# system ("ECH", or Electron Cyclotron Heating), which is configured
# with two kinds of loads:
# - an upkeep load (`upk` in the file), and
# - an injection load (`inj` in the file).
#
# Both loads are not imported from any other module, so the field
# "module" in the configuration file contains the value `none`. The
# "variables_map" field contains all information necessary to build
# all PhaseLoad instances associated with each load kind.
#
# For the `upk` load, in the order displayed below:
# - "phases" indicates in which phase that load must be maintained;
# - "normalize" indicates whether its time length must be normalized to
#   the duration of that phase;
# - "unit" indicates the unit of the values in "data";
# - "consumption" indicates whether those values will be considered
#   negative when computing the net power load;
# - "efficiencies" indicates a list of scalar values that should
#   (a) divide a consumption load or (b) multiply a production load;
# - "loads" contains all the parameters necessary to build `LoadData`
#   and `LoadModel` instances to characterize each `PhaseLoad` with
#   `PowerLoad` instances.
#
# A similar set of inputs can be seen for the `inj` load, although its
# load is only maintained during the flat-top

# %%
# Heating & Current Drive group configuration file
hcd_config_path_from_manager_file = manager_config["HCD"]["config_path"]
hcd_config_crumbs = (project_path, hcd_config_path_from_manager_file)
hcd_config_path = os.path.join(*hcd_config_crumbs)

# Read group configuration file
hcd_config = read_json(hcd_config_path)

# Visualize group configuration
pprint(hcd_config, width=90, sort_dicts=False)

# %% [markdown]
# # Compute net power loads
#
# A similar set of configurations can be given for other system groups,
# such as Magnetic and Balance-of-Plant systems. For this simple plant
# example:
# - MAG loads are reduced to a (consumption) electrical power by all
#   coils and the Central Solenoid Coils recharge during dwell.
# - BOP loads are reduced to (consumption) pumping powers in the Primary
#   Heat Transfer System (PHTS) and a (production) power in the Power
#   Conversion System computed for a generic Rankine Cycle after some
#   power (~10%) has been stored in the IHTS for the Dwell phase.
#
# Of note, only "active" power loads are considered in this example.
#

# %%
# Create the manager
manager = PowerCycleManager(scenario_config_path, manager_config_path)

# Plot net loads
ax = prepare_plot("Net Power Loads")
ax, _ = manager.plot(ax=ax)
ax = finalize_plot(ax)

# %% [markdown]
# # Analyze active pulse load
#
# One can study any part of the pulse load in more detail. The property
# `net_active` of the manager contains the `PulseLoad` instance that is
# composed by all `PhaseLoad` objects built during the initialization of
# the manager.
#

# %%
# Active load
active_pulseload = manager.net_active

# Plot active loads by phase, in green
active_phaseloads = active_pulseload.phaseload_set
for phaseload in active_phaseloads:
    phase_name = phaseload.phase.name
    ax = prepare_plot(phase_name)
    ax, _ = phaseload.plot(ax=ax, c="g")
    ax = finalize_plot(ax)

# %% [markdown]
# # Analyze individual phase load
#
# Additionally, each `PhaseLoad` object is a "resulting" load, composed
# of multiple power loads that can be plotted individually. Each power
# load that composes the resulting load can plotted in dashed lines in
# the same graph by using the `detailed` parameter.
#
# The same can be done to an individual `PowerLoad` object, to verify
# that particular input without time normalization due to the phase
# length or time shift due to phase ordering in the pulse.
# In this example, all individual `PowerLoad` objects have a single
# `LoadData` object as parameter, so data points coincide with the
# `PowerLoad` curve, as can be seen by the different colors in the
# "island control" plot.
#

# %%
# Chose flat-top phase by its label
label = "ftt"

# Retrieve `PhaseLoad` for flat-top phase
ftt_load = [load for load in active_phaseloads if load.phase.label == label]
ftt_load = ftt_load[0]

# Plot detailed `PhaseLoad` for flat-top
phase_name = ftt_load.phase.name
ax = prepare_plot(f"Detailed plot of {phase_name!r} load")
ax, _ = ftt_load.plot(ax=ax, detailed=True)
ax = finalize_plot(ax)

# Retrieve ECH `PowerLoad` that characterizes "island control"
load_name = "island control"
ftt_powerloads = ftt_load.powerload_set
control_load = [load for load in ftt_powerloads if load.name == load_name]
control_load = control_load[0]

# Plot detailed "island control" load during flat-top
ax = prepare_plot(f"Detailed plot of {phase_name!r} load for {load_name!r}")
ax, _ = control_load.plot(ax=ax, c="m", detailed=True)
ax = finalize_plot(ax)
