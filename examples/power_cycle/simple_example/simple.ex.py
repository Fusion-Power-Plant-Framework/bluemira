# %%
# COPYRIGHT PLACEHOLDER

"""
Time evolution of production and consumption power loads in simplified
power plant using the Power Cycle module.
"""

# Run with:
# python examples/power_cycle/simple_example/simple.ex.py

import import_kits

# %%
from bluemira.power_cycle.net.manager import PowerCycleManager
from bluemira.power_cycle.tools import read_json

try:
    from kits_for_examples import DisplayKit, PathKit, ScenarioKit

    import_kits.successfull_import()

except ImportError:
    import_kits.sys.exit(import_kits.failed_import())


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
def build_manager_config_path():
    """
    Read Power Cycle manager configuration file.
    """
    manager_config_filename = "manager_config.json"
    manager_config_path = PathKit.path_from_crumbs(
        PathKit.examples_crumbs,
        PathKit.simple_folder,
        manager_config_filename,
    )
    return manager_config_path


def read_manager_config(manager_config_path):
    """
    Read Power Cycle manager configuration file.
    """
    manager_config = read_json(manager_config_path)
    return manager_config


def print_manager_config_file(manager_config):
    """
    Visualize manager configuration file.
    """
    DisplayKit.pp(manager_config)


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
def read_hcd_config_from_manager_config(manager_config):
    """
    Read Heating & Current Drive group configuration file.
    """
    hcd_config_path = manager_config["HCD"]["config_path"]
    hcd_config_path = PathKit.path_from_crumbs(
        PathKit.project_path,
        hcd_config_path,
    )
    hcd_config = read_json(hcd_config_path)
    return hcd_config


def print_hcd_config_file(hcd_config):
    """
    Visualize H&CD configuration file.
    """
    DisplayKit.pp(hcd_config)


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
def build_manager(scenario_config_path, manager_config_path):
    """
    Create the Power Cycle manager.
    """
    scenario_config_path = ScenarioKit.build_scenario_config_path()
    manager_config_path = build_manager_config_path()
    manager = PowerCycleManager(scenario_config_path, manager_config_path)
    return manager


def plot_manager(manager):
    """
    Plot net loads computed by the Power Cycle manager.
    """
    ax = DisplayKit.prepare_plot("Net Power Loads")
    ax, _ = manager.plot(ax=ax)
    ax = DisplayKit.finalize_plot(ax)


# %% [markdown]
# # Analyze individual pulse load
#
# One can study any part of the pulse load in more detail. Each of the
# `net_active` and `net_passive` properties of the `PowerCycleManager`
# contain the `PulseLoad` instance that is composed by all `PhaseLoad`
# objects built during the initialization of the manager for that kind
# of load.
#


# %%
def plot_phaseload_set(phaseload_set, color):
    """
    Plot all 'PhaseLoad' objects in a list.
    """
    for phaseload in phaseload_set:
        phase_name = phaseload.phase.name
        title = f"Load in phase {phase_name}"
        ax = DisplayKit.prepare_plot(title)
        ax, _ = phaseload.plot(ax=ax, c=color)
        ax = DisplayKit.finalize_plot(ax)


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
def extract_phaseload_for_single_phase(phaseload_set, phase_label):
    """
    Extract 'PhaseLoad' from list that has a 'phase' attribute whose
    label matches 'phase_label'.
    """
    match = [p for p in phaseload_set if p.phase.label == phase_label]
    phaseload_of_single_phase = match[0]
    return phaseload_of_single_phase


def plot_detailed_phaseload(phaseload):
    """
    Plot 'PhaseLoad' in 'detailed' mode.
    """
    phase_name = phaseload.phase.name
    ax = DisplayKit.prepare_plot(f"Detailed plot of {phase_name!r} load")
    ax, _ = phaseload.plot(ax=ax, detailed=True)
    ax = DisplayKit.finalize_plot(ax)


# %% [markdown]
# # Analyze individual power load
#
# The same can be done to an individual `PowerLoad` object, to verify
# that particular input without time normalization due to the phase
# length or time shift due to phase ordering in the pulse.
# In this example, all individual `PowerLoad` objects have a single
# `LoadData` object as parameter, so data points coincide with the
# `PowerLoad` curve, as can be seen by the different colors in the plot.
#


# %%
def extract_powerload_for_single_name(powerload_set, load_name):
    """
    Extract 'PowerLoad' from list that has a 'name' attribute that
    matches 'load_name'.
    """
    match = [p for p in powerload_set if p.name == load_name]
    powerload_of_single_name = match[0]
    return powerload_of_single_name


def plot_detailed_powerload(powerload):
    """
    Plot 'PowerLoad' in 'detailed' mode.
    """
    load_name = powerload.name
    ax = DisplayKit.prepare_plot(f"Detailed plot of load for {load_name!r}")
    ax, _ = powerload.plot(ax=ax, c="m", detailed=True)
    ax = DisplayKit.finalize_plot(ax)


# %%
if __name__ == "__main__":
    # Set up Power Cycle Manager to compute net power loads
    manager_config_path = build_manager_config_path()
    manager_config = read_manager_config(manager_config_path)
    print_manager_config_file(manager_config)

    # Group configuration files
    hcd_config = read_hcd_config_from_manager_config(manager_config)
    print_hcd_config_file(hcd_config)

    # Compute net power loads
    manager = build_manager(manager_config_path, manager_config_path)
    plot_manager(manager)

    # Analyze active pulse load
    active_pulseload = manager.net_active
    active_phaseload_set = active_pulseload.phaseload_set
    plot_phaseload_set(active_phaseload_set, "g")

    # Analyze "flat-top" phase load
    ftt_active_phaseload = extract_phaseload_for_single_phase(
        active_phaseload_set,
        "ftt",
    )
    plot_detailed_phaseload(ftt_active_phaseload)

    # Analyze "island control" power load of ECH
    ftt_active_powerload_set = ftt_active_phaseload.powerload_set
    island_control_ftt_active_powerload = extract_powerload_for_single_name(
        ftt_active_powerload_set,
        "island control",
    )
    plot_detailed_powerload(island_control_ftt_active_powerload)
