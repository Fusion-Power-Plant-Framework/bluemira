# %%
# COPYRIGHT PLACEHOLDER

"""
Construct a simplified scenario to run the Power Cycle module.

Reference for scenario inputs:
------------------------------
F. Franza, Development and Validation of a Computational Tool for
Fusion Reactors' System Analysis, Karlsruher Institut für Technologie,
2019.
"""

from kits_for_examples import DisplayKit, PathKit

# Run with:
# python examples/power_cycle/scenario.ex.py
from tabulate import tabulate

from bluemira.power_cycle.time import ScenarioBuilder

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
# Finally, a custom function can be used to display the total duration
# of the pulse and of each of its phases, displayed in tabulated format.
#


# %%
def build_scenario(scenario_config_path):
    """
    Build scenario by calling the 'ScenarioBuilder' class.
    """
    scenario_builder = ScenarioBuilder(scenario_config_path)
    return scenario_builder.scenario


def print_scenario_summary(scenario):
    """
    Visualize scenario by printing tables.
    """
    # Gather durations that make up phases in scenario
    phase_library = scenario.build_phase_library()
    phase_durations = {p.name: p.duration for p in phase_library.values()}

    # Gather durations that make up pulses in scenario
    pulse_library = scenario.build_pulse_library()
    pulse_durations = {p.name: p.duration for p in pulse_library.values()}

    # Visualization configuration
    phase_headers = ["Phase", "duration (s)"]
    pulse_headers = ["Pulse", "duration (s)"]
    phase_table = tabulate(phase_durations.items(), headers=phase_headers)
    pulse_table = tabulate(pulse_durations.items(), headers=pulse_headers)

    # Visualize scenario configuration
    DisplayKit.p(phase_table)
    DisplayKit.p(pulse_table)


# %%
if __name__ == "__main__":
    # Build scenario
    scenario_config_path = PathKit.build_scenario_config_path()
    scenario = build_scenario(scenario_config_path)

    # Visualize scenario results
    print_scenario_summary(scenario)
