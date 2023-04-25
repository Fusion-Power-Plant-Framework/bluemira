# COPYRIGHT PLACEHOLDER

"""
Useful variables and functions for running examples in the Power Cycle module.
"""

import os
from pprint import pprint

import matplotlib.pyplot as plt

from bluemira.base.file import get_bluemira_root
from bluemira.power_cycle.net.manager import PowerCycleManager
from bluemira.power_cycle.time import ScenarioBuilder
from bluemira.power_cycle.tools import adjust_2d_graph_ranges, validate_axes


class PathKit:
    """
    Kit of methods to get paths for example files.
    """

    # Path of BLUEMIRA project
    project_path = get_bluemira_root()

    # Directory elements of Power Cycle module examples
    examples_crumbs = [project_path, "examples", "power_cycle"]
    simple_folder = "simple_example"
    eudemo17_folder = "EUDEMO17_example"

    @staticmethod
    def path_from_crumbs(*args):
        """Return file path from directory elements."""
        crumbs = []
        for element in args:
            if isinstance(element, str):
                crumbs.append(element)
            elif isinstance(element, tuple):
                crumbs = crumbs + list(element)
            elif isinstance(element, list):
                crumbs = crumbs + element
        return os.path.join(*tuple(crumbs))


class DisplayKit:
    """
    Kit of methods to display example results.
    """

    @staticmethod
    def p(results):
        """Print variables with a"""
        print("\n")
        print(results)

    @staticmethod
    def pp(argument):
        """Print variables with the pprint package"""
        print("\n")
        pprint(argument, width=90, sort_dicts=False)

    @staticmethod
    def prepare_plot(figure_title):
        """Prepare axes for an example plot."""
        ax = validate_axes()
        plt.grid()
        plt.title(figure_title)
        return ax

    @staticmethod
    def finalize_plot(ax):
        """Finalize axes and display example plot."""
        adjust_2d_graph_ranges(ax=ax)
        plt.show()
        return ax


class ScenarioKit:
    """
    Kit of methods to build and visualize scenarios.
    """

    scenario_config_filename = "scenario_config.json"

    @classmethod
    def build_scenario_config_path(cls):
        """
        Build scenario configuration path.
        """
        scenario_config_path = PathKit.path_from_crumbs(
            PathKit.examples_crumbs,
            cls.scenario_config_filename,
        )
        return scenario_config_path

    @classmethod
    def build_scenario(cls):
        """
        Build scenario by calling the 'ScenarioBuilder' class.
        """
        scenario_config_path = cls.build_scenario_config_path()
        scenario_builder = ScenarioBuilder(scenario_config_path)
        return scenario_builder.scenario


class ManagerKit:
    """
    Kit of methods to build and visualize Power Cycle managers.
    """

    @staticmethod
    def build_manager(manager_config_path):
        """
        Create the Power Cycle manager.
        """
        scenario_config_path = ScenarioKit.build_scenario_config_path()
        return PowerCycleManager(scenario_config_path, manager_config_path)

    @staticmethod
    def plot_manager(title, manager):
        """
        Plot net loads computed by the Power Cycle manager.
        """
        ax = DisplayKit.prepare_plot(title)
        ax, _ = manager.plot(ax=ax)
        ax = DisplayKit.finalize_plot(ax)
        return ax

    @staticmethod
    def extract_phaseload_for_single_phase(phaseload_set, phase_label):
        """
        Extract 'PhaseLoad' from list that has a 'phase' attribute whose
        label matches 'phase_label'.
        """
        match = [p for p in phaseload_set if p.phase.label == phase_label]
        phaseload_of_single_phase = match[0]
        return phaseload_of_single_phase

    @staticmethod
    def plot_detailed_phaseload(phaseload):
        """
        Plot 'PhaseLoad' in 'detailed' mode.
        """
        phase_name = phaseload.phase.name
        ax = DisplayKit.prepare_plot(f"Detailed plot of {phase_name!r} load")
        ax, _ = phaseload.plot(ax=ax, detailed=True)
        ax = DisplayKit.finalize_plot(ax)
        return ax


if __name__ == "__main__":
    print("what is happening?")
