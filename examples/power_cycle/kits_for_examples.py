# COPYRIGHT PLACEHOLDER

"""
Useful variables and functions for running examples in the Power Cycle module.
"""
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt

from bluemira.base.file import get_bluemira_root
from bluemira.power_cycle.net.manager import PowerCycleManager
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

    # Recurring filenames
    scenario_config_filename = "scenario_config.json"
    manager_config_filename = "manager_config.json"

    @classmethod
    def build_scenario_config_path(cls):
        """
        Build scenario configuration path.
        """
        scenario_config_path = Path(
            *cls.examples_crumbs,
            cls.scenario_config_filename,
        )
        return scenario_config_path

    @classmethod
    def build_eudemo_manager_config_path(cls, manager_config_filename):
        """
        Build manager configuration file path.
        """
        manager_config_path = Path(
            *cls.examples_crumbs,
            cls.eudemo17_folder,
            manager_config_filename,
        )
        return manager_config_path


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


class ManagerKit:
    """
    Kit of methods to build and visualize Power Cycle managers.
    """

    @staticmethod
    def build_manager(manager_config_path):
        """
        Create the Power Cycle manager.
        """
        scenario_config_path = PathKit.build_scenario_config_path()
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
    def plot_detailed_phaseload(phaseload, monicker=""):
        """
        Plot 'PhaseLoad' in 'detailed' mode, but transform unit
        """
        title = (
            f"Detailed plot of {monicker!r} phase load "
            f"in phase {phaseload.phase.name!r}"
        )
        ax = DisplayKit.prepare_plot(title)
        ax, _ = phaseload.plot(ax=ax, detailed=True)
        ax = DisplayKit.finalize_plot(ax)
        return ax


if __name__ == "__main__":
    print("what is happening?")
