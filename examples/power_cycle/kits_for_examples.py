# COPYRIGHT PLACEHOLDER

"""
Useful variables and functions for running examples in the Power Cycle module.
"""

import os
from pprint import pprint

import matplotlib.pyplot as plt

from bluemira.base.file import get_bluemira_root
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


if __name__ == "__main__":
    print("what is happening?")
