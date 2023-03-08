# COPYRIGHT PLACEHOLDER

import copy
import os

import pytest

from bluemira.power_cycle.errors import ScenarioBuilderError
from bluemira.power_cycle.net.manager import ScenarioBuilder
from tests.power_cycle.kits_for_tests import (  # NetLoadsTestKit,; TimeTestKit,; ToolsTestKit,
    NetManagerTestKit,
)

# import matplotlib.pyplot as plt

# from bluemira.power_cycle.tools import adjust_2d_graph_ranges

# tools_testkit = ToolsTestKit()
# time_testkit = TimeTestKit()
# netloads_testkit = NetLoadsTestKit()
manager_testkit = NetManagerTestKit()


class TestPowerCycleSystem:
    def setup_method(self):
        pass


class TestScenarioBuilder:
    def setup_method(self):
        scenario_json_path = manager_testkit.scenario_json_path
        sample = ScenarioBuilder(scenario_json_path)
        self.sample = sample

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    def test_constructor(self):
        sample = self.sample
        assert isinstance(sample, ScenarioBuilder)

        # every element of libraries must have correct class


"""

    def test_split_pulse_config(self):
        all_libraries = self.config_libraries
        for library in all_libraries:
            library_is_dict = type(library) == dict
            assert library_is_dict

        pulse_config = self.pulse_config
        wrong_config = copy.deepcopy(pulse_config)
        wrong_config.pop("pulse-library")
        with pytest.raises(PowerCycleManagerError):
            (
                pulse_library,
                phase_library,
                breakdown_library,
            ) = PowerCycleManager._split_pulse_config(wrong_config)

    def test_build_pulse_from_config(self):
        all_libraries = self.config_libraries

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # ARITHMETICS
    # ------------------------------------------------------------------
"""


"""
class TestPowerCycleImporter:
    def setup_method(self):
        importer_class = PowerCycleImporter
        self.importer_class = importer_class

        example_number = 10
        example_unit = "minute"
        example_equilibria_desired_data = "CS-recharge-time"
        example_pumping_desired_data = "pumpdown-time"

        import_parameters = {
            "None": {
                "duration": example_number,
                "unit": example_unit,
                "reference": "?",
            },
            "equilibria": {
                "desired_data": example_equilibria_desired_data
            },
            "pumping": {
                "desired_data": example_pumping_desired_data
            },
            "X": {},
        }
        self.import_parameters = import_parameters

    def test_duration_from_module(self):
        importer_class = self.importer_class
        import_parameters = self.import_parameters
        available_modules = import_parameters.keys()

        for module in available_modules:
            variable_map = import_parameters[module]
            duration = importer_class.duration_from_module(module, variable_map)
            self.assert_value_is_nonnegative(duration)

        unavailable_module = "not-implemented"
        example_variable_map = dict()
        with pytest.raises(PowerCycleImporterError):
            duration = importer_class.duration_from_module(
                unavailable_module,
                example_variable_map,
            )
"""
