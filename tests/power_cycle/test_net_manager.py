# COPYRIGHT PLACEHOLDER

import copy
import os

import pytest

from bluemira.power_cycle.errors import PowerCycleManagerError  # PowerCycleSystemError,
from bluemira.power_cycle.net_manager import PowerCycleManager  # PowerCycleSystem,
from tests.power_cycle.kits_for_tests import (  # NetLoadsTestKit,; TimeTestKit,; ToolsTestKit,
    NetManagerTestKit,
)

# import matplotlib.pyplot as plt

# from bluemira.power_cycle.tools import adjust_2d_graph_ranges

# tools_testkit = ToolsTestKit()
# time_testkit = TimeTestKit()
# netloads_testkit = NetLoadsTestKit()
netmanager_testkit = NetManagerTestKit()


class TestPowerCycleSystem:
    def setup_method(self):
        pass

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # ARITHMETICS
    # ------------------------------------------------------------------


class TestPowerCycleManager:
    def setup_method(self):
        self.pulse_json_path = netmanager_testkit.pulse_json_path

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    def test_validate_file(self):
        relative_path = self.pulse_json_path
        absolute_path = os.path.abspath(relative_path)

        return_path = PowerCycleManager._validate_file(relative_path)
        assert return_path == absolute_path

        wrong_path = absolute_path.replace("json", "doc")
        with pytest.raises(PowerCycleManagerError):
            return_path = PowerCycleManager._validate_file(wrong_path)

    def test_read_pulse_config(self):
        file_path = self.pulse_json_path
        pulse_config = PowerCycleManager._read_pulse_config(file_path)
        pulse_config_is_dict = type(pulse_config) == dict
        assert pulse_config_is_dict

        wrong_path = file_path.replace("json", "txt")
        with pytest.raises(PowerCycleManagerError):
            PowerCycleManager._read_pulse_config(wrong_path)

    def test_split_pulse_config(self):
        file_path = self.pulse_json_path
        pulse_config = PowerCycleManager._read_pulse_config(file_path)

        (
            pulse_library,
            phase_library,
            breakdown_library,
        ) = PowerCycleManager._split_pulse_config(pulse_config)

        all_libraries = [pulse_library, phase_library, breakdown_library]
        for library in all_libraries:
            library_is_dict = type(library) == dict
            assert library_is_dict

        wrong_config = copy.deepcopy(pulse_config)
        wrong_config.pop("pulse-library")
        with pytest.raises(PowerCycleManagerError):
            (
                pulse_library,
                phase_library,
                breakdown_library,
            ) = PowerCycleManager._split_pulse_config(wrong_config)

    def test_build_pulse_from_config(self):
        file_path = self.pulse_json_path
        pulse_config = PowerCycleManager._read_pulse_config(file_path)
        (
            pulse_library,
            phase_library,
            breakdown_library,
        ) = PowerCycleManager._split_pulse_config(pulse_config)

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # ARITHMETICS
    # ------------------------------------------------------------------
