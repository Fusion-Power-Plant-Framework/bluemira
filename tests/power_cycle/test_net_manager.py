# COPYRIGHT PLACEHOLDER

"""
import pytest
import copy
import matplotlib.pyplot as plt

from bluemira.power_cycle.errors import (
    PlantSystemError,
    PowerLoadManagerError,
)
from bluemira.power_cycle.net_manager import (
    PlantSystem,
    PowerLoadManager,
)
from bluemira.power_cycle.tools import adjust_2d_graph_ranges
"""
from tests.power_cycle.kits_for_tests import (
    NetLoadsTestKit,
    NetManagerTestKit,
    TimeTestKit,
    ToolsTestKit,
)

tools_testkit = ToolsTestKit()
time_testkit = TimeTestKit()
netloads_testkit = NetLoadsTestKit()
netmanager_testkit = NetManagerTestKit()


class TestPlantSystem:
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


class TestPowerLoadManager:
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
