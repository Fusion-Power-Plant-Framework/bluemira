# COPYRIGHT PLACEHOLDER

import copy
import os

import pytest

from bluemira.power_cycle.errors import PowerCycleManagerError, PowerCycleSystemError
from bluemira.power_cycle.net.manager import PowerCycleManager, PowerCycleSystem
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
        pass
