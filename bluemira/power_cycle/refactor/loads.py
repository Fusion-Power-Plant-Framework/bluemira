from typing import Dict, List

from bluemira.power_cycle.refactor.load_manager import PowerCycleManagerConfig
from bluemira.power_cycle.refactor.time import PhaseConfig


class PulseSystemLoad:
    def __init__(
        self,
        pulse: List[PhaseConfig],
        manager_configs: Dict[str, PowerCycleManagerConfig],
    ):
        # active = dict[loads]
        # reactive = dict[loads]

        # Create new PowerCycleLoadConfig(s) for a given PulseSystemLoad


def func(scenario_config, manager_config):
    ...
