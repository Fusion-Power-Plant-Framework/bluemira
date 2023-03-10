# COPYRIGHT PLACEHOLDER

"""
Classes for the calculation of net power in the Power Cycle model.
"""

from bluemira.power_cycle.base import PowerCycleABC
from bluemira.power_cycle.errors import (
    PowerCycleGroupError,
    PowerCycleManagerError,
    PowerCycleSystemError,
)
from bluemira.power_cycle.tools import read_json, validate_file


class PowerCycleSystem(PowerCycleABC):
    """
    Class to build the PowerLoad instances associated with the power
    production and consumption of a single plant system, used to
    represent that system in the time-dependent power balance of the
    Power Cycle module.

    Parameters
    ----------
    name: str
        Description of the 'PowerCycleSystem' instance.
    system_config: dict
        Dictionary that contains the necessary inputsto define
        objects of the PowerLoad class that characterize the power
        production and consumption of time-dependent power balance of the
        Power Cycle module.

    Attributes
    ----------

    """

    # Build active & reactive PowerLoads for a Plant System

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    _system_dict = {
        "name": str,
        "production": dict,
        "reactive": dict,
        "active": dict,
    }

    def __init__(self, system_config: dict):

        (
            name,
            production_config,
            reactive_config,
            active_config,
        ) = self._validate_config(system_config)
        self._system_config = system_config

        super().__init__(name)

        active_loads = self._build_loads_from_config(active_config)
        reactive_loads = self._build_loads_from_config(reactive_config)
        production_loads = self._build_loads_from_config(production_config)

        self.active_loads = active_loads
        self.reactive_loads = reactive_loads
        self.production_loads = production_loads

    @staticmethod
    def _validate_config(system_config):
        pass

    @staticmethod
    def _build_loads_from_config(load_config):
        pass


class PowerCycleGroup(PowerCycleABC):
    # Build Power Cycle representations of each Plant System in the group
    pass


class PowerCycleManager:
    # Call ScenarioBuilder
    # Read load inputs JSON files (inputs for each Plant System)
    # Build all Plant Systems
    # Build PhaseLoads and PulseLoad

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
    pass
