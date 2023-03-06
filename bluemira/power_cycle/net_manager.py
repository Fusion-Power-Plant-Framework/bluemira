# COPYRIGHT PLACEHOLDER

"""
Classes for the calculation of net power in the power cycle model.
"""
import json
import os

from bluemira.base.file import get_bluemira_root
from bluemira.power_cycle.base import PowerCycleABC
from bluemira.power_cycle.errors import PowerCycleManagerError


class PowerCycleSystem:
    # Build active & reactive PowerLoads for a Plant System

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


class PowerCycleManager(PowerCycleABC):
    """
    Class to read inputs for the Power Cycle module, and calculate the
    net power produced.

    Parameters
    ----------
    name: str
        Description of the 'PowerCycleManager' instance.
    pulse_config_path: str
        Path to JSON file that contains all inputs necessary to define
        a PowerCyclePulse object, for characterizing the time-dependent
        power balance of the Power Cycle module.

    Attributes
    ----------
    pulse: PowerCyclePulse
        Pulse defined for a Power Cycle scenario.

    """

    # Read inputs.JSON
    # Read inputs for each Plant System, build all Plant Systems
    # Build PhaseLoads and PulseLoad

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    def __init__(self, name, pulse_config_path: str):
        super().__init__(name)

        self.pulse_config_path = self._validate_file(pulse_config_path)
        pulse_config = self._read_pulse_config(self.pulse_config_path)

        (
            self.pulse_library,
            self.phase_library,
            self.breakdown_library,
        ) = self._split_pulse_config(pulse_config)

        self._build_pulse_from_config()
        # self._sanity()

    @staticmethod
    def _validate_file(file_path):
        path_is_relative = not os.path.isabs(file_path)
        if path_is_relative:
            project_path = get_bluemira_root()
            absolute_path = os.path.join(project_path, file_path)
        else:
            absolute_path = file_path

        file_exists = os.path.isfile(absolute_path)
        if file_exists:
            return absolute_path
        else:
            raise PowerCycleManagerError("file")

    @staticmethod
    def _read_pulse_config(pulse_config_path):
        try:
            with open(pulse_config_path) as pulse_config_json:
                pulse_config = json.load(pulse_config_json)
        except json.decoder.JSONDecodeError:
            raise PowerCycleManagerError(
                "pulse_config",
                "The file could not be read.",
            )
        return pulse_config

    @staticmethod
    def _split_pulse_config(pulse_config):
        try:
            pulse_library = pulse_config["pulse-library"]
            phase_library = pulse_config["phase-library"]
            breakdown_library = pulse_config["breakdown-library"]
        except KeyError:
            raise PowerCycleManagerError(
                "pulse_config",
                "The expected fields were not present in the file.",
            )
        return pulse_library, phase_library, breakdown_library

    def build_pulse_from_config(self):
        pass

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # ARITHMETICS
    # ------------------------------------------------------------------
