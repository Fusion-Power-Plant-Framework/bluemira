# COPYRIGHT PLACEHOLDER

"""
Classes to define the timeline for Power Cycle simulations.
"""

from typing import Dict, List, Union

from bluemira.base.constants import raw_uc
from bluemira.power_cycle.base import PowerCycleTimeABC
from bluemira.power_cycle.errors import PowerCyclePhaseError, ScenarioBuilderError
from bluemira.power_cycle.net.importers import EquilibriaImporter, PumpingImporter
from bluemira.power_cycle.net.manager import ModuleType
from bluemira.power_cycle.tools import (
    FormattedDict,
    FormattedLibrary,
    read_json,
    validate_file,
    validate_list,
    validate_nonnegative,
)


class PowerCyclePhase(PowerCycleTimeABC):
    """
    Class to define phases for a Power Cycle pulse.

    Parameters
    ----------
    name: str
        Description of the 'PowerCyclePhase' instance.
    duration_breakdown: dict[str, int | float]
        Dictionary of descriptions and durations of time lengths. [s]
        The dictionary defines all time lenghts of sub-phases that
        compose the duration of a pulse phase.
    """

    def __init__(
        self,
        name,
        duration_breakdown: Dict[str, Union[int, float]],
        label=None,
    ):
        breakdown = self._validate_breakdown(duration_breakdown)
        durations_list = list(breakdown.values())

        super().__init__(name, durations_list, label=label)
        self.duration_breakdown = breakdown

    @staticmethod
    def _validate_breakdown(duration_breakdown):
        """
        Validate 'duration_breakdown' input to be a dictionary with
        keys of the 'str' class and non-negative numbers as values.
        """
        for key, value in duration_breakdown.items():
            if not isinstance(key, str):
                raise PowerCyclePhaseError("breakdown")
            validate_nonnegative(value)
        return duration_breakdown


class PowerCyclePulse(PowerCycleTimeABC):
    """
    Class to define pulses for a Power Cycle scenario.

    Parameters
    ----------
    name: str
        Description of the 'PowerCyclePulse' instance.
    phase_set: PowerCyclePhase | list[PowerCyclePhase]
        List of phases that compose the pulse, in chronological order.
    """

    def __init__(
        self,
        name,
        phase_set: Union[PowerCyclePhase, List[PowerCyclePhase]],
        label=None,
    ):
        self.phase_set = self._validate_phase_set(phase_set)
        durations_list = self._build_durations_list(phase_set)
        super().__init__(name, durations_list, label=label)

    @staticmethod
    def _validate_phase_set(phase_set):
        """
        Validate 'phase_set' input to be a list of 'PowerCyclePhase'
        instances.
        """
        phase_set = validate_list(phase_set)
        for element in phase_set:
            PowerCyclePhase.validate_class(element)
        return phase_set

    def build_phase_library(self):
        """
        Returns a 'dict' with phase labels as keys and the phases
        themselves as values.
        """
        phase_library = {phase.label: phase for phase in self.phase_set}
        return phase_library


class PowerCycleScenario(PowerCycleTimeABC):
    """
    Class to define scenarios for the Power Cycle module.

    Parameters
    ----------
    name: str
        Description of the 'PowerCycleScenario' instance.
    pulse_set: PowerCyclePulse | list[PowerCyclePulse]
        List of pulses that compose the scenario, in chronological
        order.
    TBD - repetition: int | list[int]
        List of integer values that defines how many repetitions occur
        for each element of 'pulse_set' when building the scenario.
    """

    def __init__(
        self,
        name,
        pulse_set: Union[PowerCyclePulse, List[PowerCyclePulse]],
        label=None,
    ):
        self.pulse_set = self._validate_pulse_set(pulse_set)
        durations_list = self._build_durations_list(self.pulse_set)
        super().__init__(name, durations_list, label=label)

    @staticmethod
    def _validate_pulse_set(pulse_set):
        """
        Validate 'pulse_set' input to be a list of 'PowerCyclePulse'
        instances.
        """
        pulse_set = validate_list(pulse_set)
        for element in pulse_set:
            PowerCyclePulse.validate_class(element)
        return pulse_set

    def build_phase_library(self):
        """
        Returns a 'dict' with phase labels as keys and the phases
        themselves as values.
        """
        phase_library = {}
        for pulse in self.pulse_set:
            phase_library = {**phase_library, **pulse.build_phase_library()}
        return phase_library

    def build_pulse_library(self):
        """
        Returns a 'dict' with pulse labels as keys and the pulses
        themselves as values.
        """
        pulse_library = {pulse.label: pulse for pulse in self.pulse_set}
        return pulse_library


class ScenarioBuilder:
    """
    Class to read time inputs for the Power Cycle module, and build
    a scenario.

    Parameters
    ----------
    config_path: str
        Path to JSON file that contains all inputs necessary to define
        objects children of the PowerCycleTimeABC class, to enable
        characterization of the time-dependent power balance of the
        Power Cycle module.

    Attributes
    ----------
    scenario: PowerCycleScenario
        Representation of a scenario for Power Cycle simulations.

    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    _config_format = FormattedDict.Format(
        {
            "scenario": dict,
            "pulse-library": dict,
            "phase-library": dict,
            "breakdown-library": dict,
        }
    )

    _scenario_format = FormattedDict.Format(
        {
            "name": str,
            "pulses": list,
            "repetition": list,
        }
    )

    _pulse_format = FormattedLibrary.Format(
        {
            "name": str,
            "phases": list,
        }
    )

    _phase_format = FormattedLibrary.Format(
        {
            "name": str,
            "logical": str,
            "breakdown": list,
        }
    )

    _breakdown_format = FormattedLibrary.Format(
        {
            "name": str,
            "module": [None, str],
            "variables_map": dict,
        }
    )

    def __init__(self, config_path: str):
        validated_path = validate_file(config_path)
        json_contents = read_json(validated_path)
        (
            scenario_config,
            pulse_config,
            phase_config,
            breakdown_config,
        ) = self._validate_config(json_contents)

        breakdown_library = self._build_breakdown_library(
            breakdown_config,
        )

        phase_library = self._build_phase_library(
            phase_config,
            breakdown_library,
        )

        pulse_library = self._build_pulse_library(
            pulse_config,
            phase_library,
        )

        scenario = self._build_scenario(
            scenario_config,
            pulse_library,
        )

        self._scenario_config_path = validated_path
        self._breakdown_library = breakdown_library
        self._phase_library = phase_library
        self._pulse_library = pulse_library

        self.scenario = scenario

    @classmethod
    def _validate_config(cls, json_contents):
        json_contents = FormattedDict(
            cls._config_format,
            dictionary=json_contents,
        )
        for key, key_contents in json_contents.items():
            if key == "scenario":
                scenario_config = FormattedDict(
                    cls._scenario_format,
                    dictionary=key_contents,
                )
            elif key == "pulse-library":
                pulse_config = FormattedLibrary(
                    cls._pulse_format,
                    dictionary=key_contents,
                )
            elif key == "phase-library":
                phase_config = FormattedLibrary(
                    cls._phase_format,
                    dictionary=key_contents,
                )
            elif key == "breakdown-library":
                breakdown_config = FormattedLibrary(
                    cls._breakdown_format,
                    dictionary=key_contents,
                )
            else:
                raise ScenarioBuilderError(
                    "config",
                    f"Validation for the json field {key!r} has not "
                    "been implemented yet.",
                )

        return (
            scenario_config,
            pulse_config,
            phase_config,
            breakdown_config,
        )

    @staticmethod
    def import_duration(module, variables_map):
        """
        Method that unpacks the 'variables_map' field of a JSON input
        file.
        """
        if module is ModuleType.EQUILIBRIA:
            duration = EquilibriaImporter.duration(variables_map)
        elif module is ModuleType.EQUILIBRIA:
            duration = PumpingImporter.duration(variables_map)
        else:
            duration = raw_uc(variables_map["duration"], variables_map["unit"], "second")
        return duration

    @classmethod
    def _build_breakdown_library(cls, breakdown_config):
        return {
            element_label: {
                element_specs["name"]: cls.import_duration(
                    element_specs["module"], element_specs["variables_map"]
                )
            }
            for element_label, element_specs in breakdown_config.items()
        }

    @staticmethod
    def _build_phase_breakdown(breakdown_library, breakdown_list, operator):
        phase_breakdown = {}
        last_value = 0
        for label in breakdown_list:
            try:
                element = breakdown_library[label]
            except KeyError:
                raise ScenarioBuilderError(
                    "library",
                    f"Breakdown element {label!r} has not been defined.",
                )

            element_key = list(element.keys())[0]
            element_value = element[element_key]
            if operator == "&":
                pass
            elif operator == "|":
                if element_value > last_value:
                    phase_breakdown = {}
                    last_value = element_value
            else:
                raise ScenarioBuilderError(
                    "operator",
                    f"Unknown routine for {operator!r} operator.",
                )
            phase_breakdown[element_key] = element_value

        return phase_breakdown

    @classmethod
    def _build_phase_library(cls, phase_config, breakdown_library):
        return {
            phase_label: PowerCyclePhase(
                phase_specs["name"],
                cls._build_phase_breakdown(
                    breakdown_library, phase_specs["breakdown"], phase_specs["logical"]
                ),
                label=phase_label,
            )
            for phase_label, phase_specs in phase_config.items()
        }

    @staticmethod
    def _build_time_set(time_library, time_list):
        return [time_library[label] for label in time_list]

    @classmethod
    def _build_pulse_library(cls, pulse_config, phase_library):
        return {
            pulse_label: PowerCyclePulse(
                pulse_specs["name"],
                cls._build_time_set(phase_library, pulse_specs["phases"]),
                label=pulse_label,
            )
            for pulse_label, pulse_specs in pulse_config.items()
        }

    @classmethod
    def _build_scenario(cls, scenario_config, pulse_library):
        """
        Currently ignores the 'repetition' input and just uses a single
        pulse as the scenario. To be altered to contain multiple pulses.
        """
        # pulse_repetitions = scenario_config["repetition"]

        return PowerCycleScenario(
            scenario_config["name"],
            cls._build_time_set(pulse_library, scenario_config["pulses"]),
        )
