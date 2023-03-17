# COPYRIGHT PLACEHOLDER

"""
Classes to define the timeline for Power Cycle simulations.
"""

from typing import Dict, List, Union

from bluemira.base.constants import raw_uc
from bluemira.power_cycle.base import PowerCycleTimeABC
from bluemira.power_cycle.errors import PowerCyclePhaseError, ScenarioBuilderError
from bluemira.power_cycle.net.importers import EquilibriaImporter, PumpingImporter
from bluemira.power_cycle.tools import (
    read_json,
    validate_dict,
    validate_file,
    validate_list,
    validate_subdict,
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

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

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
        keys of the 'str' class.
        """
        for key in duration_breakdown:
            if not isinstance(key, str):
                raise PowerCyclePhaseError("breakdown")
        return duration_breakdown

    # ------------------------------------------------------------------
    #  OPERATIONS
    # ------------------------------------------------------------------


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

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    def __init__(
        self,
        name,
        phase_set: Union[PowerCyclePhase, List[PowerCyclePhase]],
        label=None,
    ):
        self.phase_set = self._validate_phase_set(phase_set)
        durations_list = self._build_durations_list(self.phase_set)
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

    # ------------------------------------------------------------------
    #  OPERATIONS
    # ------------------------------------------------------------------

    def build_phase_library(self):
        """
        Returns a 'dict' with phase labels as keys and the phases
        themselves as values.
        """
        phase_set = self.phase_set
        phase_library = dict()
        for phase in phase_set:
            phase_label = phase.label
            phase_library[phase_label] = phase
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
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    #  OPERATIONS
    # ------------------------------------------------------------------

    def build_phase_library(self):
        """
        Returns a 'dict' with phase labels as keys and the phases
        themselves as values.
        """
        pulse_set = self.pulse_set
        phase_library = dict()
        for pulse in pulse_set:
            phase_library_for_pulse = pulse.build_phase_library()
            phase_library = {**phase_library, **phase_library_for_pulse}
        return phase_library


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

    _config_format = {
        "scenario": dict,
        "pulse-library": dict,
        "phase-library": dict,
        "breakdown-library": dict,
    }

    _scenario_format = {
        "name": str,
        "pulses": list,
        "repetition": list,
    }

    _pulse_format = {
        "name": str,
        "phases": list,
    }

    _phase_format = {
        "name": str,
        "logical": str,
        "breakdown": list,
    }

    _breakdown_format = {
        "name": str,
        "module": [type(None), str],
        "variables_map": dict,
    }

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
        allowed_format = cls._config_format
        json_contents = validate_dict(
            json_contents,
            allowed_format,
        )
        json_keys = json_contents.keys()
        for key in json_keys:
            contents_in_key = json_contents[key]

            if key == "scenario":
                allowed_format = cls._scenario_format
                scenario_config = validate_dict(
                    contents_in_key,
                    allowed_format,
                )
            elif key == "pulse-library":
                allowed_format = cls._pulse_format
                pulse_config = validate_subdict(
                    contents_in_key,
                    allowed_format,
                )
            elif key == "phase-library":
                allowed_format = cls._phase_format
                phase_config = validate_subdict(
                    contents_in_key,
                    allowed_format,
                )
            elif key == "breakdown-library":
                allowed_format = cls._breakdown_format
                breakdown_config = validate_subdict(
                    contents_in_key,
                    allowed_format,
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
        if module is None:
            duration = variables_map["duration"]
            unit = variables_map["unit"]
            duration = raw_uc(duration, unit, "second")

        elif module == "equilibria":
            duration = EquilibriaImporter.duration(variables_map)

        elif module == "pumping":
            duration = PumpingImporter.duration(variables_map)

        else:
            raise ScenarioBuilderError(
                "import",
                "Unknown routine for importing a duration from "
                f"the {module!r} module.",
            )
        return duration

    @classmethod
    def _build_breakdown_library(cls, breakdown_config):
        all_elements = breakdown_config.keys()
        breakdown_library = dict()
        for element_label in all_elements:
            element_specs = breakdown_config[element_label]

            element_name = element_specs["name"]
            module = element_specs["module"]
            variables_map = element_specs["variables_map"]

            duration = cls.import_duration(module, variables_map)
            breakdown_library[element_label] = {element_name: duration}
        return breakdown_library

    @staticmethod
    def _build_phase_breakdown(breakdown_library, breakdown_list, operator):
        phase_breakdown = dict()
        last_value = 0
        for label in breakdown_list:
            try:
                element = breakdown_library[label]
            except KeyError:
                raise ScenarioBuilderError(
                    "library",
                    f"Breakdown element {label!r} has not been defined.",
                )

            element_key = list(element.keys())
            element_key = element_key[0]
            element_value = element[element_key]
            if operator == "&":
                pass
            elif operator == "|":
                if element_value > last_value:
                    phase_breakdown = dict()
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
        all_phases = phase_config.keys()
        phase_library = dict()
        for phase_label in all_phases:
            phase_specs = phase_config[phase_label]
            phase_name = phase_specs["name"]

            breakdown_operator = phase_specs["logical"]
            breakdown_list = phase_specs["breakdown"]
            phase_breakdown = cls._build_phase_breakdown(
                breakdown_library, breakdown_list, breakdown_operator
            )

            phase = PowerCyclePhase(
                phase_name,
                phase_breakdown,
                label=phase_label,
            )
            phase_library[phase_label] = phase
        return phase_library

    @staticmethod
    def _build_time_set(time_library, time_list):
        time_set = []
        for time_label in time_list:
            time_instance = time_library[time_label]
            time_set.append(time_instance)
        return time_set

    @classmethod
    def _build_pulse_library(cls, pulse_config, phase_library):
        all_pulses = pulse_config.keys()
        pulse_library = dict()
        for pulse_label in all_pulses:
            pulse_specs = pulse_config[pulse_label]
            pulse_name = pulse_specs["name"]

            phase_list = pulse_specs["phases"]
            phase_set = cls._build_time_set(phase_library, phase_list)

            pulse = PowerCyclePulse(
                pulse_name,
                phase_set,
                label=pulse_label,
            )
            pulse_library[pulse_label] = pulse
        return pulse_library

    @classmethod
    def _build_scenario(cls, scenario_config, pulse_library):
        """
        Currently ignores the 'repetition' input and just uses a single
        pulse as the scenario.
        """
        scenario_name = scenario_config["name"]
        pulse_list = scenario_config["pulses"]
        # pulse_repetitions = scenario_config["repetition"]
        pulse_set = cls._build_time_set(pulse_library, pulse_list)

        scenario = PowerCycleScenario(scenario_name, pulse_set)
        return scenario
