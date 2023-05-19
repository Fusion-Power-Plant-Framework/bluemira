# COPYRIGHT PLACEHOLDER

"""
Classes to define the timeline for Power Cycle simulations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
from numpy.typing import ArrayLike

from bluemira.base.constants import raw_uc
from bluemira.power_cycle.base import BaseConfig, ModuleType, PowerCycleTimeABC
from bluemira.power_cycle.errors import PowerCycleError, ScenarioBuilderError
from bluemira.power_cycle.net.importers import EquilibriaImporter, PumpingImporter
from bluemira.power_cycle.tools import read_json, validate_file


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
        duration_breakdown: float,
        label=None,
    ):
        super().__init__(name, np.array([duration_breakdown]), label=label)
        if duration_breakdown < 0:
            raise ValueError("duration_breakdown must be positive")
        self.duration_breakdown = duration_breakdown


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
        phase_set: List[PowerCyclePhase],
        label=None,
    ):
        super().__init__(name, self._build_durations_list(phase_set), label=label)
        self.phase_set = phase_set

    def build_phase_library(self):
        """
        Returns a 'dict' with phase labels as keys and the phases
        themselves as values.
        """
        return {phase.label: phase for phase in self.phase_set}


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
        pulse_set: List[PowerCyclePulse],
        label=None,
    ):
        super().__init__(name, self._build_durations_list(pulse_set), label=label)
        self.pulse_set = pulse_set

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
        return {pulse.label: pulse for pulse in self.pulse_set}


@dataclass
class ScenarioConfig:
    name: str
    pulses: list
    repetition: list


@dataclass
class PulseConfig:
    name: str
    phases: list
    label: Optional[str] = None


@dataclass
class PhaseConfig:
    name: str
    logical: str
    breakdown: list


@dataclass
class BreakdownConfig(BaseConfig):
    """Breakdown Config"""


@dataclass
class Breakdown:
    name: str
    duration: ArrayLike


@dataclass
class ScenarioBuilderConfig:
    scenario: Union[dict, ScenarioConfig]
    pulse_library: dict
    phase_library: dict
    breakdown_library: dict

    _scenario: ScenarioConfig = field(init=False, repr=False)
    _pulse_library: Dict[str, PulseConfig] = field(init=False, repr=False)
    _phase_library: Dict[str, PhaseConfig] = field(init=False, repr=False)
    _breakdown_library: Dict[str, BreakdownConfig] = field(init=False, repr=False)

    @property
    def scenario(self):
        return self._scenario

    @scenario.setter
    def scenario(self, value: Union[dict, ScenarioConfig]):
        if isinstance(value, ScenarioConfig):
            self._scenario = value
        else:
            self._scenario = ScenarioConfig(**value)

    @property
    def pulse_library(self):
        return self._pulse_library

    @pulse_library.setter
    def pulse_library(self, value: dict):
        for k, v in value.items():
            if not isinstance(v, PulseConfig):
                value[k] = PulseConfig(**v)
        self._pulse_library = value

    @property
    def phase_library(self):
        return self._phase_library

    @phase_library.setter
    def phase_library(self, value: dict):
        for k, v in value.items():
            if not isinstance(v, PhaseConfig):
                value[k] = PhaseConfig(**v)
        self._phase_library = value

    @property
    def breakdown_library(self):
        return self._breakdown_library

    @breakdown_library.setter
    def breakdown_library(self, value: dict):
        for k, v in value.items():
            if not isinstance(v, BreakdownConfig):
                value[k] = BreakdownConfig(**v)
        self._breakdown_library = value


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

    def __init__(self, config_path: str):
        self._scenario_config_path = validate_file(config_path)
        self.scenario_config = ScenarioBuilderConfig(
            **read_json(self._scenario_config_path)
        )

        self._build_breakdown_library()

        self._build_phase_library()

        self._build_pulse_library()

        self._build_scenario()

    def _build_breakdown_library(self):
        self._breakdown_library = {
            k: Breakdown(
                v.name,
                self.import_duration(v.module, v.variables_map),
            )
            for k, v in self.scenario_config.breakdown_library.items()
        }

    @staticmethod
    def import_duration(module: ModuleType, variables_map: dict) -> ArrayLike:
        """
        Method that unpacks the 'variables_map' field of a JSON input
        file.
        """
        if module is ModuleType.EQUILIBRIA:
            duration = EquilibriaImporter.duration(variables_map)
        elif module is ModuleType.PUMPING:
            duration = PumpingImporter.duration(variables_map)
        else:
            try:
                duration = raw_uc(
                    variables_map["duration"], variables_map["unit"], "second"
                )
            except KeyError:
                raise PowerCycleError(
                    msg=f'Variables map incomplete no keys {set(("duration", "unit")) - variables_map.keys()}'
                )
        return duration

    def _build_phase_library(self):
        self._phase_library = {
            k: PowerCyclePhase(
                v.name,
                self._build_phase_logical(
                    [self._breakdown_library[br].duration for br in v.breakdown],
                    v.logical,
                ),
            )
            for k, v in self.scenario_config.phase_library.items()
        }

    def _build_phase_logical(self, durations, operator) -> float:
        if operator == "&":
            return sum(durations)
        elif operator == "|":
            return max(durations)
        else:
            raise ScenarioBuilderError(
                "operator",
                f"Unknown routine for {operator!r} operator.",
            )

    def _build_pulse_library(self):
        self._pulse_library = {
            k: PowerCyclePulse(
                v.name,
                [self._phase_library[label] for label in v.phases],
                v.label,
            )
            for k, v in self.scenario_config.pulse_library.items()
        }

    def _build_scenario(self):
        """
        Currently ignores the 'repetition' input and just uses a single
        pulse as the scenario. To be altered to contain multiple pulses.
        """
        # pulse_repetitions = scenario_config["repetition"]

        self.scenario = PowerCycleScenario(
            self.scenario_config.scenario.name,
            [
                self._pulse_library[label]
                for label in self.scenario_config.scenario.pulses
            ],
        )
