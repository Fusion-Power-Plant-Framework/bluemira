from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Union

import numpy as np

from bluemira.base.constants import raw_uc
from bluemira.power_cycle.refactor.base import (
    Config,
    Descriptor,
    LibraryConfigDescriptor,
)
from bluemira.power_cycle.tools import read_json

if TYPE_CHECKING:
    from pathlib import Path


class ScenarioConfigDescriptor(Descriptor):
    """Scenario config descriptor for use with dataclasses"""

    def __get__(self, obj: Any, _) -> ScenarioConfig:
        """Get the scenario config"""
        return getattr(obj, self._name)

    def __set__(self, obj: Any, value: Union[dict, ScenarioConfig]):
        """Set the scenario config"""
        if not isinstance(value, ScenarioConfig):
            value = ScenarioConfig(**value)

        setattr(obj, self._name, value)


@dataclass
class ScenarioConfig(Config):
    pulses: dict[str, int]
    description: str = ""


@dataclass
class PulseConfig(Config):
    phases: list[str]
    description: str = ""


@dataclass
class PhaseConfig(Config):
    operation: str
    breakdown: list[str]
    description: str = ""


@dataclass
class PowerCycleBreakdownConfig(Config):
    """Breakdown Config"""

    duration: Union[float, str]
    unit: str = "s"
    description: str = ""
    reference: str = ""

    def __post_init__(self):
        if isinstance(self.duration, (float, int)):
            self.duration = raw_uc(self.duration, self.unit, "second")
            self.unit = "second"


@dataclass
class ScenarioBuilderConfig:
    scenario: ScenarioConfigDescriptor = ScenarioConfigDescriptor()
    pulse_library: LibraryConfigDescriptor = LibraryConfigDescriptor(
        library_config=PulseConfig
    )
    phase_library: LibraryConfigDescriptor = LibraryConfigDescriptor(
        library_config=PhaseConfig
    )
    breakdown_library: LibraryConfigDescriptor = LibraryConfigDescriptor(
        library_config=PowerCycleBreakdownConfig
    )

    def __post_init__(self):
        bl_keys = self.breakdown_library.keys()
        for ph_c in self.phase_library.values():
            if unknown := ph_c.breakdown - bl_keys:
                raise ValueError(f"Unknown breakdown configurations {unknown}")

    @classmethod
    def from_file(cls, config_path: Union[Path, str]):
        return cls(**read_json(config_path))

    def import_breakdown_data(self, breakdown_duration_params):
        for br in self.breakdown_library.values():
            if isinstance(br.duration, str):
                br.duration = getattr(
                    breakdown_duration_params, br.duration.replace("-", "_")
                )

    def breakdown_durations(self):
        return {k: br.duration for k, br in self.breakdown_library.items()}

    def phase_operations(self):
        return {k: ph_c.operation for k, ph_c in self.phase_library.items()}

    def phase_breakdowns(self):
        return {k: ph_c.breakdown for k, ph_c in self.phase_library.items()}

    def get_scenario_pulses(
        self,
    ) -> Dict[str, List[PhaseConfig]]:
        return {
            k: [self.phase_library[phase] for phase in v.phases]
            for k, v in self.pulse_library.items()
            if k in self.scenario.pulses
        }

    def build_phase_breakdowns(self):
        """
        Build pulse from 'PowerCyclePhase' objects stored in the
        'phase' attributes of each 'PhaseLoad' instance in the
        'phaseload_set' list.
        """
        phase_op = self.phase_operations()
        durations = self.breakdown_durations()
        phase_breakdowns = {}
        for phase, breakdowns in self.phase_breakdowns().items():
            phase_breakdowns[phase] = getattr(np, phase_op[phase])(
                [durations[br] for br in breakdowns]
            )
            if phase_breakdowns[phase] < 0:
                raise ValueError(
                    f"{phase} phase duration must be positive: {phase_breakdowns[phase]}s"
                )

        return phase_breakdowns
