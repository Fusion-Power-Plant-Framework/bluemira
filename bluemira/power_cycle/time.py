# COPYRIGHT PLACEHOLDER

"""
Classes to define the timeline for Power Cycle simulations.
"""

from typing import Dict, List, Union

from bluemira.power_cycle.base import PowerCycleTimeABC
from bluemira.power_cycle.errors import PowerCyclePhaseError


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
    ):
        breakdown = self._validate_breakdown(duration_breakdown)
        self.duration_breakdown = breakdown
        durations_list = list(breakdown.values())
        super().__init__(name, durations_list)

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


class PowerCyclePulse(PowerCycleTimeABC):
    """
    Class to define pulses for a Power Cycle timeline.

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
    ):
        self.phase_set = self._validate_phase_set(phase_set)
        durations_list = self._extract_phase_durations()
        super().__init__(name, durations_list)

    @staticmethod
    def _validate_phase_set(phase_set):
        """
        Validate 'phase_set' input to be a list of 'PowerCyclePhase'
        instances.
        """
        owner = PowerCyclePulse
        phase_set = super(owner, owner).validate_list(phase_set)
        for element in phase_set:
            PowerCyclePhase.validate_class(element)
        return phase_set

    def _extract_phase_durations(self):
        phase_set = self.phase_set
        durations_list = []
        for phase in phase_set:
            durations_list.append(phase.duration)
        return durations_list


class PowerCycleTimeline(PowerCycleTimeABC):
    """
    Class to define pulses for a Power Cycle timeline.

    Parameters
    ----------
    name: str
        Description of the 'PowerCycleTimeline' instance.
    pulse_set: PowerCyclePulse | list[PowerCyclePulse]
        List of pulses that compose the timeline, in chronological
        order.
    """

    def __init__(
        self,
        name,
        pulse_set: Union[PowerCyclePulse, List[PowerCyclePulse]],
    ):
        self.pulse_set = self._validate_pulse_set(pulse_set)
        durations_list = self._extract_pulse_durations()
        super().__init__(name, durations_list)

    @staticmethod
    def _validate_pulse_set(pulse_set):
        """
        Validate 'pulse_set' input to be a list of 'PowerCyclePulse'
        instances.
        """
        owner = PowerCycleTimeline
        pulse_set = super(owner, owner).validate_list(pulse_set)
        for element in pulse_set:
            PowerCyclePulse.validate_class(element)
        return pulse_set

    def _extract_pulse_durations(self):
        pulse_set = self.pulse_set
        durations_list = []
        for pulse in pulse_set:
            durations_list.append(pulse.duration)
        return durations_list
