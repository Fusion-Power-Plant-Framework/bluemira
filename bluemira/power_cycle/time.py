"""
Classes to define the timeline for Power Cycle simulations.
"""
from enum import Enum
from typing import Dict, List, Union

from bluemira.power_cycle.base import PowerCycleABC, PowerCycleError


class PowerCycleTimeABCError(PowerCycleError):
    """
    Exception class for 'PowerCycleTimeABC' class of the Power Cycle
    module.
    """

    def _errors(self):
        errors = {}
        return errors


class PowerCycleTimeABC(PowerCycleABC):
    """
    Abstract base class for classes in the Power Cycle module that are
    used to describe a timeline.

    Parameters
    ----------
    durations_list: int | float | list[ int | float ]
        List of all numerical values that compose the duration of an
        instance of a child class. Values must be non-negative.

    Attributes
    ----------
    duration: float
        Total duration. [s]
        Sum of all numerical values in the 'durations_list' attribute.
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------
    def __init__(self, name, durations_list):

        super().__init__(name)
        self.durations_list = self._validate_durations(durations_list)
        self.duration = sum(self.durations_list)

    @staticmethod
    def _validate_durations(argument):
        """
        Validate 'durations_list' input to be a list of non-negative
        numerical values.
        """
        owner = PowerCycleTimeABC
        durations_list = super(owner, owner).validate_list(argument)
        for value in durations_list:
            value = super(owner, owner).validate_nonnegative(value)
        return durations_list


class PowerCyclePhaseError(PowerCycleError):
    """
    Exception class for 'PowerCyclePhase' class of the Power Cycle
    module.
    """

    def _errors(self):
        errors = {
            "breakdown": [
                "The argument given for 'duration_breakdown' is not "
                "valid. It must be a dictionary with keys that are "
                "instances of the 'str' class. Each key should "
                "describe its associated time length value in the "
                "dictionary that characterizes a period that composes "
                "the full duration of an instance of the "
                f"{self._source} class."
            ],
        }
        return errors


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


class PowerCyclePulseError(PowerCycleError):
    """
    Exception class for 'PowerCyclePulse' class of the Power Cycle
    module.
    """

    def _errors(self):
        errors = {}
        return errors


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


class PowerCycleTimelineError(PowerCycleError):
    """
    Exception class for 'PowerCycleTimeline' class of the Power Cycle
    module.
    """

    def _errors(self):
        errors = {}
        return errors


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


class BOPPhaseDependency(Enum):
    """
    Members define possible classifications of an instance of the
    'BOPPhase' class. This classification is to establish the procedure
    taken by the Power Cycle model in terms of time-dependent
    calculations.

    The 'name' of a member describes a time-dependent calculation
    approach to be used in models, while its associated 'value' is
    used in methods throughout the module as a label to quickly assess
    the type of time dependency.
    """

    STEADY_STATE = "ss"
    TRANSIENT = "tt"


class BOPPhase(PowerCyclePhase):
    pass


class BOPPulse(PowerCyclePulse):
    pass


class BOPTimeline(PowerCycleTimeline):
    pass
