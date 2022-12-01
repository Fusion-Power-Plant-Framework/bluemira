"""
Classes to define the timeline for Power Cycle simulations.
"""
from enum import Enum
from typing import Dict, Union

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
        Description of the `PowerCyclePhase` instance.
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


class BOPPhaseDependency(Enum):
    """
    Possible classifications of an instance of the `PowerCyclePhase`
    class in terms of time-dependent calculation.
    """

    STEADY_STATE = "ss"
    TRANSIENT = "tt"


class BOPPhase(PowerCyclePhase):
    pass
