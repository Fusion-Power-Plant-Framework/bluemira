# COPYRIGHT PLACEHOLDER

"""
Classes to define the timeline for BOP simulations.
"""
from enum import Enum

from bluemira.power_cycle.errors import BOPPhaseError
from bluemira.power_cycle.time import (
    PowerCyclePhase,
    PowerCyclePulse,
    PowerCycleTimeline,
)


class BOPPhaseDependency(Enum):
    """
    Members define possible classifications of an instance of the
    'BOPPhase' class. This classification is to establish the procedure
    taken by the Power Cycle model in terms of time-dependent
    calculations.

    The 'name' of a member is a 'str' that describes a time-dependent
    calculation approach to be used in models, while its associated
    'value' is a 'str' used in methods throughout the module as a label
    to quickly assess the type of time dependency.
    """

    STEADY_STATE = "ss"
    TRANSIENT = "tt"


class BOPPhase(PowerCyclePhase):
    """
    Class to define pulses for a Power Cycle timeline, to be used by
    the BOP submodule.

    This class is a child of the 'PowerCyclePhase' class and uses that
    documentation in addition to the one below.

    Parameters
    ----------
    dependency: BOPPhaseDependency
        Classification of the instance in regards to methodology for
        computing time-dependent responses.
    """

    def __init__(
        self,
        name,
        duration_breakdown,
        dependency: BOPPhaseDependency,
    ):
        super().__init__(name, duration_breakdown)
        self.dependency = self._validate_dependency(dependency)

    @staticmethod
    def _validate_dependency(dependency):
        """
        Validate 'dependency' input to be an instance of the
        'BOPPhaseDependency' class.
        """
        if type(dependency) != BOPPhaseDependency:
            dependency_class = type(dependency)
            raise BOPPhaseError(
                "dependency",
                "The argument provided is an instance of "
                f"the '{dependency_class}' class instead.",
            )


class BOPPulse(PowerCyclePulse):
    pass


class BOPTimeline(PowerCycleTimeline):
    pass
