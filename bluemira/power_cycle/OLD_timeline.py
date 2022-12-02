"""
Classes to define the timeline for Power Cycle simulations.
"""

'''
from typing import Union  # , List

from bluemira.power_cycle.base import (
    PowerCycleABC,
    PowerCycleError,
)


class PowerCyclePhase(PowerCycleABC):
    """
    Class to define phases for a Power Cycle pulse.

    Parameters
    ----------
    name: str
        Description of the `PowerCyclePhase` instance.
    label: str
        Shorthand label for addressing the `PowerCyclePhase` instance.
    dependency: PowerCycleDependency
        Classification of the `PowerCyclePhase` instance in terms of
        time - dependent calculation.
    duration: float
        Phase duration. [s]
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES
    # ------------------------------------------------------------------

    # Error messages
    @classproperty
    def _errors(cls):
        class_name = cls.__name__
        valid_dependencies = PowerCycleUtilities._join_valid_values(
            cls._valid_dependencies
        )
        e = {
            "label": PowerCycleError(
                "Value",
                f"""
                The argument given for the attribute `label` is not a
                valid value. Instances of the class {class_name} must
                be labeled with strings of 3 characters.
                """,
            ),
            "dependency": PowerCycleError(
                "Value",
                f"""
                The argument given for the attribute `dependency` is not
                a valid value. Only the following values are accepted:
                {valid_dependencies}.
                """,
            ),
            "duration": PowerCycleError(
                "Value",
                """
                The argument given for the attribute `duration` is not
                a valid value. This attribute can only hold non - negative
                numeric values.
                """,
            ),
        }
        return e

    # ------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------
    def __init__(
        self,
        name,
        label: str,
        dependency: str,
        duration: Union[int, float],
    ):

        # Call superclass constructor
        super().__init__(name)

        # Validate label
        self.label = self._validate_label(label)

        # Validate dependency
        self.dependency = self._validate_dependency(dependency)

        # Validate duration
        self.duration = self._validate_duration(duration)

    @classmethod
    def _validate_dependency(cls, dependency):
        """
        Validate `dependency` input for class instance creation to be
        one of the valid values.
        """
        valid_dependencies = cls._valid_dependencies.keys()
        if dependency not in valid_dependencies:
            cls._issue_error("dependency")
        return input

    @classmethod
    def _validate_duration(cls, duration, bad_value=False):
        """
        Validate `duration` input for class instance creation to be
        non-negative numeric value.
        """
        check_numeric = isinstance(duration, (int, float))
        if not check_numeric:
            bad_value = True
        else:
            check_nonnegative = duration >= 0
            if not check_nonnegative:
                bad_value = True

        if bad_value:
            cls._issue_error("duration")
        return duration

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------


# ######################################################################
# POWER CYCLE PULSE
# ######################################################################


class PowerCyclePulse(PowerCycleABC):
    """
    Class to define pulses for a Power Cycle timeline.

    Parameters
    ----------
    name: str
        Description of the `PowerCyclePulse` instance.
    phase_set: PowerCyclePhase | list[PowerCyclePhase]
        List of phases that compose the pulse, ordered.
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES
    # ------------------------------------------------------------------

    _valid_dependencies = {
        "ss": "steady-state",
        "tt": "transient",
    }

    # ------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------

    def __init__(self, name, phase_set):

        # Call superclass constructor
        super().__init__(name)

        # Validate phase set
        self.phase_set = self._validate_phase_set(phase_set)

    @classmethod
    def _validate_phase_set(cls, phase_set):
        """
        Validate 'phase_set' input to be a list of `PowerCyclePhase`
        instances.
        """
        load_set = super()._validate_list(phase_set)
        for element in load_set:
            PowerCyclePhase._validate(element)
        return phase_set

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    def duration(self):
        """
        Retrieves the duration of each phase in the pulse, ordered in a
        dictionary.
        """

        # Retrieve phase_set
        phase_set = self.phase_set

        # Number of phases
        n_phases = len(phase_set)

        # Preallocate output
        duration = {}

        # For each phase index
        for p in range(n_phases):

            # Retrieve phase duration
            phase = phase_set[p]
            phase_duration = phase.duration

            # Store data
            duration[p + 1] = phase_duration

        # Output dictionary
        return duration

    def total_duration(self):
        """
        Compute the total duration of the pulse, based on the duration
        of each pulse phase.
        """

        # Retrieve pulse duration dictionary
        duration = self.duration()

        # Number of phases
        n_phases = len(duration)

        # Preallocate output
        total_duration = 0

        # For each phase index
        for p in range(n_phases):

            # Retrieve phase duration
            phase_duration = duration[p + 1]

            # Add to total duration
            total_duration = total_duration + phase_duration

        # Output sum
        return total_duration


# ######################################################################
# POWER CYCLE TIMELINE
# ######################################################################


class PowerCycleTimeline(PowerCycleABC):
    """
    Class to define a timeline for Power Cycle simulations.

    Parameters
    ----------
    name: str
        Description of the `PowerCycleTimeline` instance.
    pulse_set: list[PowerCyclePulse]

    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES
    # ------------------------------------------------------------------

    _valid_phases = []

    # ------------------------------------------------------------------
    # CONSTRUCTOR
    # ------------------------------------------------------------------
    def __init__(self, name, pulse_set):

        # Call superclass constructor
        super().__init__(name)

        # Validate set of pulses
        self.pulse_set = self._validate_pulse_set(pulse_set)

    @classmethod
    def _validate_pulse_set(cls, pulse_set):
        """
        Validate `pulse_set` input to be a list of instances of the
        `PowerCyclePulse` class.
        """
        return pulse_set

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

'''
