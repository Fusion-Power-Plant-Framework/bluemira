# COPYRIGHT PLACEHOLDER

"""
Exception classes for the power cycle model.
"""
from abc import abstractmethod
from typing import Dict

from bluemira.base.error import BluemiraError


class PowerCycleError(BluemiraError):
    """
    Exception class for Power Cycle classes.

    Class should be named in the following format: '(Class)Error',
    where '(Class)' is the name of the associated class (source) for
    which the exception class is being defined.
    """

    def __init__(self, case=None, msg=""):
        try:
            msg += f" {self._errors()[case]}"
        except KeyError:
            raise ValueError(
                f"The requested error case '{case}' has not been "
                f"defined for the error class '{self.__class__}'."
            ) from None
        super().__init__(msg)

    def _source(self):
        class_name = self.__class__
        source_class = class_name.replace("Error", "")
        source_class = "'" + source_class + "'"
        return source_class

    @abstractmethod
    def _errors(self) -> Dict:
        pass


# ######################################################################
# BASE
# ######################################################################


class PowerCycleABCError(PowerCycleError):
    """
    Exception class for 'PowerCycleABC' class of the Power Cycle module.
    """

    def _errors(self):
        errors = {
            "name": (
                "Invalid 'name' parameter. The 'name' attribute of ' "
                f"instances of the {self._source} class must be of "
                "the 'str' class."
            ),
            "class": [
                "Invalid instance. The tested object is not an "
                f"instance of the {self._source} class."
            ],
            "numerical": (
                "Invalid value. The tested value must be an instance "
                "of either the 'int' or 'float' classes to be "
                "processed by this instance of a child class of the "
                f"{self._source} class."
            ),
            "nonnegative": [
                "Invalid value. The tested value must be a non-"
                "numerical negative value to be processed by this "
                f"instance of a child class of the {self._source} "
                "class."
            ],
        }
        return errors


class PowerCycleTimeABCError(PowerCycleError):
    """
    Exception class for 'PowerCycleTimeABC' class of the Power Cycle
    module.
    """

    def _errors(self):
        errors = {}
        return errors


class NetPowerABCError(PowerCycleError):
    """
    Exception class for 'NetPowerABC' class of the Power Cycle module.
    """

    def _errors(self):
        errors = {
            "n_points": [
                "The argument given for 'n_points' is not a valid "
                f"value for plotting an instance of the {self._source} "
                "class. Only non-negative integers are accepted."
            ],
            "refine_vector": [
                "The argument given for 'vector' is not a valid "
                "value. Only lists of numerical values are accepted."
            ],
        }
        return errors


# ######################################################################
# TIME
# ######################################################################


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


class PowerCyclePulseError(PowerCycleError):
    """
    Exception class for 'PowerCyclePulse' class of the Power Cycle
    module.
    """

    def _errors(self):
        errors = {}
        return errors


class PowerCycleTimelineError(PowerCycleError):
    """
    Exception class for 'PowerCycleTimeline' class of the Power Cycle
    module.
    """

    def _errors(self):
        errors = {}
        return errors


# ######################################################################
# NET
# ######################################################################


class PowerDataError(PowerCycleError):
    """
    Exception class for 'PowerData' class of the Power Cycle module.
    """

    def _errors(self):
        errors = {
            "increasing": [
                "The 'time' parameter used to create an instance of "
                f"the {self._source} class must be an increasing list.",
            ],
            "sanity": [
                "The attributes 'data' and 'time' of an instance of "
                f"the {self._source} class must have the same length."
            ],
            "time": [
                "The 'intrinsic_time' property of an instance of the "
                f"{self._source} class cannot be set.",
            ],
        }
        return errors


class PowerLoadError(PowerCycleError):
    """
    Exception class for 'PowerLoad' class of the Power Cycle module.
    """

    def _errors(self):
        errors = {
            "model": [
                "The argument given for the attribute 'model' is not "
                "a valid value. A model must be specified with an "
                "instance of the 'PowerLoadModel' 'Enum' class."
            ],
            "sanity": [
                "The attributes 'load' and 'model' of an instance of "
                f"the {self._source} class must have the same length."
            ],
            "curve": [
                "The 'time' input used to create a curve with an "
                f"instance of the {self._source} class must be numeric "
                "or a list of numeric values.",
            ],
            "time": [
                "The 'intrinsic_time' property of an instance of the "
                f"{self._source} class cannot be set; it is instead "
                "built from the 'intrinsic_time' attributes of the "
                "'PowerData' objects stored in the 'powerdata_set' "
                "attribute.",
            ],
        }
        return errors


class PhaseLoadError(PowerCycleError):
    """
    Exception class for 'PhaseLoad' class of the Power Cycle module.
    """

    def _errors(self):
        errors = {
            "normalize": [
                "The argument given for 'normalize' is not a valid "
                f"value for an instance of the {self._source} class. "
                "Each element of 'normalize' must be a boolean."
            ],
            "sanity": [
                "The attributes 'load_set' and 'normalize' of an "
                f"instance of the {self._source} class must have the "
                "same length."
            ],
            "display_data": [
                "The argument passed to the 'display_data' method of "
                f"the {self._source} class for the input 'option' is "
                "not valid. Only the strings 'load' and 'normal' are "
                "accepted.",
            ],
            "normalized_set": [
                "The 'normalized_set' property of an instance of the "
                f"{self._source} class cannot be set; it is instead "
                "calculated from the 'powerload_set' and 'phase' "
                "attributes.",
            ],
            "time": [
                "The time properties of an instance of the "
                f"{self._source} class cannot be set.",
            ],
        }
        return errors


class PulseLoadError(PowerCycleError):
    """
    Exception class for 'PulseLoad' class of the Power Cycle module.
    """

    def _errors(self):
        errors = {
            "shifted_set": [
                "The 'shifted_set' property of an instance of the "
                f"{self._source} class cannot be set; it is instead "
                "calculated from the 'phaseload_set' attribute.",
            ],
            "time": [
                "The time properties of an instance of the "
                f"{self._source} class cannot be set.",
            ],
        }
        return errors


# ######################################################################
# BOP
# ######################################################################


class BOPPhaseError(PowerCycleError):
    """
    Exception class for 'BOPPhase' class of the Power Cycle module.
    """

    def _errors(self):
        errors = {
            "dependency": (
                "The 'dependency' parameter used to create an instance "
                f"of the {self._source} class must be an instance of "
                "the 'BOPPhaseDependency' class."
            ),
        }
        return errors


class BOPPulseError(PowerCycleError):
    pass


class BOPTimelineError(PowerCycleError):
    pass
