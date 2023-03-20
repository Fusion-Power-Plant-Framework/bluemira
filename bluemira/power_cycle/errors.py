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
        source_class = f"{source_class!r}"
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
            "label": (
                f"Invalid 'label' parameter for an instance of the {self._source}."
            ),
            "class": (
                "Invalid instance. The tested object is not an "
                f"instance of the {self._source} class."
            ),
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


class PowerCycleLoadABCError(PowerCycleError):
    """
    Exception class for 'PowerCycleLoadABC' class of the Power Cycle
    module.
    """

    def _errors(self):
        errors = {
            "n_points": (
                "The argument given for 'n_points' is not a valid "
                f"value for plotting an instance of the {self._source} "
                "class. Only non-negative integers are accepted."
            ),
            "refine_vector": (
                "The argument given for 'vector' is not a valid "
                "value. Only lists of numerical values are accepted."
            ),
        }
        return errors


class PowerCycleImporterABCError(PowerCycleError):
    """
    Exception class for 'PowerCycleImporterABC' class of the Power Cycle
    module.
    """

    def _errors(self):
        errors = {}
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
            "breakdown": (
                "The argument given for 'duration_breakdown' is not "
                "valid. It must be a dictionary with keys that are "
                "instances of the 'str' class. Each key should "
                "describe its associated time length value in the "
                "dictionary that characterizes a period that composes "
                "the full duration of an instance of the "
                f"{self._source} class."
            ),
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


class PowerCycleScenarioError(PowerCycleError):
    """
    Exception class for 'PowerCycleScenario' class of the Power Cycle
    module.
    """

    def _errors(self):
        errors = {}
        return errors


class ScenarioBuilderError(PowerCycleError):
    """
    Exception class for 'ScenarioBuilder' class of the Power Cycle
    module.
    """

    def _errors(self):
        errors = {
            "config": (
                "An incorrect file has been passed as 'config'. It "
                "must be a JSON file with a structure that matches "
                f"the 'struct' class attributes of the {self._source} "
                "class."
            ),
            "import": ("Bluemira data import failed."),
            "library": ("Requested element is not present in library."),
            "operator": ("Breakdown for phase could not be built."),
        }
        return errors


# ######################################################################
# NET LOADS
# ######################################################################


class LoadDataError(PowerCycleError):
    """
    Exception class for 'LoadData' class of the Power Cycle module.
    """

    def _errors(self):
        errors = {
            "increasing": (
                "The 'time' parameter used to create an instance of "
                f"the {self._source} class must be an increasing list."
            ),
            "sanity": (
                "The attributes 'data' and 'time' of an instance of "
                f"the {self._source} class must have the same length."
            ),
            "time": (
                "The 'intrinsic_time' property of an instance of the "
                f"{self._source} class cannot be set.",
            ),
            "add": (
                "Addition is not defined and should not be called for "
                f"instances of the {self._source} class.",
            ),
        }
        return errors


class LoadModelError(PowerCycleError):
    """
    Exception class for 'LoadModel' class of the Power Cycle module.
    """

    def _errors(self):
        errors = {}
        return errors


class PowerLoadError(PowerCycleError):
    """
    Exception class for 'PowerLoad' class of the Power Cycle module.
    """

    def _errors(self):
        errors = {
            "loadmodel": (
                "The argument given for the attribute 'model' is not "
                "a valid value. A model must be specified with an "
                "instance of the 'PowerLoadModel' 'Enum' class."
            ),
            "sanity": (
                "The attributes 'load' and 'model' of an instance of "
                f"the {self._source} class must have the same length."
            ),
            "curve": (
                "The 'time' input used to create a curve with an "
                f"instance of the {self._source} class must be numeric "
                "or a list of numeric values."
            ),
            "time": (
                "The 'intrinsic_time' property of an instance of the "
                f"{self._source} class cannot be set; it is instead "
                "built from the 'intrinsic_time' attributes of the "
                "'LoadData' objects stored in the 'loaddata_set' "
                "attribute.",
            ),
        }
        return errors


class PhaseLoadError(PowerCycleError):
    """
    Exception class for 'PhaseLoad' class of the Power Cycle module.
    """

    def _errors(self):
        errors = {
            "normalize": (
                "The argument given for 'normalize' is not a valid "
                f"value for an instance of the {self._source} class. "
                "Each element of 'normalize' must be a boolean."
            ),
            "sanity": (
                "The attributes 'load_set' and 'normalize' of an "
                f"instance of the {self._source} class must have the "
                "same length."
            ),
            "display_data": (
                "The argument passed to the 'display_data' method of "
                f"the {self._source} class for the input 'option' is "
                "not valid. Only the strings 'load' and 'normal' are "
                "accepted."
            ),
            "normalized_set": (
                "The 'normalized_set' property of an instance of the "
                f"{self._source} class cannot be set; it is instead "
                "calculated from the 'powerload_set' and 'phase' "
                "attributes."
            ),
            "time": (
                "The time properties of an instance of the "
                f"{self._source} class cannot be set."
            ),
            "addition": (
                f"Instances of the {self._source} class can only be "
                "added if their 'phase' attributes represent the same "
                "pulse phase."
            ),
        }
        return errors


class PulseLoadError(PowerCycleError):
    """
    Exception class for 'PulseLoad' class of the Power Cycle module.
    """

    def _errors(self):
        errors = {
            "shifted_set": (
                "The 'shifted_set' property of an instance of the "
                f"{self._source} class cannot be set; it is instead "
                "calculated from the 'phaseload_set' attribute."
            ),
            "time": (
                "The time properties of an instance of the "
                f"{self._source} class cannot be set."
            ),
            "addition": (
                f"Instances of the {self._source} class can only be "
                "added if their 'pulse' attributes represent the same "
                "pulse."
            ),
        }
        return errors


class ScenarioLoadError(PowerCycleError):
    """
    Exception class for 'ScenarioLoad' class of the Power Cycle module.
    """

    def _errors(self):
        errors = {}
        return errors


# ######################################################################
# NET IMPORTERS
# ######################################################################


class EquilibriaImporterError(PowerCycleError):
    """
    Exception class for 'EquilibriaImporter' class of the Power Cycle
    module.
    """

    def _errors(self):
        errors = {
            "duration": ("Unable to import duration from 'equilibria' module."),
            "load": ("Unable to import load from 'equilibria' module."),
        }
        return errors


class PumpingImporterError(PowerCycleError):
    """
    Exception class for 'PumpingImporter' class of the Power Cycle
    module.
    """

    def _errors(self):
        errors = {
            "duration": ("Unable to import duration from 'pumping' module."),
        }
        return errors


# ######################################################################
# NET MANAGER
# ######################################################################


class PowerCycleSystemError(PowerCycleError):
    """
    Exception class for 'PowerCycleSystem' class of the Power Cycle
    module.
    """

    def _errors(self):
        errors = {
            "scenario": ("Scenario is incorrect."),
            "import": ("Bluemira data import failed."),
        }
        return errors


class PowerCycleGroupError(PowerCycleError):
    """
    Exception class for 'PowerCycleGroup' class of the Power Cycle
    module.
    """

    def _errors(self):
        errors = {}
        return errors


class PowerCycleManagerError(PowerCycleError):
    """
    Exception class for 'PowerCycleManager' class of the Power Cycle
    module.
    """

    def _errors(self):
        errors = {
            "load-type": ("The argument passed as 'load_type' is incorrect."),
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
    """
    Exception class for 'BOPPulse' class of the Power Cycle module.
    """

    def _errors(self):
        errors = {}
        return errors


class BOPScenarioError(PowerCycleError):
    """
    Exception class for 'BOPScenario' class of the Power Cycle module.
    """

    def _errors(self):
        errors = {}
        return errors
