"""
Base classes for the power cycle model.
"""
from abc import ABC, abstractmethod
from typing import Dict

from bluemira.base.error import BluemiraError


class PowerCycleError(BluemiraError):
    """
    Exception class for PowerCycle objects.
    """

    def __init__(self, case=None, msg=""):
        message = msg
        if case:
            all_errors = self._errors()
            if case in all_errors:
                extra_msg = all_errors[case]
                message = message + extra_msg
            else:
                raise ValueError(
                    f"The requested error case '{case}' has not been "
                    f"defined for the error class '{self.__class__}'."
                )
        super().__init__(message)

    @abstractmethod
    def _errors(self) -> Dict:
        pass


class PowerCycleABCError(PowerCycleError):
    def _errors(self, case):
        errors = {
            "name": [
                "Invalid 'name' parameter.",
            ],
            "class": [
                "Invalid instance.",
            ],
        }
        return errors


class PowerCycleABC(ABC):
    """
    Abstract base class for all classes in the Power Cycle module.

    Parameters
    ----------
    name: str
        Description of the instance.
    """

    def __init__(self, name: str):
        self.name = self._validate_name(name)

    # ------------------------------------------------------------------
    #  METHODS
    # ------------------------------------------------------------------

    def _validate_name(self, argument):
        """
        Validate an argument to be an instance of the 'str' class to be
        considered a valid name for an instance of a child class of the
        PowerCycleABC class.
        """
        if not isinstance(argument, str):
            raise PowerCycleABCError(
                None,
                "The 'name' attribute of instances of the "
                f"{self.__class__} class must be of the 'str' class.",
            )
        return argument

    @staticmethod
    def validate_list(argument):
        """
        Validate an argument to be a list. If the argument is just a
        single value, insert it in a list.
        """
        if not isinstance(argument, list):
            argument = [argument]
        return argument

    @classmethod
    def validate_class(cls, instance):
        """
        Validate `instance` to be an object of the class that calls
        this method.
        """
        class_name = cls.__name__
        if type(instance) != cls:
            raise PowerCycleABCError(
                None,
                "Invalid object. The tested object is not an "
                f"instance of the '{class_name}' class.",
            )
        return instance
