"""
Base classes for the power cycle model.
"""
from abc import ABC, abstractmethod
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
        message = msg
        if case:
            all_errors = self._errors()
            if case in all_errors:
                extra_msg = str(all_errors[case])
                message = message + " " + extra_msg
            else:
                raise ValueError(
                    f"The requested error case '{case}' has not been "
                    f"defined for the error class '{self.__class__}'."
                )
        super().__init__(message)

    def _source(self):
        class_name = self.__class__
        source_class = class_name.replace("Error", "")
        source_class = "'" + source_class + "'"
        return source_class

    @abstractmethod
    def _errors(self) -> Dict:
        pass


class PowerCycleABCError(PowerCycleError):
    """
    Exception class for 'PowerCycleABC' class of the Power Cycle module.
    """

    def _errors(self):
        errors = {
            "name": [
                "Invalid 'name' parameter. The 'name' attribute of ' "
                f"instances of the {self._source} class must be of "
                "the 'str' class.",
            ],
            "class": [
                "Invalid instance. The tested object is not an "
                f"instance of the {self._source} class."
            ],
            "nonnegative": [
                "Invalid value. The tested value must be a non-"
                "numerical negative value to be processed by this "
                f"instance of a child class of the {self._source} "
                "class."
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
            raise PowerCycleABCError("name")
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

    @staticmethod
    def validate_nonnegative(argument):
        """
        Validate an argument to be a nonnegative numerical value (i.e.
        an instance of either the 'int' or the 'float' classes).
        """
        if isinstance(argument, int) or isinstance(argument, float):
            if argument >= 0:
                return argument
            else:
                raise PowerCycleABCError(
                    "nonnegative",
                    "The value is negative.",
                )
        else:
            raise PowerCycleABCError(
                "nonnegative",
                "The value is not an instance of either the 'int' nor "
                "the 'float' classes.",
            )

    @classmethod
    def validate_class(cls, instance):
        """
        Validate 'instance' to be an object of the class that calls
        this method.
        """
        if not isinstance(instance, cls):
            raise PowerCycleABCError("class")
        return instance
