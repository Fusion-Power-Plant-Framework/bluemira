"""
Base classes for the power cycle model.
"""
from abc import ABC, ABCMeta
from typing import Union

from bluemira.base.error import BluemiraError


class PowerCycleError(BluemiraError):
    """
    Exception class for PowerCycle objects.
    """

    pass


class PowerCycleABC(ABC):
    """
    Abstract base class for all classes in the Power Cycle module.

    Parameters
    ----------
    name: str
        Description of the instance.
    """

    def __init__(self, name: str):
        if isinstance(name, str):
            self.name = name
        else:
            raise TypeError(
                "The 'name' attribute of instances of the "
                f"{self.__class__} class must be of the 'str' class."
            )

    # ------------------------------------------------------------------
    #  METHODS
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_list(argument):
        """
        Validate a subclass argument to be a list. If the argument is
        just a single value, insert it in a list.
        """
        if not isinstance(argument, list):
            argument = [argument]
        return argument

    @classmethod
    def _validate(cls, instance):
        """
        Validate `instance` to be an object of the class that calls
        this method.
        """
        class_name = cls.__name__
        if not type(instance) == cls:
            raise TypeError(
                "The tested object is not an instance of the " f"{class_name} class."
            )
        return instance


class NetPowerABC(PowerCycleABC, metaclass=ABCMeta):
    """
    Abstract base class for classes in the Power Cycle module that are
    used to account, sum and manage power loads.

    Parameters
    ----------
    name: str
        Description of the instance.
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES
    # ------------------------------------------------------------------

    # Default number of points (for any plotting method)
    n_points = 50

    # Plot defaults (arguments for `matplotlib.pyplot.plot`)
    _plot_kwargs = {
        "c": "k",  # Line color
        "lw": 2,  # Line width
        "ls": "-",  # Line style
    }
    _scatter_kwargs = {
        "c": "k",  # Marker color
        "s": 100,  # Marker size
        "marker": "x",  # Marker style
    }

    # Plot text settings (for `matplotlib.pyplot.text`)
    _text_angle = 45  # rotation angle
    _text_index = -1  # index of (time,data) point used for location

    # ------------------------------------------------------------------
    # METHODS
    # ------------------------------------------------------------------

    def _validate_n_points(self, n_points: Union[int, None]):
        """
        Validate an 'n_points' argument that specifies a "number of
        points". If the argument is `None`, retrieves the default of
        the class; else it must be a non-negative integer.
        """
        if not n_points:
            n_points = self._n_points
        else:
            n_points = int(n_points)
            if n_points < 0:
                raise PowerCycleError(
                    "The argument given for `n_points` is not a "
                    "valid value for plotting an instance of the "
                    f"{self.__class__} class. Only non-negative "
                    "integers are accepted."
                )
        return n_points

    def _make_secondary_in_plot(self):
        """
        Alters the `_plot_kwargs` and `_text_index` attributes of an
        instance of this class to enforce:
            - more subtle plotting characteristics for lines; and
            - a different location for texts;
        that are displayed on a plot, as to not coincide with the
        primary plot.
        """
        self._text_index = 0
        self._plot_kwargs = {
            "c": "k",  # Line color
            "lw": 1,  # Line width
            "ls": "--",  # Line style
        }
