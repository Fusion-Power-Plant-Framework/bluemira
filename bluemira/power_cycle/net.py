"""
Classes for the calculation of net power in the power cycle model.
"""
from abc import ABCMeta
from typing import Union

from bluemira.power_cycle.base import PowerCycleABC, PowerCycleError


class NetPowerABCError(PowerCycleError):
    def _errors(self, case):
        errors = {
            "n_points": [
                "The argument given for 'n_points' is not a valid "
                "value; only non-negative integers are accepted."
            ],
        }
        return errors


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
    _n_points = 50

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
            try:
                n_points = int(n_points)
                if n_points < 0:
                    raise NetPowerABCError(
                        None,
                        f"An instance of the {self.__class__} class "
                        f"cannot be plotted with '{n_points}' points. ",
                    )
            except (TypeError, ValueError):
                raise NetPowerABCError(
                    None,
                    f"The non-numeric value '{n_points}' cannot be "
                    f"used to plot an instance of the {self.__class__} "
                    "class.",
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
