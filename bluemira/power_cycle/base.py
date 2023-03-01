# COPYRIGHT PLACEHOLDER

"""
Base classes for the power cycle model.
"""
from abc import ABC, ABCMeta, abstractproperty
from typing import Union

import numpy as np

from bluemira.power_cycle.errors import NetPowerABCError, PowerCycleABCError
from bluemira.power_cycle.tools import unnest_list


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

    @classmethod
    def validate_class(cls, instance):
        """
        Validate 'instance' to be an object of the class that calls
        this method.
        """
        if not isinstance(instance, cls):
            raise PowerCycleABCError("class")
        return instance

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
    def validate_numerical(argument):
        """
        Validate an argument to be a numerical value (i.e. an instance
        of either the 'int' or the 'float' classes).
        """
        if isinstance(argument, int) or isinstance(argument, float):
            return argument
        else:
            argument_class = type(argument)
            raise PowerCycleABCError(
                "numerical",
                f"The value is an object of the {argument_class!r} " "class instead.",
            )

    @staticmethod
    def validate_nonnegative(argument):
        """
        Validate an argument to be a nonnegative numerical value.
        """
        argument = PowerCycleABC.validate_numerical(argument)
        if argument >= 0:
            return argument
        else:
            raise PowerCycleABCError(
                "nonnegative",
                "The value is negative.",
            )

    @staticmethod
    def validate_vector(argument):
        """
        Validate an argument to be a numerical list.
        """
        argument = PowerCycleABC.validate_list(argument)
        for element in argument:
            element = PowerCycleABC.validate_numerical(element)
        return argument


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


class NetPowerABC(PowerCycleABC, metaclass=ABCMeta):
    """
    Abstract base class for classes in the NET submodule of the Power
    Cycle module that are used to account, sum and manage power loads.
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES
    # ------------------------------------------------------------------

    # Default number of points (for any plotting method)
    _n_points = 50

    # Index of (time,data) points used as location for 'text'
    _text_index = -1

    # Pyplot defaults (kwargs arguments for 'matplotlib.pyplot' methods)
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
    _text_kwargs = {
        "c": "k",  # Font color
        "size": 10,  # Font size
        "rotation": 45,  # Rotation angle (º)
    }

    # ------------------------------------------------------------------
    # METHODS
    # ------------------------------------------------------------------

    @abstractproperty
    def intrinsic_time(self):
        pass

    def _validate_n_points(self, n_points: Union[int, None]):
        """
        Validate an 'n_points' argument that specifies a "number of
        points". If the argument is 'None', retrieves the default of
        the class; else it must be a non-negative integer.
        """
        if not n_points:
            n_points = self._n_points
        else:
            try:
                n_points = int(n_points)
                if n_points < 0:
                    raise NetPowerABCError(
                        "n_points",
                        f"The value '{n_points}' is negative.",
                    )
            except (TypeError, ValueError):
                raise NetPowerABCError(
                    "n_points",
                    f"The value '{n_points}' is non-numeric.",
                )
        return n_points

    @staticmethod
    def _refine_vector(vector, n_points):
        """
        Add 'n_point' equidistant points between each segment (defined
        by a subsequent pair of points) in the input 'vector'.
        """
        try:
            vector = PowerCycleABC.validate_vector(vector)
        except PowerCycleABCError:
            raise NetPowerABCError("refine_vector")

        number_of_curve_segments = len(vector) - 1
        if (n_points is None) or (n_points == 0):
            refined_vector = vector
        else:
            refined_vector = []
            for s in range(number_of_curve_segments):
                first_point = vector[s]
                second_point = vector[s + 1]
                refined_segment = np.linspace(
                    first_point,
                    second_point,
                    n_points + 1,
                    endpoint=False,
                )
                refined_segment = refined_segment.tolist()
                refined_vector = refined_vector + refined_segment
            refined_vector.append(vector[-1])

        return refined_vector

    @staticmethod
    def _unique_and_sorted_vector(vector):
        vector = PowerCycleABC.validate_vector(vector)
        unique_vector = list(set(vector))
        sorted_vector = sorted(unique_vector)
        return sorted_vector

    @staticmethod
    def _build_time_from_power_set(power_set):
        all_times = [power_object.intrinsic_time for power_object in power_set]
        unnested_times = unnest_list(all_times)
        time = NetPowerABC._unique_and_sorted_vector(unnested_times)
        return time

    def _make_secondary_in_plot(self):
        """
        Alters the '_text_index' and kwargs attributes of an instance
        of this class to enforce:
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
        self._scatter_kwargs = {
            "c": "k",  # Marker color
            "s": 100,  # Marker size
            "marker": "x",  # Marker style
        }
        self._text_kwargs = {
            "c": "k",  # Font color
            "size": 10,  # Font size
            "rotation": 45,  # Rotation angle (º)
        }

    def _add_text_to_point_in_plot(
        self,
        axes,
        name,
        x_list,
        y_list,
        **kwargs,
    ):
        class_calling_method = self.__class__.__name__
        text_to_be_added = f"{name} ({class_calling_method})"
        label_of_text_object = name + " (name)"

        # Set each default options in kwargs, if not specified
        default_text_kwargs = self._text_kwargs
        final_kwargs = {**default_text_kwargs, **kwargs}

        # Filter kwargs

        index_for_text_placement = self._text_index
        plot_object = axes.text(
            x_list[index_for_text_placement],
            y_list[index_for_text_placement],
            text_to_be_added,
            label=label_of_text_object,
            **final_kwargs,
        )
        return plot_object
