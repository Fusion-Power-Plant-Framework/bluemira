# COPYRIGHT PLACEHOLDER

"""
Base classes for the power cycle model.
"""
from abc import ABC, ABCMeta, abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import Union

import numpy as np

from bluemira.power_cycle.errors import (
    LoadDataError,
    PowerCycleABCError,
    PowerCycleLoadABCError,
)
from bluemira.power_cycle.tools import (
    copy_dict_without_key,
    unique_and_sorted_vector,
    unnest_list,
    validate_list,
    validate_nonnegative,
    validate_vector,
)


class PowerCycleABC(ABC):
    """
    Abstract base class for all classes in the Power Cycle module.

    Parameters
    ----------
    name: str
        Description of the instance.
    label: str
        Shorthand string to refer to instance in dictionaries or lists.
        An X number of characters is considered the maximum length for
        labels, and this limitation is defined by the class attribute
        '_label_length'. If 'None' is given, the first X characters of
        the 'name' parameter are used as a label.

    """

    _label_length = 3

    def __init__(self, name: str, label=None):
        if not isinstance(name, str):
            raise PowerCycleABCError("name")

        self.name = name
        if label is None:
            label = name[0 : self._label_length]
        elif not isinstance(label, str) and len(label) != self._label_length:
            raise PowerCycleABCError(
                "label",
                f"The argument {label!r} cannot be applied because "
                "labels for this class must be objects of the 'str' "
                f"class with an exact length of {self._label_length!r}.",
            )
        else:
            self.label = label

    @classmethod
    def validate_class(cls, instance):
        """
        Validate 'instance' to be an object of the class that calls
        this method.
        """
        if not isinstance(instance, cls):
            raise PowerCycleABCError("class")
        return instance

    def __eq__(self, other):
        """
        Power Cycle objects should be considered equal even if their
        'name' and 'label' attributes are different.
        """
        if type(self) != type(other):
            return False

        attr_to_ignore = ["name", "label"]
        this_attributes = self.__dict__
        other_attributes = other.__dict__
        for attr in attr_to_ignore:
            this_attributes = copy_dict_without_key(this_attributes, attr)
            other_attributes = copy_dict_without_key(other_attributes, attr)

        return this_attributes == other_attributes

    def __add__(self, other):
        """
        Addition is not defined for 'LoadData' instances.
        """
        raise LoadDataError("add")


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

    def __init__(self, name, durations_list, label=None):
        super().__init__(name, label=label)

        self.durations_list = self._validate_durations(durations_list)
        self.duration = sum(self.durations_list)

    @staticmethod
    def _validate_durations(argument):
        """
        Validate 'durations_list' input to be a list of non-negative
        numerical values.
        """
        durations_list = validate_list(argument)
        for value in durations_list:
            value = validate_nonnegative(value)
        return durations_list

    @staticmethod
    def _build_durations_list(load_set):
        """
        Build a list with the 'duration' attribute of each load in
        the 'load_set' list.
        """
        return [load.duration for load in load_set]


class PowerCycleLoadABC(PowerCycleABC, metaclass=ABCMeta):
    """
    Abstract base class for classes in the Power Cycle module that are
    used to represent and sum power loads.
    """

    # Default number of points (for any plotting method)
    _n_points = 50

    # Index of (time,data) points used as location for 'text'
    _text_index = -1

    # Pyplot defaults (kwargs arguments for 'matplotlib.pyplot' methods)
    _plot_kwargs = {"color": "k", "linewidth": 2, "linestyle": "-"}
    _scatter_kwargs = {"color": "k", "s": 100, "marker": "x"}
    _text_kwargs = {"color": "k", "size": 10, "rotation": 45}

    @staticmethod
    def _recursive_make_consumption_explicit(load_set):
        for element in load_set:
            element.make_consumption_explicit()

    @abstractproperty
    def intrinsic_time(self):
        """
        Every child of 'PowerCycleLoadABC' must define the
        'intrisic_time' property, that collects all of its time-related
        data in a single list.
        """
        pass

    @staticmethod
    def _build_time_from_load_set(load_set):
        return unique_and_sorted_vector(
            unnest_list([load_object.intrinsic_time for load_object in load_set])
        )

    def _validate_n_points(self, n_points: Union[int, float, bool, None]):
        """
        Validate an 'n_points' argument that specifies a "number of
        points". If the argument is 'None', retrieves the default of
        the class; else it must be a non-negative number.
        Non-integer arguments are converted into integers.
        Boolean arguments 'True' and 'False' are treated as 1 and 0
        respectively.
        """
        if not n_points:
            n_points = self._n_points
        else:
            try:
                n_points = int(n_points)
                if n_points < 0:
                    raise PowerCycleLoadABCError(
                        "n_points",
                        f"The value '{n_points}' is negative.",
                    )
            except (TypeError, ValueError):
                raise PowerCycleLoadABCError(
                    "n_points",
                    f"The value '{n_points}' is non-numeric.",
                )
        return n_points

    @staticmethod
    def _refine_vector(vector, n_points):
        """
        Add 'n_point' equidistant points between the extremeties of
        a segment of the input 'vector' (defined by a subsequent pair
        of points).
        """
        try:
            vector = validate_vector(vector)
        except PowerCycleABCError:
            raise PowerCycleLoadABCError("refine_vector")

        if (n_points is None) or (n_points == 0):
            refined_vector = vector
        else:
            refined_vector = []
            for s in range(len(vector) - 1):
                refined_vector += np.linspace(
                    vector[s],
                    vector[s + 1],
                    n_points + 1,
                    endpoint=False,
                ).tolist()
            refined_vector.append(vector[-1])

        return refined_vector

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
        self._plot_kwargs = {"color": "k", "linewidth": 1, "linestyle": "--"}
        self._scatter_kwargs = {"color": "k", "s": 100, "marker": "x"}
        self._text_kwargs = {"color": "k", "size": 10, "rotation": 45}

    def _add_text_to_point_in_plot(
        self,
        axes,
        name,
        x_list,
        y_list,
        **kwargs,
    ):
        text_to_be_added = f"{name} ({type(self).__name__})"
        label_of_text_object = f"{name} (name)"

        # Fall back on default kwargs if wrong keys are passed
        try:
            plot_object = axes.text(
                x_list[self._text_index],
                y_list[self._text_index],
                text_to_be_added,
                label=label_of_text_object,
                **{**self._text_kwargs, **kwargs},
            )
        except AttributeError:
            plot_object = axes.text(
                x_list[self._text_index],
                y_list[self._text_index],
                text_to_be_added,
                label=label_of_text_object,
                **self._text_kwargs,
            )
        return plot_object

    def plot_obj(self, ax, x, y, name, kwargs, plot_or_scatter=True):
        if plot_or_scatter:
            plot = getattr(ax, "plot")
            lab = "(curve)"
            final_kwargs = {**self._plot_kwargs, **kwargs}
        else:
            plot = getattr(ax, "scatter")
            lab = "(data)"
            final_kwargs = {**self._scatter_kwargs, **kwargs}

        return [
            plot(x, y, label=f"{name} {lab}", **final_kwargs),
            self._add_text_to_point_in_plot(ax, name, x, y, **kwargs),
        ]

    def __radd__(self, other):
        """
        The reverse addition operator, to enable the 'sum' method for
        children classes that define the '__add__' method.
        """
        if other == 0:
            return self
        else:
            return self.__add__(other)


@dataclass
class PhaseLoadConfig:
    phase_list: list
    consumption: bool
    normalise_list: list
    powerload_list: list


class PowerCycleImporterABC(metaclass=ABCMeta):
    """
    Abstract base class for classes in the NET submodule of the Power
    Cycle module that are used to import data from other Bluemira
    modules into the Power Cycle module.
    """

    @staticmethod
    @abstractmethod
    def duration(variables_map):
        """
        Every child of 'PowerCycleImporterABC' must define the
        'duration' method, that returns single time-related values
        from other BLUEMIRA modules to be used in the
        'duration_breakdown' parameter of 'PowerCyclePhase' objects.
        """
        pass

    @staticmethod
    @abstractmethod
    def phaseload_inputs(variables_map):
        """
        Every child of 'PowerCycleImporterABC' must define the
        'phaseload_inputs' method, that returns a 'dict' with the
        following keys and values:
            - "phase_list": 'list' of 'str',
            - "consumption: 'bool',
            - "normalise_list": 'list' of 'bool',
            - "powerload_list": 'list' of ''PowerLoad' objects.

        Strings in "phase_list" indicate the phase labels in which the
        phase load is applied.
        The boolean in "consumption" indicate whether these loads are of
        power consumption or production (and thus positive or negative
        contributions in the net power balance).
        The booleans in "normalise_list" enforce time normalization to
        each "PowerLoad" in the "powerload_list", when each 'PhaseLoad'
        is instantiated.


        The dictionary should be constructed with data from other
        BLUEMIRA modules to be used in the '_build_phaseloads' method
        of a 'PowerCycleManager' object.
        """
        pass
