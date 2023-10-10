# COPYRIGHT PLACEHOLDER

"""
Classes for the definition of power loads in the power cycle model.
"""
import copy
from abc import ABC
from typing import Iterable, List, Union

import numpy as np

from bluemira.base.constants import EPS
from bluemira.power_cycle.errors import (
    PhaseLoadError,
    PulseLoadError,
)
from bluemira.power_cycle.tools import validate_axes


class PowerCycleLoadABC(ABC):
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

    def __eq__(self, other):
        """
        Power Cycle objects should be considered equal even if their
        'name' and 'label' attributes are different.
        """
        if type(self) != type(other):
            return False

        attributes = list(self.__dict__.keys())
        for attr in ["name", "label"]:
            attributes.pop(attributes.index(attr))

        for attr in attributes:
            check = getattr(self, attr) == getattr(other, attr)
            if isinstance(check, Iterable):
                if not all(check):
                    return False
            elif not check:
                return False
        return True

    @staticmethod
    def build_timeseries(load_set):
        return np.unique(
            np.concatenate([load_object.intrinsic_time for load_object in load_set])
        )

    def __radd__(self, other):
        """
        The reverse addition operator, to enable the 'sum' method for
        children classes that define the '__add__' method.
        """
        if other == 0:
            return self
        return self.__add__(other)


class LoadData(PowerCycleLoadABC):
    """
    Class to store a set of time and data vectors, to be used in
    creating 'PowerLoad' objects.

    Takes a pair of (time,data) vectors and creates a 'LoadData' object
    used to build power load objects to represent the time evolution
    of a given power in the plant.
    Instances of this class do not specify any dependence between the
    data points it stores, so no method is defined for altering (e.g.
    applying a multiplicative efficiency) or calculating related values
    (e.g. interpolation). Instead, these actions are performed with
    objects of the 'PowerLoad' class, that are built with instances of
    'LoadData'.

    Parameters
    ----------
    name: str
        Description of the 'LoadData' instance.
    time: int | float | list[int | float]
        List of time values that define the LoadData. [s]
    data: int | float | list[int | float]
        List of power values that define the LoadData. [W]

    Properties
    ----------
    intrinsic_time: list[int | float]
        Deep copy of the list stored in the 'time' attribute.
    """

    def __init__(
        self,
        name,
        time: List[Union[int, float]],
        data: List[Union[int, float]],
    ):
        self.name = name
        self.data = data
        self.time = time

        self._norm = []  # Memory for time normalization
        self._shift = []  # Memory for time shifting

    def _normalise_time(self, new_end_time):
        """
        Normalize values stored in the 'time' attribute, so that the
        last time value coincides with 'new_end_time'.
        Stores the normalization factor in the attribute '_norm', which
        is always initialized as an empty list.
        """
        norm = new_end_time / self.time[-1]
        self.time *= norm
        self._norm.append(norm)

    def _shift_time(self, time_shift):
        """
        Shift all time values in the 'time' attribute by the numerical
        value 'time_shift'.
        Stores the shifting factor in the attribute '_shift', which
        is always initialized as an empty list.
        """
        self.time += time_shift
        self._shift.append(time_shift)


class PhaseLoad(PowerCycleLoadABC):
    """
    Generic representation of the total power load during a pulse phase.

    Defines the phase load with a set of 'PowerLoad' instances. Each
    instance must be accompanied by a 'normalise' specification, used to
    indicate whether that power load must have its curve normalised in
    time in respect to the 'duration' of the 'phase' Parameter. This
    enables the instance to adjust the evolution of power loads
    accordingly, if changes occur to the plant pulse.

    Parameters
    ----------
    name: str
        Description of the 'PhaseLoad' instance.
    phase: PowerCyclePhase
        Pulse phase specification, that determines in which phase the
        load happens.
    powerload_set: PowerLoad | list[PowerLoad]
        Collection of instances of the 'PowerLoad' class that define
        the 'PhaseLoad' object.
    normalise: bool | list[bool]
        List of boolean values that defines which elements of
        'powerload_set' have their time-dependence normalised in respect
        to the phase duration. A value of 'True' forces a normalization,
        while a value of 'False' does not and time values beyond the
        phase duration are ignored.

    Properties
    ----------
    intrinsic_time: list[int | float]
        List that contains all values in the 'intrinsic_time' properties
        of the different 'PowerLoad' objects contained in the
        'powerload_set' attribute, ordered and with no repetitions.
    normalised_time: list[int | float]
        List that contains all values in the 'intrinsic_time' properties
        of the different 'PowerLoad' objects contained in the
        '_normalised_set' attribute, ordered and with no repetitions.
    """

    # Override number of points
    _n_points = 100

    # Override pyplot defaults
    _plot_defaults = {"color": "k", "linewidth": 2, "linestyle": "-"}

    # Defaults for detailed plots
    _detailed_defaults = {"color": "k", "linewidth": 1, "linestyle": "--"}

    def __init__(self, name, phase, powerload_set, normalise):
        self.name = name
        self.phase = phase
        self.powerload_set = powerload_set
        self.normalise = normalise

    @classmethod
    def null(cls, phase):
        """
        Instantiates an null version of the class.
        """
        return cls(
            f"Null PhaseLoad for phase {phase.name}",
            phase,
            PowerLoad.null(),
            True,
        )

    @property
    def _normalised_time_of_set(self):
        """
        Modified 'powerload_set' attribute, in which all times are
        normalised in respect to the 'duration' of the 'phase'
        attribute.
        """
        normalised_set = copy.deepcopy(self.powerload_set)
        for index in np.where(self.normalise)[0]:
            normalised_set[index]._normalise_time(self.phase.duration)
        return normalised_set

    def curve(self, time):
        """
        Create a curve by calculating power load values at the specified
        times.

        This method applies the 'curve' method of the 'PowerLoad' class
        to the 'PowerLoad' instance that is created by the sum of all
        'PowerLoad' objects stored in the 'powerload_set' attribute.

        Parameters
        ----------
        time: int | float | list[ int | float ]
            List of time values. [s]

        Returns
        -------
        curve: list[float]
            List of power values. [W]
        """
        return sum(self._normalised_time_of_set).load_total(time)

    def make_consumption_explicit(self):
        """
        Calls 'make_consumption_explicit' on every element of the
        'powerload_set' attribute.
        """
        return [pl.get_explicit_data_consumption() for pl in self.powerload_set]

    @property
    def intrinsic_time(self, normalised=False):
        """
        Single time vector that contains all values used to define the
        different 'PowerLoad' objects contained in the 'powerload_set'
        attribute (i.e. all times are their original values).
        """
        return self.build_timeseries(
            self._normalised_time_of_set if normalised else self.powerload_set
        )

    def plot(self, ax=None, n_points=100, detailed=False, primary=True, **kwargs):
        """
        If primary, plot in respect to 'normalised_time'.
        If secondary (called from 'PulseLoad'), plot in respect to
        'intrinsic_time', since set will already have been normalised.
        """
        computed_time = interpolate(
            self._normalised_time_of_set if primary else self.intrinsic_time,
            n_points,
        )

        ax = self._plot_load(
            validate_axes(ax),
            computed_time,
            sum(self._normalised_time_of_set if primary else self.powerload_set).curve(
                computed_time
            ),
            self.name,
            kwargs,
            True,
        )
        if detailed:
            for normal_load in self._normalised_time_of_set:
                normal_load._make_secondary_in_plot()
                normal_load.plot(ax=ax)
        return ax

    def __add__(self, other):
        """
        The addition of 'PhaseLoad' instances creates a new 'PhaseLoad'
        instance with joined 'powerload_set' and 'normalise' attributes,
        but only if its phases are the same.
        """
        this = copy.deepcopy(self)
        other = copy.deepcopy(other)

        if this.phase != other.phase:
            raise PhaseLoadError(
                "addition",
                "The phases of this PhaseLoad addition represent "
                f"{this.phase.name!r} and {other.phase.name!r} "
                "respectively.",
            )
        return PhaseLoad(
            f"Resulting PhaseLoad for phase {this.phase.name!r}",
            this.phase,
            this.powerload_set + other.powerload_set,
            np.concatenate([this.normalise, other.normalise]),
        )


class PulseLoad(PowerCycleLoadABC):
    """
    Generic representation of the total power load during a pulse.

    Defines the pulse load with a set of 'PhaseLoad' instances. The list
    of 'PhaseLoad' objects given as a parameter to be stored in the
    'phaseload_set' attribute must be provided in the order they are
    expected to occur during a pulse. This ensures that the correct
    'PowerCyclePulse' object is created and stored in the 'pulse'
    attribute. This enables the instance to shift power loads in time
    accordingly.

    Time shifts of each phase load in the 'phaseload_set' occurs AFTER
    the normalization performed by each 'PhaseLoad' object. In short,
    'PulseLoad' curves are built by joining each individual 'PhaseLoad'
    curve after performing the following the manipulations in the order
    presented below:
        1) normalization, given the 'normalise' attribute of each
            'PhaseLoad', with normalization equal to the 'duration'
            of its 'phase' attribute;
        2) shift, given the sum of the 'duration' attributes of
            the 'phase' of each 'PhaseLoad' that comes before it.

    Parameters
    ----------
    name: str
        Description of the 'PulseLoad' instance.
    pulse: PowerCyclePulse
        Pulse specification, that determines the necessary phases to
        be characterized by 'PhaseLoad' objects.
    phaseload_set: PhaseLoad | list[PhaseLoad]
        Collection of 'PhaseLoad' objects that define the 'PulseLoad'
        instance. Upon initialization, phase loads are permuted to have
        the same order as the phases in the 'phase_set' attribute of
        the 'pulse' parameter. Missing cases are treated with the
        creation of an null phase load.

    Properties
    ----------
    intrinsic_time: list[int | float]
        List that contains all values in the 'intrinsic_time' properties
        of the different 'PhaseLoad' objects contained in the
        'phaseload_set' attribute, ordered and with no repetitions.
    shifted_time: list[int | float]
        List that contains all values in the 'intrinsic_time' properties
        of the different 'PhaseLoad' objects contained in the
        '_ss' attribute, ordered and with no repetitions.
    """

    # Override number of points
    _n_points = 100

    # Override pyplot defaults
    _plot_defaults = {"color": "k", "linewidth": 2, "linestyle": "-"}

    # Defaults for detailed plots
    _detailed_defaults = {"color": "k", "linewidth": 1, "linestyle": "--"}

    # Defaults for delimiter plots
    _delimiter_defaults = {"color": "darkorange"}

    # Minimal shift for time correction in 'curve' method
    epsilon = 1e6 * EPS

    def __init__(self, name, pulse, phaseload_set):
        self.name = name
        self.pulse = pulse
        self.phaseload_set = phaseload_set

    @staticmethod
    def _validate_phaseload_set(phaseload_set, phases):
        """
        Validate 'phaseload_set' input to be a list of 'PhaseLoad'
        instances. Multiple phase loads for the same phase are added.
        Mssing phase loads are filled with a null instance.
        """
        validated_phaseload_set = []
        for phase_in_pulse in phases:
            phaseloads_for_phase = []
            for phaseload in phaseload_set:
                if phaseload.phase == phase_in_pulse:
                    phaseloads_for_phase.append(phaseload)

            if len(phaseloads_for_phase) == 0:
                phaseloads_for_phase = PhaseLoad.null(phase_in_pulse)

            validated_phaseload_set.append(sum(phaseloads_for_phase))

        return validated_phaseload_set

    @classmethod
    def null(cls, pulse):
        """
        Instantiates an null version of the class.
        """
        return cls(
            f"Null PhaseLoad for pulse {pulse.name}",
            pulse,
            [PhaseLoad.null(phase) for phase in pulse.build_phase_library().values()],
        )

    @property
    def _shifted_set(self):
        """
        Modified 'phaseload_set' attribute, in which all times of each
        'PhaseLoad' object are shifted by the sum of 'duration' values
        of the 'phase' attribute of each 'PhaseLoad' that comes before
        it in the 'phaseload_set' attribute.
        Shifts are applied to the '_normalised_set' property of each
        'PhaseLoad' object.
        """
        time_shift = 0
        shifted_set = []
        for phaseload in copy.deepcopy(self.phaseload_set):
            normalised_set = phaseload._normalised_set
            for normal_load in normalised_set:
                normal_load._shift_time(time_shift)
            phaseload.powerload_set = normalised_set

            shifted_set.append(phaseload)
            time_shift += phaseload.phase.duration

        return shifted_set

    def make_consumption_explicit(self):
        """
        Calls 'make_consumption_explicit' on every element of the
        'phaseload_set' attribute.
        """
        self._recursive_make_consumption_explicit(self.phaseload_set)

    @property
    def intrinsic_time(self, shifted=False):
        """
        Single time vector that contains all values used to define the
        different 'PhaseLoad' objects contained in the 'phaseload_set'
        attribute (i.e. all times are their original values).
        """
        return self.build_timeseries(
            self._shifted_set if shifted else self.phaseload_set
        )

    def __add__(self, other):
        """
        The addition of 'PulseLoad' instances can only be performed if
        their pulses are equal. It returns a new 'PulseLoad' instance
        with a 'phaseload_set' that contains the addition of the
        respective 'PhaseLoad' objects in each original instance.
        """
        this = copy.deepcopy(self)
        other = copy.deepcopy(other)
        if this.pulse != other.pulse:
            raise PulseLoadError(
                "addition",
                "The pulses of this PulseLoad addition represent "
                f"{this.pulse.name!r} and {other.pulse.name!r} "
                "respectively.",
            )

        return PulseLoad(
            f"Resulting PulseLoad for pulse {this.pulse.name!r}",
            this.pulse,
            this.phaseload_set + other.phaseload_set,
        )
