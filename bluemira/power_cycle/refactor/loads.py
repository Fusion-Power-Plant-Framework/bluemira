from copy import copy
from typing import Dict, List, Literal

import numpy.typing as npt

from bluemira.power_cycle.refactor.load_manager import LoadType, PowerCycleManagerConfig
from bluemira.power_cycle.refactor.time import PhaseConfig


class PulseSystemLoad:
    def __init__(
        self,
        pulse: List[PhaseConfig],
        manager_configs: Dict[str, PowerCycleManagerConfig],
    ):
        self.pulse = pulse
        self._manager_configs = manager_configs
        self.get_all_loads()

    @property
    def phase_names(self):
        return {p.name for p in self.pulse}

    def get_all_loads(self):
        self.psl = {
            loadtype: {
                k: {
                    sys_k: [
                        load
                        for load in getattr(sys, loadtype.name.lower()).values()
                        if self.phase_names.intersection(load.phases)
                    ]
                    for sys_k, sys in conf.systems.items()
                }
                for k, conf in self._manager_configs.items()
            }
            for loadtype in LoadType
        }

        # active = dict[loads]
        # reactive = dict[loads]

        # Create new PowerCycleLoadConfig(s) for a given PulseSystemLoad
        ...


#### PowerCycleLoadConfig
class Tmp:
    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other: float):
        this = copy.deepcopy(self)
        for load in this.loads.values():
            load.data += other
        return this

    def __mul__(self, other: float):
        this = copy.deepcopy(self)
        for load in this.loads.values():
            load.data *= other
        return this

    def __truediv__(self, other: float):
        this = copy.deepcopy(self)
        for load in this.loads.values():
            load.data /= other
        return this

    def plot(
        self,
        ax=None,
        n_points=100,
        color: Literal["r", "b"] = "r",
        detailed=False,
        **kwargs,
    ):
        """
        Plot a 'PowerLoad' curve, built using the attributes that define
        the instance. The number of points interpolated in each curve
        segment can be specified.


        This method can also plot the individual 'LoadData' objects
        stored in the 'loaddata_set' attribute that define the
        'PowerLoad' instance.

        Parameters
        ----------
        ax: Axes
            Instance of the 'matplotlib.axes.Axes' class, in which to
            plot. If 'None' is given, a new instance of axes is created.
        n_points: int
            Number of points interpolated in each curve segment. The
            default value is 'None', which indicates to the method
            that the default value should be used, defined as a class
            attribute.
        detailed: bool
            Determines whether the plot will include all individual
            'LoadData' objects (computed with their respective
            'loadmodel_set' entries), that summed result in the normal
            plotted curve. These objects are plotted as secondary plots,
            as defined in 'PowerCycleABC' class. By default this input
            is set to 'False'.
        **kwargs: dict
            Options for the 'plot' method.

        """
        computed_time = interpolate(self.build_timeseries(), n_points)

        ax = self._plot_load(
            validate_axes(ax),
            computed_time,
            self.load_total(computed_time),
            self.name,
            kwargs,
            True,
        )

        if detailed:
            for ld in self.loads.values():
                # Plot current LoadData with seconday kwargs
                ld.plot(ax=ax, marker="x")

                # Plot current curve as line with secondary kwargs
                kwargs.update({"color": "k", "linewidth": 1, "linestyle": "--"})
                ax.plot(
                    computed_time,
                    ld.interpolate(computed_time),
                    **kwargs,
                )

        return ax

    ### PowerCycleSystemLoadConfig

    def plot(self, ax=None, n_points=100, detailed=False, **kwargs):
        for load in self.active.values():
            ax = load.plot(
                ax=ax, n_points=n_points, detailed=detailed, color="r", **kwargs
            )
        for load in self.reactive.values():
            load.plot(ax=ax, n_points=n_points, detailed=detailed, color="b", **kwargs)
        return ax


###


def interpolate(vector: npt.NDArray, n_points: int):
    """
    Add points between each point in a vector.
    """
    if n_points == 0:
        return vector

    return np.concatenate(
        [
            *(
                np.linspace(vector[s], vector[s + 1], n_points + 1, endpoint=False)
                for s in range(len(vector) - 1)
            ),
            np.atleast_1d(vector[-1]),
        ]
    )


def _shift_time(subload, time):
    new_subload = copy.deepcopy(subload)
    new_subload.time += time
    new_subload._shift = time
    return new_subload


def _normalise_time(subload, new_end_time):
    new_subload = copy.deepcopy(subload)
    new_subload._norm = new_end_time / subload.time[-1]
    new_subload.time *= new_subload._norm
    return new_subload


def _normalise_time(load, new_end_time):
    for subload in load.loads.values():
        subload, _normalise_time(new_end_time)


def _shift_time(load, new_end_time):
    for subload in load.loads.values():
        subload, _shift_time(new_end_time)


def _add_text_to_point_in_plot(ax, function_name, name, x, y, index=0, **kwargs):
    ax.text(
        x[index],
        y[index],
        f"{name} ({function_name})",
        label=f"{name} (name)",
        **{**{"color": "k", "size": 10, "rotation": 45}, **kwargs},
    )


def plot_phase_breakdowns(phase_breakdowns):
    curve = []
    modified_time = []
    time = interpolate(self.shifted_time, n_points)
    for shifted_load in self._shifted_set:
        intrinsic_time = shifted_load.intrinsic_time
        max_t = max(intrinsic_time)
        min_t = min(intrinsic_time)

        load_time = time[(time >= min_t) & (time <= max_t)]
        load_time[-1] = load_time[-1] - self.epsilon

        modified_time.append(load_time)
        curve.append(shifted_load._curve(load_time, primary=False))

    modified_time = np.concatenate(modified_time)
    computed_curve = np.concatenate(curve, axis=-1)

    ax = _plot_load(validate_axes(ax), modified_time, computed_curve, self.name, kwargs)

    if detailed:
        # make secondary plot
        self._text_index = 0
        self._plot_kwargs = {"color": "k", "linewidth": 1, "linestyle": "--"}
        self._scatter_kwargs = {"color": "k", "s": 100, "marker": "x"}
        self._text_kwargs = {"color": "k", "size": 10, "rotation": 45}
        for shifted_load in self._shifted_set:
            shifted_load._plot(primary=False, ax=ax, n_points=100, **kwargs)

        # Add phase delimiters

        axis_limits = ax.get_ylim()
        _delimiter_defaults = {"color": "darkorange"}
        line_kwargs = {
            **{"color": "k", "linewidth": 1, "linestyle": "--"},
            **_delimiter_defaults,
        }
        text_kwargs = {
            **{"color": "k", "size": 10, "rotation": 45},
            **_delimiter_defaults,
        }

        for shifted_load in self._shifted_set:
            intrinsic_time = shifted_load.intrinsic_time
            last_time = intrinsic_time[-1]

            label = f"Phase delimiter for {shifted_load.phase.name}"

            ax.plot(
                [last_time, last_time],
                axis_limits,
                label=label,
                **line_kwargs,
            )

            ax.text(
                last_time,
                axis_limits[-1],
                f"End of {shifted_load.phase.name}",
                label=label,
                **text_kwargs,
            )

    return ax
