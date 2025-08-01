# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Power cycle net loads"""

from __future__ import annotations

from enum import Enum, auto
from typing import TYPE_CHECKING, Any, TypedDict

import numpy as np
import numpy.typing as npt
from numpydantic import NDArray, Shape  # noqa: TC002
from numpydantic.dtype import Number  # noqa: TC002
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeFloat,
    PrivateAttr,
    RootModel,
    model_validator,
)
from scipy.interpolate import interp1d

from bluemira.base.constants import raw_uc
from bluemira.base.look_and_feel import bluemira_debug, bluemira_warn

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


class PCBaseModel(BaseModel):
    """Base model for PowerCycle"""

    model_config = ConfigDict(validate_assignment=True)


class PCRootModel(RootModel):
    """Root model for PowerCycle"""

    model_config = ConfigDict(validate_assignment=True)


def interpolate_extra(vector: npt.NDArray, n_points: int) -> npt.NDArray:
    """
    Add points between each point in a vector.

    Returns
    -------
    :
        Interpolated vector
    """
    if n_points == 0:
        return vector

    return np.concatenate([
        *(
            np.linspace(vector[s], vector[s + 1], n_points + 1, endpoint=False)
            for s in range(len(vector) - 1)
        ),
        np.atleast_1d(vector[-1]),
    ])


def _normalise_timeseries(
    time: npt.ArrayLike, end_time: float | None = None
) -> tuple[npt.NDArray, float | None]:
    """
    Returns
    -------
    :
        Normalised array optionally to end time
    """
    time = np.asarray(time)
    if min(time) < 0:
        raise NotImplementedError("Negative time not supported")

    if max(time) > 1:
        mx_time = max(time)
        return time / mx_time, mx_time if end_time is None else end_time
    return time, end_time


def _consumption_flag(*, consumption: bool | None = None) -> set[bool]:
    """
    Returns
    -------
    :
        Consumption boolean set
    """
    return {True, False} if consumption is None else {consumption}


def _gettime(
    load: Load,
    load_type: str | LoadTypeOptions | None,
    *,
    consumption: bool | None = None,
):
    """
    Yields
    ------
    :
        Time array from load
    """
    load_type = None if load_type is None else LoadTypeOptions(load_type)
    if load.consumption in _consumption_flag(consumption=consumption):
        if load_type is None:
            yield from dict(iter(load.time)).values()
        else:
            yield load.time[load_type]


class Efficiency(PCBaseModel):
    """Efficiency Model"""

    value: LoadType
    description: str = ""

    @model_validator(mode="before")
    def value_validation(self) -> Any:
        """
        Returns
        -------
        :
            Validated value
        """
        if not isinstance(self, dict):
            return {"value": self}

        if "value" not in self:
            self = {"value": self}  # noqa: PLW0642

        if isinstance(self["value"], dict):
            if "active" not in self["value"]:
                self["value"]["active"] = 1
            if "reactive" not in self["value"]:
                self["value"]["reactive"] = 1
        return self


class Duration(PCBaseModel):
    """Duration Model"""

    value: NonNegativeFloat
    unit: str = "s"

    @model_validator(mode="before")
    def value_validator(self) -> Duration:
        """
        Returns
        -------
        :
            Validated value
        """
        if not isinstance(self, dict):
            return {"value": self}
        return self


class Durations(PCRootModel):
    """Durations Model"""

    root: dict[str, Duration]
    _revalidate: Callable[[], None] = PrivateAttr(default=lambda: None)

    def __getattr__(self, name):
        """Getattr for root model

        Returns
        -------
        :
            the duration
        """
        return self.root[name]

    def __setattr__(self, name, value):
        """Setattr for root model that revalidates the durations in the powercycle"""
        if name == "_revalidate":
            object.__setattr__(self, "_revalidate", value)
        else:
            self.root[name] = (
                value if isinstance(value, Duration) else Duration(value=value)
            )
            self._revalidate()


class SubSystem(PCBaseModel):
    """Sub system model"""

    loads: list[str]
    description: str = ""


class SubSystemLibrary(PCRootModel):
    """Sub system library model"""

    root: dict[str, SubSystem]


class System(PCBaseModel):
    """System Model"""

    subsystems: list[str]
    description: str = ""


class SystemLibrary(PCRootModel):
    """System library model"""

    root: dict[str, System]


class PhaseConfig(PCBaseModel):
    """Phase configuration"""

    description: str = ""
    operation: str = "max"
    subphases: list[str]

    def duration(self, subphase_library: SubphaseLibrary) -> npt.NDArray:
        """
        Returns
        -------
        :
            Duration of phase
        """
        return getattr(np, self.operation)([
            subphase_library.root[s_ph].duration for s_ph in self.subphases
        ])


class PhaseLibrary(PCRootModel):
    """Phase library model"""

    root: dict[str, PhaseConfig]


class PulseConfig(PCBaseModel):
    """Pulse Model"""

    description: str = ""
    phases: list[str]


class PulseLibrary(PCRootModel):
    """Pulse library model"""

    root: dict[str, PulseConfig]


class PulseRuns(PCRootModel):
    """Pulse runs model"""

    root: dict[str, int]


class Scenario(PCBaseModel):
    """Scenario Model"""

    name: str
    description: str = ""
    pulses: PulseRuns


class SubPhase(PCBaseModel):
    """Sub phase model"""

    duration: NonNegativeFloat | str
    loads: list[str] = Field(default_factory=list)
    efficiencies: dict[str, list[Efficiency]] = Field(default_factory=dict)
    unit: str = "s"
    description: str = ""
    reference: str = ""
    _duration_key: str = PrivateAttr(default="")

    @model_validator(mode="after")
    def duration_unit(self) -> SubPhase:
        """
        Returns
        -------
        :
            Enforced unit conversion
        """
        if isinstance(self.duration, float | int):
            object.__setattr__(  # noqa: PLC2801
                self, "duration", raw_uc(self.duration, self.unit, "second")
            )
        object.__setattr__(self, "unit", "s")  # noqa: PLC2801
        return self

    def import_durations(self, durations: Durations):
        """Import durations from external keys"""
        if isinstance(self.duration, str):
            key = self.duration.replace("$", "")
            self._duration_key = key
            dur = durations.root[key]
            self.duration = raw_uc(dur.value, dur.unit, "second")
        elif self._duration_key:
            dur = durations.root[self._duration_key]
            self.duration = raw_uc(dur.value, dur.unit, "second")

    def build_timeseries(
        self,
        load_library: LoadLibrary,
        load_type: str | LoadTypeOptions | None = None,
        end_time: float | None = None,
        *,
        consumption: bool | None = None,
    ) -> npt.NDArray:
        """Build a combined time series based on loads

        Returns
        -------
        :
            Time array
        """
        times = []
        for load_name in self.loads:
            load = load_library.root[load_name]
            for time in _gettime(load, load_type, consumption=consumption):
                if load.normalised:
                    times.append(time)
                else:
                    times.append(time / (max(time) if end_time is None else end_time))
        if len(times) > 0:
            return np.unique(np.concatenate(times))
        return times

    def _find_duplicate_loads(self):
        """Add duplication efficiency.

        If a load is duplicated in subphase.loads,
        the resultant data array is multiplied by the number of repeats

        Returns
        -------
        :
            Duplicated loads
        :
            Number of duplicates
        """
        u, c = np.unique(self.loads, return_counts=True)
        return u[c > 1].tolist(), c[c > 1]

    def get_load_data_with_efficiencies(
        self,
        load_library: LoadLibrary,
        timeseries: npt.NDArray,
        load_type: str | LoadTypeOptions | None,
        unit: str | None = None,
        end_time: float | None = None,
        *,
        consumption: bool | None = None,
    ) -> dict[str, npt.NDArray]:
        """
        Get load data taking into account efficiencies and consumption

        Parameters
        ----------
        timeseries:
            time array
        load_type:
            Type of load
        unit:
            return unit, defaults to [W] or [var]
        end_time:
            for unnormalised loads this assures the load is
            applied at the right point in time
        consumption:
            return only consumption loads

        Returns
        -------
        :
            Load data per load
        """
        load_type = LoadTypeOptions(load_type)
        data = self.get_explicit_data_consumption(
            load_library, timeseries, load_type, unit, end_time, consumption=consumption
        )
        dup, count = self._find_duplicate_loads()
        for ld_name in data:
            for eff in load_library.root[ld_name].efficiencies:
                for eff_type in LoadTypeOptions.check(load_type, dict(eff.value).keys()):
                    data[ld_name] *= eff.value[eff_type]
            if ld_name in dup:
                dup_eff = count[dup.index(ld_name)]
                bluemira_debug(
                    f"Duplicate load {ld_name}, duplication efficiency of {dup_eff}"
                )
                data[ld_name] *= dup_eff

        for eff_name in self.efficiencies.keys() & data.keys():
            for eff in self.efficiencies[eff_name]:
                for eff_type in LoadTypeOptions.check(load_type, dict(eff.value).keys()):
                    data[eff_name] *= eff.value[eff_type]
        return data

    def get_explicit_data_consumption(
        self,
        load_library: LoadLibrary,
        timeseries: npt.NDArray,
        load_type: str | LoadType,
        unit: str | None = None,
        end_time: float | None = None,
        *,
        consumption: bool | None = None,
    ) -> dict[str, npt.NDArray]:
        """
        Get data with consumption resulting in an oppositely signed load

        Parameters
        ----------
        timeseries:
            time array
        load_type:
            Type of load
        unit:
            return unit, defaults to [W] or [var]
        end_time:
            for unnormalised loads this assures the load is
            applied at the right point in time
        consumption:
            return only consumption loads

        Returns
        -------
        :
            Load data per load
        """
        return {
            ld_n: -ld if load_library.root[ld_n].consumption else ld
            for ld_n, ld in self.get_interpolated_loads(
                load_library,
                timeseries,
                load_type,
                unit,
                end_time,
                consumption=consumption,
            ).items()
        }

    def get_interpolated_loads(
        self,
        load_library: LoadLibrary,
        timeseries: npt.NDArray,
        load_type: str | LoadTypeOptions,
        unit: str | None = None,
        end_time: float | None = None,
        *,
        consumption: bool | None = None,
    ) -> dict[str, npt.NDArray]:
        """
        Get loads for a given time series

        Parameters
        ----------
        timeseries:
            time array
        load_type:
            Type of load
        unit:
            return unit, defaults to [W] or [var]
        end_time:
            for unnormalised loads this assures the load is
            applied at the right point in time
        consumption:
            return only consumption loads

        Returns
        -------
        :
            Load data per load
        """
        timeseries, end_time = _normalise_timeseries(timeseries, end_time)
        load_type = LoadTypeOptions(load_type)
        _cnsmptn = _consumption_flag(consumption=consumption)
        return {
            ld_name: load.interpolate(timeseries, end_time, load_type)
            if unit is None
            else raw_uc(
                load.interpolate(timeseries, end_time, load_type),
                load.unit,
                unit,
            )
            for ld_name, load in load_library.root.items()
            if ld_name in self.loads and load.consumption in _cnsmptn
        }


class SubphaseLibrary(PCRootModel):
    """Subphase library model"""

    root: dict[str, SubPhase]


class LoadModel(Enum):
    """
    Members define possible models used.

    Maps model names to 'interp1d' interpolation behaviour.
    """

    RAMP = "linear"
    STEP = "previous"

    @classmethod
    def _missing_(cls, value: str) -> LoadModel:
        try:
            return cls[value.upper()]
        except (KeyError, AttributeError):
            raise ValueError(
                f"{cls.__name__} has no type {value}."
                f" Select from {(*cls._member_names_,)}"
            ) from None


class LoadTypeOptions(Enum):
    """Possibly load types"""

    ACTIVE = auto()
    REACTIVE = auto()

    @classmethod
    def _missing_(cls, value: str) -> LoadTypeOptions:
        try:
            return cls[value.upper()]
        except (KeyError, AttributeError):
            raise ValueError(
                f"{cls.__name__} has no type {value}."
                f" Select from {(*cls._member_names_,)}"
            ) from None

    @classmethod
    def check(
        cls, self: LoadTypeOptions | None, other: list[str] | LoadTypeOptions
    ) -> set[LoadTypeOptions]:
        """Checks allowed Load types

        Returns
        -------
        :
            Allowed load types
        """
        ops = set(cls) if self is None else {self}
        return ops & {cls(o) for o in other}


class LoadType(PCBaseModel):
    """Load type model"""

    active: int | float | NDArray[Shape["*"], Number] | None = None  # noqa: F722
    reactive: int | float | NDArray[Shape["*"], Number] | None = None  # noqa: F722

    @model_validator(mode="before")
    def _(self):
        if not isinstance(self, dict):
            return {"active": self, "reactive": self}

        return self

    @model_validator(mode="after")
    def missing_lt(self):
        """
        Returns
        -------
        :
            Validated load
        """
        if self.active is None:
            object.__setattr__(  # noqa: PLC2801
                self, "active", np.atleast_1d(np.zeros_like(self.reactive, dtype=float))
            )

        if self.reactive is None:
            object.__setattr__(  # noqa: PLC2801
                self, "reactive", np.atleast_1d(np.zeros_like(self.active, dtype=float))
            )
        if not isinstance(self.reactive, np.ndarray):
            object.__setattr__(self, "reactive", np.atleast_1d(self.reactive))  # noqa: PLC2801

        if not isinstance(self.active, np.ndarray):
            object.__setattr__(self, "active", np.atleast_1d(self.active))  # noqa: PLC2801
        return self

    def __getitem__(self, value: str | LoadTypeOptions) -> npt.NDArray:
        """
        Item access for load

        Returns
        -------
        :
            Item
        """
        if isinstance(value, LoadTypeOptions):
            value = value.name.lower()
        return getattr(self, value)


class Load(PCBaseModel):
    """Load Model"""

    time: LoadType = Field(default_factory=LoadType)
    data: LoadType = Field(default_factory=LoadType)
    efficiencies: Efficiency | list[Efficiency] = Field(default_factory=list)
    model: LoadModel = LoadModel.RAMP
    unit: str = "W"
    description: str = ""
    normalised: bool = True
    consumption: bool = True

    @model_validator(mode="after")
    def load_validator(self):
        """
        Raises
        ------
        ValueError
            Time and data are not of the same size

        Returns
        -------
        :
            Validated load
        """
        default_unit = type(self).model_fields["unit"].default
        for lt in LoadTypeOptions:
            if self.data[lt].size != self.time[lt].size:
                if self.data[lt].size == 1:
                    # therefore time.size != 1
                    setattr(
                        self.data,
                        lt.name.lower(),
                        np.full(self.time[lt].shape, self.data[lt].item()),
                    )
                if all(self.time[lt] == 0):
                    setattr(self.time, lt.name.lower(), np.zeros_like(self.data[lt]))

                if self.data[lt].size != self.time[lt].size:
                    raise ValueError(
                        f"time and data must be the same length: {self.data[lt]}"
                    )

            if any(np.diff(self.time[lt]) < 0):
                raise ValueError("time must increase")

            object.__setattr__(  # noqa: PLC2801
                self.data,
                lt.name.lower(),
                raw_uc(self.data[lt], self.unit, default_unit),
            )
        object.__setattr__(self, "unit", default_unit)  # noqa: PLC2801

        if not isinstance(self.efficiencies, list):
            object.__setattr__(self, "efficiencies", [self.efficiencies])  # noqa: PLC2801

        return self

    def interpolate(
        self,
        time: npt.ArrayLike,
        end_time: float | None = None,
        load_type: str | LoadTypeOptions = LoadTypeOptions.ACTIVE,
    ) -> npt.NDArray:
        """
        Returns
        -------
        :
            Interpolate load for a given time vector

        Notes
        -----
        The interpolation type is set by model.
        Any out-of-bound values are set to zero.
        """
        load_type = LoadTypeOptions(load_type)
        return interp1d(
            self.time[load_type],
            self.data[load_type],
            kind=self.model.value,
            bounds_error=False,  # turn-off error for out-of-bound
            fill_value=(0, 0),  # below-/above-bounds extrapolations
        )(time if self.normalised or end_time is None else np.array(time) * end_time)


class LoadLibrary(PCRootModel):
    """Load library Model"""

    root: dict[str, Load]


class Phase:
    """Phase definition object"""

    def __init__(
        self, config: PhaseConfig, subphases: SubphaseLibrary, loads: LoadLibrary
    ):
        self._config = config
        self.subphases = subphases
        self.loads = loads

    @property
    def duration(self) -> float:
        """Duration of phase"""
        return self._config.duration(self.subphases)

    def timeseries(
        self,
        load_type: str | LoadTypeOptions | None = None,
        *,
        consumption: bool | None = None,
    ) -> npt.NDArray:
        """
        Returns
        -------
        :
            A combined time series based on loads
        """
        return np.unique(
            np.concatenate([
                sp.build_timeseries(
                    self.loads,
                    load_type=load_type,
                    end_time=self.duration,
                    consumption=consumption,
                )
                for sp in self.subphases.root.values()
            ])
            * self.duration
        )

    def _load(
        self,
        timeseries: npt.NDArray,
        load_type: str | LoadTypeOptions,
        unit: str | None = None,
        *,
        consumption: bool | None = None,
    ) -> dict[str, npt.NDArray]:
        load_data = [
            sp.get_load_data_with_efficiencies(
                self.loads,
                timeseries,
                load_type,
                unit,
                end_time=self.duration,
                consumption=consumption,
            )
            for sp in self.subphases.root.values()
        ]
        if self._config.operation == "sum":
            data = {}
            for ld_name in self.loads.root:
                for dat in load_data:
                    if ld_name in dat and ld_name not in data:
                        data[ld_name] = dat[ld_name]
            return data
        if self._config.operation == "max":
            # could just do this
            ind, _sp = max(
                enumerate(self.subphases.root.values()),
                key=lambda k_sp: k_sp[1].duration,
            )

            data = load_data[ind]

            # for missing loads
            for ld_name in self.loads.root:
                for i, dat in enumerate(load_data):
                    _sp_d = None
                    if ld_name in dat and ld_name not in data:
                        if _sp_d is None:
                            _sp_d = list(self.subphases.root.values())[i].duration

                        data[ld_name] = np.where(dat[ld_name] > _sp_d, 0, dat[ld_name])

            return data
        raise NotImplementedError("Only max and sum operations are supported on a phase")

    def total_load(
        self,
        load_type: str | LoadTypeOptions,
        unit: str | None = None,
        *,
        timeseries: npt.NDArray | None = None,
        consumption: bool | None = None,
    ) -> npt.NDArray:
        """Total load for each timeseries point for a given load_type

        Parameters
        ----------
        load_type:
            Type of load
        unit:
            return unit, defaults to [W] or [var]
        consumption:
            return only consumption loads

        Returns
        -------
        :
            Total load
        """
        if timeseries is None:
            timeseries = self.timeseries(load_type, consumption=consumption)
        timeseries, _ = _normalise_timeseries(timeseries, self.duration)
        return np.sum(
            list(
                self._load(timeseries, load_type, unit, consumption=consumption).values()
            ),
            axis=0,
        )

    def load(
        self,
        load_type: str | LoadTypeOptions,
        unit: str | None = None,
        *,
        timeseries: npt.NDArray | None = None,
        consumption: bool | None = None,
    ) -> dict[str, npt.NDArray]:
        """
        Get load data taking into account efficiencies and consumption

        Parameters
        ----------
        load_type:
            Type of load
        unit:
            return unit, defaults to [W] or [var]
        consumption:
            return only consumption loads

        Returns
        -------
        :
            Load
        """
        if timeseries is None:
            timeseries = self.timeseries(load_type, consumption=consumption)
        if timeseries.size == 0:
            return {}
        return self._load(
            _normalise_timeseries(timeseries, self.duration)[0],
            load_type,
            unit,
            consumption=consumption,
        )


class PulseDictType(TypedDict):
    """Pulse dictionary typing"""

    repeat: int
    data: Pulse


class Pulse:
    """Pulse definition object"""

    def __init__(
        self,
        config: PulseConfig,
        phases: dict[str, Phase],
    ):
        self._config = config
        self.phases = phases

    @property
    def duration(self) -> float:
        """Duration of pulse"""
        return sum(phase.duration for phase in self.phases)

    def phase_timeseries(
        self,
        load_type: str | LoadTypeOptions | None = None,
        *,
        consumption: bool | None = None,
    ) -> list[npt.NDArray]:
        """
        Returns
        -------
        :
            Timeseries for each phase of the pulse
        """
        return [
            self.phases[phase_name].timeseries(load_type, consumption=consumption)
            for phase_name in self._config.phases
        ]

    def timeseries(
        self,
        load_type: str | LoadTypeOptions | None = None,
        *,
        consumption: bool | None = None,
    ) -> npt.NDArray:
        """
        Returns
        -------
        :
            Timeseries of pulse
        """
        time = self.phase_timeseries(load_type, consumption=consumption)
        for no, t in enumerate(time[1:]):
            t += max(time[no])  # noqa: PLW2901
        return np.concatenate(time)

    def _timecheck(
        self,
        timeseries: npt.NDArray | list[npt.NDArray] | None,
        load_type: str | LoadTypeOptions,
        *,
        consumption: bool,
    ) -> list[npt.NDArray]:
        """Check timeseries inputs"""  # noqa: DOC201
        if timeseries is None:
            return self.phase_timeseries(load_type, consumption=consumption)
        if isinstance(timeseries, np.ndarray):
            durations = np.cumsum([
                self.phases[phase_name].duration for phase_name in self._config.phases
            ])
            phase_timeseries = [timeseries[np.nonzero(timeseries <= durations[0])]]
            for no, d in enumerate(durations[:-1], start=1):
                phase_timeseries.append(
                    timeseries[
                        np.nonzero((timeseries >= d) & (timeseries <= durations[no]))
                    ]
                )

            if max(phase_timeseries[-1]) < max(timeseries):
                bluemira_warn("Timeseries longer than pulse, clipping to end of pulse")
            return phase_timeseries
        return timeseries

    def _load(
        self,
        phase_timeseries: npt.NDArray,
        load_type: str | LoadTypeOptions,
        unit: str | None = None,
        *,
        consumption: bool | None = None,
    ) -> dict[str, npt.NDArray]:
        phase_data = [
            self.phases[phase_name].load(
                load_type, unit, timeseries=time, consumption=consumption
            )
            for time, phase_name in zip(
                phase_timeseries, self._config.phases, strict=False
            )
        ]

        load_names = np.unique([load for pd in phase_data for load in pd])
        pulse_load = {}
        for load in load_names:
            for pd in phase_data:
                if pd == {}:
                    continue
                if load in pd:
                    if load in pulse_load:
                        pulse_load[load] = np.concatenate([pulse_load[load], pd[load]])
                    else:
                        pulse_load[load] = pd[load]
                elif load in pulse_load:
                    pulse_load[load] = np.concatenate([
                        pulse_load[load],
                        np.zeros_like(next(iter(pd.values()))),
                    ])
                else:
                    pulse_load[load] = np.zeros_like(next(iter(pd.values())))
        return pulse_load

    def total_load(
        self,
        load_type: str | LoadTypeOptions,
        unit: str | None = None,
        *,
        timeseries: npt.NDArray | list[npt.NDArray] | None = None,
        consumption: bool | None = None,
    ) -> npt.NDArray:
        """Total load for each timeseries point for a given load_type

        Parameters
        ----------
        load_type:
            Type of load
        unit:
            return unit, defaults to [W] or [var]
        consumption:
            return only consumption loads

        Returns
        -------
        :
            Total load
        """
        phase_timeseries = self._timecheck(
            timeseries, load_type, consumption=consumption
        )
        return np.sum(
            list(
                self._load(
                    phase_timeseries, load_type, unit, consumption=consumption
                ).values()
            ),
            axis=0,
        )

    def load(
        self,
        load_type: str | LoadTypeOptions,
        unit: str | None = None,
        *,
        timeseries: npt.NDArray | list[npt.NDArray] | None = None,
        consumption: bool | None = None,
    ) -> dict[str, npt.NDArray]:
        """
        Get load data taking into account efficiencies and consumption

        Parameters
        ----------
        load_type:
            Type of load
        unit:
            return unit, defaults to [W] or [var]
        consumption:
            return only consumption loads

        Returns
        -------
        :
            Load
        """
        phase_timeseries = self._timecheck(
            timeseries, load_type, consumption=consumption
        )
        return self._load(phase_timeseries, load_type, unit, consumption=consumption)


class PowerCycle(PCBaseModel):
    """Power cycle Model"""

    scenario: Scenario
    pulse_library: PulseLibrary
    phase_library: PhaseLibrary
    subphase_library: SubphaseLibrary
    system_library: SystemLibrary
    sub_system_library: SubSystemLibrary
    load_library: LoadLibrary
    durations: Durations = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_config(self):
        """
        Returns
        -------
        :
            Validated powercycle configuration

        Raises
        ------
        ValueError
            Missing entries for pulses, phases etc
        """
        ph_keys = self.phase_library.root.keys()
        sph_keys = self.subphase_library.root.keys()
        ss_keys = self.sub_system_library.root.keys()
        loads = self.load_library.root.keys()
        # scenario has known pulses
        if (
            unknown_pulse := self.scenario.pulses.root.keys()
            - self.pulse_library.root.keys()
        ):
            raise ValueError(f"Unknown pulses {unknown_pulse}")
        # pulses have known phases
        for pulse in self.pulse_library.root.values():
            if unknown_phase := pulse.phases - ph_keys:
                raise ValueError(f"Unknown phases {unknown_phase}")
        # phases have known subphases
        for ph_c in self.phase_library.root.values():
            if unknown_s_ph := ph_c.subphases - sph_keys:
                raise ValueError(f"Unknown subphase configurations {unknown_s_ph}")
        # subphases have known loads
        for subphase in self.subphase_library.root.values():
            if unknown_load := subphase.loads - loads:
                raise ValueError(f"Unknown loads {unknown_load}")

        # systems have known subsystems
        for sys_c in self.system_library.root.values():
            if unknown := sys_c.subsystems - ss_keys:
                raise ValueError(f"Unknown subsystem configurations {unknown}")
        # subsystems have known loads
        for s_sys_c in self.sub_system_library.root.values():
            if unknown_load := s_sys_c.loads - loads:
                raise ValueError(f"Unknown loads {unknown_load}")
        return self

    @model_validator(mode="after")
    def duration_validation(self):
        """Import subphase data and validate duration

        Returns
        -------
        :
            Model with imported durations

        Raises
        ------
        ValueError
            Phase duration < 0
        """
        self.durations._revalidate = self.duration_validation
        for s_ph in self.subphase_library.root.values():
            s_ph.import_durations(self.durations)

        for ph_name, phase in self.phase_library.root.items():
            dur = phase.duration(self.subphase_library)
            if dur < 0:
                raise ValueError(f"{ph_name} phase duration must be positive: {dur}s")
        return self

    def get_phase(self, phase: str) -> Phase:
        """
        Returns
        -------
        :
            A single phase object
        """
        phase_config = self.phase_library.root[phase]
        subphases = SubphaseLibrary({
            k: self.subphase_library.root[k] for k in phase_config.subphases
        })
        return Phase(
            phase_config,
            subphases,
            LoadLibrary({
                ld: self.load_library.root[ld]
                for subphase in subphases.root.values()
                for ld in subphase.loads
            }),
        )

    def get_scenario(self) -> dict[str, PulseDictType]:
        """
        Returns
        -------
        :
            A scenario dictionary
        """
        return {
            pulse: {"repeat": reps, "data": self.get_pulse(pulse)}
            for pulse, reps in self.scenario.pulses.root.items()
        }

    def get_pulse(self, pulse: str) -> Pulse:
        """
        Returns
        -------
        :
            A pulse dictionary
        """
        return Pulse(
            self.pulse_library.root[pulse],
            {
                phase: self.get_phase(phase)
                for phase in self.pulse_library.root[pulse].phases
            },
        )

    def add_load(
        self,
        name: str,
        load: Load,
        subphases: str | Iterable[str] | None = None,
        subphase_efficiency: list[Efficiency] | None = None,
    ):
        """Add load config"""
        self.load_library.root[name] = load
        self.link_load_to_subphase(name, subphases or [], subphase_efficiency)

    def link_load_to_subphase(
        self,
        name: str,
        subphases: str | Iterable[str],
        subphase_efficiency: list[Efficiency] | None = None,
    ):
        """Link a load to a specific subphase"""
        if isinstance(subphases, str):
            subphases = [subphases]
        for subphase in subphases:
            self.subphase_library.root[subphase].loads.append(name)
            if subphase_efficiency is not None:
                self.subphase_library.root[subphase].efficiencies[name] = (
                    Efficiency.model_validate(sp_e) for sp_e in subphase_efficiency
                )


def make_power_cycle(
    scenario_name: str,
    pulse_runs: PulseRuns | None = None,
    pulses: dict[str, PulseConfig] | None = None,
    phases: dict[str, PhaseConfig] | None = None,
    subphases: dict[str, SubPhase] | None = None,
    systems: dict[str, System] | None = None,
    subsystems: dict[str, SubSystem] | None = None,
    loads: dict[str, Load] | None = None,
    durations: Durations | None = None,
) -> PowerCycle:
    """Create a power cycle with minimal inputs

    Returns
    -------
    :
        Initialised powercycle
    """
    return PowerCycle(
        scenario={"name": scenario_name, "pulses": pulse_runs or {}},
        pulse_library=pulses or {},
        phase_library=phases or {},
        subphase_library=subphases or {},
        system_library=systems or {},
        sub_system_library=subsystems or {},
        load_library=loads or {},
        durations=durations or {},
    )
