# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Full fuel cycle model object
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from bluemira.base.constants import T_LAMBDA, T_MOLAR_MASS, YR_TO_S, raw_uc
from bluemira.base.look_and_feel import bluemira_print
from bluemira.fuel_cycle.blocks import FuelCycleComponent, FuelCycleFlow
from bluemira.fuel_cycle.timeline import Timeline
from bluemira.fuel_cycle.tools import (
    _speed_recycle,
    discretise_1d,
    find_max_load_factor,
    find_noisy_locals,
    legal_limit,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    import numpy.typing as npt


class EUDEMOFuelCycleModel:
    """
    Tritium fuel cycle object.

    Takes a lifecycle timeline and interprets it to reach tritium start-up
    inventory and inventory doubling time estimates.

    Parameters
    ----------
    params:
        The parameters for the model. See
        :class:`~bluemira.fuel_cycle.cycle.EDFCMParams` for list of
        available parameters.
    build_config:
        Configuration options for the model. Options are:

            * verbose: bool (False)
                Print debugging information.
            * timestep: float (1200.0)
                The time step in the cycle [s].
            * conv_thresh: float (2e-4)
                The convergence threshold in the cycle.
            * n: int (length of timeline['time'])
                The no. of time steps to use from the the timeline.
    """

    params: EDFCMParams
    verbose: bool
    timestep: float
    conv_thresh: float
    n: int | None
    iterations: int
    A_global: float

    m_T: np.ndarray
    m_dot_release: float
    m_T_in: np.ndarray
    m_T_req: float
    m_T_start: float
    M_T_bred: float
    M_T_stack: float
    I_blanket: np.ndarray
    I_plasma: np.ndarray
    I_stack: np.ndarray
    I_tfv: np.ndarray

    grate: np.ndarray
    brate: np.ndarray
    prate: np.ndarray
    DEMO_rt: np.ndarray
    DEMO_t: np.ndarray
    DD_rate: np.ndarray
    DT_rate: np.ndarray
    bci: int
    arg_t_d: int | None
    arg_t_infl: int
    t: np.ndarray
    t_d: float
    t_infl: float

    def __init__(
        self,
        params: EDFCMParams | dict[str, float] | None = None,
        build_config: dict[str, Any] | None = None,
    ):
        # Handle parameters
        if isinstance(params, EDFCMParams):
            self.params = params
        elif isinstance(params, dict):
            self.params = EDFCMParams(**params)
        elif params is None:
            self.params = EDFCMParams()
        else:
            raise TypeError(
                "Invalid type for 'params'. Must be one of 'dict', "
                f"'EDFCMParams', or 'None'; found '{type(params).__name__}'."
            )

        # Handle calculation information
        build_config = {} if build_config is None else build_config
        self.verbose = build_config.get("verbose", False)
        self.timestep = build_config.get("timestep", 1200)
        self.conv_thresh = build_config.get("conv_thresh", 2e-4)
        self.n = build_config.get("n")
        self.iterations = 0
        self.A_global = 0.0

    def run(self, timeline: Mapping[str, Any] | Timeline) -> EUDEMOFuelCycleResults:
        """
        Run the fuel cycle model.

        Parameters
        ----------
        timeline:
            Timeline with which to run the model

        Returns
        -------
        :
            Results of EuDEMo fuelcycle
        """
        if isinstance(timeline, Timeline):
            timeline = timeline.to_dict()

        if self.n is None:
            self.n = len(cast("npt.NDArray", timeline["time"]))

        self.A_global = cast("float", timeline["A_global"])

        self.initialise_arrays(timeline, self.n)
        # Initialise model with a reasonable number for decay
        self.seed_t()
        self.iterations = 0
        self.recycle()
        max_T, min_T, arg_t_d, t_d, arg_t_infl, t_infl = self.finalise()  # noqa: N806
        self.m_T_req += float(self.I_tfv[0])  # Add TFV fountain inventory

        return EUDEMOFuelCycleResults(
            self.params.TBR,
            self.DEMO_t,
            self.m_T_req,
            self.t,
            t_d,
            arg_t_d,
            t_infl,
            arg_t_infl,
            min_T,
            max_T,
            self.m_T,
            self.I_tfv,
            self.params.I_tfv_min,
            self.I_stack,
            self.I_plasma,
            self.I_blanket,
            self.brate,
            self.prate,
            self.calc_m_release(),
        )

    def finalise(self):
        """
        Perform clean-up fudge to ensure all tritium returns to stores after
        the end of the reactor life.

        Returns
        -------
        max_T:
            maximum tritium inventory
        min_T:
                minimum tritium inventory
        arg_t_d:

        t_d:

        arg_t_infl:

         t_infl:

        """
        n_bins = max(int(len(self.m_T) / 4500), 400)
        # Hand tweak to get findnoisylocals looking good
        max_T = find_noisy_locals(self.m_T, x_bins=n_bins, mode="max")  # noqa: N806
        min_T = find_noisy_locals(self.m_T, x_bins=n_bins)  # noqa: N806
        # self.max_T[1][-1] = self.max_T[1][-2]   #plothack
        self.m_T[-1] = max_T[1][-1]
        arg_t_d, t_d = self.calc_t_d()
        arg_t_infl, t_infl = self.calc_t_infl()
        return max_T, min_T, arg_t_d, t_d, arg_t_infl, t_infl

    def initialise_arrays(self, timeline: Mapping[str, Any], n: int):
        """
        Initialise timeline arrays for TFV model.

        Notes
        -----
        Gas puff timeline mapped to burn signal.
        """
        self.DEMO_t = np.asarray(timeline["time"][:n])
        self.DEMO_rt = np.asarray(timeline["fusion_time"][:n])
        self.DT_rate = np.asarray(timeline["DT_rate"][:n])
        self.DD_rate = np.asarray(timeline["DD_rate"][:n])
        m_gas = T_MOLAR_MASS * raw_uc(self.params.m_gas, "Pa.m^3/s", "mol/s") / 1000
        self.grate = m_gas * self.DT_rate / max(self.DT_rate)
        self.bci = int(timeline["blanket_change_index"])
        # Burn rate of T [kgs of T per second]
        self.brate = raw_uc(T_MOLAR_MASS, "amu", "kg") * self.DT_rate
        # T production rate from D-D reaction channel [kgs of T per second]
        self.prate = raw_uc(T_MOLAR_MASS, "amu", "kg") * self.DD_rate / 2  # Only 50%!

    def seed_t(self):
        """
        Seed an initial value to the model.
        """
        self.m_T_start = 5.0
        self.m_T = np.array([self.m_T_start])

    def plasma(
        self, eta_iv: float, max_inventory: float, flows: list[np.ndarray] | None = None
    ) -> np.ndarray:
        """
        In-vessel environment

        Parameters
        ----------
        eta_iv:
            In-vessel accumulation efficiency 0 <= float <= 1

        max_inventory:
            T inventory limit of in-vessel environment [kg]
            (e.g. 0.7 kg in ITER in-vessel)
        flows:
            Additional flows to add into plasma

        Returns
        -------
        :
            Flow-rate out of the system [kg/s]
        """
        plasma = FuelCycleComponent(
            "Plasma",
            self.DEMO_t,
            eta_iv,
            max_inventory,
            bci=self.bci,
            retention_model="sqrt_bathtub",
            summing=False,
        )
        if flows is not None:
            for flow in flows:
                plasma.add_in_flow(flow)
        plasma.run()
        self.I_plasma = cast("npt.NDArray", plasma.inventory)
        return cast("npt.NDArray", plasma.m_out)

    def blanket(self, eta_b: float, max_inventory: float) -> np.ndarray:
        """
        The tritium breeding system. Dumps stored inventory at blanket change.

        Parameters
        ----------
        eta_b:
            The retention efficiency parameter for the blanket
        max_inventory:
            The maximum inventory in the blanket

        Returns
        -------
        Flow-rate out of the system [kg/s]
        """
        m_T_bred = self.params.TBR * self.brate
        blanket = FuelCycleComponent(
            "Blanket", self.DEMO_t, eta_b, max_inventory, bci=self.bci, summing=True
        )
        blanket.add_in_flow(m_T_bred)
        blanket.run()

        self.M_T_bred = blanket.sum_in
        self.I_blanket = cast("npt.NDArray", blanket.inventory)
        return cast("npt.NDArray", blanket.m_out)

    def tfv(
        self, eta_tfv: float, flows: list[np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        The TFV system where the tritium flows from the BB and plasma are combined.

        Parameters
        ----------
        eta_tfv:
            Detritiation factor of the system
        flows:
            The flows to be added to the TFV block

        Returns
        -------
        Flow-rate out of the system [kg/s]
        """
        # Runs in compressed time
        tfv = FuelCycleComponent(
            "TFV systems",
            self.t,
            eta_tfv,
            self.params.I_tfv_max,
            min_inventory=self.params.I_tfv_min,
            retention_model="fountaintub",
        )
        for flow in flows:
            tfv.add_in_flow(flow)
        tfv.run()

        self.I_tfv = cast("npt.NDArray", tfv.inventory)
        m_tfv_out = FuelCycleFlow(self.t, cast("npt.NDArray", tfv.m_out), 0)
        # Exhaust processing
        m_in_isotope_re, m_in_exhaust_det = m_tfv_out.split(2, [self.params.f_exh_split])
        # Isotope rebalancing
        # Storage and Gas Distribution and Control are the same for me
        # Fließt direkt zum Injektor
        # Exhaust detritiation
        # Combines Water Detritiation and Isotope Separation
        m_in_exhaust_det = FuelCycleFlow(self.t, m_in_exhaust_det, self.params.t_detrit)
        m_exh_stor, m_ex_stack = m_in_exhaust_det.split(2, [self.params.f_detrit_split])
        return m_in_isotope_re + m_exh_stor, m_ex_stack

    def stack(self, flows: list[np.ndarray]):
        """
        Exhaust to environment
        """
        stack = FuelCycleComponent(
            "Stack", self.t, 0, float("inf"), retention_model="bathtub", summing=True
        )
        for flow in flows:
            stack.add_in_flow(flow)
        stack.run()
        self.I_stack = cast("npt.NDArray", stack.inventory)
        # Total release to the environment
        self.M_T_stack = stack.sum_in

    def injector(self, flows: list[np.ndarray]) -> np.ndarray:
        """
        Pellet injection system assumed

        Returns
        -------
        :
            injector mass out
        """
        injector = FuelCycleComponent("Injector", self.t, 1, 0)
        for flow in flows:
            if flow is not None:
                injector.add_in_flow(flow)
        injector.run()
        return cast("npt.NDArray", injector.m_out)

    def recycle(self):
        """
        The main loop of the fuel cycle, which is called recursively until the
        convergence criterion is met.
        """
        # Fuelling (fuel in)
        # Fuel pump built in (ghosted component)
        self.m_T_in = self.brate / (self.params.f_b * self.params.eta_f)
        # m_T_out = (1/self.f_b-1)*self.brate
        # In-vessel flow lost from fuelling lines as gas (does not enter core)
        iv_loss_flow = (
            (1 - self.params.eta_fuel_pump) * (1 - self.params.eta_f) * self.m_T_in
        )
        # Gas puff flow
        gpuff = self.grate

        # Plasma/in-vessel block

        flows = [
            self.brate / self.params.f_b,
            iv_loss_flow,
            gpuff,
            self.prate,  # D-D T production from plasma
            -self.brate,
        ]

        m_T_out = self.plasma(self.params.eta_iv, self.params.I_miv, flows=flows)
        # Resolution - Not used everywhere for speed
        n_ts = round(self.DEMO_t[-1] * YR_TO_S / self.timestep)
        self.t, m_pellet_in = discretise_1d(self.DEMO_t, self.m_T_in, n_ts)

        # Flow out of the vacuum vessel
        t, m_T_out = discretise_1d(self.DEMO_t, m_T_out, n_ts)
        # Direct Internal Recycling
        m_plasma_out = FuelCycleFlow(t, m_T_out, 0)  # Initialise flow 0 t

        m_direct, m_indirect = m_plasma_out.split(2, [self.params.f_dir])
        # DIR separation
        # Flow 9
        direct = FuelCycleFlow(t, m_direct, self.params.t_pump)
        # Flow 10 with (time delay t_10+t_11/12)
        indirect = FuelCycleFlow(t, m_indirect, self.params.t_pump + self.params.t_exh)
        # Blanket
        m_T_bred = self.blanket(self.params.eta_bb, self.params.I_mbb)
        t, m_bred = discretise_1d(self.DEMO_t, m_T_bred, n_ts)
        m_T_bred_flow = FuelCycleFlow(t, m_bred, self.params.t_ters)
        # Tritium extraction and recovery system + coolant water purification
        m_T_bred_totfv, m_T_bred_tostack = m_T_bred_flow.split(
            2, [self.params.f_terscwps]
        )
        # TFV systems - runs in t
        m_tfv_out, m_tfv_stack = self.tfv(self.params.eta_tfv, flows=[indirect.out_flow])
        # Release to environment
        self.stack([m_T_bred_tostack, m_tfv_stack])
        m_tfv_out = FuelCycleFlow(t, m_tfv_out, 0)
        # Store
        store = FuelCycleComponent("Store", t, 1, np.inf)
        # Flow 11+13
        store.add_in_flow(m_tfv_out.out_flow)
        # Flow 16
        store.add_in_flow(m_T_bred_totfv)
        # Pump which compensates fuelling efficiency loss in pellet
        # injection flight tubes (Flow 3)
        store.add_in_flow(
            m_pellet_in * (1 - self.params.eta_f) * self.params.eta_fuel_pump
        )
        # Flow 9
        store.add_in_flow(direct.out_flow)
        store.run()
        # This is conservative... need to find a way to make gas available
        # instantaneously. At present this means gas puffs get "frozen" first
        m_store = FuelCycleFlow(
            t, cast("npt.NDArray", store.m_out), self.params.t_freeze
        ).out_flow
        # Add a correction flow for instantaneous gas puffing

        # m_store += gpuff_corr
        # Fuelling requirements
        # Adds gas flow now (not accounted for in ghosted fuel
        # line pumps)
        _, m_in = discretise_1d(self.DEMO_t, self.m_T_in + gpuff, n_ts)
        # Completes the loop in numba
        m_T = _speed_recycle(self.m_T_start, t, m_in, m_store)
        self.m_T = m_T + self.params.I_tfv_min  # !!!!

        min_tritium = np.min(m_T)
        self.m_T_req = self.m_T_start - min_tritium

        while abs(self.m_T_req - self.m_T_start) / self.m_T_req > self.conv_thresh:
            # Recursively called until start-up inventory is roughly equal to
            # the initial seeded (and re-calculated) value. This is important
            # to accurately calculate decay losses (which are a function of
            # mass)

            self.iterations += 1
            if self.verbose:
                old_m_start = self.m_T_start + self.params.I_tfv_min
                new_m_start = self.m_T_start - min_tritium + self.params.I_tfv_min
                bluemira_print(
                    f"m_T_start old: {old_m_start:.2f} kg \n"
                    f"m_T_start new: {new_m_start:.2f}"
                    f" kg\niterations: {self.iterations}"
                )
            self.m_T_start -= min_tritium
            self.recycle()

    def calc_t_d(self) -> tuple[int | None, float]:
        """
        Calculate the doubling time of a fuel cycle timeline, assuming that a future
        tokamak requires the same start-up inventory as the present one.

        Returns
        -------
        arg_t_d:
            index of doubling time in time array
        :
            Doubling time of the tritium fuel cycle [y]

        \t:math:`t_{d} = t[\\text{max}(\\text{argmin}\\lvert m_{T_{store}}-I_{TFV_{min}}-m_{T_{start}}\\rvert))]`
        """  # noqa: W505, E501
        t_req = self.m_T[0] + self.params.I_tfv_min
        m_temp = self.m_T[::-1]
        try:
            arg_t_d_temp = next(i for i, v in enumerate(m_temp) if v < t_req)
        except StopIteration:
            # Technically, an infinte doubling time is correct here, however it
            # does make the database rather annoying to build reduced laws from
            # TODO: Consider another way...
            return None, float("Inf")
        else:
            arg_t_d = len(self.t) - arg_t_d_temp
            # Check a little around
            if not any(x > t_req for x in self.m_T[arg_t_d - 10 : arg_t_d + 10]):
                return None, float("Inf")
        try:
            return arg_t_d, self.t[arg_t_d]
        except IndexError:
            return arg_t_d - 1, self.t[-1]

    def calc_t_infl(self) -> tuple[int, float]:
        """
        Calculate the inflection time of the reactor tritium inventory

        Returns
        -------
        arg_t_infl:
            index of inflection time in time array
        :
            inflection time
        """
        arg_t_infl = int(np.argmin(self.m_T))
        return arg_t_infl, self.t[arg_t_infl]

    def calc_m_release(self) -> float:
        """
        Calculate the tritium release rate from the entire system to the environment.

        Returns
        -------
        :
            Tritium release rate [kg/yr]
        """
        max_load_factor = find_max_load_factor(self.DEMO_t, self.DEMO_rt)
        if max_load_factor is None:
            max_load_factor = 0.0
        mb = 1000 * raw_uc(max(self.brate), "g/s", "kg/s")
        m_gas = 1000 * raw_uc(max(self.grate), "g/s", "kg/s")
        return legal_limit(
            max_load_factor,
            self.params.f_b,
            m_gas,
            self.params.eta_f,
            self.params.eta_fuel_pump,
            self.params.f_dir,
            self.params.f_exh_split,
            self.params.f_detrit_split,
            self.params.f_terscwps,
            self.params.TBR,
            mb=mb,
        )


@dataclass
class EUDEMOFuelCycleResults:
    """Results of the demo fuel cycle calculation"""

    TBR: float
    DEMO_t: npt.NDArray
    m_T_req: float
    t: npt.NDArray
    t_d: float
    arg_t_d: int | None
    t_infl: float
    arg_t_infl: int
    max_T: npt.NDArray  # noqa: N815
    min_T: npt.NDArray  # noqa: N815
    m_T: npt.NDArray
    I_tfv: npt.NDArray
    I_tfv_min: float
    I_stack: npt.NDArray
    I_plasma: npt.NDArray
    I_blanket: npt.NDArray
    brate: npt.NDArray
    prate: npt.NDArray
    m_dot_release: float

    def tbreed(self, TBR: float, m_T_0: float) -> np.ndarray:
        """
        Ideal system without T sequestration. Used for plotting and sanity.

        Parameters
        ----------
        TBR:
            Tritium breeding ratio
        m_T_0:
            Initial tritium mass

        Returns
        -------
        :
            Tritium mass over time
        """
        m_T = m_T_0 * np.ones(len(self.DEMO_t))
        for i in range(1, len(self.DEMO_t)):
            dt = self.DEMO_t[i] - self.DEMO_t[i - 1]
            dts = YR_TO_S * dt
            t_bred = TBR * self.brate[i] * dts
            t_bred += self.prate[i] * dts
            t_burnt = self.brate[i] * dts
            t_DD = self.prate[i] * dts
            m_T[i] = (m_T[i - 1]) * np.exp(-T_LAMBDA * dt) - t_burnt + t_bred + t_DD
        return m_T

    def plot(self):
        """
        Plot the results of the fuel cycle model.
        """
        _, ax = plt.subplots(2, 1)
        self.plot_m_T(ax=ax[0])
        self.plot_inventory(ax=ax[1])

    def plot_m_T(self, **kwargs):
        """
        Plot the evolution of the tritium masses over time.

        Returns
        -------
        :
            The plot axis
        """
        ax = kwargs.get("ax", plt.gca())
        ax.plot(
            self.DEMO_t,
            self.tbreed(self.TBR, self.m_T_req),
            label="Total T inventory",
        )

        (c,) = ax.plot(
            self.t[self.max_T[0].astype(int)],
            self.max_T[1],
            label="Total unsequestered T inventory",
        )
        ax.plot(
            self.t,
            self.m_T,
            color="gray",
            alpha=0.8,
            linewidth=0.1,
            label="Unsequestered T inventory in stores",
        )
        ax.plot(self.t[self.max_T[0].astype(int)], self.max_T[1], color=c.get_color())
        leg = ax.legend()
        for line in leg.get_lines():
            line.set_linewidth(3)
            line.set_alpha(1)
        ax.set_ylabel("$m_{{T}}$ [kg]")
        if "ax" not in kwargs:
            ax.set_xlabel("Elapsed plant lifetime [years]")
        ax.annotate(
            "$m_{T_{start}}$",
            xy=(0.0, self.m_T_req),
            xytext=(1.0, self.m_T_req + 4),
            arrowprops={"headwidth": 0.5, "width": 0.5, "facecolor": "k", "shrink": 0.1},
        )

        if np.isfinite(self.t_d):
            s = 1.0 if self.t_d < 0.8 * self.DEMO_t[-1] else -1.5
            ax.annotate(
                "$t_{d}$",
                xy=(self.t_d, 0.0),
                xytext=(self.t_d + s, 4.0),
                arrowprops={
                    "headwidth": 0.5,
                    "width": 0.5,
                    "facecolor": "k",
                    "shrink": 0.1,
                },
            )
            if self.arg_t_d is not None:
                self._plot_t_d(ax=ax)
        else:
            ax.annotate(r"$t_{d}=\infty$", xy=(self.t[-1] - 3, 2.0))

        if np.isfinite(self.t_infl):
            s = 1.0 if self.t_d < 0.8 * self.DEMO_t[-1] else 1.5
            ax.annotate(
                "$t_{infl}$",
                xy=(self.t_infl, 0.0),
                xytext=(self.t_infl + s, 2.0),
                arrowprops={
                    "headwidth": 0.5,
                    "width": 0.5,
                    "facecolor": "k",
                    "shrink": 0.1,
                },
            )
            if self.arg_t_infl is not None:
                self._plot_t_infl(self.arg_t_infl, ax=ax)

        ax.set_xlim(0.0, self.t[-1])
        ax.set_ylim(bottom=0)

        return ax

    def plot_inventory(self, **kwargs):
        """
        Plot the evolution of the tritium inventories (including sequestered)
        over time.
        """
        ax = kwargs.get("ax", plt.gca())
        chop = -2

        inventory = self._adjust_inv_plot(self.DEMO_t[:chop], self.I_plasma[:chop])
        ax.plot(self.DEMO_t[:chop], inventory, label="In-vessel")
        inventory = self._adjust_inv_plot(self.DEMO_t[:chop], self.I_blanket[:chop])

        ax.plot(self.DEMO_t[:chop], inventory, label="Blanket")
        ax.plot(self.t[:chop], self.I_tfv[:chop], label="TFV systems")
        # ax.plot(self.t[:chop], self.I_stack[:chop],
        #        label='Environment')
        ax.legend(bbox_to_anchor=[0.99, 0.89])
        if "ax" not in kwargs:
            ax.set_title("Trapped T [kg]")
        ax.set_ylabel("$m_{{T}}$ [kg]")
        ax.set_xlabel("Elapsed plant lifetime [years]")
        ax.set_xlim(left=0)

    @staticmethod
    def _adjust_inv_plot(
        t: np.ndarray, inventory: np.ndarray, thresh: float = 0.2
    ) -> np.ndarray:
        """
        Plot correction for compressed time inventories

        Returns
        -------
        :
            inventory
        """
        inventory = inventory.copy()
        for i in np.nonzero(t - np.roll(t, 1) > thresh)[0] - 2:
            inventory[i + 1] = inventory[i]
        return inventory

    def _plot_t_d(self, **kwargs):
        ax = kwargs.get("ax", plt.gca())
        ax._get_lines.get_next_color()
        vlinex = [self.t_d, self.t_d]
        vliney = [0.0, self.m_T[0] + self.I_tfv_min]
        (c,) = ax.plot(self.t_d, self.m_T[0] + self.I_tfv_min, marker="o", markersize=10)
        ax.plot(vlinex, vliney, color=c.get_color(), linestyle="--")

    def _plot_t_infl(self, arg: int, **kwargs):
        ax = kwargs.get("ax", plt.gca())
        ax._get_lines.get_next_color()
        vlinex = [self.t[arg], self.t[arg]]
        vliney = [0.0, self.m_T[arg]]
        (c,) = ax.plot(self.t[arg], self.m_T[arg], marker="o", markersize=10)
        ax.plot(vlinex, vliney, color=c.get_color(), linestyle="--")

    def sanity(self):
        """
        Check that no tritium is lost (graphically).
        """
        _, ax = plt.subplots()
        m_ideal = self.tbreed(self.TBR, self.m_T_req)
        inter = interp1d(self.DEMO_t, self.I_blanket)
        bb_inventory = inter(self.t)
        inter = interp1d(self.DEMO_t, self.I_plasma)
        pl_inventory = inter(self.t)
        m_tritium = (
            self.m_T
            + bb_inventory
            + pl_inventory
            + self.I_tfv
            - self.I_tfv_min
            + self.I_stack
        )
        ax.plot(self.t, m_tritium, label="max with sequestered")
        ax.plot(self.DEMO_t, m_ideal, label="ideal")
        ax.plot(self.t[self.max_T[0].astype(int)], self.max_T[1], label="max yellow")
        ax.legend()


@dataclass
class EDFCMParams:
    """
    Parameters required to run :class:`~bluemira.fuel_cycle.cycle.EUDEMOFuelCycleModel`.
    """

    TBR: float = 1.05
    """Tritium breeding ratio [dimensionless]."""
    f_b: float = 0.015
    """Burn-up fraction [dimensionless]."""
    m_gas: float = 50
    """
    Gas puff flow rate [Pa m^3/s]. To maintain detachment - no chance of
    fusion from gas injection.
    """
    A_global: float = 0.3
    """Load factor [dimensionless]."""
    r_learn: float = 1
    """Learning rate [dimensionless]."""
    t_pump: float = 100
    """
    Time in DIR loop [s]. Time between exit from plasma and entry into
    plasma through DIR loop.
    """
    t_exh: float = 3600
    """
    Time in INDIR loop [s]. Time between exit from plasma and entry into
    TFV systems INDIR.
    """
    t_ters: float = 18000
    """Time from BB exit to TFV system [s]."""
    t_freeze: float = 1800
    """Time taken to freeze pellets [s]."""
    f_dir: float = 0.9
    """Fraction of flow through DIR loop [dimensionless]."""
    t_detrit: float = 36000
    """Time in detritiation system [s]."""
    f_detrit_split: float = 0.9999
    """Fraction of detritiation line tritium extracted [dimensionless]."""
    f_exh_split: float = 0.99
    """Fraction of exhaust tritium extracted [dimensionless]."""
    eta_fuel_pump: float = 0.9
    """
    Efficiency of fuel line pump [dimensionless]. Pump which pumps down
    the fuelling lines.
    """
    eta_f: float = 0.5
    """
    Fuelling efficiency [dimensionless]. Efficiency of the fuelling
    lines prior to entry into the VV chamber.
    """
    I_miv: float = 0.3
    """Maximum in-vessel T inventory [kg]."""
    I_tfv_min: float = 2
    """
    Minimum TFV inventory [kg]. Without which e.g. cryodistillation
    columns are not effective.
    """
    I_tfv_max: float = 2.2
    """
    Maximum TFV inventory [kg]. Account for T sequestration inside the T
    plant.
    """
    I_mbb: float = 0.055
    """Maximum BB T inventory [kg]."""
    eta_iv: float = 0.9995
    """In-vessel bathtub parameter [dimensionless]."""
    eta_bb: float = 0.995
    """BB bathtub parameter [dimensionless]."""
    eta_tfv: float = 0.998
    """TFV bathtub parameter [dimensionless]."""
    f_terscwps: float = 0.9999
    """TERS and CWPS cumulated factor [dimensionless]."""
