# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.
"""
Fusion power reactor lifecycle object.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import json
from typing import Type
from scipy.optimize import brentq
from bluemira.base.look_and_feel import bluemira_warn, bluemira_print
from BLUEPRINT.base import ReactorSystem, ParameterFrame
from BLUEPRINT.base.file import get_BP_path
from bluemira.base.constants import S_TO_YR, YR_TO_S
from BLUEPRINT.utilities.tools import delta, is_num
from BLUEPRINT.neutronics.simpleneutrons import NeutronicsRulesOfThumb as nROT
from BLUEPRINT.fuelcycle.timeline import f_gompertz, histify, Timeline


class LifeCycle(ReactorSystem):
    """
    Builds a lifecycle for DEMO

    Parameters
    ----------
    A_global: float
        Target load factor (not always used)
    I_p: float
        Plasma current [MA]
    bmd: float
        Blanket maintenance duration [days]
    dmd: float
        Divertor maintenance duration [days]
    t_pulse: float
        Pulse length duration [s]
    t_cs_recharge: float
        CS recharge duration [s] (NOTE: Should be seen as minimum dwell t)
    s_ramp_down: float
        Plasma current ramp-down rate [MA/s]
    s_ramp_up: float
        Plasma current ramp-up rate [MA/s]
    n_reactions: float
        Number of D-T reactions per second
    """

    config: Type[ParameterFrame]
    inputs: dict

    # fmt: off
    default_params = [
        ["A_global", "Global load factor", 0.3, "N/A", "Not always used", "Input"],
        ["I_p", "Plasma current", 19, "MA", None, "Input"],
        ["bmd", "Blanket maintenance duration", 150, "days", "Full replacement intervention duration", "Input"],
        ["dmd", "Divertor maintenance duration", 90, "days", "Full replacement intervention duration", "Input"],
        ["t_pulse", "Pulse length", 7200, "s", "Includes ramp-up and ramp-down time", "Input"],
        ["t_cs_recharge", "CS recharge time", 600, "s", "Presently assumed to dictate minimum dwell period", "Input"],
        ["t_pumpdown", "Pump down duration of the vessel in between pulses", 599, "s", "Presently assumed to take less time than the CS recharge", "Input"],
        ["s_ramp_up", "Plasma current ramp-up rate", 0.1, "MA/s", None, "R. Wenninger"],
        ["s_ramp_down", "Plasma current ramp-down rate", 0.1, "MA/s", None, "R. Wenninger"],
        ["n_DT_reactions", "D-T fusion reaction rate", 7.078779946428698e20, "1/s", "At full power", "Input"],
        ["n_DD_reactions", "D-D fusion reaction rate", 8.548069652616976e18, "1/s", "At full power", "Input"],
        ["a_min", "Minimum operational load factor", 0.1, "N/A", "Otherwise nobody pays", "Input"],
        ["a_max", "Maximum operational load factor", 0.5, "N/A", "Can be violated", "Input"],
        ["r_learn", "Learning curve rate", 1, "1/fpy", "Looks good", "Input"],
        ["blk_1_dpa", "Starter blanket life limit (EUROfer)", 20, "dpa", "http://iopscience.iop.org/article/10.1088/1741-4326/57/9/092002/pdf", "Input"],
        ["blk_2_dpa", "Second blanket life limit (EUROfer)", 50, "dpa", "http://iopscience.iop.org/article/10.1088/1741-4326/57/9/092002/pdf", "Input"],
        ["div_dpa", "Divertor life limit (CuCrZr)", 5, "dpa", "http://iopscience.iop.org/article/10.1088/1741-4326/57/9/092002/pdf", "Input"],
        ["vv_dpa", "Vacuum vessel life limit (SS316-LN-IG)", 3.25, "dpa", "RCC-Mx or whatever it is called", "Input"],
        ["tf_fluence", "Insulation fluence limit for ITER equivalent to 10 MGy", 3.2e21, "n/m^2", "http://ieeexplore.ieee.org/document/6374236/", "Input"],
    ]
    # fmt: on

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs

        self.params = ParameterFrame(self.default_params.to_records())
        self.params.update_kw_parameters(self.config)

        # Derive/convert inputs
        self.maintenance_l = self.params.bmd * 24 * 3600  # [s]
        self.maintenance_s = self.params.dmd * 24 * 3600  # [s]
        self.t_rampup = self.params.I_p / self.params.s_ramp_up  # [s]
        self.t_rampdown = self.params.I_p / self.params.s_ramp_down  # [s]
        self.t_flattop = self.params.t_pulse - self.t_rampup - self.t_rampdown  # [s]
        self.t_min_down = max(self.params.t_cs_recharge, self.params.t_pumpdown)
        # TODO: Neutronics and design requirements still from outdated methods
        # Output preparation
        datadir = get_BP_path(subfolder="data")
        file = "DEMO_lifecycle"
        self.filename = datadir + "/" + file + ".json"
        # Build timeline
        self.life_neutronics()
        self.set_availabilities(self.params.A_global, mode="Global")
        self.timeline()
        self.sanity()

    def set_availabilities(self, load_factor, mode="Global"):
        """
        Sets availability and distributes it between the two phases of\
        planned operation. The planned maintenance windows are substracted\
        from the availability which needs to be achieved during the phase of\
        operation. The target overall plant lifetime availability as specified\
        in input parameter A remains the same.
        \t:math:`A_{overall}=\\dfrac{t_{on}}{t_{on}+t_{off}}`\n
        \t:math:`A_{operations}=\\dfrac{t_{on}}{t_{on}+t_{ramp}+t_{CS_{recharge}}+t_{m_{unplanned}}}`
        """

        def ff(x):
            """
            Optimisation objective for chunky fit to Gompertz

            \t:math:`a_{min}+(a_{max}-a_{min})e^{\\dfrac{-\\text{ln}(2)}{e^{-ct_{infl}}}}`
            """
            # NOTE: Fancy analytical integral objective of Gompertz function
            # was a resounding failure. Do not touch this again.
            # The brute force is strong in this one.
            a_ops_i = self.params.a_min + f_gompertz(
                t, (self.params.a_max - self.params.a_min), x, self.params.r_learn
            )
            a_ops_i = np.array(
                [np.mean(a_ops_i[arg_dates[i] : d]) for i, d in enumerate(arg_dates[1:])]
            )
            return self.fpy / self.params.A_global - (
                sum(p / a_ops_i) + S_TO_YR * self.m
            )

        self.m = self.maintenance_l * self.n_blk_replace + (
            self.maintenance_s * self.n_div_replace
        )
        self.t_interdown = sum(self.n_pulse_p) * self.t_min_down
        self.total_ramptime = sum(self.n_pulse_p) * (self.t_rampup + self.t_rampdown)
        self.min_downtime = self.m + self.t_interdown + self.total_ramptime
        self.unplanned = self.t_on_total / load_factor - (
            self.m + self.t_interdown + self.total_ramptime + self.t_on_total
        )
        if mode == "Global":
            # Sets lifetime load factor `A` (input) subtracts planned
            # maintenance intervals, and sets phase load factors accordingly
            # NOTE: a_max can be violated if A is high (not nec. > a_max)
            # Handle funky inputs
            if load_factor >= self.params.a_max:
                # Fudge upwards - works for up to .7
                self.params.a_max += 0.1 + 1.1 * (load_factor - self.params.a_max)
            elif load_factor < self.params.a_min:
                self.params.a_min = 0
            t = np.linspace(0, self.fpy, 100)
            dates = [0]
            for p in self.get_op_phases():
                dates.append(dates[-1] + p)
            arg_dates = [np.argmin(abs(t - i)) for i in dates]
            p = self.get_op_phases()
            b = brentq(ff, 0, 10e10)
            a_ops = self.params.a_min + f_gompertz(
                t, self.params.a_max - self.params.a_min, b, self.params.r_learn
            )
            dates = [0]
            for p in self.get_op_phases():
                dates.append(dates[-1] + p)
            arg_dates = np.array([np.argmin(abs(t - i)) for i in dates])
            self._t_argdates = t[arg_dates]  # Plotting use
            self._a = a_ops  # Plotting use
            # Get phase operational load factors
            self.a_ops = np.array(
                [np.mean(a_ops[arg_dates[i] : d]) for i, d in enumerate(arg_dates[1:])]
            )
        elif mode == "Life":
            spreadf = 0.8
            self.unplanned = self.t_on_total / load_factor - (
                self.min_downtime - self.t_on_total
            )
            a_ops = self.t_on_total / (
                self.t_on_total + self.total_ramptime + self.cs_down + self.unplanned
            )
            if a_ops > 1:
                bluemira_warn(
                    "Plant availability target cannot be met "
                    "with input maintenance durations."
                )
                a_ops = self.t_on_total / (self.t_on_total + self.min_downtime)
                bluemira_print(
                    "Recalculated plant availability based on input"
                    " CS recharge time and maintenance durations as "
                    "{0}".format(round(a_ops, 2))
                )

            def spread(a_ops, spreadf):
                """Distribute availabilities between two phases"""
                # Ratio of phase lengths
                r_t = self.blk_2_dpa / self.blk_1_dpa
                a_p1 = spreadf * a_ops
                a_p2 = r_t / ((r_t + 1) / a_ops - 1 / a_p1)
                return a_p1, a_p2

            a_ops = spread(a_ops, spreadf)
            op_phases = [p for p in self.phases if "Phase P" in p[1]]
            a_p1 = [phase[0] for phase in op_phases if "P1" in phase[1]]
            a_p2 = [phase[0] for phase in op_phases if "P2" in phase[1]]
            self.a_ops = a_p1 + a_p2
        elif mode == "Operational":
            # Here we set availabilities on operational phases only, and work
            # out the global A.

            self.a_ops = [0.1, 0.2, 0.2, 0.4, 0.4, 0.5]
            ops_pulse_p = [n for n in self.n_pulse_p if n > 1]
            p_unplanned = []

            total = self.t_flattop + self.t_rampdown + self.t_rampup + self.t_min_down

            for i, p in enumerate(self.a_ops):
                p_unplanned.append(
                    ops_pulse_p[i] * self.t_flattop / self.a_ops[i]
                    - ops_pulse_p[i] * total
                )
            self.t_unplanned_m = sum(p_unplanned)
            # Calculate global load factor
            self.A_global = self.t_on_total / (
                self.t_on_total + self.min_downtime + self.t_unplanned_m
            )

    def life_neutronics(self):
        """
        Calculate the lifetime of various components based on their damage limits
        and fluences.
        """
        # Neutron flux in TF coil insulation [MC: based on PP email 20/02/17 -
        # difficult to read figure] 2:actual 1.4:make it look good
        neutrons = nROT()
        self.tf_ins_nflux = neutrons.tf_ins_nflux  # [n/m^2/s]
        # Blanket outboard midplane EUROFER dmg rate [PP: 2M7HN3 fig. 20]
        self.blk_dmg = neutrons.blk_dmg  # [dpa/FPY]
        # Divertor region CuCrZr dmg rate [MC: CB assumption]
        self.div_dmg = neutrons.div_dmg  # [dpa/MW.annum/m^2]
        # VV peak SS316LN-IG dmg rate [MC: PP 2M7HN3 fig. 18]
        self.vv_dmg = neutrons.vv_dmg  # [dpa/MW.annum/m^2]
        divl = self.params.div_dpa / self.div_dmg  # [fpy] Divertor life
        blk1l = self.params.blk_1_dpa / self.blk_dmg  # [fpy] 1st Blanket life
        blk2l = self.params.blk_2_dpa / self.blk_dmg  # [fpy] 2nd Blanket life
        # vvl = self.params.vv_dpa / self.vv_dmg  # [fpy] VV life
        # Number of divertor changes in 1st blanket life
        ndivch_in1blk = int(blk1l / divl)
        # Number of divertor changes in 2nd blanket life (1 for div reset)
        ndivch_in2blk = int(blk2l / divl)
        self.n_blk_replace = 1  # HLR
        self.n_div_replace = ndivch_in1blk + ndivch_in2blk
        m_short = self.maintenance_s * S_TO_YR
        m_long = self.maintenance_l * S_TO_YR
        phases = []
        for i in range(ndivch_in1blk):

            p_str = "Phase P1." + str(i + 1)
            m_str = "Phase M1." + str(i + 1)
            phases.append([divl, p_str])
            phases.append([m_short, m_str])
            count = i
        if ndivch_in1blk == 0:
            count = 0
        phases.append([blk1l % divl, "Phase P1." + str(count + 2)])
        phases.append([m_long, "Phase M1." + str(count + 2)])
        for i in range(ndivch_in2blk):

            p_str = "Phase P2." + str(i + 1)
            m_str = "Phase M2." + str(i + 1)
            phases.append([divl, p_str])
            phases.append([m_short, m_str])
            count2 = i
        phases.append([blk2l % divl, "Phase P2." + str(count2 + 1)])
        self.phase_durations = [p[0] for p in phases]
        self.phase_names = [p[1] for p in phases]
        self.calc_n_pulses(phases)
        fpy = 0
        for i in range(len(phases)):
            if phases[i][1].startswith("Phase P"):
                fpy += phases[i][0]
        self.fpy = fpy
        # Irreplaceable components life checks
        self.t_on_total = self.fpy * YR_TO_S  # [s] total fusion time
        tf_ins_life_dose = self.tf_ins_nflux * self.t_on_total / self.params.tf_fluence
        if tf_ins_life_dose > 1:
            self.tf_lifeend = round(
                self.params.tf_fluence / (self.tf_ins_nflux) / (3600 * 24 * 365), 2
            )
            tflifeperc = round(100 * self.tf_lifeend / self.fpy, 1)
            bluemira_warn(
                "TF coil insulation fried after {0} full-power years"
                ", or {1} % of neutron budget"
                ".".format(self.tf_lifeend, tflifeperc)
            )
        vv_life_dmg = self.vv_dmg * self.fpy / self.params.vv_dpa
        if vv_life_dmg > 1:
            self.vv_lifeend = round(self.params.vv_dpa / self.vv_dmg, 2)
            vvlifeperc = round(100 * self.vv_lifeend / self.fpy, 1)
            bluemira_warn(
                "VV fried after {0} full-power"
                " years, or {1} % of neutron budget"
                ".".format(self.vv_lifeend, vvlifeperc)
            )
        self.add_parameter(
            "n_cycles",
            "Total number of D-T pulses",
            self.fpy * YR_TO_S / self.t_flattop,
            "",
            None,
            "BLUEPRINT",
        )

    def calc_n_pulses(self, phases):
        """
        Calculate the number of pulses per phase.
        """
        n_pulse_p = []
        for i in range(len(phases)):
            if phases[i][1].startswith("Phase P"):
                # TODO: Change to //
                n_pulse_p.append(int(round(YR_TO_S * phases[i][0] / self.t_flattop, 0)))
            else:
                pass
        self.n_pulse_p = n_pulse_p

    def get_op_phases(self):
        """
        Get the operational phases for the LifeCycle.
        """
        return [
            d for n, d in zip(self.phase_names, self.phase_durations) if "Phase P" in n
        ]

    def timeline(self):
        """
        Builds a Timeline instance
        """
        n = len(self.n_pulse_p)

        for k in ["t_rampup", "t_flattop", "t_rampdown", "t_min_down"]:
            v = getattr(self, k)
            if is_num(v):
                setattr(self, k, v * np.ones(n))

        n_DT_reactions = self.params.n_DT_reactions * np.ones(n)
        n_DD_reactions = self.params.n_DD_reactions * np.ones(n)
        Ip = self.params.I_p * np.ones(n)

        timeline = Timeline(
            self.phase_names,
            self.phase_durations,
            self.a_ops,
            self.n_pulse_p,
            self.t_rampup,
            self.t_flattop,
            self.t_rampdown,
            self.t_min_down,
            n_DT_reactions,
            n_DD_reactions,
            Ip,
            self.params.A_global,
            self.blk_dmg,
            self.params.blk_1_dpa,
            self.params.blk_2_dpa,
            self.div_dmg,
            self.params.div_dpa,
            self.tf_ins_nflux,
            self.params.tf_fluence,
            self.vv_dmg,
            self.params.vv_dpa,
        )
        self.T = timeline
        self.t_unplanned_m = self.T.t_unplanned_m
        return self.T

    def sanity(self):
        """
        Perform a sanity check. Will raise warnings if the LifeCycle generates
        results that violate the tolerances.
        """
        life = self.fpy / self.params.A_global
        actual_life = S_TO_YR * (
            self.t_on_total
            + self.total_ramptime
            + self.t_interdown
            + self.m
            + self.t_unplanned_m
        )
        actual_lf = self.fpy / actual_life
        delt = delta(actual_life, life)
        delta2 = delta(actual_lf, self.params.A_global)
        if delt > 0.015:
            bluemira_warn(
                "FuelCycle::Lifecyle: discrepancy between actual and planned\n"
                "reactor lifetime\n"
                f"Actual: {actual_life:.2f}\n"
                f"Planned: {life:.2f}\n"
                f"% diff: {100*delt:.4f}\n"
                "the problem is probably related to unplanned maintenance."
            )
            self.__init__(self.config, self.inputs)  # Phoenix

        if delta2 > 0.015:
            bluemira_warn(
                "FuelCycle::Lifecyle: availability discrepancy greated that\n"
                "specified tolerance\n"
                f"Actual: {actual_lf:.4f}\n"
                f"Planned: {self.params.A_global:.4f}\n"
                f"% diff: {100*delta2:.4f}\n"
                "the problem is probably related to unplanned maintenance."
            )
            self.__init__(self.config, self.inputs)  # Phoenix

        if self.params.A_global > self.fpy / (self.fpy + S_TO_YR * self.min_downtime):
            bluemira_warn("FuelCycle::Lifecyle: Input availability is unachievable.")
        # Re-assign A
        self.params.update_kw_parameters({"A_global": actual_lf})
        # self.A_global = actual_A

    def plot_pulse(self):
        """
        Plot a typical pulse in the LifeCycle.
        """
        t_dwell = self.t_min_down
        self.t.extend(self.t + np.sum(t_dwell + self.t_pulse))
        self.I.extend(self.I)
        self.frate.extend(self.frate)
        f, ax = plt.subplots(2, 1, sharex=True)
        ax = np.ravel(ax)
        ax[0].plot(self.t, self.I)
        ax[0].set_ylim([-0.1, 25])
        ax[0].set_xlim([-200, self.t_pulse + t_dwell + 1000])
        ax[0].set_ylabel("$I_P$ [MA]")
        ax[0].annotate("$t_{rampup} = %s s$" % self.t_rampup, xy=[200, 10])
        ax[0].annotate("$t_{rampdown} = %s s$" % self.t_rampdown, xy=[5500, 10])
        ax[0].annotate("$t_{flattop} = %s s$" % (self.t_flattop), xy=[3000, 17.5])
        ax[0].annotate(
            "$t_{pulse} = %s s$" % (self.t_pulse), xy=[3000, 20.5], color="red"
        )
        ax[0].annotate("$t_{dwell}=%s s$" % (t_dwell), xy=[7000, 20.5], color="blue")
        ax[0].axvspan(0, self.t_pulse, color="red", alpha=0.1)
        ax[0].axvspan(self.t_pulse, self.t_pulse + t_dwell, color="blue", alpha=0.1)
        ax[0].axvspan(-200, 0, color="blue", alpha=0.1)
        ax[0].axvspan(
            self.t_pulse + t_dwell,
            self.t_pulse + t_dwell + 1000,
            color="red",
            alpha=0.1,
        )
        ax[1].plot(self.t, self.frate)
        ax[1].set_xlabel("$t$ [s]")
        ax[1].set_ylabel("Number of reactions")

    def summary(self):
        """
        Plot the load factor breakdown and learning curve
        """
        f, ax = plt.subplots(1, 2)
        self.plot_learning(ax=ax[0])
        self.plot_load_factor(ax=ax[1])

    def plot_learning(self, ax=None):
        """
        Plot the load factor learning curve and its discretisation into
        operational load factors over phases
        """
        if ax is None:
            ax = plt.gca()
        a = self.a_ops
        x, y = histify(self._t_argdates, a)
        t = np.linspace(0, self.fpy, len(self._a))
        g = ax.plot(t, self._a, linestyle="--")
        ax.plot(
            x,
            y,
            color=g[-1].get_color(),
            label=f"{self.params.A_global:.1f}",
            marker="o",
        )
        # Put a legend to the right of the current axis
        ax.legend(loc="best", title="$A_{glob}$")
        ax.set_xlabel("Full power years")
        ax.set_ylabel("Operational load factor")

    def plot_life(self):
        """
        Plot the different maintenance events in the Lifecycle, and the slope
        of each operational phase
        """
        f, ax = plt.subplots(1, 1)
        # Plotting crutches
        fs, s, h = 0, 0, 0.95 * self.fpy
        ft, rt = [0], [0]
        j = 0
        for p_n, p_d in zip(self.phase_names, self.phase_durations):
            if p_n.startswith("Phase P"):
                c = "b"
                ft.append(fs + p_d)
                fs += p_d
                length = p_d / self.a_ops[j]
                rt.append(s + length)
                j += 1
            elif p_n.startswith("Phase M"):
                ft.append(ft[-1])
                c = "r"
                length = p_d
                rt.append(rt[-1] + p_d)
                if p_n.startswith("Phase M1.2"):
                    m = "s"
                else:
                    m = "o"
                ax.plot(rt[-1] - p_d / 2, 0.3, marker=m, color="r", ms=25)
                ax.axvspan(s, s + length, color=c, alpha=0.2)
            h -= 0.08 * self.fpy
            s += length

        legend = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label="Divertor replacement",
                markerfacecolor="r",
                markersize=25,
            ),
            Line2D(
                [0],
                [0],
                marker="s",
                color="w",
                label="Blanket and divertor \nreplacement",
                markerfacecolor="r",
                markersize=25,
            ),
        ]
        ax.legend(handles=legend)
        ax.plot(rt, ft, color="#0072bd")
        ax.set_xlabel("Elapsed plant lifetime [years]")
        ax.set_ylabel("Full power years [fpy]")
        ax.set_xlim([0, self.fpy / self.params.A_global])
        ax.set_ylim(bottom=0)

    def plot_load_factor(self, typ="pie", ax=None):
        """
        Plots a pie or bar chart of the breakdown of the reactor lifetime

        Parameters
        ----------
        typ: str from ['pie', 'bar']
            Whether to plot a pie or bar chart
        """
        if ax is None:
            ax = plt.gca()
        c = ["#0072bd", "#d95319", "#edb120", "#7e2f8e", "#77ac30"]
        labels = [
            "Fusion time",
            "Ramp-up and\nramp-down",
            "CS recharge",
            "Planned\nmaintenance",
            "Unallocated\ndowntime",
        ]
        sizes = [
            self.t_on_total,
            self.total_ramptime,
            self.t_interdown,
            self.m,
            self.t_unplanned_m,
        ]
        if typ == "pie":
            plt.pie(
                sizes,
                labels=labels,
                colors=c,
                startangle=90,
                autopct="%.2f",
                counterclock=False,
            )
            plt.axis("equal")
        elif typ == "bar":
            bottom = 0
            for i, s in enumerate(sizes):
                ax.bar(1, s / sum(sizes) * 100, bottom=bottom, label=labels[i])
                bottom += s / sum(sizes) * 100
            ax.set_xticklabels([""])
            ax.set_xlim([0, 6])
            ax.legend()
        plt.title(
            "Breakdown of DEMO reactor lifetime\n A = {0:.2f},"
            "{1:.2f} fpy, {2:.2f} years".format(
                self.params.A_global, self.fpy, self.T.plant_life
            )
        )

    def write(self):
        """
        Save a Timeline to a JSON file.
        """
        bluemira_print("Writing {0}".format(self.filename))
        data = self.T.to_dict()
        with open(self.filename, "w") as f_h:
            json.dump(data, f_h, indent=4)

    def read(self):
        """
        Load a Timeline from a JSON file.
        """
        bluemira_print("Reading {0}".format(self.filename))
        with open(self.filename) as f_h:
            data = json.load(f_h)
        return data


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
