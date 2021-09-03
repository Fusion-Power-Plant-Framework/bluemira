# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
Tritium fuelling and vacuum system
"""
from typing import Type
import numpy as np
import time
import matplotlib.pyplot as plt
from BLUEPRINT.base import ParameterFrame, ReactorSystem
from BLUEPRINT.base.lookandfeel import KEY_TO_PLOT
from bluemira.base.look_and_feel import plot_defaults
from BLUEPRINT.utilities.plottools import savefig
from BLUEPRINT.utilities.powerlearn import PowerLaw
from BLUEPRINT.utilities.pypetdatabase import DataBase
from BLUEPRINT.fuelcycle.cycle import FuelCycle

plot_defaults()


class TFVSystem(ReactorSystem):
    """
    Tritium Fuelling and Vacuum System
    Initialises the TFV system object without reactor information
    Runs fuel cycle model separately in `run_model`

    Attributes
    ----------
    timestep : int
        Timestep duration in seconds
    conv_thresh : float << 1
        Convergence threshold for the recursive T fuel cycle start-up inventory
        calls
    learn : bool
        Store results into DB and use to train surrogate model?
    verbose : bool
        Make noise?

    Parameters
    ----------
    see p
    """

    config: Type[ParameterFrame]
    # fmt: off
    default_params = [
        ['TBR', 'Tritium breeding ratio', 1.05, 'N/A', None, 'Input'],
        ['f_b', 'Burn-up fraction', 0.015, 'N/A', None, 'Input'],
        ['m_gas', 'Gas puff flow rate', 50, 'Pam^3/s', 'To maintain detachment - no chance of fusion from gas injection', 'Discussions with Chris Day and Yannick Hörstenmeyer'],
        ['A_global', 'Load factor', 0.3, 'N/A', None, 'Silent input'],
        ['r_learn', 'Learning rate', 1, 'N/A', None, 'Silent input'],
        ['t_pump', 'Time in DIR loop', 2 * 3600, 's', 'Time between exit from plasma and entry into plasma through DIR loop', 'Discussions with Chris Day and Yannick Hörstenmeyer'],
        ['t_exh', 'Time in INDIR loop', 4 * 3600, 's', 'Time between exit from plasma and entry into TFV systems INDIR', None],
        ['t_ters', 'Time from BB exit to TFV system', 5 * 3600, 's', None, None],
        ['t_freeze', 'Time taken to freeze pellets', 3600 / 2, 's', None, 'Discussions with Chris Day and Yannick Hörstenmeyer'],
        ['f_dir', 'Fraction of flow through DIR loop', 0.9, 'N/A', None, 'Discussions with Chris Day and Yannick Hörstenmeyer'],
        ['t_detrit', 'Time in detritiation system', 10 * 3600, 's', None, None],
        ['f_detrit_split', 'Fraction of detritiation line tritium extracted', 0.9999, 'N/A', None, None],
        ['f_exh_split', 'Fraction of exhaust tritium extracted', 0.99, 'N/A', None, None],
        ['eta_fuel_pump', 'Efficiency of fuel line pump', 0.9, 'N/A', 'Pump which pumps down the fuelling lines', None],
        ['eta_f', 'Fuelling efficiency', 0.5, 'N/A', 'Efficiency of the fuelling lines prior to entry into the VV chamber', None],
        ['I_miv', 'Maximum in-vessel T inventory', 0.2, 'kg', None, None],
        ['I_tfv_min', 'Minimum TFV inventory', 3, 'kg', 'Without which e.g. cryodistillation columns are not effective', None],
        ['I_tfv_max', 'Maximum TFV inventory', 5, 'kg', None, None],
        ['I_mbb', 'Maximum BB T inventory', 5, 'kg', None, None],
        ['eta_iv', 'In-vessel bathtub parameter', 0.9995, 'N/A', None, None],
        ['eta_bb', 'BB bathtub parameter', 0.995, 'N/A', None, None],
        ['eta_tfv', 'TFV bathtub parameter', 0.998, 'N/A', None, None],
        ['f_terscwps', 'TERS and CWPS cumulated factor', 0.9999, 'N/A', None, None]
    ]
    # fmt: on
    DATA = "Tritium DB.csv"
    build_tweaks = {
        "timestep": 1200,
        "n": None,
        "conv_thresh": 0.0002,
        "learn": False,
        "verbose": False,
    }

    def __init__(self, config):
        self.config = config

        self.params = ParameterFrame(self.default_params.to_records())
        self.params.update_kw_parameters(self.config)

    def get_startup_inventory(self, query="max", method="run"):
        """
        Get the tritium start-up inventory.

        Parameters
        ----------
        query: str
            The type of statistical value to return
            - [min, max, mean, median, 95th]

        Returns
        -------
        m_T_start: float
            The tritium start-up inventory [kg]
        """
        # TODO: predict
        if method == "run":
            return self._query("m_T_req", query)
        elif method == "predict":
            return self.prediction("m_T_startup")

    def get_doubling_time(self, query="max", method="run"):
        """
        Get the reactor doubling time.

        Parameters
        ----------
        query: str
            The type of statistical value to return
            - [min, max, mean, median, 95th]

        Returns
        -------
        t_d: float
            The reactor doubling time [years]
        """
        # TODO: predict
        if method == "run":
            return self._query("t_d", s=query)
        elif method == "predict":
            return self.prediction("t_d")

    def _query(self, p: str, s: str):
        if s == "min":
            return min(self.__dict__[p])
        if s == "max":
            return max(self.__dict__[p])
        elif s == "mean":
            return np.mean(self.__dict__[p])
        elif s == "median":
            return np.median(self.__dict__[p])
        elif s == "95th":
            return np.percentile(self.__dict__[p], 95)
        else:
            raise ValueError(f"Unknown query: '{s}'")

    def run_model(self, timelines):
        """
        Runs the tritium fuel cycle model

        Parameters
        ----------
        timelines : dict
            Timeline dict from LifeCycle object:
                DEMO_t : np.array
                    Real time signal in seconds
                DEMO_rt : np.array
                    Fusion time signal in years
                DEMO_DT_rate : np.array
                    D-T fusion reaction rate signal (NOTE: P_fus is wrapped in
                    here, along with maintenance outages, etc.)
                DEMO_DD_rate : np.array
                    D-D fusion reaction rate signal (NOTE: P_fus is wrapped in
                    here, along with maintenance outages, etc.)
                bci : int
                    Blanket change index
        timelines : list
            List of timelines dicts described above
        """
        m_T_req, t_d, m_dot_release = [], [], []
        if type(timelines) is dict:  # Single timeline
            self.T = FuelCycle(self.params, self.build_tweaks, timelines)
            self.m_T_req, self.t_d = [self.T.m_T_req], [self.T.t_d]
            self.m_dot_release = [self.T.m_dot_release]

        elif type(timelines) is list:  # Multiple timelines
            for timeline in timelines:
                self.T = FuelCycle(self.params, self.build_tweaks, timeline)
                m_T_req.append(self.T.m_T_req)
                t_d.append(self.T.t_d)
                m_dot_release.append(self.T.m_dot_release)
            self.m_T_req, self.t_d = m_T_req, t_d
            self.m_dot_release = m_dot_release

    def prediction(self, parameter, **kwargs):
        """
        Predict the value of the start-up inventory or doubling time based on
        a power law regression fit.

        Parameters
        ----------
        parameter: str
            The parameter to predict 'm_T_startup' or 't_d'

        Notes
        -----
        Interpret machine learning T start-up inventory estimate as a seed to
        minimise Picard iterations.
        """
        if parameter not in ["m_T_startup", "t_d"]:
            raise ValueError
        self.m_T_PL = PowerLaw(
            datafile=self.DATA, targets=["m_T_startup", "t_d"], target=parameter
        )
        return self.m_T_PL.predict(**kwargs)[0]

    def dist_plot(self, figsize=[12, 6], bins=20, **kwargs):
        """
        Plot the distributions of m_T_start and t_d.
        """
        f, ax = plt.subplots(1, 2, sharey=True, figsize=figsize)
        ax[0].hist(self.m_T_req, bins=bins, **kwargs)
        ax[0].set_xlabel("$m_{T_{start}}$ [kg]")
        ax[0].set_ylabel("$n$")

        if np.all(self.t_d == np.inf):
            # If all the doubling times are infinite, make a special hist plot
            t_d = 100 * np.ones(len(self.t_d))
            ax[1].hist(
                t_d,
                bins=bins,
                color=next(ax[0]._get_lines.prop_cycler)["color"],
                **kwargs,
            )
            ax[1].set_xticks([100 + 0.5 / bins])
            ax[1].set_xticklabels([r"$\infty$"])
        else:
            ax[1].hist(
                self.t_d,
                bins=bins,
                color=next(ax[0]._get_lines.prop_cycler)["color"],
                **kwargs,
            )
        ax[1].set_xlabel("$t_{d}$ [yr]")
        f.tight_layout()
        savefig(f, "mt_distribution", save=KEY_TO_PLOT)

    def plot(self):
        """
        Plot a typical fuel cycle run.
        """
        f, ax = plt.subplots(2, 1, sharex=True)
        self.T.plot_m_T(ax=ax[0])
        self.T.plot_inventory(ax=ax[1])
        f.tight_layout()
        savefig(f, "default_mtI_new", save=KEY_TO_PLOT)


def tlearn(datafile, **kwargs):
    """
    Interpret machine learning T start-up inventory estimate as a seed to
    minimise Picard iterations.
    """
    m_T_PL = PowerLaw(
        datafile=datafile + ".csv", targets=["m_T_startup"], target="m_T_startup"
    )
    startup = m_T_PL.predict(kwargs["kwargs"])[0]
    return startup


def build_TFVDB():
    """
    Builds a DataBase of TFV fuel cycle analyses.
    """
    lc_database = None  # LCDataBase()

    def f(traj):
        m_T_start, t_d, m_dot_release = [], [], []
        input_set = {"A": traj.A_global, "r_learn": traj.r_learn}
        life = lc_database.get_result(input_set)
        life["A"] = life["A_new"]  # Tweak pass
        TFV = TFVSystem(
            {
                "TBR": traj.TBR,
                "f_b": traj.f_b,
                "I_mbb": traj.I_mbb,
                "I_miv": traj.I_miv,
                "I_mtfv": traj.I_mtfv,
                "eta_bb": traj.eta_tfv,
                "eta_f": traj.eta_f,
                "eta_fuel_pump": traj.eta_fuel_pump,
                "eta_iv": traj.eta_iv,
                "eta_tfv": traj.eta_tfv,
                "f_detrit_split": traj.f_detrit_split,
                "f_dir": traj.f_dir,
                "f_exh_split": traj.f_exh_split,
                "t_ters": traj.t_ters,
                "t_detrit": traj.t_detrit,
                "t_pump": traj.t_pump,
                "t_freeze": traj.t_freeze,
                "t_exh": traj.t_exh,
            }
        )
        for i in range(len(life["bci"])):
            life_dict = dict([(k, v[i]) for k, v in life.items()])
            TFV.run_model(life_dict)
            m_T_start.append(TFV.T.m_T_req)
            t_d.append(TFV.T.t_d)
            m_dot_release.append(TFV.T.m_dot_release)
        traj.f_add_result("m_T_start", np.array(m_T_start))
        traj.f_add_result("t_d", np.array(t_d))
        traj.f_add_result("m_dot_release", np.array(m_dot_release))

    inp = TFVSystem.p.get_parameter_list()
    t = time.time()
    database = DataBase(
        "TritiumFuellingVacuum", "TFV_v2", f, inp, ["m_T_start", "t_d", "m_dot_release"]
    )
    par_range_dict = {"f_b": [0.01, 0.03], "f_dir": np.linspace(0.7, 0.9, 2)}
    database.add_ranges(par_range_dict)
    database.run()
    print(f"{time.time()-t:.2f} seconds")


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
