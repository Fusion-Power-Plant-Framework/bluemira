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
Parameter scan tools for the fuel cycle model
"""

import pandas as pd
import numpy as np
from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.interpolate import griddata
from BLUEPRINT.systems.tfv import TFVSystem
from BLUEPRINT.fuelcycle.lifecycle import LifeCycle
from BLUEPRINT.utilities.pypetdatabase import DataBase
from BLUEPRINT.utilities.plottools import savefig, weather_front, mathify

# flake8: noqa  This isn't really source code, more of a collection of procedures
# used for building reduced models of the fuel cycle, and associated parameter
# explorations and plotting.


class TFVDataBase(DataBase):
    """
    Database for TFV fuelcycle modelling explorations.
    """

    def __init__(
        self,
        subdir="TritiumFuellingVacuum",
        name="TFV",
        function=None,
        variables=TFVSystem.default_params.get_parameter_list(),
        results=["m_T_start", "t_d", "m_dot_release"],
        **kwargs,
    ):
        super().__init__(subdir, name, function, variables, results, **kwargs)
        self.function = self.f
        self.DLC = None  # LCDataBase(name="LC_paperA500")  # big_run on LC_paper_r1
        # self.DLC.load()

    def f(self, traj):
        m_start_up, t_d, m_dot_release, = (
            [],
            [],
            [],
        )
        input_set = {"A": traj.A, "r_learn": traj.r_learn}
        life = self.DLC.get_result(input_set)
        life["A"] = life["A_new"]  # Tweak pass
        tfv_system = TFVSystem(
            {
                "TBR": traj.TBR,
                "f_b": traj.f_b,
                "I_mbb": traj.I_mbb,
                "I_miv": traj.I_miv,
                "I_tfv_min": traj.I_tfv_min,
                "I_tfv_max": traj.I_tfv_max,
                "eta_bb": traj.eta_bb,
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
            tfv_system.run_model(life_dict)
            m_start_up.append(tfv_system.T.m_T_req)
            t_d.append(tfv_system.T.t_d)
            m_dot_release.append(tfv_system.T.m_dot_release)
        traj.f_add_result("m_T_start", np.array(m_start_up))
        traj.f_add_result("t_d", np.array(t_d))
        traj.f_add_result("m_dot_release", np.array(m_dot_release))

    def load_into_df(self):
        df = self.frame_data()
        # Drop unique columns
        nunique = df.apply(pd.Series.nunique)
        drop = nunique[nunique == 1].index
        # TODO: Fix full infinity drop...
        self.df = df.drop(drop, axis=1)
        # Drop infinite values in t_d (don't know how to handle these yet)
        # self.df = self.df.replace(np.inf, np.nan)
        # self.df.dropna(inplace=True)

    def get_mmm_dfs(self):
        results = ["m_T_start", "t_d", "m_dot_release"]
        var = list(self.df.columns.drop(results))
        gb = self.df.groupby(var, as_index=False)
        self.df_mean = gb.mean()
        self.df_max = gb.max()
        self.df_median = gb.median()
        self.df_95 = gb.agg(lambda x: np.percentile(x, 95))


def determine_n_montecarlo():
    """
    ⴷⴻⵜⴻⵔⵎⵉⵏⴻⵙ ⵜⵀⴻ ⵏⵓⵎⴱⴻⵔ ⴲⴼ ⵎⴲⵏⵜⴻ ⴽⴰⵔⵍⴲ ⵔⵓⵏⵙ ⴼⴲⵔ ⵜⵀⴻ ⴼⵓⴻⵍ ⴽⵢⴽⵍⴻ
    ⵎⴲⴷⴻⵍ ⵜⴲ ⴽⴲⵏⵠⴻⵔⴳⴻ ⵎⴲⵜⵀ ⵎ_ⵜ_ⵙⵜⴰⵔⵜ ⴰⵏⴷ ⵜ_ⴷ
    ...
    200 ⵍⴲⴲⴾⴻⴷ ⴳⴲⴲⴷ
    """
    nmax = 500
    lives = [
        LifeCycle({"A": 0.3, "r_learn": 1}).get_timeline_dict() for _ in range(nmax)
    ]

    tfv_system = TFVSystem(
        {
            "I_mbb": 3.0,
            "I_miv": 0.3,
            "I_tfv_min": 3.0,
            "I_tfv_max": 5.0,
            "TBR": 1.05,
            "eta_bb": 0.995,
            "eta_f": 0.7,
            "eta_fuel_pump": 0.6,
            "eta_iv": 0.99950000000000006,
            "eta_tfv": 0.9999,
            "f_b": 0.015,
            "f_detrit_split": 0.99999000000000005,
            "f_dir": 0.8,
            "f_exh_split": 0.98999999999999999,
            "t_ters": 18000.0,
            "t_detrit": 20 * 3600,
            "t_pump": 150.0,
            "t_freeze": 3600 / 2,
            "t_exh": 5 * 3600,
        }
    )
    tfv_system.run_model(lives)
    m, t = tfv_system.m_T_req, tfv_system.t_d
    mtmax, mt95 = [], []
    tdmax, td95 = [], []
    for n in range(1, nmax):
        mtmax.append(max(m[:n]))
        mt95.append(np.percentile(m[:n], 95))
        tdmax.append(max(t[:n]))
        td95.append(np.percentile(t[:n], 95))
    f, ax = plt.subplots()
    ax.plot(mtmax / mtmax[-1], label="max(m_T_req)")
    ax.plot(mt95 / mt95[-1], label="95th(m_T_req)")
    ax.plot(tdmax / tdmax[-1], label="max(t_d)")
    ax.plot(td95 / td95[-1], label="95th(t_d)")
    ax.legend()
    return tfv_system


def parameter_scan(defaults, timelines, var, sigma, mu, n=10):
    a = np.linspace(sigma - mu, sigma + mu, num=n + 1)
    mm, tt, rr = [], [], []
    for val in a:
        defaults[var] = val
        tfv_system = TFVSystem(defaults)
        tfv_system.run_model(timelines)
        m, t, r = tfv_system.m_T_req, tfv_system.t_d, tfv_system.m_dot_release
        mm.append(m)
        tt.append(t)
        rr.append(r)
    return mm, tt, rr


def fullscan():
    default = {
        "A": 0.3,
        "r_learn": 1.0,
        "I_mbb": 3.0,
        "I_miv": 0.3,
        "I_tfv_min": 3.0,
        "I_tfv_max": 5.0,
        "TBR": 1.05,
        "eta_bb": 0.995,
        "eta_f": 0.7,
        "eta_fuel_pump": 0.6,
        "eta_iv": 0.9995,
        "eta_tfv": 0.9995,
        "f_b": 0.015,
        "f_detrit_split": 0.9995,
        "f_dir": 0.8,
        "f_exh_split": 0.99,
        "f_terscwps": 0.99995,
        "t_ters": 10 * 3600.0,
        "t_detrit": 20 * 3600.0,
        "t_pump": 150.0,
        "t_freeze": 1800.0,
        "t_exh": 5 * 3600.0,
    }
    sig2 = {
        "A": 0.15,
        "r_learn": 0.5,
        "I_mbb": 2.0,
        "I_miv": 0.1,
        "I_tfv_min": 1.0,
        "I_tfv_max": 1.5,
        "TBR": 0.03,
        "eta_bb": 0.04,
        "eta_f": 0.2,
        "eta_fuel_pump": 0.3,
        "eta_iv": 0.0004,
        "eta_tfv": 0.0004,
        "f_b": 0.01,
        "f_detrit_split": 0.0004,
        "f_dir": 0.15,
        "f_exh_split": 0.009,
        "f_terscwps": 0.00004,
        "t_ters": 5 * 3600.0,
        "t_detrit": 5 * 3600.0,
        "t_pump": 100.0,
        "t_freeze": 900.0,
        "t_exh": 2 * 3600.0,
    }
    ranges = {
        k: list(np.linspace(default[k] - sig2[k], default[k] + sig2[k], 11))
        for k in default.keys()
    }
    for i, k in enumerate(default.keys()):
        rang = {k: [v] for k, v in default.items()}
        rang[k] = ranges[k]
        database = TFVDataBase(name="TFV_scan" + "_" + k)
        database.add_ranges(rang)
        database.run()
    del database


def twinscan(k1, k2, s1, e1, s2, e2, n=11):
    default = {
        "A": 0.3,
        "r_learn": 1.0,
        "I_mbb": 3.0,
        "I_miv": 0.3,
        "I_tfv_min": 3.0,
        "I_tfv_max": 5.0,
        "TBR": 1.05,
        "eta_bb": 0.995,
        "eta_f": 0.7,
        "eta_fuel_pump": 0.6,
        "eta_iv": 0.9995,
        "eta_tfv": 0.9995,
        "f_b": 0.015,
        "f_detrit_split": 0.9995,
        "f_dir": 0.8,
        "f_exh_split": 0.99,
        "f_terscwps": 0.99995,
        "t_ters": 10 * 3600.0,
        "t_detrit": 20 * 3600.0,
        "t_pump": 150.0,
        "t_freeze": 1800.0,
        "t_exh": 5 * 3600.0,
    }

    ranges = {k: [v] for k, v in default.items()}
    database = TFVDataBase(name="TFV_scan" + "_" + k1 + "_" + k2, ncpu=6)
    for k, s, e in zip([k1, k2], [s1, s2], [e1, e2]):
        ranges[k] = np.linspace(s, e, n)
    database.add_ranges(ranges)
    database.run()


def load_tfvdb(k1, k2, method="max"):
    database = TFVDataBase(name="TFV_scan_" + k1 + "_" + k2)
    database.load_into_df()
    database.get_mmm_dfs()
    if method == "max":
        df = database.df_max
    elif method == "mean":
        df = database.df_mean
    elif method == "95th":
        df = database.df_95
    elif method == "median":
        df = database.df_median
    else:
        raise ValueError
    return df


def get_2dcontour(_x, _y, _z1, _z2, x, y, m1, m2):
    n = 1000
    ix, iy = np.linspace(x[0], x[-1], n), np.linspace(y[0], y[-1], n)
    xi, yi = np.meshgrid(ix, iy)
    zzg = griddata(_x, _y, _z1, xi, yi, interp="linear").T
    zzzg = griddata(_x, _y, _z2, xi, yi, interp="linear").T
    mzzz = zzzg.copy()
    mzz = zzg.copy()
    mzzz[(np.where(zzzg > m2) or np.where(zzg > m1))] = np.inf
    mzz[(np.where(zzzg > m2) or np.where(zzg > m1))] = np.inf
    mzzz[(np.where(zzzg > m2) and np.where(zzg > m1))] = np.inf
    mzz[(np.where(zzzg > m2) and np.where(zzg > m1))] = np.inf
    line = []
    for i, row in enumerate(mzzz):
        try:
            idx = iy[int(np.where(row < np.inf)[0][0])]
            if idx == iy[0]:  # End contour
                pass
            else:
                line.append((ix[int(i)], idx))
        except IndexError:  # No contour
            pass
    return np.array(line)[::-1]


def get_def_point(k1, k2):
    default = {
        "A": 0.3,
        "r_learn": 1.0,
        "I_mbb": 3.0,
        "I_miv": 0.3,
        "I_tfv_min": 3.0,
        "I_tfv_max": 5.0,
        "TBR": 1.05,
        "eta_bb": 0.995,
        "eta_f": 0.7,
        "eta_fuel_pump": 0.6,
        "eta_iv": 0.9995,
        "eta_tfv": 0.9995,
        "f_b": 0.015,
        "f_detrit_split": 0.9995,
        "f_dir": 0.8,
        "f_exh_split": 0.99,
        "f_terscwps": 0.99995,
        "t_ters": 10 * 3600.0,
        "t_detrit": 20 * 3600.0,
        "t_pump": 150.0,
        "t_freeze": 1800.0,
        "t_exh": 5 * 3600.0,
    }
    return default[k1], default[k2]


def plot_twinscan_old(df, k1, k2, nlevels=20, mask=False, tdmax=20, mmax=8):
    k1save, k2save = k1, k2
    result = "m_T_start"
    x = df[k1].values
    y = df[k2].values
    z = df[result].values
    a = df["t_d"].values
    _x, _y, _z, _td = x, y, z, a
    x = np.unique(x)
    y = np.unique(y)
    zz = np.zeros([len(y), len(x)])
    zzz = np.zeros([len(y), len(x)])
    count = 0
    for i, xv in enumerate(x):
        for j, yv in enumerate(y):
            zz[j, i] = z[i + j + count]
            zzz[j, i] = a[i + j + count]
        count += len(y) - 1
    xx, yy = np.meshgrid(x, y)
    f, ax = plt.subplots(figsize=[10, 8])
    if k1 == "A":
        k1 = "A_glob"
    elif k2 == "A":
        k2 = "A_glob"
    if k1 == "TBR":
        k1 = "{\\Lambda}"
    if k2 == "TBR":
        k2 = "{\\Lambda}"
    if k1 == "f_dir":
        k1 = "f_DIR"
    if k2 == "f_dir":
        k2 = "f_DIR"
    if k1 == "f_b":  # %

        def tick_format(value, n):
            return value * 100

        ax.xaxis.set_major_formatter(plt.FuncFormatter(tick_format))
        ax.set_xlabel(mathify(k1) + " [%]")
    else:
        ax.set_xlabel(mathify(k1))
    if k2 == "f_DIR":
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 0.95])

        def tick_format(value, n):
            if value == 0:
                return "0  "
            else:
                return value

        ax.yaxis.set_major_formatter(plt.FuncFormatter(tick_format))
        ax.set_ylabel(mathify(k2))
    elif k2 == "f_b":

        def tick_format(value, n):
            return value * 100

        ax.yaxis.set_major_formatter(plt.FuncFormatter(tick_format))
        ax.set_ylabel(mathify(k2) + " [%]")
    else:
        ax.set_ylabel(mathify(k2))
    sm = ax.contourf(xx, yy, zzz, nlevels, cmap="viridis")
    sm2 = ax.contour(xx, yy, zz, nlevels, cmap="plasma_r")
    sm.set_clim([np.nanmin(zzz), np.nanmax(zzz[zzz != np.inf])])
    # sm2.set_clim([np.nanmin(zz), np.nanmax(zzz[zz!=np.inf])])
    # ax.plot(xx, yy, 's', color='crimson', marker='s', ms=6)
    ax_cb1 = f.add_axes([0.125, 0.9, 0.775, 0.04])
    ax_cb2 = f.add_axes([0.92, 0.1075, 0.04, 0.771])
    cb = f.colorbar(sm, cax=ax_cb1, orientation="horizontal")
    cb.ax.xaxis.set_label_position("top")
    cb.ax.xaxis.set_ticks_position("top")
    cb2 = f.colorbar(sm2, cax=ax_cb2)
    cb2.set_label(mathify("m_T_start") + " [kg]")
    cb.set_label(mathify("t_d") + " [yr]")
    tick_locator = ticker.MaxNLocator(nbins=nlevels)
    cb.locator = tick_locator
    cb2.locator = tick_locator
    cb.ax.xaxis.set_major_locator(ticker.AutoLocator())
    cb2.ax.yaxis.set_major_locator(ticker.AutoLocator())
    cb.update_ticks()
    cb2.update_ticks()
    if mask:
        line = get_2dcontour(_x, _y, _z, _td, x, y, mmax, tdmax)
        # line = np.concatenate((line[:70], line[105:]))
        weather_front(line, ax=ax, n=7, scale=True, ends=False)
        lin2 = get_2dcontour(_x, _y, _z, _td, x, y, 5, 15)
        weather_front(lin2, ax=ax, n=7, scale=True, ends=False, color="b")
    x, y = get_def_point(k1save, k2save)
    ax.plot(x, y, "o", color="k")
    savefig(f, "_".join([k1save, k2save]), save=globals()["KEY_TO_PLOT"], dpi=600)


def plot_twinscan(
    df,
    k1,
    k2,
    f1="t_d",
    f2="m_T_start",
    nlevels=20,
    mask=False,
    f1max=20,
    f2max=8,
    double=True,
    figname="",
):
    k1save, k2save = k1, k2
    x = df[k1].values
    y = df[k2].values
    z = df[f2].values
    a = df[f1].values
    _x, _y, _z, _td = x, y, z, a
    x = np.unique(x)
    y = np.unique(y)
    zz = np.zeros([len(y), len(x)])
    zzz = np.zeros([len(y), len(x)])
    count = 0
    for i, xv in enumerate(x):
        for j, yv in enumerate(y):
            zz[j, i] = z[i + j + count]
            zzz[j, i] = a[i + j + count]
        count += len(y) - 1
    xx, yy = np.meshgrid(x, y)
    f, ax = plt.subplots(figsize=[10, 8])
    if k1 == "A":
        k1 = "A_glob"
    elif k2 == "A":
        k2 = "A_glob"
    if k1 == "TBR":
        k1 = "{\\Lambda}"
    if k2 == "TBR":
        k2 = "{\\Lambda}"
    if k1 == "f_dir":
        k1 = "f_DIR"
    if k2 == "f_dir":
        k2 = "f_DIR"
    if k1 == "f_b":  # %

        def tick_format(value, n):
            return np.round(value * 100, 2)

        ax.xaxis.set_major_formatter(plt.FuncFormatter(tick_format))
        ax.set_xlabel(mathify(k1) + " [%]")
    else:
        ax.set_xlabel(mathify(k1))
    if k2 == "f_DIR":
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 0.95])

        def tick_format(value, n):
            if value == 0:
                return "0  "
            else:
                return value

        ax.yaxis.set_major_formatter(plt.FuncFormatter(tick_format))
        ax.set_ylabel(mathify(k2))
    elif k2 == "f_b":

        def tick_format(value, n):
            return np.round(value * 100, 2)

        ax.yaxis.set_major_formatter(plt.FuncFormatter(tick_format))
        ax.set_ylabel(mathify(k2) + " [%]")
    else:
        ax.set_ylabel(mathify(k2))
    sm = ax.contourf(xx, yy, zzz, nlevels, cmap="viridis")
    sm.set_clim([np.nanmin(zzz), np.nanmax(zzz[zzz != np.inf])])

    # Colorbar formatting
    tick_locator = ticker.MaxNLocator(nbins=nlevels)
    ax_cb1 = f.add_axes([0.92, 0.515, 0.04, 0.365])
    cb = f.colorbar(sm, cax=ax_cb1)
    cb.set_label(mathify(f1) + " [yr]")
    cb.locator = tick_locator
    cb.ax.yaxis.set_major_locator(ticker.AutoLocator())
    cb.update_ticks()
    if double:
        sm2 = ax.contour(xx, yy, zz, nlevels, cmap="plasma_r")
        ax_cb2 = f.add_axes([0.92, 0.11, 0.04, 0.365])
        cb2 = f.colorbar(sm2, cax=ax_cb2)
        cb2.set_label(mathify(f2) + " [kg]")
        cb2.locator = tick_locator
        cb2.ax.yaxis.set_major_locator(ticker.AutoLocator())
        cb2.update_ticks()
    if mask:
        line = get_2dcontour(_x, _y, _z, _td, x, y, f2max, f1max)
        weather_front(line, ax=ax, n=7, scale=True, ends=False, color="k")
        lin2 = get_2dcontour(_x, _y, _z, _td, x, y, 5, 15)
        weather_front(lin2, ax=ax, n=7, scale=True, ends=False, color="w")
    x, y = get_def_point(k1save, k2save)
    ax.plot(x, y, "o", color="r")
    savefig(
        f,
        "_".join([k1save, k2save, figname]),
        save=globals()["KEY_TO_PLOT"],
        folder=globals()["PLOTFOLDER"],
        dpi=600,
    )


def keyscan(key, start, stop):
    default = {
        "A": 0.3,
        "r_learn": 1.0,
        "I_mbb": 3.0,
        "I_miv": 0.3,
        "I_tfv_min": 3.0,
        "I_tfv_max": 5.0,
        "TBR": 1.05,
        "eta_bb": 0.995,
        "eta_f": 0.7,
        "eta_fuel_pump": 0.6,
        "eta_iv": 0.9995,
        "eta_tfv": 0.9995,
        "f_b": 0.01,
        "f_detrit_split": 0.9995,
        "f_dir": 0.8,
        "f_exh_split": 0.99,
        "f_terscwps": 0.99995,
        "t_ters": 10 * 3600.0,
        "t_detrit": 20 * 3600.0,
        "t_pump": 150.0,
        "t_freeze": 1800.0,
        "t_exh": 5 * 3600.0,
    }
    rang = {k: [v] for k, v in default.items()}
    rang[key] = list(np.linspace(start, stop, 11))
    # rang['A'] = [0.3]
    # rang['r_learn'] = [1.0]
    database = TFVDataBase(name="TFV_scan_" + key)
    database.add_ranges(rang)
    database.run()


def _plot_pscan(ax, var, result, n):
    r = [max(i) for i in result]
    x = np.linspace(0, 2, n + 1)
    ax.plot(x, r / r[int(n / 2)], label=var)


def plot_p_scan(results):
    f, ax = plt.subplots(1, 2, sharey=True, figsize=[14, 7])
    for k, v in results.items():
        for a, s in zip(ax, v[:2]):
            _plot_pscan(a, k, s, 10)
    ax[0].set_title("$m_{T_{start}}$")
    ax[0].set_ylabel("Normalised value")
    ax[1].set_title("$t_{d}$")

    def tick_format(value, n):
        if value == 0:
            return r"$\mu-\sigma$"
        elif value == 2:
            return r"$\mu+\sigma$"
        elif value == 1:
            return r"$\mu$"
        else:
            return ""

    for a in ax:
        a.xaxis.set_major_formatter(plt.FuncFormatter(tick_format))
        a.legend()
    f.tight_layout()


def get_summary(default):
    summary = {}
    for k in default.keys():
        database = TFVDataBase(name="TFV_scan_" + k)
        database.load_into_df()
        database.get_mmm_dfs()
        summary[k] = database.df_max
    return summary


def plot_fullscan(summary, var="m_T_start"):
    f, ax = plt.subplots()
    super_summ = {}
    for k in summary.keys():
        values = summary[k][var]
        ref = values[5]
        if abs(min(values) / ref) >= 1.05 or abs(max(values) / ref) >= 1.05:
            super_summ[k] = values
    super_summ = dict(sorted(super_summ.items(), key=lambda x: -x[1][10] / x[1][5]))
    mc = cycle("o>^*h")
    for i, k in enumerate(super_summ.keys()):
        values = super_summ[k]
        ref = values[5]
        if k == "TBR":
            k = "\\Lambda"
        if k == "eta_f":
            k = "\\eta_f"
        if k == "eta_bb":
            k = "\\eta_{BB}"
        if k == "f_dir":
            k = "f_DIR"
        if k == "A":
            k = "A_{glob}"
        if var == "t_d" and i == 2:
            i *= 1.6
        o = 0.3 - i * 0.09
        x = np.linspace(0, 2, 11)
        m = next(mc)
        a = ax.plot(x, values / ref, ls=(0, (1, 1)), lw=3, ms=12, marker=m)
        ax.annotate(
            mathify(k),
            xy=[2, values[10] / ref],
            xytext=[2.2, values[10] / ref + o],
            arrowprops=dict(headwidth=0.5, width=0.5, facecolor="k", shrink=0.1),
            color=a[-1].get_color(),
        )
        if var == "t_d" and k == "\\Lambda":
            ax.arrow(
                x[2],
                values[2] / ref,
                0,
                3.1 - values[2] / ref - 0.03,
                lw=3,
                color="r",
                head_width=0.009,
                head_length=0.0175,
            )
            ax.annotate("$\\infty$", xy=[x[2] + 0.035, 3], color="r")

    def tick_format(value, n):
        if value == 0:
            return r"$\mu-2\sigma$"
        elif value == 2:
            return r"$\mu+2\sigma$"
        elif value == 1:
            return r"$\mu$"
        else:
            return ""

    ax.set_xlim([0, 2.44])
    if var == "m_T_start":
        ax.set_ylim([0.2, 2.6])
    else:
        ax.set_ylim([0, 3.1])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(tick_format))
    ax.set_ylabel(f"Normalised value of {mathify(var)}")
    savefig(f, "fullscan_" + var, save=globals().get("KEY_TO_PLOT"))
    # ax.legend()
    # f.tight_layout()


def build_TFVDB_v2():  # noqa (N802)
    database = TFVDataBase(name="TFV_v3")
    ranges = {
        "A": [0.15, 0.24, 0.3, 0.39, 0.45],
        "r_learn": [1.0],
        "I_mbb": [1.0, 3.0],
        "I_miv": [0.3],
        "I_tfv_min": [3.0],  # varied linearly later
        "I_tfv_max": [5.0],
        "TBR": [1.02, 1.03, 1.05, 1.07],
        "eta_bb": [0.991, 0.995, 0.999],
        "eta_f": [0.5, 0.7, 0.9],
        "eta_fuel_pump": [0.3, 0.6],
        "eta_iv": [0.9995],
        "eta_tfv": [0.9995],
        "f_b": np.linspace(0.005, 0.025, 5),
        "f_detrit_split": [0.9995],
        "f_dir": np.linspace(0.65, 0.95, 5),
        "f_exh_split": [0.99],
        "f_terscwps": [0.99995],
        "t_ters": [10 * 3600.0],
        "t_detrit": [20 * 3600.0],
        "t_pump": [150.0],
        "t_freeze": [900.0, 1800.0, 3600.0],
        "t_exh": [3 * 3600.0, 5 * 3600.0, 7 * 3600.0],
    }
    # runs = np.product([len(v) for v in ranges.values()])
    database.add_ranges(ranges)
    database.run()


def load_all_scans():
    # dfATBR = load_tfvdb('A', 'TBR')
    # Get rid of horrible outlier ruining twin_scan plot
    database = TFVDataBase(name="TFV_scan_A_TBR")
    database.load_into_df()
    for n in list(range(0, 5499, 500)):
        i = database.df.loc[22000:27499].iloc[n : n + 500]["m_T_start"].idxmax()
        database.df.drop(i, inplace=True)
    # cycle through groups
    # find maximum row
    # delete maximum row
    database.get_mmm_dfs()
    df_atbr = database.df_max

    database = TFVDataBase(name="TFV_scan_A_f_b")
    database.load_into_df()
    for n in list(range(0, 5499, 500)):
        i = database.df.loc[22000:27499].iloc[n : n + 500]["m_T_start"].idxmax()
        database.df.drop(i, inplace=True)
    # cycle through groups
    # find maximum row
    # delete maximum row
    database.get_mmm_dfs()
    df_afb = database.df_max

    df_afdir = load_tfvdb("A", "f_dir")

    dffbfdir = load_tfvdb("f_b", "f_dir")
    dftbrfdir = load_tfvdb("TBR", "f_dir")
    dftbrfb = load_tfvdb("TBR", "f_b")
    return dffbfdir, dftbrfb, dftbrfdir, df_atbr, df_afdir, df_afb


def run_all_plots(dffbfdir, dftbrfb, dftbrfdir, dfatbr, dfafdir, dfafb):
    plot_twinscan(dffbfdir, "f_b", "f_dir", nlevels=8, mask=True)
    plot_twinscan(dftbrfb, "TBR", "f_b", nlevels=8, mask=True)
    plot_twinscan(dftbrfdir, "TBR", "f_dir", nlevels=8, mask=True)
    plot_twinscan(dfatbr, "A", "TBR", nlevels=8, mask=True)
    plot_twinscan(dfafdir, "A", "f_dir", nlevels=8, mask=True)
    plot_twinscan(dfafb, "A", "f_b", nlevels=8, mask=True)


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
