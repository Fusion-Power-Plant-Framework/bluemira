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
Database and DOE tools wrapping pypet (HDF5 files)
"""
import os  # To allow file paths working under Windows and Linux
import numpy as np
import logging
from pandas import DataFrame
import pandas as pd
from pypet import Environment
from pypet.utils.explore import cartesian_product
from pypet.utils.hdf5compression import compact_hdf5_file
from pypet.trajectory import Trajectory
from pypet import pypetconstants
from BLUEPRINT.base.lookandfeel import bprint


logger = logging.getLogger()
logger.setLevel("INFO")

I_GLOBAL = 0


class DataBase:
    """
    DataBase object to build design studies using the pypet package. Compiles
    the results in an HDF5 file.

    Parameters
    ----------
    directory: str
        The directory in which to load/build the DataBase
    name: str
        The name of the DataBase (also the filename - no extension)
    function: callable
        The evalution function for the DataBase
    variables: List[Parameter]
        The variables to use in the exploration
    results

    Other Parameters
    ----------------
    ncpu: int
        The number of CPUs to use when building the DataBase
    """

    __loaded__ = False

    def __init__(self, directory, name, function, variables, results, **kwargs):
        self.name = name
        self.ncpu = kwargs.get("ncpu", os.cpu_count() - 2)
        if self.ncpu == 1:
            self._multiproc = False
        else:
            self._multiproc = True

        self.filename = os.path.join(directory, f"{self.name}.hdf5")
        self.function = function
        self.variables = variables
        self.results = results

        self.df = None
        self.env = None
        self.traj = None
        self.lock = False
        self.env0 = None
        self.traj_old = None

        if os.path.exists(self.filename):
            self.load_existing()
            self.runs = DataFrame.from_dict(self.get_run_inputs(self.traj_old))
            self.initialise(name=self.name + "_temp")
        else:
            self.initialise(name=self.name)

    def initialise(self, name):
        """
        Set up the pypet Environment and Trajectory, with the DataBase Parameters.
        """
        self.env = self.set_env(name)
        self.traj = self.env.trajectory
        for p in self.variables:
            if "Parameter" in str(type(p)):
                self.traj.f_add_parameter(
                    p.var, float(p.value), comment=p.name + " " + p.unit
                )
            else:
                raise ValueError(f"Need a Parameter, not a {type(p)}")

    def set_env(self, name):
        """
        Set up the pypet Environment.
        """
        return Environment(
            trajectory=name,
            filename=self.filename,
            add_time=False,
            log_stdout=True,
            log_level=logging.INFO,
            overwrite_file=False,
            multiproc=self._multiproc,
            ncores=self.ncpu,
            large_overview_tables=False,
        )

    def add_ranges(self, parameter_dict):
        """
        Add ranges of variables to the DataBase. Builds a Cartesian product
        exploration.

        Parameters
        ----------
        parameter_dict: dict
            The dictionary of ranges for variables to add to the DataBase
        """
        exploration = cartesian_product(parameter_dict)
        new_runs = self.check_for_duplicates(exploration)
        if new_runs is None:
            bprint("Already fully explored this parameter set.")
            self.lock = True
        else:
            self.traj.f_explore(new_runs.to_dict(orient="list"))
            self.lock = False

    def check_for_duplicates(self, exploration):
        """
        Check the exploration for duplicates in the DataBase

        Parameters
        ----------
        exploration: dict
            The dictionary of parameter ranges to explore

        Returns
        -------
        exploration: dict
            The dictionary of unique parameter ranges to explore
        """
        exploration = self.pad_exploration(exploration)
        exploration = DataFrame.from_dict(exploration)
        if hasattr(self, "runs"):
            c = pd.merge(self.runs, exploration, how="outer", indicator=True)
            new = c[c["_merge"] == "right_only"]
            if len(new) == 0:
                return None
            else:
                bprint("Dropping duplicate parameter entries.")
                return new.drop("_merge", axis=1)
        else:
            return exploration

    def pad_exploration(self, exploration):
        """
        Adds default values to varied values so that check duplicates
        finds proper duplicates
        """
        defaults = self.get_run_inputs(self.traj)
        length = len(next(iter(exploration.items()))[-1])  # Length of expl
        for k, v in defaults.items():
            defaults[k] = v[0] * np.ones(length)
        return {**defaults, **exploration}  # Latter ** overwrites!

    def run(self):
        """
        Build the DataBase with the specified ranges of variables.
        """
        # self.env.disable_logging()
        if self.lock:
            bprint("Add new parameter explorations before running.")
            return None
        self.env.run(self.function)
        if self.traj_old:
            self.traj_old.f_merge(
                self.traj,
                backup_filename=False,
                delete_other_trajectory=True,
                move_data=True,
            )
        # self.env.disable_logging()

    def load(self):
        """
        Load up results from the existing DataBase file.
        """
        if self.traj is None:
            self.traj = Trajectory(self.name)
        self.traj_old.f_load(load_results=pypetconstants.LOAD_DATA, force=True)
        self.__loaded__ = True

    def load_existing(self):
        """
        Load up the existing DataBase from a file and populate the old Trajectory.
        """
        self.env0 = self.set_env(self.name)
        self.traj_old = self.env0.traj
        # Just load explored parameters
        self.traj_old.f_load(
            load_parameters=2,
            load_derived_parameters=0,
            load_results=0,
            load_other_data=0,
            force=True,
        )

    def build_results_summary(self):
        """
        Build a summary DataFrame of the results.
        """
        res = [list() for _ in range(len(self.results))]
        for i, run in enumerate(self.traj_old.f_get_run_names()):
            self.traj_old.f_set_crun(run)
            for k in range(len(self.results)):
                res[k].append(self.traj_old.crun[self.results[k]])
        self.traj_old.f_restore_default()  # Clears the run

        # Build dictionaries, then DataFrame
        new_res = {}
        for i, r in enumerate(self.results):
            new_res[r] = res[i]
        var_dict = self.get_run_inputs(self.traj_old)
        self.df = DataFrame.from_dict(dict(**var_dict, **new_res))

    def get_run_inputs(self, traj):
        """
        Get the inputs of a run.

        Parameters
        ----------
        traj: Trajectory
            The trajectory for which to get the inputs

        Returns
        -------
        var_dict: dict
            The dictionary of variables
        """
        var = [np.zeros(len(traj)) for _ in range(len(self.variables))]
        for i, run in enumerate(traj.f_get_run_names()):
            traj.f_set_crun(run)
            for j in range(len(var)):
                var[j][i] = traj[self.variables[j].var]
        var_dict = {}
        for j, v in enumerate(self.variables):
            var_dict[v.var] = var[j]
        return var_dict

    def get_result(self, input_set):
        """
        Loads an individual result dynamically
        """
        self.traj_old.f_load(load_results=1)  # Load results skeleton
        self.traj_old.v_auto_load = True  # Dynamic loading (noisy in Ipython)
        df = self.runs
        for k, v in input_set.items():
            df = df[np.isclose(df[k], v)]
        i = df.index[0]
        name = "run_" + str("0" * (8 - len(str(i)))) + str(i)
        result = self.traj_old.results.f_get(name)
        res_dict = {}
        for k in self.results:
            res_dict[k] = result[k]
        return res_dict

    def compact_file(self):
        """
        Compress the HDF5 file. Takes a long time.
        """
        # NOTE: You tried this once on a 3.7 GB file (with naive storage style)
        # It took several minutes and reduced the file to 3.7 GB...
        compact_hdf5_file(self.filename)

    def frame_data(self):
        """
        Build a total DataFrame of the results.

        Returns
        -------
        df: DataFrame
            The DataFrame of results
        """
        if not self.__loaded__:
            self.load()
        var = [i.var for i in self.variables]
        res = self.results
        row_list = []
        for run in self.traj_old.f_get_run_names():
            self.traj_old.f_set_crun(run)
            row = {}
            for i in var:
                row[i] = self.traj_old[i]
            for k in range(len(self.traj_old.results[0][res[0]])):
                newrow = row.copy()
                for j in res:
                    try:
                        newrow[j] = self.traj_old.results[run][j][k]
                    except AttributeError:  # Run failed
                        row[j] = None
                        newrow = row
                row_list.append(newrow)
            self.traj_old.f_restore_default()
        return DataFrame(row_list)


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
