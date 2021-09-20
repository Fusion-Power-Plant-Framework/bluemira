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
The BLUEPRINT-JETTO wrapper.
"""
import tarfile
import os
import shutil
import numpy as np

from enum import Enum, auto
from importlib import util as imp_u
from unittest.mock import patch

from jetto_tools.job import JobManager

from BLUEPRINT.equilibria.eqdsk import EQDSKInterface
from bluemira.base.look_and_feel import bluemira_warn, bluemira_print

####################
# Setup and Run currently a skeleton of what will possibly happen
# not completely working code
####################

# BLUEPRINT -> JETTO
# with JETTO short descriptions
mapping = {
    None: "q_min",  # minimum safety factor
    "q_95": "q_95",  # edge safety factor
    None: "n_GW",  # Greenwalf fraction
    None: "n_GW_95",  # Greenwald fraction @rho=0.95
    None: "s",  # magnetic shear
    None: None,  # Troyon limit
    None: None,  # resistive wall mode (no wall) limit
    "beta_N": "beta_N",  # Normalised Beta
    "l_i": "li3",  # Internal inductance   - UNSURE
    None: "dWdt",  # ...
    None: None,  # Fast particle pressure
    "kappa": "kappa",  # Elongaton
    "delta": "delta",  # Triangularity (absolute value)
    "B_0": "Btor",  # Toroidal field on axis
    "I_p": "Ip",  # Plasma current
    "A": "A",  # Aspect Ratio
    None: "H_98",  # confinement relative to scaling law
    None: "alpha",  # Normalised pedistal pressure
    None: "P_fus",  # fusion power
    None: "f_boot",  # bootstrap fraction
    None: "P_aux",  # Heating and CD power (coupled)
    None: "I_aux",  # Current drive
    None: "eta_CD",  # Current drive efficiency
    None: "Q_fus",  # fusion gain
    None: "Q_eng",  # Net energy gain
    None: "P_sep",  # Power crossing separatrix
    None: "f_rad",  # Radiation fraction
    None: "Q_max",  # target heat load
    None: None,  # Target incident angle
    None: None,  # Flux expansion
    None: "Q_max_fw",  # Maximum heat flux to the first wall
    None: "eta_pll",  # Transient energy fluence to the divertor
}


class RunMode(Enum):
    PROMINENCE = auto()
    BATCH = auto()
    MOCK = auto()

    def __call__(self, obj, *args, **kwargs):
        """
        Call function of object with lowercase name of
        enum

        Parameters
        ----------
        obj: instance
            instance of class the function will come from
        *args
           args of function
        **kwargs
           kwargs of function

        Returns
        -------
        function result
        """
        func = getattr(obj, f"_{self.name.lower()}")
        return func(*args, **kwargs)


class Setup:
    """

    TODO

    This is how a jetto run is set up, taken from quickstart:

    template = jetto_tools.template.from_directory(path)
    config = jetto_tools.config.RunConfig(template)

    then config is submitted in our Run class

    Things to modify in Setup:

    jetto.ex file  - experimental data useful for speeding up jetto run
                     (better guesses)
    jetto.bnd file - full boundary contour


    using:

    jetto_tools.binary.read_binary_file()
    jetto_tools.binary.write_binary_exfile()

    bnd file saving is not in jettotools yet see jettotools#32


    Parameters
    ----------
    runmode: str
        The running method for jetto see BLUEPRINT.codes.jetto.RunMode for possible values
    save_path: str
        path to save jeto input and output data
    """

    def __init__(self, runmode, save_path="data/BLUEPRINT"):
        self.set_runmode(runmode)
        self.save_path = save_path

        self.runmode(self)

    def process_input(self):
        raise NotImplementedError

    def set_runmode(self, runmode):
        self.runmode = RunMode[runmode]

    def _prominence(self):
        raise NotImplementedError

        from prominence import ProminenceClient

        self.client = ProminenceClient()
        self.client.authenticate_user()

    def _batch(self):
        pass

    def _mock(self):
        pass

    def boundary_contour_to_bnd_file(
        self, bndry_cntr, conversion=100, *, bnd_spec_version=3, time=10
    ):
        """
        Write boundary contour to bnd file

        Parameters
        ----------
        bndry_cntr: np.array(2, N)
            boundary contour array in x, z
        conversion: float
            factor to convert contour to cm
        bnd_spec_version: int
            bnd file specification version
        time: float
            for multiple contours (time dependent), arbitrary for 1 contour


        Notes
        -----
        This function only deals with a single contour therefore 'time' as
        implemented is always arbitrary

        The bnd file format is

        <bnd_spec_version> <no_points> <conversion>
        <time>, <no_points>
          R_boundary
        <data len=no_points>
          Z_boundary
        <data len=no_points>
          end

        """
        no_points = bndry_cntr.shape[1]
        header = (
            f"{bnd_spec_version} {no_points} {conversion}\n{time:.5f}, {no_points}\n"
        )
        sections = ["  R_boundary", "  Z_boundary", "  end"]
        with open(os.sep.join([self.save_path, "jetto.bnd"]), "w") as bnd:
            bnd.write(header)
            for section, axis in zip(sections, bndry_cntr):
                np.savetxt(bnd, axis, fmt="%.15e", header=section, comments="")
            bnd.write(sections[-1])


class Run:
    def __init__(self, setup):
        self.jobmananger = JobManager()
        self.result = setup.runmode(self, *setup.args)

    def _prominence(self, config, run_dir):
        self.jobid = self.jobmananger.submit_job_to_prominence(config, run_dir)

    def _batch(self, config, run_dir, run=False):
        self.jobid = self.jobmananger.submit_job_to_batch(config, run_dir, run)

    def _mock(self, *args):
        pass


##########################


class Teardown:
    """
    Teardown a JETTO run for use in the wider BLUEPRINT

    TODO this could probably be coded cleaner/ refactored
    once I have a testable setup

    Parameters
    ----------
    filenames: dict
       dictionary of filenames to process

    """

    files = {"eqdsk": "jetto.eqdsk_out"}

    def __init__(self, filenames=None):

        # Process any file from JETTO run - maybe overkill
        if filenames is not None:
            # Python 3.9
            # self.filnames |= filenames
            self.files = {**self.files, **filenames}

    def process(self, setup, runner):
        """
        Process a JETTO run

        also saves a run to the save_path specified in setup

        Parameters
        ----------
        setup: Setup
            Setup instance
        runner: Run
            Run instance

        Returns
        -------
        dict
          dictionary of JETTO data (currently only eqdsk)

        """
        self._setup = setup
        self._runner = runner
        self.save_path = self._setup.save_path

        self.processed_result = self._setup.runmode(self)

        return self.processed_result

    def _prominence(self, preprocessed=False):
        """
        Download and process a prominence jetto run

        Returns
        -------
        dict
            JETTO data
        """
        if not preprocessed:

            self.jobid = self._runner.jobid

            self._check_prominence_progress()
            self._get_prominence_output()
            # TODO test
            # Might not be needed don't know whether the files
            # are written directly or to a tarfile
            # self.extract_tar_contents()

        return self.get_data()

    def _batch(self):
        """
        Check a job has finished, download and process

        TODO the function
        """
        raise NotImplementedError

        return self.get_data()

    def _mock(self):
        """
        Get data from already dowloaded Run

        Returns
        -------
        dict
            JETTO data

        """
        return self.get_data()

    def _check_prominence_progress(self):
        """
        Check if job has finished

        TODO actually run a check

        The same as 'prominence list --completed' but in python
        """
        job_running = True
        while job_running:
            self._setup.client.list_jobs(completed=True, workflow_id=self.jobid)

            job_running = False
            # check if finished in while loop with a sleep time

    def _get_prominence_output(self):
        """
        Download files using jobid

        The same as 'promience download' but directly in python
        instead of proxying through bash

        """
        if not hasattr(self, "_downloader"):
            self._downloader = ProminenceDownloader(
                self.jobid, self.save_path, force=False
            )

        self._downloader()

    def extract_tar_contents(self):
        """
        Extract files specified in the filenames dictionary
        to the specified save_path

        By default only extracts jetto.eqdsk_out
        """
        with tarfile.open(self.tarfilename, "r") as tar:

            files = []

            content_names = tar.getnames()
            content = tar.getmembers()

            for file in self.files:
                try:
                    files += [content[content_names.index(file)]]
                except ValueError:
                    bluemira_warn(f"{file} not found in JETTO output")

            tar.extractall(path=self.save_path, members=files)

    def get_data(self):
        """
        Get all required data from JETTO run

        TODO return more than just eqdsk contents

        Returns
        -------
        dict
            dictionary of eqdsk contents

        """
        return self._get_eqdsk_data()

    def _get_eqdsk_data(self):
        """
        Extract data from JETTO eqdsk_out

        TODO be more prescriptive about output instead of just dumping it all

        Returns
        -------
        dict
            dictionary of eqdsk contents

        """
        return EQDSKInterface().read(os.sep.join(self.save_path, self.files["eqdsk"]))


class ProminenceDownloader:
    """
    Object to import prominence's binary file to enable job data downloading.

    the binary file has a lot of python code directly in it that is
    not accessible from the prominence module.

    Parameters
    ----------
    jobid: int
      prominence jobid
    save_dir: str
        directory to save the jetto output
    force: bool
        overwrite existing files

    """

    def __init__(self, jobid, save_dir, force=False):
        self.id = jobid
        self.dir = False
        self.force = force

        self._save_dir = save_dir
        self._old_open = open

        self.prom_bin = self._import_binary()

    @staticmethod
    def _import_binary():
        """
        Import prominence binary file as a module in order to use download functionality

        Returns
        -------
        module
           prominence binary as a module

        """
        prom_bin_loc = shutil.which("prominence")
        spec = imp_u.spec_from_file_location(prom_bin_loc, "prominence_bin")
        try:
            prom_bin = imp_u.module_from_spec(spec)
        except AttributeError:
            raise ImportError("Prominence binary not found")
        spec.loader.exec_module(prom_bin)

        return prom_bin

    def __call__(self):
        """
        Download the data from a run.

        Temporarily changes directory so that saving happens where desired
        and not in working directory.

        """
        with patch("builtins.print", new=self.captured_print):
            with patch("builtins.open", new=self.captured_open):
                self.prom_bin.command_download(self)

    def captured_print(self, string, *args, **kwargs):
        """
        Capture prominence print statements to feed them into our logging system

        Parameters
        ----------
        string: str
            string to print`
        *args
           builtin print statement args
        *kwargs
           builtin print statement kwargs

        """
        import sys

        sys.stdout.write("HERE")
        if string.startswith("Error"):
            bluemira_warn(f"Prominence {string}")
        else:
            bluemira_print(string)

    def captured_open(self, filepath, *args, **kwargs):
        """
        Prepend save directory to filepath

        Parameters
        ----------
        filepath: str
            filepath
        *args
           builtin open statement args
        *kwargs
           builtin open statement kwargs

        Returns
        -------
        filehandle

        """
        filepath = os.path.join(self._save_dir, filepath)
        return self._old_open(filepath, *args, **kwargs)


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
