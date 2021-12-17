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
The bluemira-JETTO wrapper.
"""
import tarfile
import os
import shutil
import numpy as np

from enum import Enum, auto
from importlib import util as imp_u

from jetto_tools.job import JobManager

from bluemira.equilibria.file import EQDSKInterface
from bluemira.base.look_and_feel import bluemira_warn
import bluemira.codes.interface as interface
from bluemira.codes.jetto.prominence import ProminenceDownloader
from bluemira.codes.jetto.constants import NAME as JETTO
from bluemira.codes.jetto.mapping import mappings


####################
# Setup and Run currently a skeleton of what will possibly happen
# not completely working code
####################


class RunMode(interface.RunMode):
    PROMINENCE = auto()
    BATCH = auto()
    MOCK = auto()


class Setup(interface.Setup):
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
    save_path: str
        path to save jeto input and output data
    """

    def __init__(self, save_path="data/bluemira", params=None):
        super().__init__(params=params)
        self.save_path = save_path

    def process_input(self):
        raise NotImplementedError

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


class Run(interface.Run):

    _binary = False

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, kwargs.pop("binary", self._binary), *args, **kwargs)
        self.jobmananger = JobManager()

    def _prominence(self, config, run_dir):
        self.jobid = self.jobmananger.submit_job_to_prominence(config, run_dir)

    def _batch(self, config, run_dir, run=False):
        self.jobid = self.jobmananger.submit_job_to_batch(config, run_dir, run)


class Teardown(interface.Teardown):
    """
    Teardown a JETTO run for use in the wider bluemira

    TODO this could probably be coded cleaner/ refactored
    once I have a testable setup

    Parameters
    ----------
    filenames: dict
       dictionary of filenames to process

    """

    files = {"eqdsk": "jetto.eqdsk_out"}

    def __init__(self, parent, *args, filenames=None, **kwargs):
        super().__init__(parent)
        # Process any file from JETTO run - maybe overkill
        if filenames is not None:
            # Python 3.9
            # self.filnames |= filenames
            self.files = {**self.files, **filenames}

    def process(self):
        """
        Process a JETTO run

        also saves a run to the save_path

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
        pass

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
            self.parent.setup_obj.client.list_jobs(
                completed=True, workflow_id=self.jobid
            )

            job_running = False
            # check if finished in while loop with a sleep time maybe in async?

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


class Solver(interface.FileProgramInterface):
    """
    JETTO solver class

    Parameters
    ----------
    params: ParameterFrame
        ParameterFrame for JETTO
    build_config: Dict
        build configuration dictionary
    run_dir: str
        Plasmod run directory

    Notes
    -----
    build config keys: mode, binary, problem_settings
    """

    _setup = Setup
    _run = Run
    _teardown = Teardown
    _runmode = RunMode

    def __init__(
        self,
        params,
        build_config=None,
        run_dir: Optional[str] = None,
    ):
        super().__init__(
            JETTO,
            params,
            build_config.get("mode", "run"),
            binary=build_config.get("binary", BINARY),
            run_dir=run_dir,
            mappings=mappings,
            problem_settings=build_config.get("problem_settings", None),
        )
