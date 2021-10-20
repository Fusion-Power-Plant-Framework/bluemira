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
PROCESS run functions
"""

from enum import Enum, auto
import json
import os
import subprocess  # noqa (S404)
import string

from bluemira.codes.error import CodesError
from bluemira.base.look_and_feel import bluemira_warn, bluemira_print
from BLUEPRINT.systems.physicstoolbox import normalise_beta
from bluemira.codes.utilities import get_read_mapping, get_write_mapping
from bluemira.codes.process.api import (
    DEFAULT_INDAT,
    update_obsolete_vars,
)
from bluemira.codes.process.setup import PROCESSInputWriter
from bluemira.codes.process.teardown import BMFile
from bluemira.codes.process.constants import NAME as PROCESS


class RunMode(Enum):
    """
    Enum class to pass args and kwargs to the PROCESS functions corresponding to the
    chosen PROCESS runmode (Run, Runinput, Read, Readall, or Mock).
    """

    RUN = auto()
    RUNINPUT = auto()
    READ = auto()
    READALL = auto()
    MOCK = auto()

    def __call__(self, obj, *args, **kwargs):
        """
        Call function of object with lowercase name of enum

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


class Run:
    """
    PROCESS Run functions. Runs, loads or mocks PROCESS to generate the reactor's radial
    build as an input for the BLUEPRINT run.

    Parameters
    ----------
    reactor: Reactor class instance
        The instantiated reactor class for the run. The parameters for the run are stored
        in reactor.params; values with a mapping will be used by PROCESS. The run mode is
        in reactor.build_config.processmode.
    run_dir: str
        Path to the PROCESS run directory, where the main PROCESS executable is located
        and the input/output files will be written.
    template_indat: str
        Path to the template IN.DAT file to be used for the run.
        Default, the value specified by DEFAULT_INDAT.
    params_to_update: list
        A list of parameter names compatible with the ParameterFrame class.
        If provided, parameters included in this list will be modified to write their
        values to PROCESS inputs, while all others will be modified to not be written to
        the PROCESS inputs. By default, None.

    Notes
    -----
    - "run": Run PROCESS within a BLUEPRINT run to generate an radial build.
        Creates a new input file from a template IN.DAT modified with updated parameters
        from the BLUEPRINT run mapped with write=True. If params_to_update are provided
        then these will be modified to have write=True.
    - "runinput": Run PROCESS from an unmodified input file (IN.DAT), generating the
        radial build to use as the input to the BLUEPRINT run. Overrides the write
        mapping of all parameters to be False.
    - "read": Load the radial build from a previous PROCESS run (MFILE.DAT). Loads
        only the parameters mapped with read=True.
    - "readall": Load the radial build from a previous PROCESS run (MFILE.DAT). Loads
        all values with a BLUEPRINT mapping regardless of the mapping.read bool.
        Overrides the read mapping of all parameters to be True.
    - "mock": Run BLUEPRINT without running PROCESS, using the default radial build based
        on EU-DEMO. This option should not be used if PROCESS is installed, except for
        testing purposes.
    """

    def __init__(
        self,
        reactor,
        run_dir=None,
        template_indat=None,
        params_to_update=None,
    ):
        self.reactor = reactor
        self.param_list = self.reactor.params.get_parameter_list()

        if params_to_update is not None:
            self.params_to_update = params_to_update
        else:
            self.params_to_update = self.reactor.build_config.get(
                "params_to_update", None
            )

        if run_dir is not None:
            self.run_dir = run_dir
        else:
            self.run_dir = self.reactor.file_manager.generated_data_dirs["systems_code"]

        if template_indat is not None:
            self.template_indat = template_indat
        else:
            self.template_indat = self.reactor.build_config.get(
                "process_indat", DEFAULT_INDAT
            )

        self.parameter_mapping = get_read_mapping(
            self.reactor.params, PROCESS, read_all=True
        )
        self.read_mapping = get_read_mapping(self.reactor.params, PROCESS)
        self.write_mapping = get_write_mapping(self.reactor.params, PROCESS)
        self.set_runmode()
        self.output_files = [
            "OUT.DAT",
            "MFILE.DAT",
            "OPT.DAT",
            "SIG_TF.DAT",
        ]

        self.runmode(self)  # Run PROCESS in the given run mode

    def set_runmode(self):
        """
        Set PROCESS runmode according to the "process_mode" parameter in build_config.
        """
        mode = (
            self.reactor.build_config["process_mode"]
            .upper()
            .translate(str.maketrans("", "", string.whitespace))
        )
        self.runmode = RunMode[mode]

    def _run(self):
        self.run_PROCESS(use_bp_inputs=True)

    def _runinput(self):
        self.run_PROCESS(use_bp_inputs=False)

    def _read(self):
        self.get_PROCESS_run(
            path=self.reactor.file_manager.reference_data_dirs["systems_code"],
            read_all=False,
        )

    def _readall(self):
        self.get_PROCESS_run(
            path=self.reactor.file_manager.reference_data_dirs["systems_code"],
            read_all=True,
        )

    def _mock(self):
        self.mock_PROCESS_run()

    def run_PROCESS(self, use_bp_inputs=True):
        """
        Run the systems code to get an initial reactor solution (radial build).

        Parameters
        ----------
        use_bp_inputs: bool, optional
            Option to use BLUEPRINT values as PROCESS inputs. Used to re-run PROCESS
            within a BLUEPRINT run. If False, runs PROCESS without modifying inputs.
            Default, True
        """
        bluemira_print("Running PROCESS systems code")

        # Write the IN.DAT file and store in the main PROCESS folder
        # Note that if use_bp_inputs is True, BLUEPRINT outputs with
        # param.mapping.write == True will be written to IN.DAT.
        self.prepare_bp_inputs()
        self.write_indat(use_bp_inputs=use_bp_inputs)

        # Run PROCESS
        self._clear_PROCESS_output()
        self._run_subprocess()
        self._check_PROCESS_output()

        # Load PROCESS results into BLUEPRINT
        self._load_PROCESS(self.read_mfile(), read_all=not use_bp_inputs)

    def get_PROCESS_run(self, path, read_all=False):
        """
        Loads an existing PROCESS file (read-only). Not to be used when running PROCESS.
        """
        bluemira_print("Loading PROCESS systems code run.")

        # Load the PROCESS MFile & read selected output
        params_to_read = self.parameter_mapping if read_all else self.read_mapping
        self._load_PROCESS(BMFile(path, params_to_read), read_all)

        # Add DD fusion fraction
        self.reactor.add_parameter(
            "f_DD_fus",
            "Fraction of DD fusion",
            self.reactor.params.P_fus_DD / self.reactor.params.P_fus,
            "N/A",
            "At full power",
            "Derived",
        )

    def mock_PROCESS_run(self):
        """
        Mock PROCESS. To be used in tests and examples only!
        """
        bluemira_print("Mocking PROCESS systems code run")

        # Create mock PROCESS file.
        path = self.reactor.file_manager.reference_data_dirs["systems_code"]
        filename = os.sep.join([path, "mockPROCESS.json"])
        with open(filename, "r") as fh:
            process = json.load(fh)

        # Set mock PROCESS parameters.
        self.reactor.add_parameters(process, source="Input")
        self.reactor.add_parameter(
            "f_DD_fus",
            "Fraction of DD fusion",
            self.reactor.params.P_fus_DD / self.reactor.params.P_fus,
            "N/A",
            "At full power",
            "Derived",
        )

        beta_n = normalise_beta(
            self.reactor.params.beta,
            self.reactor.params.R_0 / self.reactor.params.A,
            self.reactor.params.B_0,
            self.reactor.params.I_p,
        )

        self.reactor.add_parameters({"beta_N": beta_n})
        self.reactor.calc_reaction_rates()
        PlasmaClass = self.reactor.get_subsystem_class("PL")
        self.reactor.PL = PlasmaClass(
            self.reactor.params, {}, self.reactor.build_config["plasma_mode"]
        )

    def _load_PROCESS(self, bm_file, read_all=False):
        """
        Loads a PROCESS output file (MFILE.DAT) and extract some or all its output data

        Parameters
        ----------
            bm_file: BMFile
                PROCESS output file (MFILE.DAT) to load
            read_all: bool, optional
                True - Read all PROCESS output mapped by BTOPVARS,
                False - reads only a subset of the PROCESS output.
                Default, False
        """
        self.reactor.__PROCESS__ = bm_file

        # Load all PROCESS vars mapped with a BLUEPRINT input
        var = self.parameter_mapping.values() if read_all else self.read_mapping.values()
        param = self.reactor.__PROCESS__.extract_outputs(var)
        self.reactor.add_parameters(dict(zip(var, param)), source=PROCESS)

    def prepare_bp_inputs(self, use_bp_inputs=True):
        """
        Update parameter mapping write values to True/False depending on use_bp_inputs.

        Parameters
        ----------
        use_bp_inputs: bool, optional
            Option to use BLUEPRINT values as PROCESS inputs. If True, sets the write
            value for params in the params_to_update list to True and sets all others to
            False. If True but no params_to_update list provided, makes no changes to
            write values. If False, sets all write values to False.
            Default, True
        """
        # Skip if True but no list provided
        if use_bp_inputs is True and self.params_to_update is None:
            return
        # Update write values to True or False
        for param in self.param_list:
            if param.mapping is not None and PROCESS in param.mapping:
                mapping = param.mapping[PROCESS]
                if mapping.name in self.parameter_mapping:
                    bp_name = self.parameter_mapping[mapping.name]
                    mapping.write = use_bp_inputs and bp_name in self.params_to_update

    def write_indat(self, use_bp_inputs=True):
        """
        Write the IN.DAT file and stores in the main PROCESS folder.

        Parameters
        ----------
        use_bp_inputs: bool, optional
            Option to use BLUEPRINT values as PROCESS inputs. Used to re-run PROCESS
            within a BLUEPRINT run. If False, runs PROCESS without modifying inputs.
            Default, True
        """
        # Load defaults in BLUEPRINT folder
        writer = PROCESSInputWriter(template_indat=self.template_indat)
        if writer.data == {}:
            raise CodesError(
                f"Unable to read template IN.DAT file at {self.template_indat}"
            )

        if use_bp_inputs is True:
            for param in self.param_list:  # Overwrite variables
                if param.mapping is not None and PROCESS in param.mapping:
                    mapping = param.mapping[PROCESS]
                    if mapping.write:
                        writer.add_parameter(
                            update_obsolete_vars(mapping.name), param.value
                        )

        filename = os.path.join(self.run_dir, "IN.DAT")
        writer.write_in_dat(output_filename=filename)

    def read_mfile(self):
        """
        Read the MFILE.DAT from the PROCESS run_dir.

        Returns
        -------
        mfile: BMFile
            The object representation of the output MFILE.DAT.
        """
        m_file = BMFile(self.run_dir, self.parameter_mapping)
        self._check_feasible_solution(m_file)
        return m_file

    def _clear_PROCESS_output(self):
        """
        Clear the output files from PROCESS run directory.
        """
        for filename in self.output_files:
            filepath = os.sep.join([self.run_dir, filename])
            if os.path.exists(filepath):
                os.remove(filepath)

    def _check_PROCESS_output(self):
        """
        Check that PROCESS has produced valid (non-zero lined) output.

        Raises
        ------
        CodesError
            If any resulting output files don't exist or are empty.
        """
        for filename in self.output_files:
            filepath = os.sep.join([self.run_dir, filename])
            if os.path.exists(filepath):
                with open(filepath) as fh:
                    if len(fh.readlines()) == 0:
                        message = (
                            f"PROCESS generated an empty {filename} "
                            f"file in {self.run_dir} - check PROCESS logs."
                        )
                        bluemira_warn(message)
                        raise CodesError(message)
            else:
                message = (
                    f"PROCESS run did not generate the {filename} "
                    f"file in {self.run_dir} - check PROCESS logs."
                )
                bluemira_warn(message)
                raise CodesError(message)

    @staticmethod
    def _check_feasible_solution(m_file):
        """
        Check that PROCESS found a feasible solution.

        Parameters
        ----------
        m_file: BMFile
            The PROCESS MFILE to check for a feasible solution

        Raises
        ------
        CodesError
            If a feasible solution was not found.
        """
        error_code = m_file.params["Numerics"]["ifail"]
        if error_code != 1:
            message = (
                f"PROCESS did not find a feasible solution. ifail = {error_code}."
                " Check PROCESS logs."
            )
            bluemira_warn(message)
            raise CodesError(message)

    def _run_subprocess(self):
        subprocess.run("process", cwd=self.run_dir)  # noqa (S603)


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
