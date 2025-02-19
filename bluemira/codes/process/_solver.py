# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import copy
from collections.abc import Mapping
from enum import auto
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame import ParameterFrame
from bluemira.codes.error import CodesError
from bluemira.codes.interface import (
    BaseRunMode,
    CodesSolver,
)
from bluemira.codes.process._inputs import ProcessInputs
from bluemira.codes.process._run import Run
from bluemira.codes.process._setup import Setup, create_template_from_path
from bluemira.codes.process._teardown import Teardown
from bluemira.codes.process.api import ENABLED, Impurities
from bluemira.codes.process.constants import BINARY as PROCESS_BINARY
from bluemira.codes.process.constants import NAME as PROCESS_NAME
from bluemira.codes.process.params import ProcessSolverParams
from bluemira.display.plotter import plot_coordinates
from bluemira.geometry.coordinates import Coordinates

BuildConfig = dict[str, Union[float, str, "BuildConfig"]]


class RunMode(BaseRunMode):
    """
    Run modes for the PROCESS solver.
    """

    RUN = auto()
    RUNINPUT = auto()
    READ = auto()
    READALL = auto()
    MOCK = auto()
    NONE = auto()


class Solver(CodesSolver):
    """
    PROCESS solver. Runs, loads or mocks PROCESS to generate a reactor's
    radial build.

    Parameters
    ----------
    params:
        ParameterFrame or dict containing parameters for running PROCESS.
        See :class:`~bluemira.codes.process.params.ProcessSolverParams` for
        parameter details.
    build_config:
        Dictionary containing the configuration for this solver.
        Expected keys are:

        * binary:
            The path to the PROCESS binary. The default assumes the
            PROCESS executable is on the system path.
        * run_dir:
            The directory in which to run PROCESS. It is also the
            directory in which to look for PROCESS input and output
            files. Default is current working directory.
        * read_dir:
            The directory from which data is read when running in read mode.
        * template_in_dat_path:
            The path to a template PROCESS IN.DAT file or and instances of
            :class:`~bluemira.codes.process._inputs.ProcessInputs`.
            By default this is an empty instance of the class. To create a new
            instance
            :class:`~bluemira.codes.process.template_builder.PROCESSTemplateBuilder`
            should be used.
        * problem_settings:
            Any PROCESS parameters that do not correspond to a bluemira
            parameter.
        * in_dat_path:
            The path to save the IN.DAT file that is run by PROCESS.
            By default this is '<run_dir>/IN.DAT'.

    Notes
    -----
    This solver has several run modes:

    * run: Run PROCESS to generate a radial build.
        Creates a new input file from the given template IN.DAT, which
        is modified with bluemira parameters that are mapped with
        :code:`send = True`.
    * runinput: Run PROCESS with an unmodified template IN.DAT.
        The template IN.DAT is not modified with bluemira parameters.
        This is equivalent to all bluemira parameters mappings having
        :code:`send = False`.
    * read: Load the radial build from a PROCESS MFILE.DAT.
        Loads only the parameters with :code:`send = True`.
        A file named 'MFILE.DAT' must exist within 'run_directory'.
    * readall: Load the radial build from a PROCESS MFILE.DAT.
        Loads all mappable parameters from the PROCESS file.
        A file named 'MFILE.DAT' must exist within 'run_directory'.
    * mock: Load bluemira parameters directly from a JSON file in the
        run directory. This does not run PROCESS.
    * none: Does nothing.
        PROCESS is not run and parameters are not updated. This is
        useful loading results form previous runs of bluemira, where
        overwriting data with PROCESS outputs would be undesirable.
    """

    name: str = PROCESS_NAME
    setup_cls: type[Setup] = Setup
    run_cls: type[Run] = Run
    teardown_cls: type[Teardown] = Teardown
    run_mode_cls: type[RunMode] = RunMode
    params_cls: type[ProcessSolverParams] = ProcessSolverParams

    def __init__(
        self,
        params: dict | ParameterFrame,
        build_config: Mapping[str, float | str | BuildConfig],
    ):
        # Init task objects on execution so parameters can be edited
        # between separate 'execute' calls.
        self._setup: Setup | None = None
        self._run: Run | None = None
        self._teardown: Teardown | None = None

        _build_config = copy.deepcopy(build_config)
        self.plot = _build_config.pop("plot", False)
        self.binary = _build_config.pop("binary", PROCESS_BINARY)
        self.run_directory = _build_config.pop("run_dir", Path.cwd().as_posix())
        self.read_directory = _build_config.pop("read_dir", Path.cwd().as_posix())
        self.template_in_dat = _build_config.pop("template_in_dat", ProcessInputs())
        self.custom_solver = _build_config.pop("custom_solver", None)
        self.problem_settings = _build_config.pop("problem_settings", {})
        self.in_dat_path = _build_config.pop(
            "in_dat_path", Path(self.run_directory, "IN.DAT").as_posix()
        )

        if isinstance(self.template_in_dat, str | Path):
            self.template_in_dat = create_template_from_path(self.template_in_dat)

        self.params = self.params_cls.from_defaults(self.template_in_dat)

        self.params.update(params)

        if len(_build_config) > 0:
            quoted_delim = "', '"
            bluemira_warn(
                f"'{self.name}' solver received unknown build_config arguments: "
                f"'{quoted_delim.join(_build_config.keys())}'."
            )

    def execute(self, run_mode: str | RunMode) -> ParameterFrame:
        """
        Execute the solver in the given run mode.

        Parameters
        ----------
        run_mode:
            The run mode to execute the solver in. See the
            :func:`~bluemira.codes.process._solver.Solver.__init__`
            docstring for details of the behaviour of each run mode.

        Returns
        -------
        :
            The modified parameters

        Raises
        ------
        CodesError
            install not found
        """
        if isinstance(run_mode, str):
            run_mode = self.run_mode_cls.from_string(run_mode)
        if not ENABLED and run_mode not in {
            self.run_mode_cls.MOCK,
            self.run_mode_cls.NONE,
        }:
            raise CodesError(f"{self.name} installation not found")

        self._setup = self.setup_cls(
            self.params,
            self.in_dat_path,
            self.problem_settings,
        )
        self._run = self.run_cls(
            self.params, self.in_dat_path, self.binary, self.custom_solver
        )
        self._teardown = self.teardown_cls(
            self.params, self.run_directory, self.read_directory
        )

        if setup := self._get_execution_method(self._setup, run_mode):
            setup()
        if run := self._get_execution_method(self._run, run_mode):
            run()
        if teardown := self._get_execution_method(self._teardown, run_mode):
            teardown()

        if self.plot:
            self.plot_radial_build()

        return self.params

    def plot_radial_build(
        self,
        width: float = 1.0,
        *,
        show: bool = True,
    ) -> plt.Axes | None:
        """
        Plot PROCESS radial build.

        Parameters
        ----------
        width:
            The relative width of the plot.
        show:
            If True then immediately display the plot,
            else delay displaying the plot until
            the user shows it, by default True.

        Returns
        -------
        The plot Axes object.
        """
        if not self._teardown:
            return None

        radial_build = self._teardown.get_radial_build_for_plotting()
        if not radial_build:
            bluemira_warn("MFILE.DAT file in old format. Cannot plot radial build.")
            return None

        R_0 = radial_build["R_0"]

        col = {
            "Gap": "w",
            "blanket": "#edb120",
            "TF coil": "#7e2f8e",
            "Vacuum vessel": "k",
            "Radiation shield": "#5dbb63",
            "Plasma": "#f77ec7",
            "first wall": "#edb120",
            "Machine bore": "w",
            "dr_cs_precomp": "#0072bd",
            "scrape-off": "#a2142f",
            "solenoid": "#0072bd",
            "Thermal shield": "#77ac30",
        }

        _, ax = plt.subplots(figsize=[14, 10])

        lpatches = []
        gkeys = [
            "blanket",
            "TF coil",
            "Vacuum vessel",
            "Radiation shield",
            "Plasma",
            "scrape-off",
            "solenoid",
            "Thermal shield",
        ]
        glabels = {
            "blanket": "Breeding blanket",
            "TF coil": "TF coil",
            "Plasma": "Plasma",
            "Vacuum vessel": "Vacuum vessel",
            "Radiation shield": "Radiation shield",
            "scrape-off": "Scrape-off layer",
            "solenoid": "Central solenoid",
            "Thermal shield": "Thermal shield",
        }

        for comp in radial_build["Radial Build"]:
            # Generate coordinates for an arbitrary
            # height radial width.
            xc = [
                comp[2] - comp[1],
                comp[2] - comp[1],
                comp[2],
                comp[2],
                comp[2] - comp[1],
            ]
            yc = [-width, width, width, -width, -width]
            yc = np.array(yc)
            coords = Coordinates({"x": xc, "y": yc})

            for key, c in col.items():
                if key.upper() in comp[0].upper():
                    ax.plot(xc, yc, color=c, linewidth=0, label=key)
                    if comp[1] > 0:
                        plot_coordinates(
                            coords, ax=ax, facecolor=c, edgecolor="k", linewidth=0
                        )
                    if key in gkeys:
                        gkeys.remove(key)
                        lpatches.append(patches.Patch(color=c, label=glabels[key]))

        ax.set_xlim([0, np.ceil(radial_build["Radial Build"][-1][-1])])
        ax.set_ylim([-width * 0.5, width * 0.5])
        ax.set_xticks([*list(ax.get_xticks()), R_0])
        ax.axes.set_axisbelow(b=False)

        def tick_format(value, n):  # noqa: ARG001
            if value == R_0:
                return "\n$R_{0}$"
            return int(value)

        def tick_formaty(value, n):  # noqa: ARG001
            if value == 0:
                return int(value)
            return ""

        ax.xaxis.set_major_formatter(plt.FuncFormatter(tick_format))
        ax.yaxis.set_major_formatter(plt.FuncFormatter(tick_formaty))
        ax.set_xlabel("$x$ [m]")
        ax.set_aspect("equal")
        ax.legend(
            handles=lpatches,
            ncol=3,
            loc="lower left",
            bbox_to_anchor=(0.0, 1.0),
            frameon=False,
        )

        if show:
            plt.show()
        return ax

    def get_raw_variables(self, params: list | str) -> list[float]:
        """
        Get raw variables from this solver's associate MFile.

        Mapped bluemira parameters will have bluemira names.

        Parameters
        ----------
        params:
            Names of parameters to access.

        Returns
        -------
        The parameter values.

        Raises
        ------
        CodesError
            Cannot read output before creation
        """
        if self._teardown:
            return self._teardown.get_raw_outputs(params)
        raise CodesError(
            "Cannot retrieve output from PROCESS MFile. "
            "The solver has not been run, so no MFile is available to read."
        )

    @staticmethod
    def get_species_data(
        impurity: str, confinement_time_ms: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get species data from PROCESS section of OPEN-ADAS database.

        The data is taken with density $n_e = 10^{19} m^{-3}$.

        Parameters
        ----------
        impurity:
            The impurity to get the species data for. This string should
            be one of the names in the
            :class:`~bluemira.codes.process.api.Impurities` Enum.
        confinement_time_ms:
            the confinement time to read the data for options are:
            [0.1, 1.0, 10.0, 100.0, 1000.0, np.inf]

        Returns
        -------
        tref:
            The temperature in eV.
        l_ref:
            The loss function value $L_z(n_e, T_e)$ in W.m3.
        z_ref:
            Average effective charge.
        """
        lz_ref, z_ref = Impurities[impurity].read_impurity_files(("lz", "z"))

        t_ref = filter(lambda lz: lz.content == "Te[eV]", lz_ref)
        lz_ref = filter(lambda lz: f"{confinement_time_ms:.1f}" in lz.content, lz_ref)
        z_av_ref = filter(lambda z: f"{confinement_time_ms:.1f}" in z.content, z_ref)
        return tuple(
            np.array(next(ref).data, dtype=float) for ref in (t_ref, lz_ref, z_av_ref)
        )

    def get_species_fraction(self, impurity: str) -> float:
        """
        Get species fraction for the given impurity.

        Parameters
        ----------
        impurity:
            The impurity to get the species data for. This string should
            be one of the names in the
            :class:`~bluemira.codes.process.api.Impurities` Enum.

        Returns
        -------
        The species fraction for the impurity taken from the PROCESS
        output MFile.
        """
        return self.get_raw_variables(Impurities[impurity].id())[0]
