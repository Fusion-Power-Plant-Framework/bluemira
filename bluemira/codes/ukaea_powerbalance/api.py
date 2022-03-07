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
API for the ukaea powerbalance model and related functions
"""

from enum import auto
from pathlib import Path
from typing import Optional

import bluemira.codes.interface as interface
from bluemira.base.look_and_feel import bluemira_debug  # , bluemira_warn

# from bluemira.codes.error import CodesError
from bluemira.codes.ukaea_powerbalance.constants import BINARY
from bluemira.codes.ukaea_powerbalance.constants import NAME as POWERBALANCE
from bluemira.codes.ukaea_powerbalance.mapping import mappings


class Inputs:
    """
    Dummy class for an IO manager
    """

    pass


class Outputs:
    """
    Dummy class for an IO manager
    """

    pass


class RunMode(interface.RunMode):
    """
    RunModes for powerbalance
    """

    RUN = auto()
    READ = auto()
    MOCK = auto()


class Setup(interface.Setup):
    """
    Setup class for PowerBalance

    Parameters
    ----------
    parent
        Parent solver class instance
    input_file: str
        input file save location
    output_file: str
        output file save location
    profiles_file: str
        profiles file save location
    kwargs: Dict
        passed to parent setup task

    """

    # TODO get inputs update inputs write inputs io_manager
    # file names
    # use _get_new_inputs, involve problem settings
    def __init__(self, parent, *args, problem_settings=None, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self._problem_settings = problem_settings if problem_settings is not None else {}
        self.input_file = "powerbalance_input.dat"
        self.output_file = "powerbalance_outputs.dat"
        self.io_manager = Inputs({**self._get_new_inputs(), **self._problem_settings})

    def _run(self):
        """
        Run powerbalance setup
        """
        self.update_inputs()
        self.write_input()


class Run(interface.Run):
    """
    Run class for powerbalance

    Parameters
    ----------
    parent
        Parent solver class instance
    kwargs: Dict
        passed to parent setup task

    """

    _binary = BINARY

    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, kwargs.pop("binary", self._binary), *args, **kwargs)

    def _run(self):
        """
        Run powerbalance runner
        """
        bluemira_debug("Mode: run")
        super()._run_subprocess(
            [
                self._binary,
                Path(self.parent.run_dir, self.parent.setup_obj.input_file),
                Path(self.parent.run_dir, self.parent.setup_obj.output_file),
            ]
        )


class Teardown(interface.Teardown):
    """
    PowerBalance Teardown Task
    """

    def _run(self):
        """
        Run powerbalance teardown
        """
        self.io_manager = Outputs()
        self.io_manager.read_output_files(
            Path(self.parent.run_dir, self.parent.setup_obj.output_file),
        )
        self.prepare_outputs()

    def _mock(self):
        """
        Mock powerbalance teardown
        """
        self.io_manager = Outputs(use_defaults=True)
        self.prepare_outputs()

    def _read(self):
        """
        Read powerbalance teardown
        """
        self.io_manager = Outputs()
        self.io_manager.read_output_files(
            Path(self.parent.read_dir, self.parent.setup_obj.output_file),
        )
        self.prepare_outputs()

    def prepare_outputs(self):
        """
        Prepare outputs for ParameterFrame
        """
        super().prepare_outputs(
            {
                bm_key: getattr(self.io_manager, pb_key)
                for pb_key, bm_key in self.parent._recv_mapping.items()
            },
            source=POWERBALANCE,
        )


class Solver(interface.FileProgramInterface):
    """
    PowerBalance solver class

    Parameters
    ----------
    params: ParameterFrame
        ParameterFrame for powerbalance
    build_config: Dict
        build configuration dictionary
    run_dir: str
        PowerBalance run directory
    read_dir: str
        Directory to read in previous run

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
        read_dir: Optional[str] = None,
    ):
        super().__init__(
            POWERBALANCE,
            params,
            build_config.get("mode", "run"),
            binary=build_config.get("binary", BINARY),
            run_dir=run_dir,
            read_dir=read_dir,
            mappings=mappings,
            problem_settings=build_config.get("problem_settings", None),
        )

    @property
    def problem_settings(self):
        """
        Get problem settings dictionary
        """
        return self.setup_obj._problem_settings
