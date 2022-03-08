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

import copy
import os
import pprint
from enum import auto
from pathlib import Path
from typing import Optional

import pandas
import power_balance.cli as ukaea_pbm
import power_balance.configs as ukaea_pbm_conf
import power_balance.core as ukaea_pbm_core
import power_balance.exceptions as ukaea_pbm_exc
import power_balance.parameters as ukaea_pbm_param
import power_balance.profiles as ukaea_pbm_prof
import toml

import bluemira.codes.interface as interface
from bluemira.base.file import get_bluemira_path
from bluemira.base.look_and_feel import bluemira_debug  # , bluemira_warn

# from bluemira.codes.error import CodesError
from bluemira.codes.ukaea_powerbalance.constants import BINARY, MODEL_NAME
from bluemira.codes.ukaea_powerbalance.constants import NAME as POWERBALANCE
from bluemira.codes.ukaea_powerbalance.mapping import mappings


class PowerBalanceSolutions:
    """
    Class for storing UKAEA Power Balance outputs
    """

    def __init__(self, output_directory):
        _hdf5_file = os.path.join(output_directory, "data", "session.hdf5")
        _hdf5_key = MODEL_NAME.lower()
        self._data_frame = pandas.read_hdf(_hdf5_file, _hdf5_key)
        self._metadata = pandas.HDFStore(_hdf5_file).get_storer(_hdf5_key).attrs

    @property
    def open_modelica_version(self):
        return self._metadata["om_version"]

    @property
    def power_balance_version(self):
        return self._metadata["pbm_version"]

    @property
    def datetime(self):
        return self._metadata["time"]

    @property
    def data(self):
        return self._data_frame


class Inputs(ukaea_pbm_param.PBMParameterSet):
    """
    Class for UKAEA Power Balance inputs
    """

    def __init__(self, run_dir, new_inputs=None):
        ukaea_pbm.new(run_dir)
        super().__init__(run_dir)
        if new_inputs:
            self.modify(new_inputs)

    def modify(self, new_inputs):
        """
        Modify the parameter values
        """
        for param, val in new_inputs.items():
            try:
                self.set_parameter(param, val)
            except ukaea_pbm_exc.UnidentifiedParameterError:
                continue

    def as_dict(self):
        """
        Returns the instance as a dictionary.
        """
        return copy.deepcopy(self._parameters)

    def __repr__(self):
        """
        Representation string of the PlasmodParameters.
        """
        return f"{self.__class__.__name__}({pprint.pformat(self._parameters)}" + "\n)"


class Outputs(PowerBalanceSolutions):
    """
    Dummy class for an IO manager
    """

    filepath = get_bluemira_path("codes/ukaea_powerbalance")
    def_outdir = Path(filepath, "default")

    def __init__(self, output_dir=None):
        super().__init__(output_dir or self.def_outdir)


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

    _default_config = toml.load(ukaea_pbm_conf.config_default)
    _default_param_dir = ukaea_pbm_param.DEFAULT_PARAM_DIR

    # TODO get inputs update inputs write inputs io_manager
    # file names
    # use _get_new_inputs, involve problem settings
    def __init__(self, parent, *args, problem_settings=None, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.default_plasma = toml.load(
            self._default_param_dir, self._default_config["simulation_options_file"]
        )
        self.default_simopts = toml.load(
            os.path.join(
                self._default_param_dir, self._default_config["simulation_options_file"]
            )
        )

        self._problem_settings = problem_settings if problem_settings is not None else {}
        self.input_file = "powerbalance_input.dat"
        self.output_file = "powerbalance_outputs.dat"
        self.io_manager = Inputs({**self._get_new_inputs(), **self._problem_settings})
        self._generate_profiles()

    def _get_new_inputs(self):
        """
        Get new key mappings from the ParameterFrame.
        """
        return {
            pl_key: self.params.get(bm_key)
            for pl_key, bm_key in self._send_mapping.items()
        }

    def _generate_hcd_profiles(self, profile_dir, plasma_tuple):
        try:
            _hcd_type = self.io_manager.get_parameter("currdrive_eff_model")
            _heat_power = self.io_manager.get_parameter("profiles.Heat.max_power")

            if _hcd_type not in [8, 10, 12]:
                raise AssertionError
        except (AssertionError, ukaea_pbm_exc.UnidentifiedParameterError):
            return

        if _hcd_type == 8:
            _gen_w_zero = ukaea_pbm_prof.gen_rfheat_profile
            _gen_w_var = ukaea_pbm_prof.gen_nbiheat_profile
        else:
            _gen_w_zero = ukaea_pbm_prof.gen_nbiheat_profile
            _gen_w_var = ukaea_pbm_prof.gen_rfheat_profile

        _isnum = isinstance(_heat_power, int)
        _isnum = _isnum or isinstance(_heat_power, float)

        _gen_w_zero(
            output_directory=profile_dir,
            max_power=0,
            stop_time=self.default_simopts["stopTime"],
            time_step=self.default_simopts["stepSize"],
            time_range=plasma_tuple,
        )

        _val_str = str(_heat_power).replace(".", "_")
        _val_str = _val_str.replace("-", "")
        _gen_w_var(
            output_directory=profile_dir,
            max_power=abs(_heat_power),
            stop_time=self.default_simopts["stopTime"],
            time_step=self.default_simopts["stepSize"],
            time_range=plasma_tuple,
        )

    def _generate_profiles(self):
        _profile_dir = os.mkdirs(os.path.join(self.parent.run_dir, "ukaea_pbm_profiles"))

        _plasma_tuple = (
            self.default_plasma["plasma_ramp_up_start"],
            self.default_plasma["plasma_flat_top_start"],
            self.default_plasma["plasma_flat_top_end"],
            self.default_plasma["plasma_ramp_down_end"],
        )

        ukaea_pbm_prof.generate_all(
            output_directory=_profile_dir,
            time_range=_plasma_tuple,
            stop_time=self.default_simopts["stopTime"],
            time_step=self.default_simopts["stepSize"],
        )

        _profiles = [
            ("profiles.TFCoil.max_current", "gen_tfcoil_current_profile"),
            ("profiles.ThermalPowerOut.max_power", "gen_thermalpowerout_profile"),
        ]

        for profile in _profiles:
            try:
                _constant = self.io_manager.get_parameter(profile[0])
                getattr(ukaea_pbm_prof, profile[1])(
                    output_directory=_profile_dir,
                    max_current=abs(_constant),
                    stop_time=self.default_simopts["stopTime"],
                    time_step=self.default_simopts["stepSize"],
                    time_range=_plasma_tuple,
                )
            except ukaea_pbm_exc.UnidentifiedParameterError:
                pass

        self._generate_hcd_profiles(_profile_dir, _plasma_tuple)

    def write_input(self):
        """
        Writes input files to run directory
        """
        _params_dir = os.path.join(self.parent.run_dir, "ukaea_pbm_parameters")
        self.io_manager.save_to_directory(_params_dir)

    def update_inputs(self):
        """
        Update input values
        """
        self.io_manager.modify({**self._get_new_inputs(), **self._problem_settings})

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
        _session = ukaea_pbm_core.PowerBalance(
            no_browser=True,
            parameter_directory=os.path.join(
                self.parent.run_dir, "ukaea_pbm_parameters"
            ),
            profiles_directory=os.path.join(self.parent.run_dir, "ukaea_pbm_profiles"),
        )
        _session.run_simulation(
            output_directory=os.path.join(self.parent.run_dir, "ukaea_pbm_outputs")
        )


class Teardown(interface.Teardown):
    """
    PowerBalance Teardown Task
    """

    def _run(self):
        """
        Run powerbalance teardown
        """
        self.io_manager = Outputs(self.parent.run_dir)

    def _mock(self):
        """
        Mock powerbalance teardown
        """
        self.io_manager = Outputs()

    def _read(self):
        """
        Read powerbalance teardown
        """
        self.io_manager = Outputs()


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
