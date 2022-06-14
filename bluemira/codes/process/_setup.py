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
Defines the setup task for running PROCESS.
"""
import os
from typing import Dict, Union

from bluemira.base.parameter import ParameterFrame
from bluemira.codes.error import CodesError
from bluemira.codes.interface import CodesSetup
from bluemira.codes.process.api import DEFAULT_INDAT, InDat, update_obsolete_vars
from bluemira.codes.process.constants import NAME as PROCESS_NAME
from bluemira.codes.process.mapping import (
    CurrentDriveEfficiencyModel,
    TFCoilConductorTechnology,
)


class Setup(CodesSetup):
    """
    Setup Task for running PROCESS.

    Parameters
    ----------
    params: ParameterFrame
        The bluemira parameters for this task.
    in_dat_path: str
        The path to where the IN.DAT file should be written.
    problem_settings: Dict[str, Union[float, str]]
        The PROCESS parameters that do not exist in Bluemira.
    template_in_dat_path: str
        The path to a template PROCESS IN.DAT file. By default this
        points to a sample one within the Bluemira repository.
    """

    MODELS = {
        "iefrf": CurrentDriveEfficiencyModel,
        "i_tf_sup": TFCoilConductorTechnology,
    }

    def __init__(
        self,
        params: ParameterFrame,
        in_dat_path: str,
        problem_settings: Dict[str, Union[float, str]] = None,
        template_in_dat_path: str = DEFAULT_INDAT,
    ):
        super().__init__(params, PROCESS_NAME)

        self.in_dat_path = in_dat_path
        self.problem_settings = problem_settings if problem_settings is not None else {}
        self.template_in_dat_path = template_in_dat_path

    def run(self):
        """
        Write the IN.DAT file and store in the main PROCESS folder.

        Bluemira params with :code:`param.mapping.send == True` will be
        written to IN.DAT.
        """
        self._write_in_dat(use_bp_inputs=True)

    def runinput(self):
        """
        Write the IN.DAT file and store in the main PROCESS folder.

        Bluemira outputs will not be written to the file.
        """
        self._write_in_dat(use_bp_inputs=False)

    def _write_in_dat(self, use_bp_inputs=True):
        """
        Write the IN.DAT file and stores in the main PROCESS folder.

        Parameters
        ----------
        use_bp_inputs: bool
            Option to use bluemira values as PROCESS inputs. Used to re-run PROCESS
            within a bluemira run. If False, runs PROCESS without modifying inputs.
            Default, True
        """
        # Load defaults in bluemira folder
        writer = _make_writer(self.template_in_dat_path)
        if writer.data == {}:
            raise CodesError(
                f"Unable to read template IN.DAT file '{self.template_in_dat_path}'."
            )

        if use_bp_inputs:
            inputs = self._get_new_inputs(remapper=update_obsolete_vars)
            for key, value in inputs.items():
                writer.add_parameter(key, value)
            for key, value in self.problem_settings.items():
                writer.add_parameter(key, value)

            self._validate_models(writer)

        writer.write_in_dat(output_filename=self.in_dat_path)

    def _validate_models(self, writer):
        """
        Loop through known models, find the PROCESS output value for the
        model, and convert the type to its corresponding Enum value.
        """
        for name, model_cls in self.MODELS.items():
            try:
                val = writer.data[name].get_value
            except KeyError:
                continue

            model = model_cls[val] if isinstance(val, str) else model_cls(val)
            writer.add_parameter(name, model.value)


def _make_writer(template_in_dat: str) -> InDat:
    if os.path.isfile(template_in_dat):
        # InDat autoloads IN.DAT without checking for existence
        return InDat(filename=template_in_dat)
    else:
        raise CodesError(f"Template IN.DAT '{template_in_dat}' is not a file.")
