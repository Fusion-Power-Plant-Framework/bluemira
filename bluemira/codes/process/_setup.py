# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Defines the setup task for running PROCESS.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Dict, Optional, Union

from bluemira.codes.error import CodesError
from bluemira.codes.interface import CodesSetup
from bluemira.codes.process._inputs import ProcessInputs
from bluemira.codes.process.api import ENABLED, InDat, _INVariable, update_obsolete_vars
from bluemira.codes.process.constants import NAME as PROCESS_NAME
from bluemira.codes.process.model_mapping import (
    CurrentDriveEfficiencyModel,
    TFCoilConductorTechnology,
)

if TYPE_CHECKING:
    from bluemira.base.parameter_frame import ParameterFrame
    from bluemira.codes.process.params import ProcessSolverParams


class Setup(CodesSetup):
    """
    Setup Task for running PROCESS.

    Parameters
    ----------
    params:
        The bluemira parameters for this task.
    in_dat_path:
        The path to where the IN.DAT file should be written.
    problem_settings:
        The PROCESS parameters that do not exist in Bluemira.
    """

    MODELS: ClassVar = {
        "iefrf": CurrentDriveEfficiencyModel,
        "i_tf_sup": TFCoilConductorTechnology,
    }

    params: ProcessSolverParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
        in_dat_path: str,
        problem_settings: Optional[Dict[str, Union[float, str]]] = None,
    ):
        super().__init__(params, PROCESS_NAME)

        self.in_dat_path = in_dat_path
        self.problem_settings = problem_settings if problem_settings is not None else {}

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

    def _write_in_dat(self, use_bp_inputs: bool = True):
        """
        Write the IN.DAT file and stores in the main PROCESS folder.

        Parameters
        ----------
        use_bp_inputs:
            Option to use bluemira values as PROCESS inputs. Used to re-run PROCESS
            within a bluemira run. If False, runs PROCESS without modifying inputs.
            Default, True
        """
        # Load defaults in bluemira folder
        writer = _make_writer(self.params.template_defaults)

        if use_bp_inputs:
            inputs = self._get_new_inputs(remapper=update_obsolete_vars)
            for key, value in inputs.items():
                if value is not None:
                    writer.add_parameter(key, value)
            for key, value in self.problem_settings.items():
                if value is not None:
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


def _make_writer(template_in_dat: Dict[str, _INVariable]) -> InDat:
    indat = InDat(filename=None)
    indat.data = template_in_dat
    return indat


def create_template_from_path(template_in_dat: Union[str, Path]) -> ProcessInputs:
    if not ENABLED:
        raise CodesError(
            f"{PROCESS_NAME} is not installed cannot read template {template_in_dat}"
        )
    if Path(template_in_dat).is_file():
        # InDat autoloads IN.DAT without checking for existence
        return ProcessInputs(**{
            k: v.value if k == "runtitle" else v.get_value
            for k, v in InDat(filename=template_in_dat).data.items()
        })
    raise CodesError(f"Template IN.DAT '{template_in_dat}' is not a file.")
