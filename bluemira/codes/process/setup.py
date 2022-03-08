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
PROCESS setup functions
"""

import os

import bluemira.codes.interface as interface
from bluemira.codes.error import CodesError

# PROCESS imports
from bluemira.codes.process.api import DEFAULT_INDAT, InDat, update_obsolete_vars
from bluemira.codes.process.constants import NAME as PROCESS


class PROCESSInputWriter(InDat):
    """
    Bluemira IN.DAT writer for PROCESS input.

    Parameters
    ----------
    template_indat: str
        Path to the IN.DAT file to use as the template for PROCESS parameters.
    """

    def __init__(self, template_indat=DEFAULT_INDAT):
        if os.path.isfile(template_indat):
            # InDat autoloads IN.DAT without checking for existence
            super().__init__(filename=template_indat)
        else:
            super().__init__(filename=None)
            self.filename = template_indat


class Setup(interface.Setup):
    """
    Setup Task for process
    """

    def _run(self):
        """
        Write the IN.DAT file and store in the main PROCESS folder
        Note that if use_bp_inputs is True, bluemira outputs with
        param.mapping.send == True will be written to IN.DAT.
        """
        self.prepare_bp_inputs()
        self.write_indat(use_bp_inputs=True)

    def _runinput(self):
        self.prepare_bp_inputs()
        self.write_indat(use_bp_inputs=False)

    def prepare_bp_inputs(self, use_bp_inputs=True):
        """
        Update parameter mapping send values to True/False depending on use_bp_inputs.

        Parameters
        ----------
        use_bp_inputs: bool, optional
            Option to use bluemira values as PROCESS inputs. If True, sets the send
            value for params in the params_to_update list to True and sets all others to
            False. If True but no params_to_update list provided, makes no changes to
            send values. If False, sets all send values to False.
            Default, True
        """
        # Skip if True but no list provided
        if use_bp_inputs is True and self.parent._params_to_update is None:
            return
        # Update send values to True or False
        for param in self.params.get_parameter_list():
            bp_name = self._parameter_mapping[param.mapping[PROCESS].name]
            param.mapping[PROCESS].send = (
                use_bp_inputs and bp_name in self.parent._params_to_update
            )

    def write_indat(self, use_bp_inputs=True):
        """
        Write the IN.DAT file and stores in the main PROCESS folder.

        Parameters
        ----------
        use_bp_inputs: bool, optional
            Option to use bluemira values as PROCESS inputs. Used to re-run PROCESS
            within a bluemira run. If False, runs PROCESS without modifying inputs.
            Default, True
        """
        # Load defaults in bluemira folder
        writer = PROCESSInputWriter(template_indat=self.parent._template_indat)
        if writer.data == {}:
            raise CodesError(
                f"Unable to read template IN.DAT file at {self.parent._template_indat}"
            )

        if use_bp_inputs is True:
            for param in self.params.get_parameter_list():
                mapping = param.mapping[PROCESS]
                if mapping.send:
                    new_mapping = update_obsolete_vars(mapping.name)
                    if isinstance(new_mapping, list):
                        for mapping in new_mapping:
                            writer.add_parameter(mapping, param.value)
                    else:
                        writer.add_parameter(new_mapping, param.value)

        filename = os.path.join(self.parent.run_dir, "IN.DAT")
        writer.write_in_dat(output_filename=filename)
