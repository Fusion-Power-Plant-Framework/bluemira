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
UKAEA Power Balance Models mappings
"""

from bluemira.base.parameter import ParameterMapping
from bluemira.codes.ukaea_powerbalance.constants import MODEL_NAME

mappings = {
    "TF_cond_tech": ParameterMapping(
        "structural.Magnets.isMagnetTFSuperconCoil", False, True
    ),
    "TF_res_bus": ParameterMapping(
        f"{MODEL_NAME}.magnetpower.magnetTF.Rfeeder", False, True
    ),
    "TF_E_tot": ParameterMapping(
        f"{MODEL_NAME}.magnetpower.magnetTF.magEnergy", False, True
    ),
    "TF_res_tot": ParameterMapping(
        f"{MODEL_NAME}.magnetpower.magnetTF.Rtot", False, True
    ),
    "TF_currpt_ob": ParameterMapping("profiles.TFCoil.max_current", False, True),
    "TF_respc_ob": ParameterMapping(
        f"{MODEL_NAME}.magnetpower.magnetTF.Roleg", False, True
    ),
    "P_bd_in": ParameterMapping("profiles.Heat.max_power", False, True),
    "condrad_cryo_heat": ParameterMapping(
        f"{MODEL_NAME}.cryogenicpower.PFcrMW", False, True
    ),
    "tk_tp_tot": ParameterMapping("profiles.ThermalPowerOut.max_power", False, True),
}
