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
An attempt at some naming conventions
"""
from bluemira.base.look_and_feel import bluemira_warn

NAME_SHORT_TO_LONG = {
    "VV": "Reactor vacuum vessel",
    "BB": "Breeding blanket",
    "TF": "Toroidal field coils",
    "PF": "Poloidal field coils",
    "CS": "Central solenoid",
    "ATEC": "Coil structures",
    "DIV": "Divertor",
    "PL": "Plasma",
    "HCD": "Heating and current drive",
    # "CR": "Cryostat",
    "CR": "Cryostat vacuum vessel",
    "TS": "Thermal shield",
    "RS": "Radiation shield",
    "EC": "Electron cyclotron",
    "NBI": "Neutral beam injector",
    "CCS": "Central column shield",
    "FW": "First wall",
}

NAME_LONG_TO_SHORT = {v: k for k, v in NAME_SHORT_TO_LONG.items()}
SHORT_NAMES = list(NAME_SHORT_TO_LONG.keys())
LONG_NAMES = list(NAME_SHORT_TO_LONG.values())


def name_mapper(name_dict, target_dict, map_dict):
    """
    Force maps a dictionary to alternative keys for miscellaneous purposes

    Parameters
    ----------
    name_dict: dict
        The dictionary of names that needs to be force mapped
    target_dict: dict
        The dictionary of desired keys and their mappings
    map_dict: dict
        The reversed dictionary key mapping

    Returns
    -------
    out: dict
        The dictionary with the alternative keys
    """
    out = {}
    for k, v in name_dict.items():
        if k not in target_dict:
            try:
                k = map_dict[k]
            except KeyError:
                bluemira_warn("Unknown key in name mapper")
        out[k] = v
    return out


def name_short_long(name_dict):
    """
    Map the short names to the long names
    """
    return name_mapper(name_dict, NAME_LONG_TO_SHORT, NAME_SHORT_TO_LONG)


def name_long_short(name_dict):
    """
    Map the long names to the short names
    """
    return name_mapper(name_dict, NAME_SHORT_TO_LONG, NAME_LONG_TO_SHORT)
