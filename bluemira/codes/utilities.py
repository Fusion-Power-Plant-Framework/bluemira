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
Utility functions for interacting with external codes
"""


from typing import Dict, Literal

from . import error as code_err


def _get_mapping(
    params, code_name: str, read_write: Literal["read", "write"], override: bool = False
) -> Dict[str, str]:
    """
    Create a dictionary to get the read or write mappings for a given code.

    Parameters
    ----------
    params: ParameterFrame
        The parameters with mappings that define what is going to be read or written.
    code_name: str
        The identifying name of the code that is being read or written from.
    read_write: Literal["read", "write"]
        Whether to generate a mapping for reading or writing.
    override: bool, optional
        If True then map variables with a mapping defined, even if read or write=False.
        By default, False.

    Yields
    ------
    mapping: Dict[str, str]
        The mapping between external code parameter names (key) and bluemira parameter
        names (value).
    """
    if read_write not in ["read", "write"]:
        raise code_err.CodesError("Mapping must be obtained for either read or write.")

    mapping = {}
    for key in params.keys():
        param = params.get_param(key)
        has_mapping = param.mapping is not None and code_name in param.mapping
        map_param = has_mapping and (
            override or getattr(param.mapping[code_name], read_write)
        )
        if map_param:
            mapping[param.mapping[code_name].name] = key
    return mapping


def get_read_mapping(params, code_name, read_all=False):
    """
    Get the read mapping for variables mapped from the external code to the provided
    input ParameterFrame.

    Parameters
    ----------
    params: ParameterFrame
        The parameters with mappings that define what is going to be read.
    code_name: str
        The identifying name of the code that is being read from.
    read_all: bool, optional
        If True then read all variables with a mapping defined, even if read=False. By
        default, False.

    Returns
    -------
    mapping: Dict[str, str]
        The mapping between external code parameter names (key) and bluemira parameter
        names (value) to use for readings.
    """
    return _get_mapping(params, code_name, "read", read_all)


def get_write_mapping(params, code_name, write_all=False):
    """
    Get the write mapping for variables mapped from the external code to the provided
    input ParameterFrame.

    Parameters
    ----------
    params: ParameterFrame
        The parameters with mappings that define what is going to be written.
    code_name: str
        The identifying name of the code that is being written to.
    write_all: bool, optional
        If True then write all variables with a mapping defined, even if write=False. By
        default, False.

    Returns
    -------
    mapping: Dict[str, str]
        The mapping between external code parameter names (key) and bluemira parameter
        names (value) to use for writing.
    """
    return _get_mapping(params, code_name, "write", write_all)
