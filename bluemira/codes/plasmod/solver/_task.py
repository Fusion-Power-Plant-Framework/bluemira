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
Defines the base Task for plasmod
"""

from bluemira.base.parameter import ParameterFrame
from bluemira.base.solver import Task
from bluemira.codes.plasmod.constants import NAME as PLASMOD_NAME
from bluemira.codes.plasmod.mapping import mappings as plasmod_mappings
from bluemira.codes.utilities import add_mapping


class PlasmodTask(Task):
    """
    A task related to plasmod.

    This adds plasmod parameter mappings to the input ParameterFrame.
    """

    def __init__(self, params: ParameterFrame) -> None:
        super().__init__(params)
        add_mapping(PLASMOD_NAME, self._params, plasmod_mappings)
