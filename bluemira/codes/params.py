# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
"""Base classes for external parameter management."""

import abc
from typing import Dict

from bluemira.base.parameter import ParameterMapping
from bluemira.base.parameter_frame import NewParameterFrame as ParameterFrame


class MappedParameterFrame(ParameterFrame):
    """
    Special ``ParameterFrame`` that contains a set of parameter mappings.

    The mappings are intended to be used to map bluemira parameters to
    parameters in an external code.
    """

    @abc.abstractproperty
    def mappings(self) -> Dict[str, ParameterMapping]:
        """
        The mappings associated with these frame's parameters.

        The keys are names of parameters in this frame, the values
        are ``ParameterMapping` objects containing the name of the
        corresponding parameter in some external code.

        TODO(hsaunders1904: link to ``add_mapping``)
        """
