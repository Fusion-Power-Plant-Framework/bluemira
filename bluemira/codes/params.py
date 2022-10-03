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
from typing import Dict, Literal

from bluemira.base.parameter import ParameterMapping
from bluemira.base.parameter_frame import NewParameterFrame as ParameterFrame
from bluemira.codes.error import CodesError


class MappedParameterFrame(ParameterFrame):
    """
    Special ``ParameterFrame`` that contains a set of parameter mappings.

    The mappings are intended to be used to map bluemira parameters to
    parameters in an external code.

    See :class:`bluemira.base.parameter_frame.ParameterFrame` for details
    on how to declare parameters.
    """

    @abc.abstractproperty
    def mappings(self) -> Dict[str, ParameterMapping]:
        """
        The mappings associated with these frame's parameters.

        The keys are names of parameters in this frame, the values
        are ``ParameterMapping`` objects containing the name of the
        corresponding parameter in some external code.
        """

    def update_mappings(
        self, new_send_recv: Dict[str, Dict[Literal["send", "recv"], bool]]
    ):
        """
        Update the mappings in this frame with new send/recv values.

        Parameters
        ----------
        new_send_recv: Dict[str, Dict[Literal["send", "recv"], bool]]
            The new send/recv values for all, or a subset, of the
            parameter mappings.
            Keys are parameter names (as defined in this class, not the
            external code), the values are a dictionary, optionally
            containing keys 'send' and/or 'recv'. The values for the
            inner dictionary are a boolean.

        Raises
        ------
        CodesError:
            If a parameter name in the input does not match the name of
            a parameter in this frame.
        """
        for param_name, send_recv_mapping in new_send_recv.items():
            try:
                param_mapping = self.mappings[param_name]
            except KeyError:
                raise CodesError(
                    "Cannot update parameter mapping. "
                    f"No parameter with name '{param_name}' in '{type(self).__name__}'."
                )
            if (send_mapping := send_recv_mapping.get("send", None)) is not None:
                param_mapping.send = send_mapping
            if (recv_mapping := send_recv_mapping.get("recv", None)) is not None:
                param_mapping.recv = recv_mapping
