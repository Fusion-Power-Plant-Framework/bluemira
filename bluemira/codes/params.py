# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Base classes for external parameter management."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Literal

from bluemira.base.parameter_frame import ParameterFrame
from bluemira.codes.error import CodesError


class MappedParameterFrame(ParameterFrame):
    """
    Special ``ParameterFrame`` that contains a set of parameter mappings.

    The mappings are intended to be used to map bluemira parameters to
    parameters in an external code.

    See :class:`~bluemira.base.parameter_frame.ParameterFrame` for details
    on how to declare parameters.
    """

    @abc.abstractproperty
    def defaults(self) -> dict:
        """
        Default values for the ParameterFrame
        """

    @classmethod
    def from_defaults(
        cls, data: dict, source: str = "bluemira codes default"
    ) -> MappedParameterFrame:
        """
        Create ParameterFrame with default values for external codes.

        External codes are likely to have variables that are not changed often
        therefore in some cases sane defaults are needed.

        If a default value is not found for a given mapping it is set to NaN

        """
        new_param_dict = {}
        for bm_map_name, param_map in cls._mappings.items():
            new_param_dict[bm_map_name] = {
                "value": data.get(param_map.name, None),
                "unit": param_map.unit,
                "source": source,
            }

        return cls.from_dict(new_param_dict)

    @abc.abstractproperty
    def mappings(self) -> dict[str, ParameterMapping]:
        """
        The mappings associated with these frame's parameters.

        The keys are names of parameters in this frame, the values
        are ``ParameterMapping`` objects containing the name of the
        corresponding parameter in some external code.
        """

    def update_mappings(
        self, new_send_recv: dict[str, dict[Literal["send", "recv"], bool]]
    ):
        """
        Update the mappings in this frame with new send/recv values.

        Parameters
        ----------
        new_send_recv:
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
                ) from None
            if (send_mapping := send_recv_mapping.get("send", None)) is not None:
                param_mapping.send = send_mapping
            if (recv_mapping := send_recv_mapping.get("recv", None)) is not None:
                param_mapping.recv = recv_mapping


@dataclass
class ParameterMapping:
    """
    Simple class containing information on mapping of a bluemira parameter to one in
    external software.

    Parameters
    ----------
    name:
       name of mapped parameter
    recv:
        receive data from mapped parameter (to overwrite bluemira parameter)
    send:
        send data to mapped parameter (from bluemira parameter)
    """

    name: str
    out_name: str | None = None
    send: bool = True
    recv: bool = True
    unit: str | None = None

    _frozen = ()

    def __post_init__(self):
        """
        Freeze the dataclass
        """
        if self.out_name is None:
            self.out_name = self.name
        self._frozen = ("name", "out_name", "unit", "_frozen")

    def to_dict(self) -> dict:
        """
        Convert this object to a dictionary with attributes as values.
        """
        return {
            "name": self.name,
            "out_name": self.out_name,
            "send": self.send,
            "recv": self.recv,
            "unit": self.unit,
        }

    @classmethod
    def from_dict(cls, the_dict: dict) -> ParameterMapping:
        """
        Create a ParameterMapping using a dictionary with attributes as values.
        """
        return cls(**the_dict)

    def __str__(self):
        """
        Create a string representation of of this object which is more compact than that
        provided by the default `__repr__` method.
        """
        return repr(self.to_dict())

    def __setattr__(self, attr: str, value: bool | str):
        """
        Protect against additional attributes

        Parameters
        ----------
        attr:
            Attribute to set (name can only be set on init)
        value:
            Value of attribute
        """
        if (
            attr not in {"send", "recv", "name", "out_name", "unit", "_frozen"}
            or attr in self._frozen
        ):
            raise KeyError(f"{attr} cannot be set for a {type(self).__name__}")
        if attr in {"send", "recv"} and not isinstance(value, bool):
            raise ValueError(f"{attr} must be a bool")
        super().__setattr__(attr, value)
