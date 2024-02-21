# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Coil Types
"""

from enum import Enum, EnumMeta, auto


class CoilTypeEnumMeta(EnumMeta):
    """
    Allow override of KeyError error string
    """

    def __getitem__(self, name):
        """
        Coil Type __getitem__
        """
        try:
            return super().__getitem__(name)
        except KeyError:
            raise KeyError(f"Unknown CoilType {name}") from None


class CoilType(Enum, metaclass=CoilTypeEnumMeta):
    """
    CoilType Enum
    """

    PF = auto()
    CS = auto()
    DUM = auto()
    NONE = auto()
