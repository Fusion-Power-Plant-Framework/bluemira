# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Coil Types
"""

from enum import Enum, auto


class CoilType(Enum):
    """
    CoilType Enum
    """

    PF = auto()
    CS = auto()
    DUM = auto()
    NONE = auto()

    @classmethod
    def _missing_(cls, value):
        if not isinstance(value, str):
            raise TypeError("Input must be a string.")
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(
                f"{value} is not a valid CoilType. Choose from: PF, CS, DUM, NONE"
            ) from None
