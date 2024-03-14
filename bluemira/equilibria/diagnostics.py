# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Diagnostic Options for use in the equilibria module.
"""

from dataclasses import dataclass
from enum import Enum, auto


class PicardDiagnosticType(Enum):
    """Type of plot to view dusring optimisation."""

    EQ = auto()
    CONVERGENCE = auto()
    PSI_COMPARISON = auto()


@dataclass
class PicardDiagnosticOptions:
    """Diagnostic plotting options for the Picard Iterator"""
