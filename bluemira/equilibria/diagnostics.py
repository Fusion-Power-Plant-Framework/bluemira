# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Diagnostic options for use in the equilibria module.
"""

from dataclasses import dataclass
from enum import Enum, auto


class PicardDiagnostic(Enum):
    """Type of plot to view during optimisation."""

    EQ = auto()
    CONVERGENCE = auto()
    PSI_COMPARISON = auto()


@dataclass
class PicardDiagnosticOptions:
    """Diagnostic plotting options for the Picard Iterator"""
