# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Diagnostic options for use in the equilibria module.
"""

from dataclasses import dataclass


@dataclass
class EqDiagnosticOptions:
    """Diagnostic plotting options for Equilibrium."""

    psi_diff: bool = False
    split_psi_plots: bool = False
