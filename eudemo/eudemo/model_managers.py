# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
EUDEMO model manager classes
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bluemira.equilibria.run import Snapshot

from bluemira.base.look_and_feel import bluemira_warn


class EquilibriumManager:
    """
    Manager for free-boundary equilibria
    """

    REFERENCE = "Reference"
    BREAKDOWN = "Breakdown"
    SOF = "SOF"  # Start of flat-top
    EOF = "EOF"  # End of flat-top

    def __init__(self):
        self.states = {
            self.REFERENCE: None,
            self.BREAKDOWN: None,
            self.SOF: None,
            self.EOF: None,
        }

    def add_state(self, name: str, snapshot: Snapshot):
        """
        Add an equilibrium state to the Equilibrium manager.
        """
        if self.states.get(name, None) is not None:
            bluemira_warn(f"Over-writing equilibrium state: {name}!")
        self.states[name] = snapshot

    def get_state(self, name: str) -> None | Snapshot:
        """
        Get an equilibrium state from the Equilibrium manager.
        """
        return self.states.get(name, None)
