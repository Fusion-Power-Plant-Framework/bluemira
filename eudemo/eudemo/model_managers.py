# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
EUDEMO model manager classes
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from bluemira.base.look_and_feel import bluemira_warn

if TYPE_CHECKING:
    from bluemira.codes.openmc.output import OpenMCResult
    from bluemira.equilibria.run import Snapshot
    from eudemo.eudemo.neutronics.run import EUDEMONeutronicsCSGReactor


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


class NeutronicsManager:
    """Manager for neutronics"""

    def __init__(
        self,
        csg_reactor: EUDEMONeutronicsCSGReactor,
        results: OpenMCResult | dict[int, float],
    ):
        self.csg_reactor = csg_reactor
        self.results = results

    def plot(self):
        """Plot neutronics results"""
        _f, ax = plt.subplots()
        ax.axis("off")
        ax.imshow(
            np.asarray(
                Image.open(
                    Path(__file__).parent.parent
                    / "config"
                    / "neutronics"
                    / "plot"
                    / "plot_1.png"
                )
            )
        )

        self.csg_reactor.plot_2d()

        plt.show()

    def __str__(self) -> str:
        """String Representation"""
        if hasattr(self.results, "_tabulate"):
            # Avoid openmc related import
            return self.results.__str__()
        return super().__str__()
