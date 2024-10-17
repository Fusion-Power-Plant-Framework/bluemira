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
from tabulate import tabulate

from bluemira.base.constants import raw_uc
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

    def get_state(self, name: str) -> Snapshot | None:
        """
        Returns
        -------
        :
            An equilibrium state from the Equilibrium manager.

        """
        return self.states.get(name, None)

    def plot(self):
        """
        Plot the states of the equilibria
        """
        f, ax = plt.subplots(1, 3)
        for i, case in enumerate([self.BREAKDOWN, self.SOF, self.EOF]):
            state = self.get_state(case)
            state.eq.plot(ax[i])
            state.eq.coilset.plot(ax[i], label=True)
        return f

    def summary(self):
        """
        Produce a summary of the coils and currents in different states.
        """
        coilset = self.get_state(self.SOF).coilset
        columns = [
            "Coil name",
            "x",
            "z",
            "x_width",
            "z_width",
            "Breakdown currents [MA]",
            "SOF currents [MA]",
            "EOF currents [MA]",
        ]

        records = list(
            zip(
                coilset.name,
                coilset.x,
                coilset.z,
                2.0 * coilset.dx,
                2.0 * coilset.dz,
                raw_uc(self.get_state(self.BREAKDOWN).coilset.current, "A", "MA"),
                raw_uc(self.get_state(self.SOF).coilset.current, "A", "MA"),
                raw_uc(self.get_state(self.EOF).coilset.current, "A", "MA"),
                strict=False,
            )
        )

        return tabulate(
            records,
            headers=columns,
            tablefmt="fancy_grid",
            showindex=False,
            numalign="right",
        )


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

    def tabulate(self) -> str:
        """
        Returns
        -------
        :
            Tabulated results
        """
        if hasattr(self.results, "_tabulate"):
            return str(self.results)
        raise NotImplementedError("Tabulate not available for volume calculation")

    def __str__(self) -> str:
        """
        Returns
        -------
        :
            String representation
        """
        try:
            return self.tabulate()
        except NotImplementedError:
            return super().__str__()
