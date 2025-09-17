# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Winding pack module"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matproplib import OperationalConditions

from bluemira.magnets.conductor import Conductor


class WindingPack:
    """
    Represents a winding pack composed of a grid of conductors.

    Attributes
    ----------
    conductor:
        The base conductor type used in the winding pack.
    nx:
        Number of conductors along the x-axis.
    ny:
        Number of conductors along the y-axis.
    """

    def __init__(
        self, conductor: Conductor, nx: int, ny: int, name: str = "WindingPack"
    ):
        """
        Initialise a WindingPack instance.

        Parameters
        ----------
        conductor:
            The conductor instance.
        nx:
            Number of conductors along the x-axis.
        ny:
            Number of conductors along the y-axis.
        name:
            Name of the winding pack instance.
        """
        self.conductor = conductor
        self.nx = nx
        self.ny = ny
        self.name = name

    @property
    def dx(self) -> float:
        """Return the width of the winding pack [m]."""
        return self.conductor.dx * self.nx

    @property
    def dy(self) -> float:
        """Return the height of the winding pack [m]."""
        return self.conductor.dy * self.ny

    @property
    def area(self) -> float:
        """Return the total cross-sectional area [m²]."""
        return self.dx * self.dy

    @property
    def n_conductors(self) -> int:
        """Return the total number of conductors."""
        return self.nx * self.ny

    @property
    def jacket_area(self) -> float:
        """Return the total jacket material area [m²]."""
        return self.conductor.area_jacket * self.n_conductors

    def Kx(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Compute the equivalent stiffness along the x-axis.

        Parameters
        ----------
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Stiffness along the x-axis [N/m].
        """
        return self.conductor.Kx(op_cond) * self.ny / self.nx

    def Ky(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Compute the equivalent stiffness along the y-axis.

        Parameters
        ----------
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Stiffness along the y-axis [N/m].
        """
        return self.conductor.Ky(op_cond) * self.nx / self.ny

    def plot(
        self,
        xc: float = 0,
        yc: float = 0,
        *,
        show: bool = False,
        ax: plt.Axes | None = None,
        homogenised: bool = True,
    ) -> plt.Axes:
        """
        Plot the winding pack geometry.

        Parameters
        ----------
        xc:
            Center x-coordinate [m].
        yc:
            Center y-coordinate [m].
        show:
            If True, immediately show the plot.
        ax:
            Axes object to draw on.
        homogenised:
            If True, plot as a single block. Otherwise, plot individual conductors.

        Returns
        -------
        :
            Axes object containing the plot.
        """
        if ax is None:
            _, ax = plt.subplots()

        pc = np.array([xc, yc])
        a = self.dx / 2
        b = self.dy / 2

        p0 = np.array([-a, -b])
        p1 = np.array([a, -b])
        p2 = np.array([a, b])
        p3 = np.array([-a, b])

        points_ext = np.vstack((p0, p1, p2, p3, p0)) + pc

        ax.fill(points_ext[:, 0], points_ext[:, 1], "gold", snap=False)
        ax.plot(points_ext[:, 0], points_ext[:, 1], "k")

        if not homogenised:
            for i in range(self.nx):
                for j in range(self.ny):
                    xc_c = xc - self.dx / 2 + (i + 0.5) * self.conductor.dx
                    yc_c = yc - self.dy / 2 + (j + 0.5) * self.conductor.dy
                    self.conductor.plot(xc=xc_c, yc=yc_c, ax=ax)

        if show:
            plt.show()
        return ax

    def to_dict(self) -> dict[str, Any]:
        """
        Serialise the WindingPack to a dictionary.

        Returns
        -------
        :
            Serialised dictionary of winding pack attributes.
        """
        return {
            "name": self.name,
            "conductor": self.conductor.to_dict(),
            "nx": self.nx,
            "ny": self.ny,
        }
