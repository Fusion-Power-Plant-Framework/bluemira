# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Winding pack module"""

import matplotlib.pyplot as plt
import numpy as np

from bluemira.magnets.conductor import Conductor


class WindingPack:
    """
    A class to represent a winding pack which contains multiple conductors arranged in
    a grid pattern.
    """

    def __init__(self, conductor: Conductor, nx: int, ny: int):
        """
        Initializes the WindingPack with the given conductor, number of conductors
        along x and y axes, and an optional name.

        Parameters
        ----------
        conductor : Conductor
            An instance of the Conductor class.
        nx : int
            Number of conductors along the x-axis.
        ny : int
            Number of conductors along the y-axis.
        """
        self.nx = int(nx)
        self.ny = int(ny)
        self.conductor = conductor

    @property
    def dx(self):
        """
        Total horizontal size of the winding pack [m].

        Returns
        -------
        float
            Width of the winding pack in the x-direction.
        """
        return self.conductor.dx * self.nx

    @property
    def dy(self):
        """
        Total vertical size of the winding pack [m].

        Returns
        -------
        float
            Height of the winding pack in the y-direction.
        """
        return self.conductor.dy * self.ny

    @property
    def area(self):
        """
        Total cross-sectional area of the winding pack.

        Returns
        -------
        float
            Total area of the winding pack [m²].
        """
        return self.dx * self.dy

    @property
    def n_conductors(self):
        """
        Total number of conductors in the winding pack.

        Returns
        -------
        int
            Number of conductors (nx * ny).
        """
        return self.nx * self.ny

    @property
    def jacket_area(self):
        """
        Total jacket material area in the winding pack.

        Returns
        -------
        float
            Combined jacket area of all conductors in the pack [m²].
        """
        return self.conductor.area_jacket * self.n_conductors

    def Kx(self, **kwargs) -> float:  # noqa: N802
        """
        Compute the equivalent mechanical stiffness in the x-direction.

        This models the stiffness as a composite structure based on
        conductor properties and arrangement.

        Parameters
        ----------
        **kwargs : dict
            Additional arguments passed to the conductor stiffness model.

        Returns
        -------
        float
            Effective axial stiffness along the x-axis [N/m].
        """
        return self.conductor.Kx(**kwargs) * self.ny / self.nx

    def Ky(self, **kwargs) -> float:  # noqa: N802
        """
        Calculates the total equivalent stiffness of the winding pack along the y-axis.

        Parameters
        ----------
        **kwargs
            Additional parameters to pass to the stiffness calculation methods.

        Returns
        -------
            The total equivalent stiffness along the y-axis.
        """
        return self.conductor.Ky(**kwargs) * self.nx / self.ny

    def plot(
        self,
        xc: float = 0,
        yc: float = 0,
        *,
        show: bool = False,
        ax=None,
        homogenized: bool = True,
    ):
        """
        Plots the winding pack and its conductors.

        Parameters
        ----------
        xc :
            The x-coordinate of the center of the winding pack (default is 0).
        yc :
            The y-coordinate of the center of the winding pack (default is 0).
        show :
            Whether to show the plot immediately (default is False).
        ax :
            The axes on which to plot (default is None, which creates a new figure and
            axes).
        homogenized :
            Whether to plot the winding pack as a homogenized block or show individual
            conductors (default is True).

        Returns
        -------
            The axes with the plotted winding pack.
        """
        if ax is None:
            _, ax = plt.subplots()

        ax.plot([0], [0])

        pc = np.array([xc, yc])
        a = self.dx / 2
        b = self.dy / 2

        p0 = np.array([-a, -b])
        p1 = np.array([a, -b])
        p2 = np.array([a, b])
        p3 = np.array([-a, b])

        points_ext = np.vstack((p0, p1, p2, p3, p0)) + pc

        ax.fill(points_ext[:, 0], points_ext[:, 1], "gold")
        ax.plot(points_ext[:, 0], points_ext[:, 1], "k")

        if not homogenized:
            for i in range(self.nx):
                for j in range(self.ny):
                    xc_c = xc - self.dx / 2 + (i + 0.5) * self.conductor.dx
                    yc_c = yc - self.dy / 2 + (j + 0.5) * self.conductor.dy
                    self.conductor.plot(xc=xc_c, yc=yc_c, ax=ax)

        if show:
            plt.show()
        return ax
