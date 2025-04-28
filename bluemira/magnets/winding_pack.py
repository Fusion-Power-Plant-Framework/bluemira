# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Winding pack module"""

from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np

from bluemira.magnets.conductor import Conductor, create_conductor_from_dict
from bluemira.magnets.registry import RegistrableMeta

# Global registries
WINDINGPACK_REGISTRY = {}


class WindingPack(metaclass=RegistrableMeta):
    """
    Represents a winding pack composed of a grid of conductors.

    Attributes
    ----------
    conductor : Conductor
        The base conductor type used in the winding pack.
    nx : int
        Number of conductors along the x-axis.
    ny : int
        Number of conductors along the y-axis.
    """

    _registry_: ClassVar[dict] = WINDINGPACK_REGISTRY
    _name_in_registry_: ClassVar[str] = "WindingPack"

    def __init__(
        self, conductor: Conductor, nx: int, ny: int, name: str = "WindingPack"
    ):
        """
        Initialize a WindingPack instance.

        Parameters
        ----------
        conductor : Conductor
            The conductor instance.
        nx : int
            Number of conductors along the x-direction.
        ny : int
            Number of conductors along the y-direction.
        name : str, optional
            Name of the winding pack instance.
        """
        self.conductor = conductor
        self.nx = int(nx)
        self.ny = int(ny)
        self.name = name

    @property
    def dx(self) -> float:
        """Return the total width of the winding pack [m]."""
        return self.conductor.dx * self.nx

    @property
    def dy(self) -> float:
        """Return the total height of the winding pack [m]."""
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

    def Kx(self, **kwargs) -> float:  # noqa: N802
        """
        Compute the equivalent stiffness along the x-axis.

        Parameters
        ----------
        **kwargs
            Additional arguments forwarded to the conductor.

        Returns
        -------
        float
            Stiffness along the x-axis [N/m].
        """
        return self.conductor.Kx(**kwargs) * self.ny / self.nx

    def Ky(self, **kwargs) -> float:  # noqa: N802
        """
        Compute the equivalent stiffness along the y-axis.

        Parameters
        ----------
        **kwargs
            Additional arguments forwarded to the conductor.

        Returns
        -------
        float
            Stiffness along the y-axis [N/m].
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
        Plot the winding pack geometry.

        Parameters
        ----------
        xc : float
            Center x-coordinate [m].
        yc : float
            Center y-coordinate [m].
        show : bool, optional
            If True, immediately show the plot.
        ax : matplotlib.axes.Axes, optional
            Axes object to draw on.
        homogenized : bool, optional
            If True, plot as a single block. Otherwise, plot individual conductors.

        Returns
        -------
        matplotlib.axes.Axes
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

    def to_dict(self) -> dict:
        """
        Serialize the WindingPack to a dictionary.

        Returns
        -------
        dict
            Serialized dictionary of winding pack attributes.
        """
        return {
            "name_in_registry": getattr(
                self, "_name_in_registry_", self.__class__.__name__
            ),
            "name": self.name,
            "conductor": self.conductor.to_dict(),
            "nx": self.nx,
            "ny": self.ny,
        }

    @classmethod
    def from_dict(
        cls,
        windingpack_dict: dict[str, Any],
        name: str | None = None,
    ) -> "WindingPack":
        """
        Deserialize a WindingPack from a dictionary.

        Parameters
        ----------
        windingpack_dict : dict
            Serialized winding pack dictionary.
        name : str
            Name for the new instance. If None, attempts to use the 'name' field from
            the dictionary.

        Returns
        -------
        WindingPack
            Reconstructed WindingPack instance.

        Raises
        ------
        ValueError
            If 'name_in_registry' does not match the expected class.
        """
        # Validate name_in_registry
        name_in_registry = windingpack_dict.get("name_in_registry")
        expected_name_in_registry = getattr(cls, "_name_in_registry_", cls.__name__)

        if name_in_registry != expected_name_in_registry:
            raise ValueError(
                f"Cannot create {cls.__name__} from dictionary with name_in_registry "
                f"'{name_in_registry}'. Expected '{expected_name_in_registry}'."
            )

        # Deserialize conductor
        conductor = create_conductor_from_dict(
            conductor_dict=windingpack_dict["conductor"],
            name=None,
        )

        return cls(
            conductor=conductor,
            nx=windingpack_dict["nx"],
            ny=windingpack_dict["ny"],
            name=name or windingpack_dict.get("name"),
        )
