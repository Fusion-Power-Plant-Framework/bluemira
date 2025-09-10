# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Winding pack module"""

from typing import Any, ClassVar

import matplotlib.pyplot as plt
import numpy as np
from matproplib import OperationalConditions

from bluemira.magnets.conductor import Conductor, create_conductor_from_dict


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

    _name_in_registry_: ClassVar[str] = "WindingPack"

    def __init__(
        self, conductor: Conductor, nx: int, ny: int, name: str = "WindingPack"
    ):
        """
        Initialize a WindingPack instance.

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
        homogenized: bool = True,
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
        homogenized:
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

        if not homogenized:
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
        Serialize the WindingPack to a dictionary.

        Returns
        -------
        :
            Serialized dictionary of winding pack attributes.
        """
        return {
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
        windingpack_dict:
            Serialized winding pack dictionary.
        name:
            Name for the new instance. If None, attempts to use the 'name' field from
            the dictionary.

        Returns
        -------
        :
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


def create_wp_from_dict(
    windingpack_dict: dict[str, Any],
    name: str | None = None,
) -> WindingPack:
    """
    Factory function to create a WindingPack or its subclass from a serialized
    dictionary.

    Parameters
    ----------
    windingpack_dict:
        Dictionary containing serialized winding pack data.
        Must include a 'name_in_registry' field matching a registered class.
    name:
        Optional name override for the reconstructed WindingPack.

    Returns
    -------
    :
        An instance of the appropriate WindingPack subclass.

    Raises
    ------
    ValueError
        If 'name_in_registry' is missing from the dictionary.
        If no matching registered class is found.
    """
    name_in_registry = windingpack_dict.get("name_in_registry")
    if name_in_registry is None:
        raise ValueError(
            "Serialized winding pack dictionary must contain a 'name_in_registry' field."
        )

    cls = WINDINGPACK_REGISTRY.get(name_in_registry)
    if cls is None:
        available = ", ".join(WINDINGPACK_REGISTRY.keys())
        raise ValueError(
            f"No registered winding pack class with registration name '"
            f"{name_in_registry}'. "
            f"Available: {available}."
        )

    return cls.from_dict(windingpack_dict=windingpack_dict, name=name)
