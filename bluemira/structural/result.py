# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
FE result object
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.pyplot import Axes

    from bluemira.structural.geometry import Geometry
    from bluemira.structural.loads import LoadCase
    from bluemira.structural.symmetry import CyclicSymmetry

import numpy as np

from bluemira.structural.geometry import DeformedGeometry
from bluemira.structural.transformation import cyclic_pattern
from bluemira.utilities.plot_tools import Plot3D


class Result:
    """
    Container class for storing results
    """

    __slots__ = (
        "_cycle_sym",
        "_max_deflections",
        "_max_stresses",
        "_safety_factors",
        "_stresses",
        "deflections",
        "deflections_xyz",
        "geometry",
        "load_case",
        "reactions",
    )

    def __init__(
        self,
        geometry: Geometry,
        load_case: LoadCase,
        deflections: np.ndarray,
        reactions: np.ndarray,
        cyclic_symmetry: CyclicSymmetry | None,
    ):
        self.geometry = geometry
        self.load_case = load_case
        self.deflections = deflections
        self.deflections_xyz = deflections.reshape(self.geometry.n_nodes, 6)
        self.reactions = reactions
        self._cycle_sym = cyclic_symmetry

        self._stresses = None
        self._max_stresses = None
        self._safety_factors = None
        self._max_deflections = None
        self._get_values()

    def _get_values(self):
        """
        Get the maximum values of stress, deflection, and safety factors over
        all the Elements in the Geometry.
        """
        s = []
        max_stresses = np.zeros(self.geometry.n_elements)
        safety_factors = np.zeros(self.geometry.n_elements)
        max_deflections = np.zeros(self.geometry.n_elements)
        for i, element in enumerate(self.geometry.elements):
            s.append(element.stresses)
            safety_factors[i] = element.safety_factor
            max_stresses[i] = element.max_stress
            d_n1 = element.node_1.displacements[:3]
            d_n2 = element.node_2.displacements[3:]

            d1 = np.sqrt(np.sum(d_n1**2))
            d2 = np.sqrt(np.sum(d_n2**2))

            max_deflections[i] = max(d1, d2)
        self._safety_factors = safety_factors
        self._max_stresses = max_stresses
        self._stresses = s
        self._max_deflections = max_deflections

    def make_deformed_geometry(self, scale: float = 30.0) -> DeformedGeometry:
        """
        Make deformed geometry of the result

        Parameters
        ----------
        scale:
            The scale for the deformations

        Returns
        -------
        The deformed geometry of the Result at the specified scale
        """
        return DeformedGeometry(self.geometry, scale=scale)

    def _make_cyclic_geometry(
        self, geometry: Geometry | None = None
    ) -> list[Geometry | DeformedGeometry]:
        if geometry is None:
            geometry = self.geometry

        n = self._cycle_sym.n
        theta = self._cycle_sym.theta
        axis = self._cycle_sym.axis

        return cyclic_pattern(geometry, axis, theta, n, include_first=False)

    def plot(
        self,
        deformation_scale: float = 10.0,
        ax: Axes | None = None,
        *,
        stress: bool = False,
        deflection: bool = False,
        pattern: bool = False,
        **kwargs,
    ):
        """
        Plot the Result of the finite element analysis

        Parameters
        ----------
        deformation_scale:
            The scale of the deformation to be shown in the plot
        ax:
            The Axes onto which to plot (should be 3-D).
        stress:
            Whether or not to plot stresses [color map]
        deflection:
            Whether or not to plot deflection [color map]
        pattern:
            Whether or not to pattern the model (if symmetry was used)
        """
        if ax is None:
            ax = Plot3D()

        dg = self.make_deformed_geometry(deformation_scale)

        if stress:
            dg.plot(ax, stress=self._max_stresses, **kwargs)
        elif deflection:
            dg.plot(ax, stress=self._max_deflections, **kwargs)

        else:
            self.geometry.plot(ax)
            dg.plot(ax)

        if pattern:
            pdg = self._make_cyclic_geometry(dg)
            pdg.plot(ax, **kwargs)
