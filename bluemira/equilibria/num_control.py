# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Numerical vertical stability control - still not quite there!
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from bluemira.equilibria.coils import Coil, CoilGroup

if TYPE_CHECKING:
    from bluemira.equilibria.equilibrium import Equilibrium

__all__ = ["DummyController", "VirtualController"]


class DummyController:
    """
    Dummy control object to enable calculations to take place with no numerical
    vertical control scheme.

    psi() returns np.zeros(eq.psi.shape)
    """

    def __init__(self, psi: npt.NDArray[np.float64]):
        self._shape = psi.shape

    def stabilise(self, *args):
        """
        Dummy method to retain procedures with no effect on the equilibria.
        """

    def psi(self) -> npt.NDArray[np.float64]:
        """
        Dummy method to retain procedures with no effect on the equilibria.
        """  # noqa: DOC201
        return np.zeros(self._shape)

    @staticmethod
    def Bx(
        x: npt.ArrayLike,
        z: npt.ArrayLike,  # noqa: ARG004
    ) -> float | npt.NDArray[np.float64]:
        """
        Dummy method to retain procedures with no effect on the equilibria.
        """  # noqa: DOC201
        x_arr = np.asarray(x)
        if x_arr.ndim == 0:
            return 0.0
        return np.zeros_like(x_arr, dtype=np.float64)

    @staticmethod
    def Bz(
        x: npt.ArrayLike,
        z: npt.ArrayLike,  # noqa: ARG004
    ) -> float | npt.NDArray[np.float64]:
        """
        Dummy method to retain procedures with no effect on the equilibria.
        """  # noqa: DOC201
        x_arr = np.asarray(x)
        if x_arr.ndim == 0:
            return 0.0
        return np.zeros_like(x_arr, dtype=np.float64)


class VirtualController(CoilGroup):
    """
    Represents a pair of virtual coils for the numerical vertical control of
    the plasma, as described in :doi:`Jeon, 2015 <10.3938/jkps.67.843>`

    It does work to some extent (perhaps I've implemented it incorrectly). It
    seems to fall over for large numerical instabilities.
    """

    def __init__(self, eq: Equilibrium, gz: float = 1.5):
        self.eq = eq
        self.coilset = eq.coilset
        self.Xc = (self.eq.grid.x_min + self.eq.grid.x_max) / 2
        self.Zc = self.eq.grid.z_max + 2  # outside computational domain
        self.gz = gz
        self._pgreen = self.psi_response(self.eq.x, self.eq.z)
        super().__init__(
            Coil(self.Xc, self.Zc, current=1, name="V1", ctype="NONE"),
            Coil(self.Xc, -self.Zc, current=1, name="V2", ctype="NONE"),
        )

    def feedback_current(self) -> float:
        """
        Calculate feedback currents to compensate for a radial field at the
        centre of the plasma. (Vertical stability)

        \t:math:`I_{feedback}=-g_{z}\\dfrac{B_{X,vac}}{B_{X,feedback}}`
        \t:math:`\\Bigr|_{\\substack{X_{cur}, Z_{cur}}}`
        """  # noqa: DOC201
        xcur, zcur = self.eq.effective_centre()
        bx_vac = float(np.asarray(self.coilset.Bx(xcur, zcur)))
        bx_feedback = float(np.asarray(self.Bx_response(xcur, zcur)))
        return -self.gz * bx_vac / bx_feedback

    def adjust_currents(self, d_current: float | npt.NDArray[np.float64]):
        """
        Adjust the currents in the virtual control coils.
        """
        self.current = np.asarray(self.current) + d_current

    def stabilise(self):
        """
        Stabilise the equilibrium, calculating the feedback currents and applying
        them to the control coils.
        """
        currents = self.feedback_current()
        self.adjust_currents(currents)

    def psi(  # type: ignore[override]
        self,
        x: float | np.ndarray | None = None,
        z: float | np.ndarray | None = None,
    ) -> npt.NDArray[np.float64]:
        """
        Get the psi array of the VirtualController
        """  # noqa: DOC201
        if x is None and z is None:
            return np.asarray(self.current) * self._pgreen
        assert x is not None  # noqa: S101
        assert z is not None  # noqa: S101
        return np.asarray(super().psi(x, z))
