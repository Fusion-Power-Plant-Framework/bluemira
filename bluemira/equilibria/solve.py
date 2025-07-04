# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Picard iteration procedures for equilibria (and their infinite variations)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.look_and_feel import (
    bluemira_print,
    bluemira_print_flush,
    bluemira_warn,
)
from bluemira.equilibria.constants import PSI_REL_TOL
from bluemira.equilibria.diagnostics import PicardDiagnosticOptions
from bluemira.optimisation.error import OptimisationError

if TYPE_CHECKING:
    from collections.abc import Iterator

    import numpy.typing as npt

    from bluemira.equilibria.equilibrium import Equilibrium
    from bluemira.equilibria.optimisation.problem import CoilsetOptimisationProblem
    from bluemira.equilibria.optimisation.problem.base import CoilsetOptimiserResult

__all__ = [
    "CunninghamConvergence",
    "DudsonConvergence",
    "JeonConvergence",
    "JrelConvergence",
    "JsourceConvergence",
    "PicardIterator",
]


class ConvergenceCriterion(ABC):
    """
    Convergence criterion base class

    Parameters
    ----------
    limit:
        The limit at which the convergence criterion is met.
    """

    flag_psi: bool = True

    def __init__(self, limit: float, math_string: str):
        self.limit: float = limit
        self.progress: list[float] = []
        self.math_string: str = math_string
        self._relax: int = 0

    @property
    def relax_steps(self):
        """Number of relaxation steps"""
        return self._relax

    @relax_steps.setter
    def relax_steps(self, value: int):
        """Number of relaxation steps"""
        self._relax = int(value)

    @abstractmethod
    def __call__(
        self,
        old_val: npt.NDArray[np.float64],
        new_val: npt.NDArray[np.float64],
        i: int,
        *,
        print_status: bool = True,
    ) -> bool:
        """
        Carry out convergence check.

        Parameters
        ----------
        old_val:
            The value from the previous iteration.
        new_val:
            The value from the current iteration.
        i:
            The index of the iteration.
        print_status:
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        :
            True if the convergence criterion is met, else False.
        """

    def check_converged(self, value: float) -> bool:
        """
        Check for convergence.

        Parameters
        ----------
        value:
            The value of the convergence criterion

        Returns
        -------
        Whether or not convergence has been reached
        """
        self.progress.append(value)
        return value <= self.limit

    def plot(self, ax: plt.Axes | None = None):
        """
        Plot the convergence behaviour.

        Parameters
        ----------
        ax:
            The matplotlib axes onto which to plot
        """
        if ax is None:
            _f, ax = plt.subplots()
        rl = self.relax_steps + 1
        ax.semilogy([0, len(self.progress) // rl], [self.limit, self.limit])
        ax.semilogy(np.arange(len(self.progress)) / rl, self.progress)
        ax.grid(visible=True, which="both")
        ax.set_xlabel("Iterations [n]")
        ax.set_ylabel(self.math_string)


class DudsonConvergence(ConvergenceCriterion):
    """
    FreeGS convergence criterion

    Parameters
    ----------
    limit:
        The limit at which the convergence criterion is met.
    """

    def __init__(self, limit: float = PSI_REL_TOL):
        super().__init__(limit, "$\\dfrac{max|\\Delta\\psi|}{max(\\psi)-min(\\psi)}$")

    def __call__(
        self,
        psi_old: npt.NDArray[np.float64],
        psi: npt.NDArray[np.float64],
        i: int,
        *,
        print_status: bool = True,
    ) -> bool:
        """
        Carry out convergence check.

        Parameters
        ----------
        psi_old:
            The value from the previous iteration.
        psi:
            The value from the current iteration.
        i:
            The index of the iteration.
        print_status:
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        True if the convergence criterion is met, else False.
        """
        dpsi = psi_old - psi
        dpsi_max = np.amax(abs(dpsi))
        dpsi_rel = dpsi_max / (np.amax(psi) - np.amin(psi))
        if print_status:
            bluemira_print_flush(
                f"EQUILIBRIA G-S iter {i}: relative delta_psi: {100 * dpsi_rel:.2f} %"
            )
        return self.check_converged(dpsi_rel)


class JrelConvergence(ConvergenceCriterion):
    """
    FreeGS convergence criterion

    Parameters
    ----------
    limit:
        The limit at which the convergence criterion is met.
    """

    flag_psi = False

    def __init__(self, limit: float = 1e-2):
        super().__init__(limit, "$\\dfrac{max|\\Delta J|}{max(J)-min(J)}$")

    def __call__(
        self,
        j_old: npt.NDArray[np.float64],
        j_new: npt.NDArray[np.float64],
        i: int,
        *,
        print_status: bool = True,
    ) -> bool:
        """
        Carry out convergence check.

        Parameters
        ----------
        j_old:
            The value from the previous iteration.
        j_new:
            The value from the current iteration.
        i:
            The index of the iteration.
        print_status:
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        True if the convergence criterion is met, else False.
        """
        d_j = j_old - j_new
        d_j_max = np.amax(abs(d_j))
        d_j_rel = d_j_max / (np.amax(j_new) - np.amin(j_new))
        if print_status:
            bluemira_print_flush(
                f"EQUILIBRIA G-S iter {i}: relative delta_J: {100 * d_j_rel:.2f} %"
            )
        return self.check_converged(d_j_rel)


class LacknerConvergence(ConvergenceCriterion):
    """
    Karl Lackner's convergence criterion (< 10E-4)
    (Lackner, Computation of ideal MHD equilibria, 1976)

    Parameters
    ----------
    limit:
        The limit at which the convergence criterion is met.
    """

    def __init__(self, limit: float = 10e-4):
        super().__init__(limit, "$max\\dfrac{|\\Delta\\psi|}{\\psi}$")

    def __call__(
        self,
        psi_old: npt.NDArray[np.float64],
        psi: npt.NDArray[np.float64],
        i: int,
        *,
        print_status: bool = True,
    ) -> bool:
        """
        Carry out convergence check.

        Parameters
        ----------
        psi_old:
            The value from the previous iteration.
        psi:
            The value from the current iteration.
        i:
            The index of the iteration.
        print_status:
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        True if the convergence criterion is met, else False.
        """
        conv = np.amax(np.abs((psi - psi_old) / psi))
        if print_status:
            bluemira_print_flush(f"EQUILIBRIA G-S iter {i}: psi convergence: {conv:e}")
        return self.check_converged(conv)


class JeonConvergence(ConvergenceCriterion):
    """
    TES convergence criterion

    Parameters
    ----------
    limit:
        The limit at which the convergence criterion is met.
    """

    def __init__(self, limit: float = 1e-4):
        super().__init__(limit, "$||\\Delta\\psi||$")

    def __call__(
        self,
        psi_old: npt.NDArray[np.float64],
        psi: npt.NDArray[np.float64],
        i: int,
        *,
        print_status: bool = True,
    ) -> bool:
        """
        Carry out convergence check.

        Parameters
        ----------
        psi_old:
            The value from the previous iteration.
        psi:
            The value from the current iteration.
        i:
            The index of the iteration.
        print_status:
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        True if the convergence criterion is met, else False.
        """
        conv = np.linalg.norm(psi_old - psi)
        if print_status:
            bluemira_print_flush(
                f"EQUILIBRIA G-S iter {i}: psi norm convergence: {conv:e}"
            )
        return self.check_converged(conv)


class CunninghamConvergence(ConvergenceCriterion):
    """
    FIESTA convergence criterion

    Parameters
    ----------
    limit:
        The limit at which the convergence criterion is met.
    """

    flag_psi = False

    def __init__(self, limit: float = 1e-7):
        super().__init__(
            limit, "$\\dfrac{\\sum{\\Delta J_{n}^{2}}}{\\sum{J_{n+1}^{2}}}$"
        )

    def __call__(
        self,
        j_old: npt.NDArray[np.float64],
        j_new: npt.NDArray[np.float64],
        i: int,
        *,
        print_status: bool = True,
    ) -> bool:
        """
        Carry out convergence check.

        Parameters
        ----------
        j_old:
            The value from the previous iteration.
        j_new:
            The value from the current iteration.
        i:
            The index of the iteration.
        print_status:
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        True if the convergence criterion is met, else False.
        """
        d_j = j_old - j_new
        conv = np.sum(d_j**2) / np.sum(j_new**2)
        if print_status:
            bluemira_print_flush(
                f"EQUILIBRIA G-S iter {i}: J_phi source convergence: {conv:e}"
            )
        self._conv = conv
        return self.check_converged(conv)


class JsourceConvergence(ConvergenceCriterion):
    """
    Plasma current source convergence criterion.

    Parameters
    ----------
    limit:
        The limit at which the convergence criterion is met.
    """

    flag_psi = False

    def __init__(self, limit: float = 1e-4):
        super().__init__(limit, "$||\\Delta J||$")

    def __call__(
        self,
        j_old: npt.NDArray[np.float64],
        j_new: npt.NDArray[np.float64],
        i: int,
        *,
        print_status: bool = True,
    ) -> bool:
        """
        Carry out convergence check.

        Parameters
        ----------
        j_old:
            The value from the previous iteration.
        j_new:
            The value from the current iteration.
        i:
            The index of the iteration.
        print_status:
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        True if the convergence criterion is met, else False.
        """
        conv = np.linalg.norm(j_old - j_new)
        if print_status:
            # Format convergence
            bluemira_print_flush(
                f"EQUILIBRIA G-S iter {i}: ||J_phi_old-J_phi|| convergence: {conv:e}"
            )
        return self.check_converged(conv)


class PicardIterator:
    """A Picard iterative solver.

    Child classes must provide a __call__ method which carries out the
    iteration process(es)

    Parameters
    ----------
    eq:
        The equilibrium to solve for
    optimisation_problem:
        The optimisation problem to use when iterating
    convergence:
        The convergence criterion to use (defaults to Dudson)
    fixed_coils:
        Whether or not the coil positions are fixed
    relaxation:
        The relaxation parameter to use between iterations
    maxiter:
        The maximum number of iterations
    diagnostic_plotting:
        PicardDiagnosticOptions - dataclass containing plot type or not plot option,
        whether or not to make a GIF, the path where figures will be saved
        and the plot name.
    """

    def __init__(
        self,
        eq: Equilibrium,
        optimisation_problem: CoilsetOptimisationProblem,
        diagnostic_plotting: PicardDiagnosticOptions | None = None,
        convergence: ConvergenceCriterion | None = None,
        *,
        fixed_coils: bool = False,
        relaxation: float = 0,
        maxiter: int = 30,
    ):
        self.eq = eq
        self.coilset = self.eq.coilset
        self.opt_prob = optimisation_problem
        if isinstance(convergence, ConvergenceCriterion):
            self.convergence = convergence
        elif convergence is None:
            self.convergence = DudsonConvergence()
        else:
            raise ValueError(
                "Optimiser convergence specification must be a sub-class of"
                " ConvergenceCriterion."
            )
        self.convergence.relax_steps = 1
        self.fixed_coils = fixed_coils

        self.relaxation = relaxation
        self.maxiter = maxiter
        self.diagnostic_plotting = diagnostic_plotting or PicardDiagnosticOptions()
        self.i = 0

    def _optimise_coilset(self):
        self.result = None
        try:
            self.result = self.opt_prob.optimise(fixed_coils=self.fixed_coils)
            self.coilset = self.result.coilset
        except OptimisationError:
            self.coilset = None

    @property
    def psi(self) -> npt.NDArray[np.float64]:
        """
        Get the magnetic flux array.
        """
        return self._psi

    @property
    def j_tor(self) -> npt.NDArray[np.float64]:
        """
        Get the toroidal current density array.
        """
        return self._j_tor

    def __call__(self) -> CoilsetOptimiserResult:
        """
        The iteration object call handle.

        Returns
        -------
        :
            The result
        """
        iterator = iter(self)
        while self.i < self.maxiter:
            try:
                next(iterator)
            except StopIteration:  # noqa: PERF203
                bluemira_print("EQUILIBRIA G-S converged value found.")
                break
        else:
            bluemira_warn(
                "EQUILIBRIA G-S unable to find converged value after"
                f" {self.i} iterations."
            )
        self._teardown()
        return self.result

    def __iter__(self) -> Iterator:
        """
        Make the class a Python iterator.

        Returns
        -------
        The current instance as an iterator.
        """
        self._setup()
        return self

    def __next__(self):
        """
        Perform an iteration of the solver.

        Raises
        ------
        StopIteration
            if converged
        """
        if not hasattr(self, "_psi"):
            self._setup()

        if self.i > 0 and self.check_converged(print_status=False):
            self.diagnostic_plotting.finalise_plots()
            raise StopIteration

        self._psi_old = self.psi.copy()
        self._j_tor_old = self.j_tor.copy()
        self._solve()
        self._psi = self.eq.psi()
        self._j_tor = self.eq._jtor
        check = self.check_converged()
        self.diagnostic_plotting.update_figure(self.eq, self.convergence, self.i)
        if check:
            self.diagnostic_plotting.make_gif()
            raise StopIteration
        self._optimise_coilset()
        self._psi = (
            1 - self.relaxation
        ) * self.eq.psi() + self.relaxation * self._psi_old
        self.i += 1

    def iterate_once(self) -> CoilsetOptimiserResult:
        """
        Perform a single iteration and handle convergence.

        Returns
        -------
        :
            The result

        Raises
        ------
        StopIteration
            if converged
        """
        try:
            next(self)
        except StopIteration:
            bluemira_print("EQUILIBRIA G-S converged value found, nothing to do.")
            self._teardown()
        return self.result

    def check_converged(self, *, print_status: bool = True) -> bool:
        """
        Check if the iterator has converged.

        Parameters
        ----------
        print_status:
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        True if the convergence criterion is met, else False.
        """
        if self.convergence.flag_psi:
            return self.convergence(
                self._psi_old, self.psi, self.i, print_status=print_status
            )

        return self.convergence(
            self._j_tor_old, self.j_tor, self.i, print_status=print_status
        )

    def _solve(self):
        """
        Solve for this iteration.
        """
        if self.eq._li_flag and self.i > self.eq.profiles._l_i_min_iter:
            self.eq.solve_li(psi=self.psi)
        else:
            self.eq.solve(psi=self.psi)

    def _initial_optimise_coilset(self, **kwargs):
        self._optimise_coilset(**kwargs)

    def _setup(self):
        """
        Initialise psi and toroidal current values.
        """
        self._initial_optimise_coilset()
        self._psi = self.eq.psi()
        self._j_tor = self.eq._jtor
        if self._j_tor is None:
            self._j_tor = np.zeros((self.eq.grid.nx, self.eq.grid.nz))

    def _teardown(self):
        """Final clean-up for consistency between psi and jtor.

        In the case of converged equilibria, slight (artificial) improvement
        in consistency. In the case of unconverged equilibria, gives a more
        reasonable understanding of the final state.
        """
        o_points, x_points = self.eq.get_OX_points(force_update=True)
        self.eq._jtor = self.eq.profiles.jtor(
            self.eq.x, self.eq.z, self.eq.psi(), o_points, x_points
        )
