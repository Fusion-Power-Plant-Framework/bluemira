# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
Picard iteration procedures for equilibria (and their infinite variations)
"""
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

from bluemira.utilities.error import ExternalOptError
from bluemira.utilities.plot_tools import save_figure, make_gif
from bluemira.equilibria.constants import (
    PSI_REL_TOL,
    DPI_GIF,
    PLT_PAUSE,
)
from bluemira.base.look_and_feel import (
    bluemira_print_flush,
    bluemira_print,
    bluemira_warn,
)
from bluemira.base.file import try_get_bluemira_path

__all__ = [
    "DudsonConvergence",
    "CunninghamConvergence",
    "JsourceConvergence",
    "JeonConvergence",
    "JrelConvergence",
    "PicardLiAbsIterator",
    "PicardAbsIterator",
    "PicardDeltaIterator",
    "PicardLiDeltaIterator",
]


class ConvergenceCriterion(ABC):
    """
    Convergence criterion base class

    Parameters
    ----------
    limit: float
        The limit at which the convergence criterion is met.
    """

    flag_psi = True

    def __init__(self, limit):
        self.limit = limit
        self.progress = []
        self.math_string = NotImplemented

    @abstractmethod
    def __call__(self, old_val, new_val, i, print_status=True):
        """
        Carry out convergence check.

        Parameters
        ----------
        old_val: List[float]
            The value from the previous iteration.
        new_val: List[float]
            The value from the current iteration.
        i: int
            The index of the iteration.
        print_status: bool, optional
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        converged: bool
            True if the convergence criterion is met, else False.
        """
        pass

    def check_converged(self, value):
        """
        Check for convergence.

        Parameters
        ----------
        value: float
            The value of the convergence criterion

        Returns
        -------
        converged: bool
            Whether or not convergence has been reached
        """
        self.progress.append(value)
        if value <= self.limit:
            return True
        return False

    def plot(self, ax=None):
        """
        Plot the convergence behaviour.

        Parameters
        ----------
        ax: Axes
            The matplotlib axes onto which to plot
        """
        if ax is None:
            f, ax = plt.subplots()
        ax.semilogy(self.progress)
        ax.set_xlabel("Iterations [n]")
        ax.set_ylabel(self.math_string)


class DudsonConvergence(ConvergenceCriterion):
    """
    FreeGS convergence criterion

    Parameters
    ----------
    limit: float
        The limit at which the convergence criterion is met.
    """

    def __init__(self, limit=PSI_REL_TOL):
        super().__init__(limit)
        self.math_string = "$\\dfrac{max|\\Delta\\psi|}{max(\\psi)-min(\\psi)}$"

    def __call__(self, psi_old, psi, i, print_status=True):
        """
        Carry out convergence check.

        Parameters
        ----------
        psi_old: List[float]
            The value from the previous iteration.
        psi: List[float]
            The value from the current iteration.
        i: int
            The index of the iteration.
        print_status: bool, optional
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        converged: bool
            True if the convergence criterion is met, else False.
        """
        dpsi = psi_old - psi
        dpsi_max = np.amax(abs(dpsi))
        dpsi_rel = dpsi_max / (np.amax(psi) - np.amin(psi))
        if print_status:
            bluemira_print_flush(
                f"EQUILIBRIA G-S iter {i}: relative delta_psi: " f"{100*dpsi_rel:.2f} %"
            )
        return self.check_converged(dpsi_rel)


class JrelConvergence(ConvergenceCriterion):
    """
    FreeGS convergence criterion

    Parameters
    ----------
    limit: float
        The limit at which the convergence criterion is met.
    """

    flag_psi = False

    def __init__(self, limit=1e-2):
        super().__init__(limit)
        self.math_string = "$\\dfrac{max|\\Delta J|}{max(J)-min(J)}$"

    def __call__(self, j_old, j_new, i, print_status=True):
        """
        Carry out convergence check.

        Parameters
        ----------
        j_old: List[float]
            The value from the previous iteration.
        j_new: List[float]
            The value from the current iteration.
        i: int
            The index of the iteration.
        print_status: bool, optional
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        converged: bool
            True if the convergence criterion is met, else False.
        """
        d_j = j_old - j_new
        d_j_max = np.amax(abs(d_j))
        d_j_rel = d_j_max / (np.amax(j_new) - np.amin(j_new))
        if print_status:
            bluemira_print_flush(
                f"EQUILIBRIA G-S iter {i}: relative delta_J: " f"{100*d_j_rel:.2f} %"
            )
        return self.check_converged(d_j_rel)


class LacknerConvergence(ConvergenceCriterion):
    """
    Karl Lackner's convergence criterion (< 10E-4)
    (Lackner, Computation of ideal MHD equilibria, 1976)

    Parameters
    ----------
    limit: float
        The limit at which the convergence criterion is met.
    """

    def __init__(self, limit=10e-4):
        super().__init__(limit)
        self.math_string = "$max\\dfrac{|\\Delta\\psi|}{\\psi}$"

    def __call__(self, psi_old, psi, i, print_status=True):
        """
        Carry out convergence check.

        Parameters
        ----------
        psi_old: List[float]
            The value from the previous iteration.
        psi: List[float]
            The value from the current iteration.
        i: int
            The index of the iteration.
        print_status: bool, optional
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        converged: bool
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
    limit: float
        The limit at which the convergence criterion is met.
    """

    def __init__(self, limit=1e-4):
        super().__init__(limit)
        self.math_string = "$||\\Delta\\psi||$"

    def __call__(self, psi_old, psi, i, print_status=True):
        """
        Carry out convergence check.

        Parameters
        ----------
        psi_old: List[float]
            The value from the previous iteration.
        psi: List[float]
            The value from the current iteration.
        i: int
            The index of the iteration.
        print_status: bool, optional
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        converged: bool
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
    limit: float
        The limit at which the convergence criterion is met.
    """

    flag_psi = False

    def __init__(self, limit=1e-7):
        super().__init__(limit)
        self.math_string = "$\\dfrac{\\sum{\\Delta J_{n}^{2}}}{\\sum{J_{n+1}^{2}}}$"

    def __call__(self, j_old, j_new, i, print_status=True):
        """
        Carry out convergence check.

        Parameters
        ----------
        j_old: List[float]
            The value from the previous iteration.
        j_new: List[float]
            The value from the current iteration.
        i: int
            The index of the iteration.
        print_status: bool, optional
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        converged: bool
            True if the convergence criterion is met, else False.
        """
        d_j = j_old - j_new
        conv = np.sum(d_j ** 2) / np.sum(j_new)
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
    limit: float
        The limit at which the convergence criterion is met.
    """

    flag_psi = False

    def __init__(self, limit=1e-4):
        super().__init__(limit)
        self.math_string = "$||\\Delta J||$"

    def __call__(self, j_old, j_new, i, print_status=True):
        """
        Carry out convergence check.

        Parameters
        ----------
        j_old: List[float]
            The value from the previous iteration.
        j_new: List[float]
            The value from the current iteration.
        i: int
            The index of the iteration.
        print_status: bool, optional
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        converged: bool
            True if the convergence criterion is met, else False.
        """
        conv = np.linalg.norm(j_old - j_new)
        if print_status:
            # Format convergence
            bluemira_print_flush(
                f"EQUILIBRIA G-S iter {i}: ||J_phi_old-J_phi|| convergence: {conv:e}"
            )
        return self.check_converged(conv)


class CurrentOptimiser:
    """
    Mixin class for performing optimisation of currents
    """

    def _optimise_currents(self, psib=None, update_size=True):
        """
        Finds optimal currents for the coilset

        Parameters
        ----------
        psib: List[float], optional
            The boundary psi values, by default None.
        update_size: bool, optional
            If True then update the coilset size, by default True.
        """
        self.constraints(self.eq, I_not_dI=True)
        try:
            currents = self.optimiser(self.eq, self.constraints, psib)
            self.store.append(currents)
        except ExternalOptError:
            currents = self.store[-1]
        self.coilset.set_control_currents(currents, update_size)

    def _initial_optimise_currents(self, psib=None, update_size=True):
        """
        Finds optimal currents for the coilset for optimiser initialisation

        Parameters
        ----------
        psib: List[float], optional
            The boundary psi values, by default None.
        update_size: bool, optional
            If True then update the coilset size, by default True.
        """
        return self._optimise_currents(psib, update_size)


class CurrentGradientOptimiser:
    """
    Mixin class for performing optimisation of current gradients
    """

    def _optimise_currents(self, fixed_coils=False):
        """
        Finds an optimal dI for the coilset

        Parameters
        ----------
        fixed_coils: bool, optional
            If True then applies a fixed-coil constraint, by default False.
        """
        self.constraints(self.eq, fixed_coils=fixed_coils)
        try:
            d_current = self.optimiser(self.eq, self.constraints)
            self.store.append(d_current)
        except ExternalOptError:
            d_current = self.store[-1]
        self.coilset.adjust_currents(d_current)

    def _initial_optimise_currents(self, fixed_coils=False):
        """
        Finds an optimal dI for the coilset for optimiser initialisation

        Parameters
        ----------
        fixed_coils: bool, optional
            If True then applies a fixed-coil constraint, by default False.
        """
        return self._optimise_currents(fixed_coils)


class PicardBaseIterator(ABC):
    """
    Abstract base class for Picard iterative solvers

    Child classes must provide a __call__ method which carries out the
    iteration process(es)

    Parameters
    ----------
    eq: Equilibrium object
        The equilibrium to solve for
    profiles: Profile object
        The plasma profile object to solve with
    constraints: Constraint object
        The constraint to solve for
    optimiser: EquilibriumOptimiser object
        The optimiser to use
    convergence: ConvergenceCriterion
        The convergence criterion to use
    relaxation: float
        The relaxation parameter to use between iterations
    miniter: int
        The minimum number of iterations before using li optimisation
    maxiter: int
        The maximum number of iterations
    plot: bool
        Whether or not to plot
    gif: bool
        Whether or not to make a GIF
    figure_folder: str (default = None)
        The path where figures will be saved. If the input value is None (e.g. default)
        then this will be reinterpreted as the path data/plots/equilibria under the
        bluemira root folder, if that path is available.
    """

    def __init__(
        self,
        eq,
        profiles,
        constraints,
        optimiser,
        convergence=DudsonConvergence(),
        relaxation=0,
        miniter=5,
        maxiter=30,
        plot=True,
        gif=False,
        figure_folder=None,
        plot_name="default_0",
    ):
        self.eq = eq
        self.coilset = self.eq.coilset
        self.profiles = profiles
        self.constraints = constraints
        self.optimiser = optimiser
        if isinstance(convergence, ConvergenceCriterion):
            self.convergence = convergence
        else:
            raise ValueError(
                "Optimiser convergence specification must be a sub-class of ConvergenceCriterion."
            )
        self.relaxation = relaxation
        self.miniter = miniter
        self.maxiter = maxiter
        self.plot_flag = plot
        self.gif_flag = gif
        if figure_folder is None:
            figure_folder = try_get_bluemira_path(
                "plots/equilibria", subfolder="data", allow_missing=not self.gif_flag
            )
        self.figure_folder = figure_folder
        self.store = []
        self.i = 0
        if self.plot_flag:
            self.pname = plot_name
            self.f, self.ax = plt.subplots()

    @property
    @abstractmethod
    def current_optimiser_kwargs(self):
        """
        Get the kwargs for the current optimiser.

        Returns
        -------
        kwargs: dict
            The keyword arguments to use in the current optimiser.
        """
        return {}

    @property
    def psi(self):
        """
        Get the magnetic flux profile.

        Returns
        -------
        psi: List[float]
            The magnetic flux profile.
        """
        return self._psi

    @property
    def j_tor(self):
        """
        Get the toroidal current profile.

        Returns
        -------
        j_tor: List[float]
            The toroidal current profile profile.
        """
        return self._j_tor

    def __call__(self):
        """
        The iteration object call handle.
        """
        iterator = iter(self)
        while self.i < self.maxiter:
            try:
                next(iterator)
            except StopIteration:
                print()
                bluemira_print("EQUILIBRIA G-S converged value found.")
                break
        else:
            print()
            bluemira_warn(
                f"EQUILIBRIA G-S unable to find converged value after {self.i} iterations."
            )
        self._teardown()

    def __iter__(self):
        """
        Make the class a Python iterator.

        Returns
        -------
        iterator: Iterator
            The current instance as an iterator.
        """
        self._setup()
        return self

    def __next__(self):
        """
        Perform an interation of the solver.
        """
        if not hasattr(self, "_psi"):
            self._setup()

        if self.i > 0 and self.check_converged(print_status=False):
            raise StopIteration
        else:
            self._psi_old = self.psi.copy()
            self._j_tor_old = self.j_tor.copy()
            self._solve()
            self._psi = self.eq.psi()
            self._j_tor = self.eq._jtor
            check = self.check_converged()
            if self.plot_flag:
                self.update_fig()
            if check:
                if self.gif_flag:
                    make_gif(self.figure_folder, self.pname)
                raise StopIteration
            self._optimise_currents(**self.current_optimiser_kwargs)
            self._psi = (
                1 - self.relaxation
            ) * self.eq.psi() + self.relaxation * self._psi_old
            self.i += 1

    def iterate_once(self):
        """
        Perform a single iteration and handle convergence.
        """
        try:
            next(self)
        except StopIteration:
            bluemira_print("EQUILIBRIA G-S converged value found, nothing to do.")
            self._teardown()

    def check_converged(self, print_status=True):
        """
        Check if the iterator has converged.

        Parameters
        ----------
        print_status: bool, optional
            If True then prints the status of the convergence, by default True.

        Returns
        -------
        converged: bool
            True if the convergence criterion is met, else False.
        """
        if self.convergence.flag_psi:
            return self.convergence(
                self._psi_old, self.psi, self.i, print_status=print_status
            )
        else:
            return self.convergence(
                self._j_tor_old, self.j_tor, self.i, print_status=print_status
            )

    def update_fig(self):
        """
        Updates the figure if plotting is used
        """
        self.ax.clear()
        self.eq.plot(ax=self.ax)
        self.ax.figure.canvas.draw()
        save_figure(
            self.f,
            self.pname + str(self.i),
            save=self.gif_flag,
            folder=self.figure_folder,
            dpi=DPI_GIF,
        )
        plt.pause(PLT_PAUSE)

    @abstractmethod
    def _solve(self):
        """
        Solve for this iteration.
        """
        pass

    @abstractmethod
    def _optimise_currents(self, **kwargs):
        pass

    @abstractmethod
    def _initial_optimise_currents(self, **kwargs):
        pass

    def _setup(self):
        """
        Initialise psi and toroidal current values.
        """
        self._initial_optimise_currents(**self.current_optimiser_kwargs)
        self._psi = self.eq.psi()
        self._j_tor = self.eq._jtor
        if self._j_tor is None:
            self._j_tor = np.zeros((self.eq.grid.nx, self.eq.grid.nz))

    def _teardown(self):
        """
        Final clean-up to have consistency between psi and jtor
        In the case of converged equilibria, slight (artificial) improvement
        in consistency. In the case of unconverged equilibria, gives a more
        reasonable understanding of the final state.
        """
        self.eq.get_OX_points(force_update=True)
        self.eq._jtor = self.profiles.jtor(
            self.eq.x, self.eq.z, self.eq.psi(), self.eq._o_points, self.eq._x_points
        )


class PicardDeltaIterator(CurrentGradientOptimiser, PicardBaseIterator):
    """
    Picard solver for unconstrained li using dI iteration
    """

    @property
    def current_optimiser_kwargs(self):
        """
        Get the kwargs for the current optimiser.
        """
        return {"fixed_coils": False}

    def _solve(self):
        """
        Solve for this iteration.
        """
        self.eq.solve(self.profiles, psi=self.psi)


class PicardLiDeltaIterator(CurrentGradientOptimiser, PicardBaseIterator):
    """
    Picard solver for constrained plasma profiles (li) using dI iteration
    """

    @property
    def current_optimiser_kwargs(self):
        """
        Get the kwargs for the current optimiser.
        """
        return {"fixed_coils": False}

    def _solve(self):
        """
        Solve for this iteration.
        """
        if self.i < self.miniter:  # free internal plasma control
            self.eq.solve(self.profiles, psi=self.psi)
        else:  # constrain plasma li
            self.coilset.mesh_coils(d_coil=0.4)
            self.eq._remap_greens()
            self.eq.solve_li(self.profiles, psi=self.psi)


class PicardAbsIterator(CurrentOptimiser, PicardBaseIterator):
    """
    Picard solver for unconstrained plasma profiles (li) using I iteration.
    Best used for constrained coil optimisation
    """

    @property
    def current_optimiser_kwargs(self):
        """
        Get the kwargs for the current optimiser.
        """
        return {"psib": None, "update_size": True}

    def _solve(self):
        """
        Solve for this iteration.
        """
        self.coilset.mesh_coils(d_coil=0.4)
        self.eq._remap_greens()
        self.eq.solve(self.profiles, psi=self.psi)


class PicardLiAbsIterator(CurrentOptimiser, PicardBaseIterator):
    """
    Picard solver for constrained plasma profiles (li) using I iteration.
    Best used for constrained coil optimisation
    """

    @property
    def current_optimiser_kwargs(self):
        """
        Get the kwargs for the current optimiser.
        """
        return {"psib": None, "update_size": True}

    def _solve(self):
        """
        Solve for this iteration.
        """
        if self.i < self.miniter:  # free internal plasma control
            self.eq.solve(self.profiles, psi=self.psi)
        else:  # constrain plasma li
            self.coilset.mesh_coils(d_coil=0.4)
            self.eq._remap_greens()
            self.eq.solve_li(self.profiles, psi=self.psi)


class EquilibriumConverger(CurrentOptimiser, PicardBaseIterator):
    """
    Tool used to converge equilibria with fixed coil sizes for a given boundary
    flux
    """

    @property
    def current_optimiser_kwargs(self):
        """
        Get the kwargs for the current optimiser.
        """
        return {"update_size": False}

    def __call__(self, psib):
        """
        The iteration object call handle.
        """
        self.psib = psib
        super().__call__()
        return self.eq, self.profiles

    def _solve(self):
        """
        Solve for this iteration.
        """
        self.eq.solve_li(self.profiles, psi=self.psi)

    def _optimise_currents(self, **kwargs):
        super()._optimise_currents(self.psib, **self.current_optimiser_kwargs)

    def _initial_optimise_currents(self, **kwargs):
        super()._optimise_currents(self.psib, update_size=False)
