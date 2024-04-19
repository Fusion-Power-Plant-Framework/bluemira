# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import abc

import numpy as np
import numpy.typing as npt

from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Breakdown
from bluemira.equilibria.optimisation.constraints import (
    FieldConstraints,
    UpdateableConstraint,
)
from bluemira.equilibria.optimisation.objectives import MaximiseFluxObjective
from bluemira.equilibria.optimisation.problem.base import (
    CoilsetOptimisationProblem,
    CoilsetOptimiserResult,
)
from bluemira.optimisation import Algorithm, AlgorithmType, optimise


class BreakdownZoneStrategy(abc.ABC):
    """
    Abstract base class for the definition of a breakdown zone strategy.

    Parameters
    ----------
    R_0:
        Major radius of the reference plasma
    A:
        Aspect ratio of the reference plasma
    tk_sol:
        Thickness of the scrape-off layer
    """

    def __init__(self, R_0, A, tk_sol, **kwargs):  # noqa: ARG002
        self.R_0 = R_0
        self.A = A
        self.tk_sol = tk_sol

    @abc.abstractproperty
    def breakdown_point(self) -> tuple[float, float]:
        """
        The location of the breakdown point.

        Returns
        -------
        x_c:
            Radial coordinate of the breakdown point
        z_c:
            Vertical coordinate of the breakdown point
        """

    @abc.abstractproperty
    def breakdown_radius(self) -> float:
        """
        The radius of the breakdown zone.
        """

    @abc.abstractmethod
    def calculate_zone_points(self, n_points: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the discretised set of points representing the breakdown zone.
        """


class CircularZoneStrategy(BreakdownZoneStrategy):
    """
    Circular breakdown zone strategy.
    """

    def calculate_zone_points(self, n_points: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate the discretised set of points representing the breakdown zone.
        """
        x_c, z_c = self.breakdown_point
        r_c = self.breakdown_radius
        theta = np.linspace(0, 2 * np.pi, n_points - 1, endpoint=False)
        x = x_c + r_c * np.cos(theta)
        z = z_c + r_c * np.sin(theta)
        x = np.append(x, x_c)
        z = np.append(z, z_c)
        return x, z


class InboardBreakdownZoneStrategy(CircularZoneStrategy):
    """
    Inboard breakdown zone strategy.
    """

    @property
    def breakdown_point(self) -> tuple[float, float]:
        """
        The location of the breakdown point.

        Returns
        -------
        x_c:
            Radial coordinate of the breakdown point
        z_c:
            Vertical coordinate of the breakdown point
        """
        r_c = self.breakdown_radius
        x_c = self.R_0 - self.R_0 / self.A - self.tk_sol + r_c
        z_c = 0.0
        return x_c, z_c

    @property
    def breakdown_radius(self) -> float:
        """
        The radius of the breakdown zone.
        """
        return 0.5 * self.R_0 / self.A


class OutboardBreakdownZoneStrategy(CircularZoneStrategy):
    """
    Outboard breakdown zone strategy.
    """

    @property
    def breakdown_point(self) -> tuple[float, float]:
        """
        The location of the breakdown point.

        Returns
        -------
        x_c:
            Radial coordinate of the breakdown point
        z_c:
            Vertical coordinate of the breakdown point
        """
        r_c = self.breakdown_radius
        x_c = self.R_0 + self.R_0 / self.A + self.tk_sol - r_c
        z_c = 0.0
        return x_c, z_c

    @property
    def breakdown_radius(self) -> float:
        """
        The radius of the breakdown zone.
        """
        return 0.7 * self.R_0 / self.A


class InputBreakdownZoneStrategy(CircularZoneStrategy):
    """
    User input breakdown zone strategy.
    """

    def __init__(self, x_c, z_c, r_c):
        self.x_c = x_c
        self.z_c = z_c
        self.r_c = r_c

    @property
    def breakdown_point(self) -> tuple[float, float]:
        """
        The location of the breakdown point.

        Returns
        -------
        x_c:
            Radial coordinate of the breakdown point
        z_c:
            Vertical coordinate of the breakdown point
        """
        return self.x_c, self.z_c

    @property
    def breakdown_radius(self) -> float:
        """
        The radius of the breakdown zone.
        """
        return self.r_c


class BreakdownCOP(CoilsetOptimisationProblem):
    """
    Coilset optimisation problem for the premagnetisation / breakdown phase.
    """

    def __init__(
        self,
        coilset: CoilSet,
        breakdown: Breakdown,
        breakdown_strategy: BreakdownZoneStrategy,
        B_stray_max: float,
        B_stray_con_tol: float,
        n_B_stray_points: int,
        max_currents: npt.ArrayLike | None = None,
        opt_algorithm: AlgorithmType = Algorithm.SLSQP,
        opt_conditions: dict[str, float | int] | None = None,
        constraints: list[UpdateableConstraint] | None = None,
    ):
        self.coilset = coilset
        self.eq = breakdown
        self.opt_algorithm = opt_algorithm
        self.opt_conditions = opt_conditions
        self.bounds = self.get_current_bounds(self.coilset, max_currents, self.scale)

        self._args = {
            "c_psi_mat": np.array(
                coilset.psi_response(*breakdown_strategy.breakdown_point)
            ),
            "scale": self.scale,
        }

        x_zone, z_zone = breakdown_strategy.calculate_zone_points(n_B_stray_points)
        stray_field_cons = FieldConstraints(
            x_zone, z_zone, B_max=B_stray_max, tolerance=B_stray_con_tol
        )
        self._constraints = constraints
        if self._constraints is not None:
            self._constraints.append(stray_field_cons)
        else:
            self._constraints = [stray_field_cons]

        max_currents = np.atleast_1d(max_currents)
        self.bounds = (-max_currents / self.scale, max_currents / self.scale)

    def optimise(self, x0=None, *, fixed_coils=True):
        """
        Solve the optimisation problem.
        """
        self.update_magnetic_constraints(I_not_dI=True, fixed_coils=fixed_coils)

        if x0 is None:
            cs_opt_state = self.coilset.get_optimisation_state(current_scale=self.scale)
            x0 = np.clip(cs_opt_state.currents, *self.bounds)
        else:
            x0 = np.clip(x0 / self.scale, *self.bounds)

        objective = MaximiseFluxObjective(**self._args)
        eq_constraints, ineq_constraints = self._make_numerical_constraints()
        opt_result = optimise(
            f_objective=objective.f_objective,
            df_objective=objective.df_objective,
            x0=x0,
            algorithm=self.opt_algorithm,
            opt_conditions=self.opt_conditions,
            bounds=self.bounds,
            eq_constraints=eq_constraints,
            ineq_constraints=ineq_constraints,
        )

        opt_currents = opt_result.x
        self.coilset.set_optimisation_state(
            opt_currents=opt_currents,
            current_scale=self.scale,
        )

        return CoilsetOptimiserResult.from_opt_result(self.coilset, opt_result)
