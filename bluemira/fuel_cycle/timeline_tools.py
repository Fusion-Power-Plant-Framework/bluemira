# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Distribution and timeline utilities
"""

import abc
from collections.abc import Iterable
from typing import Any, overload

import numpy as np
from numpy.random import Generator, SeedSequence
from scipy.optimize import brentq

from bluemira.base.constants import RNGSeeds
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.fuel_cycle.error import FuelCycleError

__all__ = [
    "ExponentialAvailabilityStrategy",
    "GompertzLearningStrategy",
    "LogNormalAvailabilityStrategy",
    "TruncNormAvailabilityStrategy",
    "UniformLearningStrategy",
    "UserSpecifiedLearningStrategy",
]


@overload
def f_gompertz(t: float, a: float, b: float, c: float) -> float: ...
@overload
def f_gompertz(t: np.ndarray, a: float, b: float, c: float) -> np.ndarray: ...
def f_gompertz(t: float | np.ndarray, a: float, b: float, c: float) -> Any:
    """
    Gompertz sigmoid function parameterisation.

    \t:math:`a\\text{exp}(-b\\text{exp}(-ct))`
    """  # noqa: DOC201
    return a * np.exp(-b * np.exp(-c * t))


def f_logistic(t: float, value: float, k: float, x_0: float) -> float:
    """
    Logistic function parameterisation.
    """  # noqa: DOC201
    return value / (1 + np.exp(-k * (t - x_0)))


def histify(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform values into arrays usable to make histograms.
    """  # noqa: DOC201
    x, y = np.array(x), np.array(y)
    return x.repeat(2)[1:-1], y.repeat(2)


def generate_lognorm_distribution(
    n: int, integral: float, sigma: float, rng: Generator
) -> np.ndarray:
    """
    Generate a log-norm distribution for a given standard deviation of the
    underlying normal distribution. The mean value of the normal distribution
    is optimised approximately.

    Parameters
    ----------
    n:
        The size of the distribution
    integral:
        The integral value of the distribution
    sigma:
        The standard deviation of the underlying normal distribution
    rng:
        Random number generator for lognormal distribution

    Returns
    -------
    :
        The distribution of size n and of the correct integral value
    """

    def f_integral(x):
        return np.sum(rng.lognormal(x, sigma, n)) - integral

    mu = brentq(f_integral, -1e3, 1e3, maxiter=200)
    distribution = rng.lognormal(mu, sigma, n)
    # Correct distribution integral
    error = np.sum(distribution) - integral
    distribution -= error / n
    return distribution


def generate_truncnorm_distribution(
    n: int, integral: float, sigma: float, rng: Generator
) -> np.ndarray:
    """
    Generate a truncated normal distribution for a given standard deviation.

    Parameters
    ----------
    n:
        The size of the distribution
    integral:
        The integral value of the distribution
    sigma:
        The standard deviation of the underlying normal distribution
    rng:
        random number generator for the log normal distribution

    Returns
    -------
    :
        The distribution of size n and of the correct integral value
    """
    distribution = rng.normal(0, sigma, n)
    # Truncate distribution by 0-folding
    distribution = np.abs(distribution)
    # Correct distribution integral
    distribution /= np.sum(distribution)
    distribution *= integral
    return distribution


def generate_exponential_distribution(
    n: int, integral: float, lambdda: float, rng: Generator
) -> np.ndarray:
    """
    Generate an exponential distribution for a given rate parameter.

    Parameters
    ----------
    n:
        The size of the distribution
    integral:
        The integral value of the distribution
    lambdda:
        The rate parameter of the distribution

    Returns
    -------
    :
        The distribution of size n and of the correct integral value
    """
    distribution = rng.exponential(lambdda, n)
    # Correct distribution integral
    distribution /= np.sum(distribution)
    distribution *= integral
    return distribution


class LearningStrategy(abc.ABC):
    """
    Abstract base class for learning strategies distributing the total operational
    availability over different operational phases.
    """

    @abc.abstractmethod
    def generate_phase_availabilities(
        self, lifetime_op_availability: float, op_durations: Iterable[float]
    ) -> Iterable[float]:
        """
        Generate operational availabilities for the specified phase durations.

        Parameters
        ----------
        lifetime_op_availability:
            Operational availability averaged over the lifetime
        op_durations:
            Durations of the operational phases [fpy]

        Returns
        -------
        :
            Operational availabilities at each operational phase
        """
        ...


class UniformLearningStrategy(LearningStrategy):
    """
    Uniform learning strategy
    """

    @staticmethod
    def generate_phase_availabilities(
        lifetime_op_availability: float, op_durations: Iterable[float]
    ) -> Iterable[float]:
        """
        Generate operational availabilities for the specified phase durations.

        Parameters
        ----------
        lifetime_op_availability:
            Operational availability averaged over the lifetime
        op_durations:
            Durations of the operational phases [fpy]

        Returns
        -------
        :
            Operational availabilities at each operational phase
        """
        return lifetime_op_availability * np.ones(len(list(op_durations)))


class UserSpecifiedLearningStrategy(LearningStrategy):
    """
    User-specified learning strategy to hard-code the operational availabilities at
    each operational phase.
    """

    def __init__(self, operational_availabilities: Iterable[float]):
        """
        Parameters
        ----------
        operational_availabilities:
            Operational availabilities to prescribe
        """
        self.operational_availabilities = list(operational_availabilities)

    def generate_phase_availabilities(
        self, lifetime_op_availability: float, op_durations: Iterable[float]
    ) -> Iterable[float]:
        """
        Generate operational availabilities for the specified phase durations.

        Parameters
        ----------
        lifetime_op_availability:
            Lifetime operational availability
        op_durations:
            Durations of the operational phases [fpy]

        Returns
        -------
        :
            Operational availabilities at each operational phase

        Raises
        ------
        FuelCycleError
            Number of phases should be equal to the number of operational availabilities
        """
        op_durs = np.asarray(list(op_durations))
        op_avails = np.asarray(self.operational_availabilities)
        if len(op_durs) != len(op_avails):
            raise FuelCycleError(
                "The number of phases is not equal to the number of user-specified"
                " operational availabilities."
            )

        total_fpy = float(np.sum(op_durs))
        fraction = (total_fpy / lifetime_op_availability) / (op_durs / op_avails)
        if not np.all(np.isclose(fraction, 1.0)):
            bluemira_warn(
                "User-specified operational availabilities do not match the specified"
                f" lifetime operational : {fraction} != 1.0. Normalising to adjust"
                " to meet the specified lifetime operational availability."
            )

        return fraction * op_avails


class GompertzLearningStrategy(LearningStrategy):
    """
    Gompertz learning strategy.
    """

    def __init__(
        self, learn_rate: float, min_op_availability: float, max_op_availability: float
    ):
        """
        Parameters
        ----------
        learn_rate:
            Gompertz distribution learning rate
        min_op_availability:
            Minimum operational availability within any given operational phase
        max_op_availability:
            Maximum operational availability within any given operational phase
        """
        self.learn_rate = learn_rate
        self.min_op_a = min_op_availability
        self.max_op_a = max_op_availability
        super().__init__()

    def _f_op_availabilities(
        self, t: np.ndarray, x: float, arg_dates: np.ndarray
    ) -> np.ndarray:
        a_ops: np.ndarray = self.min_op_a + f_gompertz(
            t, self.max_op_a - self.min_op_a, x, self.learn_rate
        )

        return np.array([
            float(np.mean(a_ops[int(arg_dates[i]) : int(d)]))
            for i, d in enumerate(arg_dates[1:])
        ])

    def generate_phase_availabilities(
        self, lifetime_op_availability: float, op_durations: Iterable[float]
    ) -> Iterable[float]:
        """
        Generate operational availabilities for the specified phase durations.

        Parameters
        ----------
        lifetime_op_availability:
            Operational availability averaged over the lifetime
        op_durations:
            Durations of the operational phases [fpy]

        Returns
        -------
        :
            Operational availabilities at each operational phase

        Raises
        ------
        FuelCycleError
            Input lifetimes must be in range
        """
        if not self.min_op_a < lifetime_op_availability < self.max_op_a:
            raise FuelCycleError(
                "Input lifetime operational availability must be within the specified"
                " bounds on the phase operational availability."
            )

        durations = np.append(0.0, np.asarray(op_durations, dtype=float))
        total_fpy = float(np.sum(durations))
        cum_fpy = np.cumsum(durations)

        t = np.linspace(0, total_fpy, 100)
        arg_dates = np.array([np.argmin(abs(t - i)) for i in cum_fpy])

        def f_opt(x):
            """
            Optimisation objective for chunky fit to Gompertz

            \t:math:`a_{min}+(a_{max}-a_{min})e^{\\dfrac{-\\text{ln}(2)}{e^{-ct_{infl}}}}`

            Returns
            -------
            :
                Objective
            """
            a_ops_i = self._f_op_availabilities(t, x, arg_dates)
            # NOTE: Fancy analytical integral objective of Gompertz function
            # was a resounding failure. Do not touch this again.
            # The brute force is strong in this one.
            return total_fpy / lifetime_op_availability - float(
                np.sum(durations[1:] / a_ops_i)
            )

        x_opt = brentq(f_opt, 0, 10e10)
        return self._f_op_availabilities(t, x_opt, arg_dates)


class OperationalAvailabilityStrategy(abc.ABC):
    """
    Abstract base class for operational availability strategies to generate
    distributions of unplanned outages.

    Parameters
    ----------
    rng_seed:
        random number generator seed for the operational availability
    """

    def __init__(self, rng_seed: int | SeedSequence):
        self.rng = np.random.default_rng(rng_seed)

    @abc.abstractmethod
    def generate_distribution(self, n: int, integral: float) -> np.ndarray:
        """
        Generate a distribution with a specified number of entries and integral.

        Parameters
        ----------
        n:
            Number of entries in the distribution
        integral:
            Integral of the distribution

        Returns
        -------
        :
            The distribution of size n and of the correct integral value
        """
        ...


class LogNormalAvailabilityStrategy(OperationalAvailabilityStrategy):
    """
    Log-normal distribution strategy

    Parameters
    ----------
    sigma:
        Standard deviation of the underlying normal distribution
    rng_seed:
        random number generator seed for the normal distribution
    """

    def __init__(
        self,
        sigma: float,
        rng_seed: int | SeedSequence = RNGSeeds.timeline_tools_lognorm.value,
    ):
        self.sigma = sigma
        super().__init__(rng_seed)

    def generate_distribution(self, n: int, integral: float) -> np.ndarray:
        """
        Generate a log-normal distribution with a specified number of entries and
        integral.

        Parameters
        ----------
        n:
            Number of entries in the distribution
        integral:
            Integral of the distribution

        Returns
        -------
        :
            The distribution of size n and of the correct integral value
        """
        return generate_lognorm_distribution(n, integral, self.sigma, self.rng)


class TruncNormAvailabilityStrategy(OperationalAvailabilityStrategy):
    """
    Truncated normal distribution strategy

    Parameters
    ----------
    sigma:
        Standard deviation of the underlying normal distribution
    rng_seed:
        random number generator seed for the normal distribution
    """

    def __init__(
        self,
        sigma: float,
        rng_seed: int | SeedSequence = RNGSeeds.timeline_tools_truncnorm.value,
    ):
        self.sigma = sigma
        super().__init__(rng_seed)

    def generate_distribution(self, n: int, integral: float) -> np.ndarray:
        """
        Generate a truncated normal distribution with a specified number of entries and
        integral.

        Parameters
        ----------
        n:
            Number of entries in the distribution
        integral:
            Integral of the distribution

        Returns
        -------
        :
            The distribution of size n and of the correct integral value
        """
        return generate_truncnorm_distribution(n, integral, self.sigma, self.rng)


class ExponentialAvailabilityStrategy(OperationalAvailabilityStrategy):
    """
    Exponential distribution strategy

    Parameters
    ----------
    lambdda:
        Rate of the distribution
    rng_seed:
        random number generator seed for the exponential distribution
    """

    def __init__(
        self,
        lambdda: float,
        rng_seed: int | SeedSequence = RNGSeeds.timeline_tools_expo.value,
    ):
        self.lambdda = lambdda
        super().__init__(rng_seed)

    def generate_distribution(self, n: int, integral: float) -> np.ndarray:
        """
        Generate an exponential distribution with a specified number of entries and
        integral.

        Parameters
        ----------
        n
            Number of entries in the distribution
        integral:
            Integral of the distribution

        Returns
        -------
        :
            The distribution of size n and of the correct integral value
        """
        return generate_exponential_distribution(n, integral, self.lambdda, self.rng)
