# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from bluemira.base.look_and_feel import bluemira_print, bluemira_warn


@dataclass
class NelderMeadParams:
    """Options for Nelder-Mead."""

    maxiter: int | None = None
    disp: bool | None = None
    maxfev: int | None = None
    return_all: bool | None = None
    initial_simplex: Any | None = None
    xatol: float | None = None
    fatol: float | None = None
    adaptive: bool | None = None


@dataclass
class PowellParams:
    """Options for Powell."""

    maxiter: int | None = None
    disp: bool | None = None
    xtol: float | None = None
    ftol: float | None = None
    maxfev: int | None = None
    direc: Any | None = None
    return_all: bool | None = None


@dataclass
class LBFGSBParams:
    """Options for L-BFGS-B."""

    maxiter: int | None = None
    disp: int | None = None  # deprecated - will be removed in v1.18.0
    maxcor: int | None = None
    ftol: float | None = None
    gtol: float | None = None
    eps: float | Any | None = None
    maxfun: int | None = None
    iprint: int | None = None
    maxls: int | None = None
    finite_diff_rel_step: Any | None = None
    workers: int | Any | None = None


@dataclass
class TNCParams:
    """Options for TNC."""

    maxiter: int | None = None
    eps: float | Any | None = None
    scale: list[float] | None = None
    offset: float | None = None
    disp: bool | None = None
    maxCGit: int | None = None  # noqa: N815
    eta: float | None = None
    stepmx: float | None = None
    accuracy: float | None = None
    minfev: float | None = None
    ftol: float | None = None
    xtol: float | None = None
    gtol: float | None = None
    rescale: float | None = None
    finite_diff_rel_step: Any | None = None
    maxfun: int | None = None
    workers: int | Any | None = None


@dataclass
class COBYLAParams:
    """Options for COBYLA."""

    maxiter: int | None = None
    rhobeg: float | None = None
    tol: float | None = None
    disp: int | None = None
    catol: float | None = None
    f_target: float | None = None


@dataclass
class COBYQAParams:
    """Options for COBYQA."""

    maxiter: int | None = None
    disp: bool | None = None
    maxfev: int | None = None
    f_target: float | None = None
    feasibility_tol: float | None = None
    initial_tr_radius: float | None = None
    final_tr_radius: float | None = None
    scale: bool | None = None


@dataclass
class SLSQPParams:
    """Options for SLSQP."""

    maxiter: int | None = None
    ftol: float | None = None
    eps: float | None = None
    disp: bool | None = None
    finite_diff_rel_step: Any | None = None
    workers: int | Any | None = None


@dataclass
class TrustConstrParams:
    """Options for Trust-Constr."""

    maxiter: int | None = None
    gtol: float | None = None
    xtol: float | None = None
    barrier_tol: float | None = None
    sparse_jacobian: bool | None = None
    initial_tr_radius: float | None = None
    initial_constr_penalty: float | None = None
    initial_barrier_parameter: float | None = None
    initial_barrier_tolerance: float | None = None
    factorization_method: str | None = None
    finite_diff_rel_step: Any | None = None
    verbose: int | None = None
    disp: bool | None = None
    workers: int | Any | None = None


def _make_alg_params(
    user_params: Mapping[str, int | float],
    param_cls: type,
    overrides: dict[str, str],
) -> Mapping[str, int | float]:
    """
    Algorithm parameter factory.

    Returns
    -------
    The dataclass associated with the given algorithm,
    with user parameters merged into the default parameters.
    """
    params = dict(user_params)

    if overrides:
        for old_name, new_name in overrides.items():
            if old_name in params:
                params[new_name] = params.pop(old_name)

    known_keys = set(asdict(param_cls()).keys())

    clean_params = {}
    extras = []

    for key, val in params.items():
        if key in known_keys:
            if val is not None:
                clean_params[key] = val
        else:
            clean_params[key] = val
            extras.append(key)

    if extras:
        bluemira_warn(
            f"Unknown parameters {extras} for {param_cls.__name__}. "
            "They will be passed through anyway."
        )
        bluemira_print(f"Available parameters are: {list(known_keys)}.")

    return clean_params


def _filter_to_scipy_options(
    params: Mapping[str, int | float],
    param_cls: type,
) -> Mapping[str, int | float]:
    """Strip *params* down to the keys ``param_cls`` declares.

    Used right before ``scipy.optimize.minimize`` so SciPy doesn't emit a
    duplicate ``OptimizeWarning: Unknown solver options`` on top of the
    bluemira_warn that ``_make_alg_params`` already raised. ``params``
    itself is preserved on the optimiser for user-facing transparency.

    Returns
    -------
    :
        ``params`` filtered down to keys present on ``param_cls``.
    """
    known_keys = set(asdict(param_cls()).keys())
    return {k: v for k, v in params.items() if k in known_keys}
