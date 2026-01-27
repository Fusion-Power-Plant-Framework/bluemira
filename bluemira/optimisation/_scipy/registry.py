# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from dataclasses import dataclass, field

from bluemira.optimisation._algorithm import Algorithm
from bluemira.optimisation._scipy.parameters import (
    COBYLAParams,
    COBYQAParams,
    LBFGSBParams,
    NelderMeadParams,
    PowellParams,
    SLSQPParams,
    TNCParams,
    TrustConstrParams,
)


@dataclass
class ScipyAlgConfig:
    """Configuration metadata for a specific SciPy algorithm."""

    alg_name: str
    param_cls: type
    condition_overrides: dict[str, str] = field(default_factory=dict)

    supports_grad: bool = False
    supports_eq_constraints: bool = False
    supports_ineq_constraints: bool = False


SCIPY_REGISTRY: dict[Algorithm, ScipyAlgConfig] = {
    Algorithm.NELDER_MEAD: ScipyAlgConfig(
        alg_name="NELDER_MEAD",
        param_cls=NelderMeadParams,
        condition_overrides={"xtol_abs": "xatol", "ftol_abs": "fatol"},
        supports_grad=True,
    ),
    Algorithm.POWELL: ScipyAlgConfig(
        alg_name="POWELL",
        param_cls=PowellParams,
        condition_overrides={"xtol_abs": "xtol", "ftol_abs": "ftol"},
        supports_grad=True,
    ),
    Algorithm.L_BFGS_B: ScipyAlgConfig(
        alg_name="L_BFGS_B",
        param_cls=LBFGSBParams,
        condition_overrides={"ftol_abs": "ftol"},
    ),
    Algorithm.TNC: ScipyAlgConfig(
        alg_name="TNC",
        param_cls=TNCParams,
        condition_overrides={"ftol_abs": "ftol"},
    ),
    Algorithm.COBYLA_SCIPY: ScipyAlgConfig(
        alg_name="COBYLA",
        param_cls=COBYLAParams,
        condition_overrides={"ftol_abs": "tol", "stop_val": "f_target"},
        supports_grad=True,
        supports_eq_constraints=True,
        supports_ineq_constraints=True,
    ),
    Algorithm.COBYQA: ScipyAlgConfig(
        alg_name="COBYQA",
        param_cls=COBYQAParams,
        condition_overrides={"stop_val": "f_target"},
        supports_eq_constraints=True,
        supports_ineq_constraints=True,
    ),
    Algorithm.SLSQP_SCIPY: ScipyAlgConfig(
        alg_name="SLSQP",
        param_cls=SLSQPParams,
        condition_overrides={"ftol_abs": "ftol"},
        supports_eq_constraints=True,
        supports_ineq_constraints=True,
    ),
    Algorithm.TRUST_CONSTR: ScipyAlgConfig(
        alg_name="TRUST_CONSTR",
        param_cls=TrustConstrParams,
        condition_overrides={},
        supports_eq_constraints=True,
        supports_ineq_constraints=True,
    ),
}
