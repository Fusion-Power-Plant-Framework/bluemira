# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Enumeration of supported optimiser algorithms.

Note that it may not be the case that all backends support all
algorithms. Each `Optimiser` implementation must state which of these
algorithms it supports.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum, auto


class Algorithm(Enum):
    """Enumeration of available optimisation algorithms."""

    DEBUG: bool
    SLSQP = auto()
    COBYLA = auto()
    SBPLX = auto()
    MMA = auto()
    BFGS = auto()
    DIRECT = auto()
    DIRECT_L = auto()
    CRS = auto()
    ISRES = auto()

    NELDER_MEAD = auto()
    POWELL = auto()
    CG = auto()
    BFGS_SCIPY = auto()
    NEWTON_CG = auto()
    L_BFGS_B = auto()
    TNC = auto()
    COBYLA_SCIPY = auto()
    SLSQP_SCIPY = auto()
    TRUST_CONSTR = auto()
    DOGLEG = auto()
    TRUST_NCG = auto()
    TRUST_EXACT = auto()
    TRUST_KRYLOV = auto()

    def __new__(cls, value):
        """Create Enum and debug attribute"""
        obj = object.__new__(cls)
        obj._value_ = value
        obj.DEBUG = False
        return obj

    @classmethod
    def _missing_(cls, value: str) -> Algorithm:
        try:
            value = value.upper()
            if value == "DIRECT-L":
                return cls.DIRECT_L
            return cls[value]
        except (KeyError, AttributeError):
            if value.startswith("DEBUG_"):
                self = cls(value.strip("DEBUG_"))
                self.DEBUG = True
                return self
            raise ValueError(f"No such Algorithm value '{value}'.") from None


AlgorithmType = str | Algorithm


@dataclass
class AlgorithmConditions:
    """Algorithm conditions container"""

    ftol_abs: float | None = None
    ftol_rel: float | None = None
    xtol_abs: float | None = None
    xtol_rel: float | None = None
    max_eval: int = 2000
    max_time: float | None = None
    stop_val: float | None = None

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary without Nones"""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class AlgorithmDefaultConditions:
    """Default Algorithm conditions"""

    SLSQP: AlgorithmConditions = field(
        default_factory=lambda: AlgorithmConditions(
            xtol_rel=1e-4, xtol_abs=1e-4, ftol_rel=1e-4, ftol_abs=1e-4
        )
    )
    COBYLA: AlgorithmConditions = field(
        default_factory=lambda: AlgorithmConditions(ftol_rel=1e-6)
    )
    SBPLX: AlgorithmConditions = field(
        default_factory=lambda: AlgorithmConditions(stop_val=1)
    )
    MMA: AlgorithmConditions = field(default_factory=AlgorithmConditions)
    BFGS: AlgorithmConditions = field(
        default_factory=lambda: AlgorithmConditions(xtol_rel=0)
    )
    DIRECT: AlgorithmConditions = field(
        default_factory=lambda: AlgorithmConditions(ftol_rel=1e-4)
    )
    DIRECT_L: AlgorithmConditions = field(default_factory=AlgorithmConditions)
    CRS: AlgorithmConditions = field(default_factory=AlgorithmConditions)
    ISRES: AlgorithmConditions = field(default_factory=AlgorithmConditions)
