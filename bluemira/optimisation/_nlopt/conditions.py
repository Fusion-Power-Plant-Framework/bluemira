# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from dataclasses import asdict, dataclass

from bluemira.base.constants import EPS
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.optimisation.error import OptimisationConditionsError


@dataclass
class NLOptConditions:
    """Hold and validate optimiser stopping conditions."""

    ftol_abs: float | None = None
    ftol_rel: float | None = None
    xtol_abs: float | None = None
    xtol_rel: float | None = None
    max_eval: int | None = None
    max_time: float | None = None
    stop_val: float | None = None

    def __post_init__(self):
        """Validate initialised values."""
        self._validate()

    def to_dict(self) -> dict[str, float]:
        """Return the data in dictionary form."""
        return asdict(self)

    def _validate(self) -> None:
        for condition in [
            self.ftol_abs,
            self.ftol_rel,
            self.xtol_abs,
            self.xtol_rel,
        ]:
            if condition and condition < EPS:
                bluemira_warn(
                    "optimisation: Setting stopping condition is too small given this "
                    "machine's precision. This condition may never be met."
                )
        if self._no_stopping_condition_set():
            raise OptimisationConditionsError(
                "Must specify at least one stopping condition for the optimiser."
            )
        if self.max_eval is not None and isinstance(self.max_eval, float):
            bluemira_warn("optimisation: max_eval must be an integer, forcing type.")
            self.max_eval = int(self.max_eval)

    def _no_stopping_condition_set(self) -> bool:
        return all(
            condition is None
            for condition in [
                self.ftol_abs,
                self.ftol_rel,
                self.xtol_abs,
                self.xtol_rel,
                self.max_eval,
                self.max_time,
                self.stop_val,
            ]
        )
