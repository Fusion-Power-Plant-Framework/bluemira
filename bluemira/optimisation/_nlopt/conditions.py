from dataclasses import asdict, dataclass
from typing import Dict, Optional

from bluemira.base.constants import EPS
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.optimisation.error import OptimisationConditionsError


@dataclass
class NLOptConditions:
    """Hold and validate optimiser stopping conditions."""

    ftol_abs: Optional[float] = None
    ftol_rel: Optional[float] = None
    xtol_abs: Optional[float] = None
    xtol_rel: Optional[float] = None
    max_eval: Optional[int] = None
    max_time: Optional[float] = None
    stop_val: Optional[float] = None

    def __post_init__(self):
        """Validate initialised values."""
        self._validate()

    def to_dict(self) -> Dict[str, float]:
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
                    "optimisation: Setting stopping condition to less than machine "
                    "precision. This condition may never be met."
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
