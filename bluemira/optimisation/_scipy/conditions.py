# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from bluemira.base.look_and_feel import bluemira_warn

CONDITION_MAP = {
    "COMMON": {
        "ftol_rel": "ftol",
        "ftol_abs": "ftol",  # override if both given
        "xtol_rel": "xtol",
        "xtol_abs": "xtol",
        "max_eval": "maxiter",
        "stop_val": "f_target",
    },
    "COBYLA": {"ftol_abs": "tol"},
    "COBYQA": {"ftol_abs": "feasibility_tol"},
}


@dataclass
class ScipyConditions:
    """Hold and validate SciPy optimiser conditions."""

    ftol: float | None = None
    xtol: float | None = None
    tol: float | None = None
    feasibility_tol: float | None = None
    maxiter: int | None = None
    f_target: float | None = None

    def __post_init__(self) -> None:
        """Validate initialised values."""
        self._validate()

    def to_dict(self) -> dict[str, int | float]:
        """
        Return used conditions as a clean dictionary.

        Returns
        -------
        :
            The data in dictionary form.
        """
        return {k: v for k, v in asdict(self).items() if v is not None}

    def _validate(self) -> None:
        if self.maxiter and not isinstance(self.maxiter, int):
            bluemira_warn("optimisation: max_eval must be an integer, forcing type.")
            self.maxiter = int(self.maxiter)


def _convert_to_scipy(
    conds: Mapping[str, int | float], alg: str
) -> dict[str, int | float]:
    """
    Translate optimiser conditions from NLopt to SciPy format.

    Returns
    -------
    :
        The data in dictionary form as each SciPy algorithm expects.
    """
    map_ = CONDITION_MAP["COMMON"].copy()
    map_.update(CONDITION_MAP.get(alg, {}))
    translated = {}
    for name, val in map_.items():
        if key := conds.get(name):
            translated[val] = key
            conds.pop(name)
    if conds:
        bluemira_warn(f"Condition(s) '{conds}' not recognised by SciPy ({alg})")
    return translated
