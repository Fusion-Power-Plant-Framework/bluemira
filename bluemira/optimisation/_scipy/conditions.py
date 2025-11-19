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
        "maxeval": "maxiter",
        "stopval": "f_target",
    },
    "COBYLA": {"ftol_abs": "tol"},
}


@dataclass
class ScipyConditions:
    """Hold and validate SciPy optimiser conditions."""

    ftol: float | None = None
    xtol: float | None = None
    tol: float | None = None
    maxiter: int | None = None
    f_target: float | None = None

    def to_dict(self) -> dict[str, int | float]:
        """
        Return used conditions as a clean dictionary.

        Returns
        -------
        :
            The data in dictionary form.
        """
        return {k: v for k, v in asdict(self).items() if v is not None}


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
    for name, val in conds.items():
        if key := map_.get(name):
            translated[key] = val
        else:
            bluemira_warn(f"Condition '{name}' not recognised by SciPy ({alg})")
    return translated
