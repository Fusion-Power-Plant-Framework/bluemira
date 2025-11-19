# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.optimisation._algorithm import Algorithm, AlgorithmDefaultConditions

CONDITION_MAP = {  # nlopt to scipy condition map
    "ftol_rel": "ftol",
    "ftol_abs": "ftol",  # override if both given
    "xtol_rel": "xtol",
    "xtol_abs": "xtol",
    "max_eval": "maxiter",
}

ALG_CONDITION_MAP = {  # algorithm-specific condition overrides
    "COBYLA": {"ftol": "tol"},
}


@dataclass
class ScipyConditions:
    """Hold and validate SciPy optimiser conditions."""

    ftol: float | None = None
    xtol: float | None = None
    gtol: float | None = None
    maxiter: int | None = None
    tol: float | None = None
    rhobeg: float | None = None
    catol: float | None = None

    def __init__(self, algorithm: Algorithm, opt_conditions) -> None:
        if opt_conditions:
            scipy_conditions = self._convert_to_scipy(opt_conditions, algorithm)
            for key, value in scipy_conditions.items():
                setattr(self, key, value)

        self._backfill(scipy_conditions, self._get_defaults(algorithm))

    @staticmethod
    def _convert_to_scipy(
        conds: Mapping[str, int | float] | None, algorithm: Algorithm
    ) -> dict[str, float]:
        """
        Translate optimiser conditions from NLopt terminology.

        Returns
        -------
        :
            The data in dictionary form as each SciPy algorithm expects.
        """
        translated = {}
        for name, val in conds.items():
            if name in CONDITION_MAP:
                if name.endswith("_rel"):
                    bluemira_warn(
                        f"{name} provided; using as {CONDITION_MAP[name]}, unless "
                        f"{CONDITION_MAP[name] + '_abs'} is provided (takes priority)"
                    )
                translated[CONDITION_MAP[name]] = val
            else:
                bluemira_warn(f"Scipy does not recognise '{name}'")
                translated[name] = val

        for k_old, k_new in ALG_CONDITION_MAP.get(
            algorithm.name.removesuffix("_SCIPY"), {}
        ).items():
            if k_old in translated:
                translated[k_new] = translated.pop(k_old)

        return translated

    @staticmethod
    def _get_defaults(algorithm: Algorithm) -> dict[str, float]:
        """
        Retrieve default conditions for the given algorithm.

        Returns
        -------
        :
            A dict of default conditions if they exist, otherwise an empty dict.
        """
        try:
            defaults = getattr(AlgorithmDefaultConditions(), algorithm.name).to_dict()
        except AttributeError:  # algorithm doesn't have defaults
            return {}
        else:
            defaults.pop("max_eval")
            return defaults

    def _backfill(self, conditions, defaults) -> None:
        """Fill in default conditions."""
        for key, val in defaults.items():
            if key not in conditions:
                setattr(self, key, val)

    def to_dict(self) -> dict[str, float]:
        """
        Return used conditions as a clean dictionary.

        Returns
        -------
        :
            The data in dictionary form.
        """
        return {k: v for k, v in asdict(self).items() if v is not None}
