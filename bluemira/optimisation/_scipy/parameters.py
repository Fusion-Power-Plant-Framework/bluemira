# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from collections.abc import Mapping
from dataclasses import asdict, dataclass


@dataclass
class COBYLAParams:
    rhobeg: float = 0.2
    catol: float = 1e-4


@dataclass
class COBYQAParams:
    initial_tr_radius: float = 0.3


PARAMETER_CLS = {
    "COBYLA": COBYLAParams,
    "COBYQA": COBYQAParams,
}


def _make_alg_params(
    user_params: Mapping[str, int | float], alg: str
) -> Mapping[str, int | float]:
    """
    Algorithm parameter factory.

    Returns
    -------
    The dataclass associated with the given algorithm,
    with user parameters merged into the default parameters.
    """
    if not (cls := PARAMETER_CLS.get(alg)):
        return user_params  # no defaults
    known = {k: user_params.get(k, v) for k, v in asdict(cls()).items()}
    extras = {k: v for k, v in user_params.items() if k not in asdict(cls())}
    return {**known, **extras}  # merge defaults with user_params
