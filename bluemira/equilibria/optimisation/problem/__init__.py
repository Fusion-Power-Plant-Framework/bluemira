# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Coilset optimisation problem classes and tools."""

from bluemira.equilibria.optimisation.problem._breakdown import (
    BreakdownCOP,
    BreakdownZoneStrategy,
    CircularZoneStrategy,
    InboardBreakdownZoneStrategy,
    InputBreakdownZoneStrategy,
    OutboardBreakdownZoneStrategy,
)
from bluemira.equilibria.optimisation.problem._maximise_connection_length import (
    MaximiseConnectionLengthCOP,
)
from bluemira.equilibria.optimisation.problem._maximise_divertor_leg_length import (
    MaximiseDivertorLegLengthCOP,
)
from bluemira.equilibria.optimisation.problem._minimal_current import MinimalCurrentCOP
from bluemira.equilibria.optimisation.problem._nested_position import (
    NestedCoilsetPositionCOP,
    PulsedNestedPositionCOP,
)
from bluemira.equilibria.optimisation.problem._position import CoilsetPositionCOP
from bluemira.equilibria.optimisation.problem._tikhonov import (
    TikhonovCurrentCOP,
    UnconstrainedTikhonovCurrentGradientCOP,
)
from bluemira.equilibria.optimisation.problem.base import CoilsetOptimisationProblem

__all__ = [
    "BreakdownCOP",
    "BreakdownZoneStrategy",
    "CircularZoneStrategy",
    "CoilsetOptimisationProblem",
    "CoilsetPositionCOP",
    "InboardBreakdownZoneStrategy",
    "InputBreakdownZoneStrategy",
    "MinimalCurrentCOP",
    "MaximiseConnectionLengthCOP",
    "MaximiseDivertorLegLengthCOP",
    "NestedCoilsetPositionCOP",
    "OutboardBreakdownZoneStrategy",
    "PulsedNestedPositionCOP",
    "TikhonovCurrentCOP",
    "UnconstrainedTikhonovCurrentGradientCOP",
]
