# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Coil and coil grouping objects
"""

from bluemira.equilibria.coils._coil import Coil
from bluemira.equilibria.coils._grouping import (
    Circuit,
    CoilGroup,
    CoilSet,
    SymmetricCircuit,
    symmetrise_coilset,
)
from bluemira.equilibria.coils._tools import (
    check_coilset_symmetric,
    get_max_current,
    make_mutual_inductance_matrix,
)
from bluemira.equilibria.coils.coil_type import CoilType

__all__ = [
    "Circuit",
    "Coil",
    "CoilGroup",
    "CoilSet",
    "CoilType",
    "SymmetricCircuit",
    "check_coilset_symmetric",
    "get_max_current",
    "make_mutual_inductance_matrix",
    "symmetrise_coilset",
]
