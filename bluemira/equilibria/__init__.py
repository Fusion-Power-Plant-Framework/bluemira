# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
The bluemira equilibria module
"""

from bluemira.equilibria.coils import Coil, CoilSet, SymmetricCircuit
from bluemira.equilibria.equilibrium import Breakdown, Equilibrium
from bluemira.equilibria.find import (
    find_LCFS_separatrix,
    find_OX_points,
    find_flux_surfs,
)
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.limiter import Limiter
from bluemira.equilibria.profiles import BetaIpProfile, CustomProfile
from bluemira.equilibria.shapes import (
    flux_surface_cunningham,
    flux_surface_johner,
    flux_surface_manickam,
)
from bluemira.equilibria.solve import PicardIterator

__all__ = [
    "BetaIpProfile",
    "Breakdown",
    "Coil",
    "CoilSet",
    "CustomProfile",
    "Equilibrium",
    "Grid",
    "Limiter",
    "PicardIterator",
    "SymmetricCircuit",
    "find_LCFS_separatrix",
    "find_OX_points",
    "find_flux_surfs",
    "flux_surface_cunningham",
    "flux_surface_johner",
    "flux_surface_manickam",
]
