# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
The bluemira equilibria module
"""

from .coils import Coil, CoilSet, SymmetricCircuit
from .eq_constraints import (
    AutoConstraints,
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
    PsiBoundaryConstraint,
    PsiConstraint,
)
from .equilibrium import Breakdown, Equilibrium
from .find import find_flux_surfs, find_LCFS_separatrix, find_OX_points
from .grid import Grid
from .limiter import Limiter
from .opt_problems import (
    BoundedCurrentCOP,
    CoilsetPositionCOP,
    NestedCoilsetPositionCOP,
    UnconstrainedCurrentCOP,
)
from .optimiser import FBIOptimiser, Norm2Tikhonov, PositionOptimiser
from .profiles import BetaIpProfile, CustomProfile
from .run import AbInitioEquilibriumProblem
from .shapes import flux_surface_cunningham, flux_surface_johner, flux_surface_manickam
from .solve import (
    PicardAbsIterator,
    PicardCoilsetIterator,
    PicardDeltaIterator,
    PicardLiAbsIterator,
    PicardLiDeltaIterator,
)
