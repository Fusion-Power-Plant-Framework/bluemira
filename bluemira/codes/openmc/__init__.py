# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""OpenMC interface"""

from bluemira.codes.openmc.solver import OpenMCCSGNeutronicsSolver as CSGSolver
from bluemira.codes.openmc.solver import OpenMCDAGMCNeutronicsSolver as DAGMCSolver

__all__ = ["CSGSolver", "DAGMCSolver"]
