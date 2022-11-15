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
Coil and coil grouping objects
"""

from bluemira.equilibria.coils._coil import Coil, CoilType
from bluemira.equilibria.coils._field import get_max_current
from bluemira.equilibria.coils._grouping import (
    CS_COIL_NAME,
    PF_COIL_NAME,
    CoilGroup,
    Solenoid,
)
from bluemira.equilibria.coils._grouping import _CoilSet as CoilSet
from bluemira.equilibria.coils._grouping import _SymmetricCircuit as SymmetricCircuit
from bluemira.equilibria.coils._grouping import symmetrise_coilset
