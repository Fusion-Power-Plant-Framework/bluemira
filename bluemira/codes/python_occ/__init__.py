# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
The API for Python OCC.
"""

from bluemira.codes.python_occ.imprint_solids import imprint_solids
from bluemira.codes.python_occ.imprintable_solid import ImprintableSolid

__all__ = ["ImprintableSolid", "imprint_solids"]
