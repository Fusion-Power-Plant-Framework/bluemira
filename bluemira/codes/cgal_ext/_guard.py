# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
API guard for the CGAL package.
"""

from bluemira.codes.utilities import code_guard, is_code_available

cgal_available = is_code_available("CGAL")

cgal_guard = code_guard(
    "CGAL",
    "CGAL is not available. Run `pip install cgal` to use this function.",
)
