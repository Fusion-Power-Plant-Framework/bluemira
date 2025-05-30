# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
API guard for the OCC package.
"""

from bluemira.codes.utilities import code_guard

occ_guard = code_guard(
    "OCC",
    "Run `conda install pythonocc-core` to use this function.",
)
