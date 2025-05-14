# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
API guard for the fast_ctd package.
"""

from bluemira.codes.utilities import code_guard

fast_ctd_guard = code_guard(
    "fast_ctd",
    "fast_ctd is not available. To install, go to https://github.com/Fusion-Power-Plant-Framework/fast_ctd",
)
