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
    "To install, go to https://github.com/Fusion-Power-Plant-Framework/fast_ctd "
    "or install directly (occt>=7.8.0 is required) "
    "run `pip install fast_ctd@git+https://github.com/Fusion-Power-Plant-Framework/fast_ctd@main`",
)
