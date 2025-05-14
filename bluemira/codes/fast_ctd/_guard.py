# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
API guard for the fast_ctd package.
"""

try:
    import fast_ctd  # noqa: F401

    fast_ctd_available = True
except ImportError:
    fast_ctd_available = False


def cgal_api_available(f):
    """API guard for CGAL functions.

    Raises
    ------
    ImportError
        If CGAL is not available.
    """  # noqa: DOC201

    def wrap(*args, **kwargs):
        if not fast_ctd_available:
            raise ImportError(
                "CGAL is not available. Run `conda install CGAL` to use this function."
            )
        return f(*args, **kwargs)

    return wrap
