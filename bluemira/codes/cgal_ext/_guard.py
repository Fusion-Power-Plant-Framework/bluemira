# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
API guard for the CGAL package.
"""

try:
    import CGAL  # noqa: F401

    cgal_available = True
except ImportError:
    cgal_available = False


def guard_cgal_available(f):
    """API guard for CGAL functions.

    Raises
    ------
    ImportError
        If CGAL is not available.
    """  # noqa: DOC201

    def wrap(*args, **kwargs):
        if not cgal_available:
            raise ImportError(
                "CGAL is not available. Run `conda install CGAL` to use this function."
            )
        return f(*args, **kwargs)

    return wrap
