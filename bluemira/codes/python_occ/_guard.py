# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
API guard for the OCC package.
"""

try:
    import OCC  # noqa: F401

    occ_available = True
except ImportError:
    occ_available = False


def guard_occ_available(f):
    """
    Check if the OCC module is available.

    Raises
    ------
    ImportError
        If the OCC module is not available.
    """  # noqa: DOC201

    def wrap(*args, **kwargs):
        if not occ_available:
            raise ImportError(
                "OCC is not available. Run `conda install pythonocc-core` "
                "to use this function."
            )
        return f(*args, **kwargs)

    return wrap
