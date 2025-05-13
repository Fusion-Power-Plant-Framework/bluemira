# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
CAD apis
"""

BACKEND = "FREECAD"

if BACKEND == "FREECAD":
    import bluemira.codes.cadapi._freecad.api as cadapi
elif BACKEND == "CadQuery":
    import bluemira.codes.cadapi._cadquery.api as cadapi

__all__ = ["cadapi"]
