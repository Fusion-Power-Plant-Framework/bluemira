# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
CAD API namespace package.

Holds the two backend implementations: ``_freecad`` (FreeCAD/Part) and
``_cadquery`` (CadQuery / OCP). The active backend is selected at import
time by the dispatcher in :mod:`bluemira.codes._geometryapi`.
"""
