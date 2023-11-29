# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Error classes for the codes module.
"""

import bluemira.base.error as base_err

__all__ = ["CodesError"]


class CodesError(base_err.BluemiraError):
    """
    Error class for use in the codes module
    """


class FreeCADError(base_err.BluemiraError):
    """
    Error class for use in the geometry module where FreeCAD throws an error.
    """


class InvalidCADInputsError(base_err.BluemiraError):
    """
    Error class for use in the geometry module where inputs are not valid.
    """
