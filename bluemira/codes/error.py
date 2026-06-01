# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Error classes for the codes module.
"""

import bluemira.base.error as base_err

__all__ = ["CADError", "CadQueryError", "CodesError", "FreeCADError"]


class CodesError(base_err.BluemiraError):
    """
    Error class for use in the codes module
    """


class CADError(base_err.BluemiraError):
    """
    Base error class for CAD backend errors.

    Catch this in backend-agnostic code (geometry layer, etc.)
    to handle errors from any CAD backend.
    """


class FreeCADError(CADError):
    """
    Error class for FreeCAD backend errors.
    """


class CadQueryError(CADError):
    """
    Error class for CadQuery backend errors.
    """


class InvalidCADInputsError(base_err.BluemiraError):
    """
    Error class for use in the geometry module where inputs are not valid.
    """
