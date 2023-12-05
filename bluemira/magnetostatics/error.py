# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Error classes for the magnetostatics module
"""

from bluemira.base.error import BluemiraError


class MagnetostaticsError(BluemiraError):
    """
    The base class for magnetostatics errors.
    """


class MagnetostaticsIntegrationError(MagnetostaticsError):
    """
    Error class for integration errors in magnetostatics.
    """
