# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Error classes for the utilities module.
"""

from bluemira.base.error import BluemiraError


class UtilitiesError(BluemiraError):
    """
    The base class for utilities errors.
    """


class PositionerError(UtilitiesError):
    """
    Error for positioner utilities.
    """


class OptUtilitiesError(UtilitiesError):
    """
    Error for optimisation utilities.
    """


class OptVariablesError(OptUtilitiesError):
    """
    Error for optimisation utilities.
    """


class InternalOptError(OptUtilitiesError):
    """
    Error class for errors inside the optimisation algorithms.
    """


class ExternalOptError(OptUtilitiesError):
    """
    Error class for errors relating to the optimisation, but not originating
    inside the optimisers.
    """
