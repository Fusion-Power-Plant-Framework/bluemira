# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Error classes for use in the fuel_cycle module
"""

from bluemira.base.error import BluemiraError


class FuelCycleError(BluemiraError):
    """
    The base fuel_cycle error class.
    """
