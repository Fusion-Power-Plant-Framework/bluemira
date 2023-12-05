# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Error classes for the balance_of_plant module
"""

from bluemira.base.error import BluemiraError


class BalanceOfPlantError(BluemiraError):
    """
    Base error for the balance_of_plant module.
    """
