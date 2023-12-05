# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Error classes for the equilibria module.
"""

from bluemira.base.error import BluemiraError


class EquilibriaError(BluemiraError):
    """
    Base class for equilibria errors.
    """


class FluxSurfaceError(EquilibriaError):
    """
    Error class for FluxSurfaces.
    """
