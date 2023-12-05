# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Public exception classes for the optimisation module."""

from bluemira.base.error import BluemiraError


class OptimisationError(BluemiraError):
    """Generic optimisation error."""


class GeometryOptimisationError(BluemiraError):
    """Generic geometry optimisation error."""


class OptimisationConditionsError(OptimisationError):
    """Error relating to optimiser conditions."""


class OptimisationParametersError(OptimisationError):
    """Error relating to optimiser parameters."""
