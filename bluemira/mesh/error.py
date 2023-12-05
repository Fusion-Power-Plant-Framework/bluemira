# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Errors for meshing module
"""

from bluemira.base.error import BluemiraError


class MeshError(BluemiraError):
    """
    Error class for use in the mesh module.
    """


class MeshOptionsError(MeshError):
    """
    Error class for use with meshing options.
    """


class MeshConversionError(MeshError):
    """
    Error class for use with mesh conversions.
    """
