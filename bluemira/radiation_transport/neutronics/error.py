# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Error classes for use in the radiation_transport.neutronics module.
"""

from bluemira.base.error import BluemiraError


class CSGGeometryError(BluemiraError):
    """Error when creating invalid CSG geometry."""


class CSGGeometryValidationError(BluemiraError):
    """Error for validating csg geometry.
    Thrown when generating incorrect CSG neutronics model.
    """


class MissingParentError(BluemiraError):
    """Error for when the ".parent" of an object is not set.
    Thrown when generating CSG neutronics model in incorrect order.
    """


class ReadOnlyAttributeError(BluemiraError):
    """Error for when trying to set an already-set write-once attribute.
    Thrown when tampering the CSG neutronics model.
    """


def check_if_read_only_attr_has_been_set(public_attr_name: str, self) -> None:
    """Throw an error

    Raises
    ------
    ReadOnlyAttributeError
        Raised when the write-once/read only atttribute has already been set.
    """
    if not getattr(self, "_" + public_attr_name):
        raise ReadOnlyAttributeError(
            f"{self}.{public_attr_name} has already been set once and cannot be re-set!"
        )
