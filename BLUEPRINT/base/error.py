# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
Errors for sub-modules
"""
from bluemira.base.error import BluemiraError


class BeamsError(BluemiraError):
    """
    Error class for use in the beams module
    """

    pass


class NeutronicsError(BluemiraError):
    """
    Error class for use in the neutronics module
    """

    pass


class CADError(BluemiraError):
    """
    Error class for use in the cad module
    """

    pass


class SystemsError(BluemiraError):
    """
    Error class for use in the systems module
    """

    pass


class UtilitiesError(BluemiraError):
    """
    Error class for use in the utilities module
    """

    pass


class MaterialsError(BluemiraError):
    """
    Error class for use in the materials module
    """

    pass


class SysCodesError(BluemiraError):
    """
    Error class for use in the syscodes module
    """

    pass


class GeometryError(BluemiraError):
    """
    Error class for use in the geometry module
    """

    pass


class NovaError(BluemiraError):
    """
    Error class for use in the nova module
    """

    pass


class BaseError(BluemiraError):
    """
    Error class for use in the base module
    """

    pass


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
