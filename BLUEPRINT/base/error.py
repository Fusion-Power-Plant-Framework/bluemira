# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
from textwrap import fill, dedent


class BLUEPRINTError(Exception):
    """
    Base exception class
    """

    def __str__(self):
        """
        Prettier handling of the Exception strings
        """
        return fill(dedent(self.args[0]))


class BeamsError(BLUEPRINTError):
    """
    Error class for use in the beams module
    """

    pass


class NeutronicsError(BLUEPRINTError):
    """
    Error class for use in the neutronics module
    """

    pass


class CADError(BLUEPRINTError):
    """
    Error class for use in the cad module
    """

    pass


class SystemsError(BLUEPRINTError):
    """
    Error class for use in the systems module
    """

    pass


class EquilibriaError(BLUEPRINTError):
    """
    Error class for use in the equilibria module
    """

    pass


class UtilitiesError(BLUEPRINTError):
    """
    Error class for use in the utilities module
    """

    pass


class MaterialsError(BLUEPRINTError):
    """
    Error class for use in the materials module
    """

    pass


class SysCodesError(BLUEPRINTError):
    """
    Error class for use in the syscodes module
    """

    pass


class GeometryError(BLUEPRINTError):
    """
    Error class for use in the geometry module
    """

    pass


class NovaError(BLUEPRINTError):
    """
    Error class for use in the nova module
    """

    pass


class BaseError(BLUEPRINTError):
    """
    Error class for use in the base module
    """

    pass


class FuelCycleError(BLUEPRINTError):
    """
    Error class for use in the fuelcycle module
    """

    pass


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
