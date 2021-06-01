# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.

"""
Simple FE material objects
"""


class Material(dict):
    """
    A simple material property dictionary (keep small for speed and memory)
    """

    __slots__ = []

    def __init__(self, e_modulus, nu, rho, alpha, sigma_y):
        self["E"] = e_modulus
        self["nu"] = nu
        self["alpha"] = alpha
        self["rho"] = rho
        self["G"] = e_modulus / (1 + nu) / 2
        self["sigma_y"] = sigma_y


# Just some simple materials to play with during tests and the like


class SS316(Material):
    """
    Typical stainless steel properties.
    """

    def __init__(self):
        super().__init__(200e9, 0.33, 8910, 18e-6, 360e6)


class Concrete(Material):
    """
    Typical concrete properties.
    """

    def __init__(self):
        super().__init__(40e9, 0.3, 2400, 12e-6, 40e6)


class ForgedSS316LN(Material):
    """
    Forged SS316LN plates: OIS structural material as defined in 2MBS88 and
    ITER SDC-MC DRG1 Annex A (values at 4K).
    """

    def __init__(self):
        super().__init__(205e9, 0.29, 8910, 10.36e-6, 800e6)


class ForgedJJ1(Material):
    """
    Forged EK1/JJ1 strengthened austenitic steel plates: TF inner leg material
    as defined in 2MBS88 and ITER SDC-MC DRG1 Annex A (values at 4K).
    """

    def __init__(self):
        super().__init__(205e9, 0.29, 8910, 10.38e-6, 1000e6)


class CastEC1(Material):
    """
    Cast EC1 strengthened austenitic steel castings: TF outer leg material as
    defined in 2MBS88 and ITER SDC-MC DRG1 Annex A (values at 4K).
    """

    def __init__(self):
        super().__init__(190e9, 0.29, 8910, 10.38e-6, 750e6)


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
