# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Simple structural material representations
"""

from matproplib.library.fluids import Void
from matproplib.material import material
from matproplib.properties.group import props

__all__ = ["Void"]


# Just some simple materials to play with during tests and the like
def sm(self, op_cond):
    return self.youngs_modulus(op_cond) / (0.5 + 0.5 * self.poissons_ratio(op_cond))


SS316 = material(
    "SS316 room temperature",
    properties=props(
        youngs_modulus=200e9,
        poissons_ratio=0.33,
        density=8910,
        coefficient_thermal_expansion=18e-6,
        average_yield_stress=360e6,
        shear_modulus=sm,
    ),
)

FORGED_SS316LN = material(
    name="forged_ss316ln",
    properties=props(
        youngs_modulus=205e9,
        poissons_ratio=0.29,
        density=8910,
        coefficient_thermal_expansion=10.36e-6,
        average_yield_stress=800e6,
        shear_modulus=sm,
    ),
)
"""Forged SS316LN plates: OIS structural material as defined in 2MBS88 and"
    "ITER SDC-MC DRG1 Annex A (values at 4K)."""


FORGED_JJ1 = material(
    name="forged_jj1",
    properties=props(
        youngs_modulus=205e9,
        poissons_ratio=0.29,
        density=8910,
        coefficient_thermal_expansion=10.38e-6,
        average_yield_stress=1000e6,
        shear_modulus=sm,
    ),
)
"""Forged EK1/JJ1 strengthened austenitic steel plates: TF inner leg material"
    " as defined in 2MBS88 and ITER SDC-MC DRG1 Annex A (values at 4K)."""


CAST_EC1 = material(
    name="cast_ec1",
    properties=props(
        youngs_modulus=190e9,
        poissons_ratio=0.29,
        density=8910,
        coefficient_thermal_expansion=10.38e-6,
        average_yield_stress=750e6,
        shear_modulus=sm,
    ),
)
""" Cast EC1 strengthened austenitic steel castings: TF outer leg material as"
    " defined in 2MBS88 and ITER SDC-MC DRG1 Annex A (values at 4K)."""

CONCRETE = material(
    name="concrete",
    properties=props(
        youngs_modulus=40e9,
        poissons_ratio=0.3,
        density=2400,
        coefficient_thermal_expansion=12e-6,
        average_yield_stress=40e6,
        shear_modulus=sm,
    ),
)
"""Typical concrete properties at room temperature"""


vacuum_void = Void()
