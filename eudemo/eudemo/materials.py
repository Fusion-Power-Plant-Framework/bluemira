# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Materials for the conventional aspect ratio example.
"""

from matproplib.material import mixture
from matproplib.library.fluids import Water
from eurofusion_materials.library.steel import SS316_LN
from matproplib.converters.neutronics import OpenMCNeutronicConfig

__all__ = ["VV_MATERIAL"]


class Thing:
    pass


WATER_MAT = Water()
SS316LN_MAT = SS316_LN()

VV_MATERIAL = mixture(
    "SteelWater",
    [(SS316LN_MAT, 0.6), (WATER_MAT, 0.4)],
    fraction_type="mass",
    mix_condition={"temperature": 300, "pressure": 101325},
    converters=OpenMCNeutronicConfig(),
)
