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
from eurofusion_materials.library.steel import SS316LN

WATER_MAT = Water()
SS316LN_MAT = SS316LN()

VV_MATERIAL = mixture("SteelWater", [(SS316LN_MAT, 0.6), (WATER_MAT, 0.4)], fraction_type="mass")