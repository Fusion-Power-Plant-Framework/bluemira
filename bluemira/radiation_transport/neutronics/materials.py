# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Create the material sets for each type of reactor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matproplib.material import Material


@dataclass
class NeutronicsMaterials:
    """A dictionary of materials according to the type of blanket used"""

    inb_vv_mat: Material
    inb_fw_mat: Material
    inb_bz_mat: Material
    inb_mani_mat: Material
    divertor_mat: Material
    div_fw_mat: Material
    outb_fw_mat: Material
    outb_bz_mat: Material
    outb_mani_mat: Material
    outb_vv_mat: Material
    tf_coil_mat: Material
    container_mat: Material
    inb_sf_mat: Material
    outb_sf_mat: Material
    div_sf_mat: Material
    rad_shield: Material
