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
    from bluemira.materials.material import MassFractionMaterial
    from bluemira.materials.mixtures import HomogenisedMixture


@dataclass
class NeutronicsMaterials:
    """A dictionary of materials according to the type of blanket used"""

    inb_vv_mat: HomogenisedMixture | MassFractionMaterial
    inb_fw_mat: HomogenisedMixture | MassFractionMaterial
    inb_bz_mat: HomogenisedMixture | MassFractionMaterial
    inb_mani_mat: HomogenisedMixture | MassFractionMaterial
    divertor_mat: HomogenisedMixture | MassFractionMaterial
    div_fw_mat: HomogenisedMixture | MassFractionMaterial
    outb_fw_mat: HomogenisedMixture | MassFractionMaterial
    outb_bz_mat: HomogenisedMixture | MassFractionMaterial
    outb_mani_mat: HomogenisedMixture | MassFractionMaterial
    outb_vv_mat: HomogenisedMixture | MassFractionMaterial
    tf_coil_mat: HomogenisedMixture | MassFractionMaterial
    container_mat: HomogenisedMixture | MassFractionMaterial
    inb_sf_mat: HomogenisedMixture | MassFractionMaterial
    outb_sf_mat: HomogenisedMixture | MassFractionMaterial
    div_sf_mat: HomogenisedMixture | MassFractionMaterial
