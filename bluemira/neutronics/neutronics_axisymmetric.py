# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Make an axis-symmetric tokamak.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bluemira.base.constants import raw_uc
from bluemira.neutronics.make_materials import MaterialsLibrary

if TYPE_CHECKING:
    from bluemira.neutronics.params import BreederTypeParameters


def create_materials(
    breeder_materials: BreederTypeParameters,
) -> MaterialsLibrary:
    """
    Parameters
    ----------
    breeder_materials:
        dataclass containing attributes: 'blanket_type', 'enrichment_fraction_Li6'
    """
    return MaterialsLibrary.create_from_blanket_type(
        breeder_materials.blanket_type,
        raw_uc(breeder_materials.enrichment_fraction_Li6, "", "%"),
    )


def create_and_export_materials(
    breeder_materials: BreederTypeParameters,
) -> MaterialsLibrary:
    """
    Parameters
    ----------
    breeder_materials:
        dataclass containing attributes: 'blanket_type', 'enrichment_fraction_Li6'
    """
    material_lib = create_materials(breeder_materials)
    material_lib.export()
    return material_lib
