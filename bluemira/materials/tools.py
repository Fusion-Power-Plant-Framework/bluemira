# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Materials tools"""

from __future__ import annotations

import json
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

import numpy as np

from bluemira.base.constants import raw_uc
from bluemira.materials.error import MaterialsError

if TYPE_CHECKING:
    import openmc


def matproperty(t_min: float, t_max: float):
    """
    Material property decorator object.

    Checks that input T vector is within bounds. Handles floats and arrays.
    """

    def decorator(f):
        def wrapper(*args, **kwargs):
            temperatures = np.atleast_1d(args[1])

            if not (temperatures <= t_max).all():
                raise ValueError(
                    "Material property not valid outside of tempe"
                    f"rature range: {temperatures} > T_max = {t_max}"
                )
            if not (temperatures >= t_min).all():
                raise ValueError(
                    "Material property not valid outside of tempe"
                    f"rature range: {temperatures} < T_min = {t_min}"
                )
            return f(args[0], temperatures, **kwargs)

        return wrapper

    return decorator


def _try_calc_property(mat, prop_name, *args, **kwargs):
    if not hasattr(mat, prop_name):
        raise MaterialsError(
            f"Property {prop_name} does not exist for material {mat.name}"
        )

    if getattr(mat, prop_name) is not None:
        return getattr(mat, prop_name)(*args, **kwargs)
    raise MaterialsError(
        f"Property {prop_name} has not been defined for material {mat.name}"
    )


def import_nmm():
    """Don't hack my json, among other annoyances."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        import neutronics_material_maker as nmm  # noqa: PLC0415

        # Really....
        json.JSONEncoder.default = nmm.material._default.default

    # hard coded value is out of date after 2019 redefinition
    nmm.material.atomic_mass_unit_in_g = raw_uc(1, "amu", "g")

    return nmm


@contextmanager
def patch_nmm_openmc():
    """Avoid creating openmc material until necessary"""
    nmm = import_nmm()
    if value := nmm.material.OPENMC_AVAILABLE:
        nmm.material.OPENMC_AVAILABLE = False
    try:
        yield nmm
    finally:
        if value:
            nmm.material.OPENMC_AVAILABLE = True


def to_openmc_material(
    name: str | None = None,
    packing_fraction: float = 1.0,
    enrichment: float | None = None,
    enrichment_target: str | None = None,
    temperature: float | None = None,
    pressure: float | None = None,
    elements: dict[str, float] | None = None,
    chemical_equation: str | None = None,
    isotopes: dict[str, float] | None = None,
    percent_type: str | None = None,
    density: float | None = None,
    density_unit: str | None = None,
    atoms_per_unit_cell: int | None = None,
    volume_of_unit_cell_cm3: float | None = None,
    enrichment_type: str | None = None,
    comment: str | None = None,
    zaid_suffix: str | None = None,
    material_id: int | None = None,
    decimal_places: int = 8,
    volume_in_cm3: float | None = None,
    additional_end_lines: dict[str, list[str]] | None = None,
    *,
    temperature_to_neutronics_code: bool = True,
) -> openmc.Material:
    """Convert Bluemira material to OpenMC material"""
    with patch_nmm_openmc() as nmm:
        return nmm.Material(
            name=name,
            packing_fraction=packing_fraction,
            enrichment=enrichment,
            enrichment_target=enrichment_target,
            temperature=temperature,
            temperature_to_neutronics_code=temperature_to_neutronics_code,
            pressure=pressure,
            elements=elements,
            chemical_equation=chemical_equation,
            isotopes=isotopes,
            percent_type=percent_type,
            density=density,
            density_unit=density_unit,
            atoms_per_unit_cell=atoms_per_unit_cell,
            volume_of_unit_cell_cm3=volume_of_unit_cell_cm3,
            enrichment_type=enrichment_type,
            comment=comment,
            zaid_suffix=zaid_suffix,
            material_id=material_id,
            decimal_places=decimal_places,
            volume_in_cm3=volume_in_cm3,
            additional_end_lines=additional_end_lines,
        ).openmc_material


def to_openmc_material_mixture(
    materials: list[openmc.Material],
    fracs: list[float],
    name: str | None = None,
    material_id: int | None = None,
    temperature: float | None = None,
    percent_type: str = "vo",
    packing_fraction: float = 1.0,
    pressure: float | None = None,
    comment: str | None = None,
    zaid_suffix: str | None = None,
    decimal_places: int = 8,
    volume_in_cm3: float | None = None,
    additional_end_lines: dict[str, list[str]] | None = None,
    *,
    temperature_to_neutronics_code: bool = True,
) -> openmc.Material:
    """Convert Bluemira material mixture to OpenMC material mixture"""
    with patch_nmm_openmc() as nmm:
        return nmm.Material.from_mixture(
            name=name,
            material_id=material_id,
            materials=materials,
            fracs=fracs,
            percent_type=percent_type,
            packing_fraction=packing_fraction,
            temperature=temperature,
            temperature_to_neutronics_code=temperature_to_neutronics_code,
            pressure=pressure,
            comment=comment,
            zaid_suffix=zaid_suffix,
            decimal_places=decimal_places,
            volume_in_cm3=volume_in_cm3,
            additional_end_lines=additional_end_lines,
        ).openmc_material
