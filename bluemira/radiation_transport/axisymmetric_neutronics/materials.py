# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Materials for generalised neutronics
"""

import json
from pathlib import Path

import mcio.material.Material
from matproplib import material
from matproplib.conditions import OperationalConditions
from matproplib.properties.dependent import Density


def to_mcio_material(
    inp_material: material,
    op_cond: OperationalConditions,
    number: int | None = None,
) -> mcio.material.Material:
    """
    MCIO does not support matproplib. Hence we would need
    to just convert our matproplib.material into
    mcio.material.Material

    Returns
    -------
    mcio.material.Material

    Raises
    ------
    ValueError
        if density unit is unsupported
    """
    # Name
    # rename mcio material as per mcio accepted format
    name = f"m{number}" if number else "m1"

    # Composition
    composition = {
        element: element_fraction.fraction
        for element, element_fraction in inp_material.elements.root.items()
    }

    if isinstance(inp_material.density, Density):
        # The inp material is NOT a mixure
        density = inp_material.density.value(op_cond)
        density_unit = inp_material.density.unit
    else:
        # Calculate Total Density

        densities = inp_material.density.dpp
        fractions = inp_material.density.fractions

        units = [dpp.unit for dpp in densities]

        if not all(unit == units[0] for unit in units):
            raise ValueError(f"Density units do not match: {units}")

        density = sum(
            fraction * dpp.value(op_cond)
            for fraction, dpp in zip(fractions, densities, strict=True)
        )
        density_unit = units[0]

    if density_unit in {"kg/m³", "kg/m3", "kilogram / meter ** 3"}:
        density *= 1e-3
        density_unit = "g/cm3"
    elif density_unit in {"g/cc", "g/cm3", "atoms/barn-cm"}:
        pass
    else:
        raise ValueError(f"Unsupported density unit: {density_unit}")
    # Fraction Types
    if inp_material.converters and "openmc" in inp_material.converters.root:
        material_fraction_type = inp_material.converters.openmc.percent_type
    else:
        material_fraction_type = "atomic"

    fraction_type = "ao" if material_fraction_type == "atomic" else "wo"

    # Reference
    if inp_material.reference and inp_material.reference.root:
        reference = "; ".join(str(ref) for ref in inp_material.reference.root.values())
    else:
        reference = ""

    return mcio.material.Material(
        density=density,
        composition=composition,
        fraction_type=fraction_type,
        density_unit=density_unit,
        name=name,
        reference=reference,
        number=number,
    )


def write_materials_json(materials: list[mcio.material.Material], filepath: Path):
    """
    Write MCIO materials to a JSON file.
    """
    data = {}

    for number, mat in enumerate(materials, start=1):
        data[f"m{number}"] = {
            "density": mat.density,
            "number": number,
            "composition": mat.composition,
            "fraction_type": mat.fraction_type,
            "density_type": mat.density_unit,
            "name": f"process_material_{number}",
        }

    Path(filepath).write_text(
        json.dumps(data, indent=4),
        encoding="utf-8",
    )
