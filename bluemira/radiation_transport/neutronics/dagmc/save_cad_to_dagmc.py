# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Saving CAD to DAGMC helper function.
"""

from collections.abc import Iterable
from pathlib import Path

from pydantic import BaseModel, Field

from bluemira.geometry.base import BluemiraGeoT
from bluemira.radiation_transport.neutronics.dagmc.dagmc_converter_fast_ctd import (
    DAGMCConverterFastCTDConfig,
)


class DAGMCConverterConfigModel(BaseModel):
    """A model to enable config validation and conversion."""

    config: DAGMCConverterFastCTDConfig = Field(discriminator="converter_type")


def save_cad_to_dagmc(
    shapes: Iterable[BluemiraGeoT],
    names: list[str],
    filename: Path,
    material_name_map: dict[str, str] | None = None,
    *,
    converter_config: dict | DAGMCConverterFastCTDConfig | None = None,
) -> None:
    """
    Save a list of shapes to a DAGMC file.

    Parameters
    ----------
    shapes:
        List of shapes to be saved.
    names:
        List of names for the shapes.
    filename:
        Path to the output DAGMC file.
    material_name_map:
        Mapping of component names to material names.
    converter_config:
        Configuration for the converter. If None, the default (fast_ctd) is used.
        If a dictionary is provided, it will be converted to the appropriate
        configuration model. If a configuration model is provided, it will be
        used directly.

    Raises
    ------
    NotImplementedError
        If the converter is not implemented.
    """
    # runs it through the pydantic model to validate the config
    # and convert it to the correct type
    return DAGMCConverterConfigModel(
        # Use fast_ctd as the default converter
        config=converter_config or DAGMCConverterFastCTDConfig()
    ).config.run_converter(shapes, names, material_name_map, filename)


if __name__ == "__main__":
    # Example usage
    from copy import deepcopy

    from bluemira.geometry.face import BluemiraFace
    from bluemira.geometry.tools import extrude_shape, make_polygon

    box_a = BluemiraFace(
        make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], closed=True
        )
    )
    box_a = extrude_shape(box_a, [0, 0, 1])
    box_b = deepcopy(box_a)
    box_b.translate([-0.6, -0.6, 1])
    box_c = deepcopy(box_a)
    box_c.translate([0.6, 0.6, 1])

    shapes = [box_a, box_b, box_c]
    names = ["box_a", "box_b", "box_c"]

    filename = Path(__file__).parent / "test.h5m"
    save_cad_to_dagmc(
        shapes,
        names,
        filename,
        # converter_config={
        #     "converter_type": "fast_ctd",
        #     "comp_name_to_material_name_map": {
        #         "box_a": "material_a",
        #         "box_b": "material_b",
        #         "box_c": "material_c",
        #     },
        #     "lin_deflection": 0.1,
        #     "clean_up": True,
        # },
    )
