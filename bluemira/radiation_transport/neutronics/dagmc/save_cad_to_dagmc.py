# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Saving CAD to DAGMC helper function.
"""

from pathlib import Path

from pydantic import BaseModel, Field

from bluemira.base.file import force_file_extension
from bluemira.geometry.base import BluemiraGeoT
from bluemira.radiation_transport.neutronics.dagmc.dagmc_converter_fast_ctd import (
    DAGMCConverterFastCTDConfig,
)


class DAGMCConverterConfigModel(BaseModel):
    """A model to enable config validation and conversion."""

    config: DAGMCConverterFastCTDConfig = Field(discriminator="converter_type")


def save_cad_to_dagmc(
    shapes: list[BluemiraGeoT],
    names: list[str],
    filename: Path,
    comp_mat_mapping: dict[str, str],
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
    comp_mat_mapping:
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
    filename = Path(force_file_extension(filename.as_posix(), ".h5m"))

    # runs it through the pydantic model to validate the config
    # and convert it to the correct type
    return DAGMCConverterConfigModel(
        # Use fast_ctd as the default converter
        config=converter_config or DAGMCConverterFastCTDConfig()
    ).config.run_converter(shapes, names, comp_mat_mapping, filename)
