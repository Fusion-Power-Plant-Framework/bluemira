# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
DAGMC converter definition.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Generic, TypeVar

from pydantic import BaseModel

from bluemira.geometry.base import BluemiraGeoT


class DAGMCConverterConfig(BaseModel, ABC):
    """
    Enum for DAGMC converters.
    """

    converter_type: str

    @abstractmethod
    def run_converter(
        self,
        shapes: Iterable[BluemiraGeoT],
        names: list[str],
        output_dagmc_model_path: str | Path,
    ) -> None:
        """
        Run the converter.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")


T = TypeVar("T", bound=DAGMCConverterConfig)


class DAGMCConverter(ABC, Generic[T]):
    """
    Class to convert a DAGMC file to a format suitable for use in the
    Bluemira radiation transport module.
    """

    def __init__(
        self,
        shapes: Iterable[BluemiraGeoT],
        names: list[str],
        comp_mat_mapping: dict[str, str],
    ):
        """
        Abstract class representing a converter from CAD to a DAGMC .h5m model file.

        Parameters
        ----------
        shapes:
            List of shapes to be converted.
        names:
            List of names for the shapes.
        comp_mat_mapping:
            Mapping of component names to material names.

        Raises
        ------
        ValueError
            If any name is not in the keys of the comp_mat_mapping.
        """
        names_set = set(names)
        keys_set = set(comp_mat_mapping.keys())
        if not names_set.issubset(keys_set):
            raise ValueError(
                "Every name must be in the keys of the comp_mat_mapping.\n"
                f"Provided set of names:\n{names_set}\n\n"
                f"Keys:\n{keys_set}"
            )

        self.shapes = shapes
        self.names = names
        self.comp_mat_mapping = comp_mat_mapping

    @abstractmethod
    def run(self, output_dagmc_model_path: str | Path, converter_config: T) -> None:
        """
        Convert the DAGMC file to a format suitable for use in Bluemira.
        """
