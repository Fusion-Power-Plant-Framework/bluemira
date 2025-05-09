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

from bluemira.base.look_and_feel import bluemira_print
from bluemira.codes import _freecadapi as cadapi
from bluemira.codes import fast_ctd
from bluemira.codes.python_occ import imprint_solids
from bluemira.geometry.base import BluemiraGeoT


class DAGMCConverter(ABC):
    """
    Class to convert a DAGMC file to a format suitable for use in the
    Bluemira radiation transport module.
    """

    def __init__(
        self,
        shapes: Iterable[BluemiraGeoT],
        names: list[str],
        *,
        comp_name_to_material_name_map: dict[str, str] | None = None,
    ):
        """
        Abstract class representing a converter from CAD to a DAGMC .h5m model file.

        Parameters
        ----------
        shapes:
            List of shapes to be converted.
        names:
            List of names for the shapes.
        comp_name_to_material_name_map:
            Mapping of component names to material names.
            If None, the component names are used as the material names.

        Raises
        ------
        ValueError
            If the keys of comp_name_to_material_name_map do not match the
            names provided.
        """
        if (
            comp_name_to_material_name_map
            and set(names) != comp_name_to_material_name_map.keys()
        ):
            raise ValueError(
                "The keys of comp_name_to_material_name_map must match "
                "the names provided.\n"
                f"Provided names: {names}, "
                f"keys: {comp_name_to_material_name_map.keys()}"
            )
        self.shapes = shapes
        self.names = names
        self.comp_name_to_material_name_map = comp_name_to_material_name_map

    @abstractmethod
    def run(self, dagmc_model_path: str | Path, **kwargs) -> None:
        """
        Convert the DAGMC file to a format suitable for use in Bluemira.
        """


class DAGMCConverterFastCTD(DAGMCConverter):
    """
    Class to convert a DAGMC file to a format suitable for use in the
    Bluemira radiation transport module using fast_ctd.
    """

    def run(
        self,
        dagmc_model_path: str | Path,
        *,
        clean_up=True,
        **kwargs,  # noqa: ARG002
    ) -> None:
        """
        Convert the DAGMC file to a format suitable for use in Bluemira.

        Parameters
        ----------
        dagmc_model_path:
            Path to the output DAGMC model file.
        kwargs:
            Additional keyword arguments to pass to the conversion function.
        """
        imp_res = imprint_solids(self.shapes, self.names)

        imprinted_geom_step_file = Path(dagmc_model_path).with_suffix("_imp.step")
        try:
            bluemira_print(f"Saving imprinted geometry to {imprinted_geom_step_file}")
            cadapi.save_cad(
                [s.shape for s in imp_res.solids],
                imprinted_geom_step_file,
                cad_format="step",
                labels=imp_res.labels,
            )
            bluemira_print(f"Converting {imprinted_geom_step_file} to DAGMC model")
            fast_ctd.step_to_dagmc_pipeline(
                step_file_path=dagmc_model_path,
                output_dagmc_model_path=dagmc_model_path,
                comp_name_to_material_map=self.comp_name_to_material_name_map,
            )
            bluemira_print(f"Conversion to DAGMC model completed: {dagmc_model_path}")
        finally:
            if clean_up:
                bluemira_print(f"Cleaning up {imprinted_geom_step_file}")
                # Remove the intermediate STEP file
                imprinted_geom_step_file.unlink()


def save_cad_to_dagmc(
    shapes: Iterable[BluemiraGeoT],
    names: list[str],
    filename: Path,
    *,
    faceting_tolerance=0.1,
):
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
    faceting_tolerance:
        Tolerance for faceting. Default is 0.1.
    """
    converter = DAGMCConverterFastCTD(shapes, names, comp_name_to_material_name_map=None)
    converter.run(filename, faceting_tolerance=faceting_tolerance)
