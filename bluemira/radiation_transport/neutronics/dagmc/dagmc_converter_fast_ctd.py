# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
fast_ctd DAGMC converter workflow definition.
"""

import json
from pathlib import Path
from typing import Literal

from bluemira.base.look_and_feel import bluemira_debug, bluemira_error, bluemira_print
from bluemira.codes import _freecadapi as cadapi
from bluemira.codes import fast_ctd
from bluemira.codes.python_occ import imprint_solids
from bluemira.geometry.base import BluemiraGeo, BluemiraGeoT
from bluemira.geometry.compound import BluemiraCompound
from bluemira.geometry.solid import BluemiraSolid
from bluemira.radiation_transport.neutronics.dagmc.dagmc_converter import (
    DAGMCConverter,
    DAGMCConverterConfig,
)


class DAGMCConverterFastCTDConfig(DAGMCConverterConfig):
    """
    Converter config model for DAGMCConverterFastCTD.
    """

    converter_type: Literal["fast_ctd"] = "fast_ctd"

    imprint_geometry: bool = True
    """If True, imprint the geometry before converting to DAGMC."""
    imprint_per_compound: bool = True
    """If True, imprint solids grouped in a compound together only.
    Set to False to imprint all solids together. This may take much longer."""

    minimum_include_volume: float = 1.0
    """Minimum volume of a solid to be included in the DAGMC model."""
    fix_step_to_brep_geometry: bool = False
    """Attempts to fix small edges and gaps, refer to the fast_ctd documentation."""
    merge_dist_tolerance: float = 0.001
    """Distance tolerance for merging entities."""
    lin_deflection_tol: float = 0.001
    """Linear edge length after which a mesh node is added.
    Refer to the OCC BRepMesh_IncrementalMesh documentation for more details."""
    lin_deflection_is_absolute: bool = False
    """Whether the linear deflection tolerance is absolute or
    relative to edge length."""
    angular_deflection_tol: float = 0.5
    """Angular extent after which a a mesh node is added.
    Refer to the OCC BRepMesh_IncrementalMesh documentation for more details."""
    run_make_watertight: bool = True
    """Run the `make_watertight` subprocess from DAGMC. Attempts to
    make the DAGMC model watertight. Useful to run `check_geometry` manually
    on the output if it's not watertight."""
    save_vtk_model: bool = True
    """Save the DAGMC model as a VTK file, viewable in ParaView."""
    enable_ext_debug_logging: bool = False
    """Enable debug logging in the fast_ctd pipeline, in the C++ extension code."""
    use_cached_files: bool = True
    """Use the cached intermediary files if they exist, picking up where
    the last run ended. This can happen if the converter failed
    during a run and the intermediate files were not cleaned up.
    """
    clean_up_cached: bool = True
    """Clean up the cach intermediate files after the conversion is
    completes successfully."""

    def run_converter(
        self,
        shapes: list,
        names: list[str],
        comp_mat_mapping: dict[str, str],
        output_dagmc_model_path: Path,
    ) -> None:
        """
        Run the converter.

        Parameters
        ----------
        shapes:
            List of shapes to be converted.
        names:
            List of names for the shapes.
        comp_mat_mapping:
            Mapping of component names to material names.
        output_dagmc_model_path:
            Path to the output DAGMC model file.
        """
        return DAGMCConverterFastCTD(shapes, names, comp_mat_mapping).run(
            output_dagmc_model_path, converter_config=self
        )


class DAGMCConverterFastCTD(DAGMCConverter[DAGMCConverterFastCTDConfig]):
    """
    fast_ctd CA to DAGMC converter workflow.
    """

    def __init__(
        self,
        shapes: list[BluemiraGeoT],
        names: list[str],
        comp_mat_mapping: dict[str, str],
    ):
        """
        Initialize the converter.

        Parameters
        ----------
        shapes:
            List of shapes to be converted.
        names:
            List of names for the shapes.
        comp_mat_mapping:
            Mapping of component names to material names.
        """
        # fast_ctd internals convert names with spaces and -'s to underscores
        # do it here so the mapping is correct
        names = [name.replace(" ", "_").replace("-", "_") for name in names]
        comp_mat_mapping = {
            k.replace(" ", "_").replace("-", "_"): comp_mat_mapping[k]
            for k in comp_mat_mapping
        }
        super().__init__(shapes, names, comp_mat_mapping)

    def _run_imprint_all(self) -> list[BluemiraSolid]:
        slds = []
        names = []
        for shape, name in zip(self.shapes, self.names, strict=True):
            if isinstance(shape, BluemiraCompound):
                slds.extend(shape.solids)
                names.extend([name] * len(shape.solids))
            elif isinstance(shape, BluemiraSolid):
                slds.append(shape)
                names.append(name)
            else:
                raise TypeError(
                    f"Shape {shape} is not a BluemiraSolid or BluemiraCompound."
                )
        return imprint_solids(slds, names).solids

    def _run_imprint_per_compound(self) -> list[BluemiraGeo]:
        imprinted_shapes: list[BluemiraGeo] = []
        for shape, name in zip(self.shapes, self.names, strict=True):
            if isinstance(shape, BluemiraCompound):
                imprinted_shapes.append(imprint_solids(shape.solids, name).as_compound)
            elif isinstance(shape, BluemiraSolid):
                imprinted_shapes.append(shape)
            else:
                raise TypeError(
                    f"Shape {shape} is not a BluemiraSolid or BluemiraCompound."
                )
        return imprinted_shapes

    def run(
        self,
        output_dagmc_model_path: Path,
        converter_config: DAGMCConverterFastCTDConfig,
    ) -> None:
        """
        Convert the DAGMC file to a format suitable for use in Bluemira.

        Parameters
        ----------
        output_dagmc_model_path:
            Path to the output DAGMC model file.
        converter_config:
            Configuration options for the converter.

        Raises
        ------
        TypeError
            If the shapes are not of type BluemiraSolid or BluemiraCompound.
        """
        bluemira_print("Running fast_ctd CAD to DAGMC workflow")

        imprinted_geom_step_file_p = output_dagmc_model_path.with_name(
            f"{output_dagmc_model_path.stem}-imprinted.stp"
        )

        try:
            if converter_config.use_cached_files and imprinted_geom_step_file_p.exists():
                bluemira_print(
                    f"Using cached imprinted geometry: '{imprinted_geom_step_file_p}'"
                )
            else:
                if converter_config.imprint_geometry:
                    bluemira_print("Imprinting shapes")
                    imprinted_shapes = (
                        self._run_imprint_per_compound()
                        if converter_config.imprint_per_compound
                        else self._run_imprint_all()
                    )
                    bluemira_print("Saving imprinted geometry to .step file")
                else:
                    bluemira_print(
                        "Skipping imprinting, saving input geometry to .step file"
                    )
                    imprinted_shapes = self.shapes
                cadapi.save_cad(
                    [s.shape for s in imprinted_shapes],
                    imprinted_geom_step_file_p.as_posix(),
                    cad_format="step",
                    labels=self.names,  # they will match the order of the shapes
                )

            bluemira_print("Converting to DAGMC model using fast_ctd")
            bom = fast_ctd.step_to_dagmc_pipeline(
                step_file_path=imprinted_geom_step_file_p,
                output_dagmc_model_path=output_dagmc_model_path,
                comp_name_to_material_name_map=self.comp_mat_mapping,
                **converter_config.model_dump(),
            )
            bom = list(set(bom))

            bluemira_print(
                "Conversion to DAGMC model complete, "
                f"model saved to: '{output_dagmc_model_path}'"
            )
        except Exception as e:
            bluemira_error(f"Error during DAGMCConverterFastCTD run: {e}")
            raise

        model_meta_json_path = output_dagmc_model_path.with_suffix(".meta.json")
        bluemira_print(f"Writing model meta.json file to: '{model_meta_json_path}'")
        with open(model_meta_json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "bom": bom,
                    "comp_mat_mapping": self.comp_mat_mapping,
                    "converter_config": converter_config.model_dump(),
                    "model_path": output_dagmc_model_path.as_posix(),
                },
                f,
                indent=2,
            )

        # Remove the intermediate STEP file
        if converter_config.clean_up_cached and imprinted_geom_step_file_p.exists():
            bluemira_debug(f"Cleaning up {imprinted_geom_step_file_p}")
            imprinted_geom_step_file_p.unlink()
