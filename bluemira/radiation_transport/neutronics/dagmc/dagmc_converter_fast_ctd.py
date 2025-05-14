# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
fast_ctd DAGMC converter workflow definition.
"""

from pathlib import Path
from typing import Literal

from bluemira.base.look_and_feel import bluemira_debug, bluemira_error, bluemira_print
from bluemira.codes import _freecadapi as cadapi
from bluemira.codes import fast_ctd
from bluemira.codes.python_occ import imprint_solids
from bluemira.radiation_transport.neutronics.dagmc.dagmc_converter import (
    DAGMCConverter,
    DAGMCConverterConfig,
)


class DAGMCConverterFastCTDConfig(DAGMCConverterConfig):
    """
    Enum for DAGMC converters.
    """

    converter_type: Literal["fast_ctd"] = "fast_ctd"

    minimum_include_volume: float = 1
    fix_step_to_brep_geometry: bool = False
    merge_dist_tolerance: float = 0.001
    lin_deflection_tol: float = 0.001
    lin_deflection_is_absolute: bool = False
    angular_deflection_tol: float = 0.5
    run_make_watertight: bool = True
    enable_debug_logging: bool = False
    clean_up: bool = True

    def run_converter(
        self,
        shapes: list,
        names: list[str],
        material_name_map: dict[str, str] | None,
        output_dagmc_model_path: str | Path,
    ) -> None:
        """
        Run the converter.

        Parameters
        ----------
        shapes:
            List of shapes to be converted.
        names:
            List of names for the shapes.
        material_name_map:
            Mapping of component names to material names.

        Returns
        -------
            DAGMCConverterFastCTD
                The converter object.
        """
        return DAGMCConverterFastCTD(shapes, names, material_name_map).run(
            output_dagmc_model_path, converter_config=self
        )


class DAGMCConverterFastCTD(DAGMCConverter[DAGMCConverterFastCTDConfig]):
    """
    Class to convert a DAGMC file to a format suitable for use in the
    Bluemira radiation transport module using fast_ctd.
    """

    def run(
        self,
        output_dagmc_model_path: str | Path,
        converter_config: DAGMCConverterFastCTDConfig,
    ) -> None:
        """
        Convert the DAGMC file to a format suitable for use in Bluemira.

        Parameters
        ----------
        output_dagmc_model_path:
            Path to the output DAGMC model file.
        kwargs:
            Additional keyword arguments to pass to the conversion function.
        """
        imp_res = imprint_solids(self.shapes, self.names)

        imprinted_geom_step_file_p = Path(output_dagmc_model_path).with_suffix(
            ".imp.stp"
        )
        try:
            bluemira_print(f"Saving imprinted geometry to {imprinted_geom_step_file_p}")
            cadapi.save_cad(
                [s.shape for s in imp_res.solids],
                imprinted_geom_step_file_p.as_posix(),
                cad_format="step",
                labels=imp_res.labels,
            )
            bluemira_print("Converting to DAGMC model")
            fast_ctd.step_to_dagmc_pipeline(
                step_file_path=imprinted_geom_step_file_p,
                output_dagmc_model_path=output_dagmc_model_path,
                comp_name_to_material_name_map=self.material_name_map,
                # Config options
                minimum_include_volume=converter_config.minimum_include_volume,
                fix_step_to_brep_geometry=converter_config.fix_step_to_brep_geometry,
                merge_dist_tolerance=converter_config.merge_dist_tolerance,
                lin_deflection_tol=converter_config.lin_deflection_tol,
                lin_deflection_is_absolute=converter_config.lin_deflection_is_absolute,
                angular_deflection_tol=converter_config.angular_deflection_tol,
                run_make_watertight=converter_config.run_make_watertight,
                enable_debug_logging=converter_config.enable_debug_logging,
                clean_up=converter_config.clean_up,
            )
            bluemira_print(
                f"Conversion to DAGMC model completed: {output_dagmc_model_path}"
            )
        except Exception as e:
            bluemira_error(f"Error during DAGMCConverterFastCTD run: {e}")
            raise
        finally:
            # Remove the intermediate STEP file
            if converter_config.clean_up and imprinted_geom_step_file_p.exists():
                bluemira_debug(f"Cleaning up {imprinted_geom_step_file_p}")
                imprinted_geom_step_file_p.unlink()
