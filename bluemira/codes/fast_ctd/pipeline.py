# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Pipeline for converting STEP files to DAGMC models using fast_ctd.
"""

from pathlib import Path

from fast_ctd import (
    facet_brep_to_dagmc,
    make_watertight,
    merge_brep_geometries,
    step_to_brep,
)

# @dataclass
# class StepToDagmcPipelineParams:


def step_to_dagmc_pipeline(
    step_file_path: str | Path,
    output_dagmc_model_path: str | Path,
    comp_name_to_material_map: dict[str, str] | None = None,
    *,
    enable_debug_logging: bool = False,
):
    """
    Convert a STEP file to a DAGMC model using fast_ctd.

    This function performs the following steps:
        1. Convert the STEP file to a BREP file.
        2. Create a CSV file mapping component names to materials.
        3. Perform a merge of similar (imprinted) entities.
        4. Facet/mesh the BREP model and convert it into DAGMC model using MOAB.
        5. Make the DAGMC model watertight.
        6. Clean up intermediate files.

    Parameters
    ----------
    step_file_path:
        The path to the input STEP file.
    output_dagmc_model_path:
        The path to the output DAGMC model file.
    comp_name_to_material_map:
        A dictionary mapping component names to material names.
        If None, the component names are used as the material names.
    enable_debug_logging:
        If True, enable debug logging within the C++ extension code. Default is False.
    """
    step_file_path = Path(step_file_path)
    output_dagmc_model_path = Path(output_dagmc_model_path)

    intm_materials_csv_file_path = step_file_path.with_suffix(".csv")
    intm_brep_file = step_file_path.with_suffix(".brep")
    intm_merged_brep_file = step_file_path.with_suffix("_merged.brep")
    intm_dagmc_file = output_dagmc_model_path.with_suffix("_nwt.h5m")

    try:
        comps_info = step_to_brep(
            step_file_path,
            intm_brep_file,
            enable_logging=enable_debug_logging,
        )

        # Map component names to materials and write to the CSV file
        # If comp_name_to_material_map is None, use the component names as the material
        mats_list = (
            [comp_name_to_material_map[e[1]] for e in comps_info]
            if comp_name_to_material_map
            else [e[1] for e in comps_info]
        )
        with open(intm_materials_csv_file_path, "w") as f:
            for i, mat_name in enumerate(mats_list):
                f.write(mat_name)
                if i != len(mats_list) - 1:
                    f.write("\n")

        merge_brep_geometries(
            intm_brep_file,
            intm_merged_brep_file,
            enable_logging=enable_debug_logging,
        )
        facet_brep_to_dagmc(
            intm_merged_brep_file,
            output_h5m_file=output_dagmc_model_path,
            materials_csv_file=intm_materials_csv_file_path,
            enable_logging=enable_debug_logging,
        )
        make_watertight(intm_dagmc_file, output_dagmc_model_path)
    finally:
        # Clean up intermediate files
        for file in [
            intm_materials_csv_file_path,
            intm_brep_file,
            intm_merged_brep_file,
            intm_dagmc_file,
        ]:
            if file.exists():
                file.unlink()
