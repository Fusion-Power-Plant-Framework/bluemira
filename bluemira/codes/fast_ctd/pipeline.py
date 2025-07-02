# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Pipeline for converting STEP files to DAGMC models using fast_ctd.
"""

import contextlib
from pathlib import Path
from subprocess import CalledProcessError  # noqa: S404
from typing import Literal

from bluemira.base.look_and_feel import (
    bluemira_critical,
    bluemira_debug,
    bluemira_print,
    bluemira_warn,
)
from bluemira.codes.fast_ctd._guard import fast_ctd_guard

with contextlib.suppress(ImportError):
    from fast_ctd import (
        check_watertight,
        dagmc_to_vtk,
        decode_tightness_checks,
        facet_brep_to_dagmc,
        make_watertight,
        merge_brep_geometries,
        step_to_brep,
    )


def _run_check_or_make_watertight(
    dagmc_file: Path,
    output_h5m_file: Path,
    *,
    check_or_make: Literal["check", "make"],
):
    # just run make or check. Their output is the same
    # as make runs check at the end internally
    cmd = "check_watertight" if check_or_make == "check" else "make_watertight"
    if check_or_make == "make" and output_h5m_file is None:
        raise ValueError(
            "output_h5m_file must be provided when check_or_make is 'make'."
        )

    bluemira_print(f"Running DAGMC `{cmd}`")
    # will raise CalledProcessError if the command fails
    sub_proc_res = (
        check_watertight(dagmc_file)
        if check_or_make == "check"
        else make_watertight(dagmc_file, output_h5m_file)
    )

    percentages = decode_tightness_checks(sub_proc_res.stdout)
    if not percentages or any(p > 0 for p in percentages):
        stout_log_dump_path = dagmc_file.with_name(
            f"{output_h5m_file.stem}-{cmd}.stdout.txt",
        )
        sterr_log_dump_path = dagmc_file.with_name(
            f"{output_h5m_file.stem}-{cmd}.stderr.txt",
        )
        with stout_log_dump_path.open("w") as log_dump:
            log_dump.write(sub_proc_res.stdout)
        with sterr_log_dump_path.open("w") as log_dump:
            log_dump.write(sub_proc_res.stderr)

        bluemira_warn(
            f"`{cmd}` completed successfully, "
            "but the model is not watertight or the output could not be parsed. "
            "Check log files for further details.:\n"
            f"stout_log: {stout_log_dump_path}\n"
            f"sterr_log: {sterr_log_dump_path}\n",
        )
    else:
        bluemira_print(f"`{cmd}` completed successfully, showing no leaky volumes.")


@fast_ctd_guard
def step_to_dagmc_pipeline(
    step_file_path: str | Path,
    output_dagmc_model_path: str | Path,
    comp_name_to_material_name_map: dict[str, str] | None = None,
    *,
    # Optional parameters for fast_ctd
    minimum_include_volume: float = 1.0,
    fix_step_to_brep_geometry: bool = False,
    merge_dist_tolerance: float = 0.001,
    lin_deflection_tol: float = 0.001,
    lin_deflection_is_absolute: bool = False,
    angular_deflection_tol: float = 0.5,
    run_make_watertight: bool = True,
    save_vtk_model: bool = True,
    enable_ext_debug_logging: bool = False,
    use_cached_files: bool = True,
    clean_up_cached: bool = True,
    **kwargs,  # noqa: ARG001
) -> list[str]:
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
    comp_name_to_material_name_map:
        A dictionary mapping component names to material names.
        If None, the component names are used as the material names.

    Returns
    -------
        list[str]:
            The list of material names used by the model.

    Raises
    ------
    CalledProcessError
        If make_watertight fails.
    """
    step_file_path = Path(step_file_path)
    output_dagmc_model_path = Path(output_dagmc_model_path)
    output_vtk_file_path = output_dagmc_model_path.with_suffix(".vtk")

    intm_materials_csv_file_path = output_dagmc_model_path.with_suffix(".mats.csv")
    intm_brep_file = output_dagmc_model_path.with_suffix(".brep")
    intm_merged_brep_file = output_dagmc_model_path.with_suffix(".merged.brep")
    intm_dagmc_file = (
        output_dagmc_model_path.with_suffix(".nwt.h5m")
        if run_make_watertight
        else output_dagmc_model_path
    )

    if (
        use_cached_files
        and intm_brep_file.exists()
        and intm_materials_csv_file_path.exists()
    ):
        bluemira_print(f"Skipping `step_to_brep`, using '{intm_brep_file}'")
        bluemira_print(
            f"Skipping materials files creation, using '{intm_materials_csv_file_path}'"
        )
        mats_list = []
    else:
        bluemira_print("Running `step_to_brep`")
        comps_info = step_to_brep(
            step_file_path,
            intm_brep_file,
            minimum_volume=minimum_include_volume,
            fix_geometry=fix_step_to_brep_geometry,
            enable_logging=enable_ext_debug_logging,
        )
        bluemira_print("Mapping components to materials and writing materials CSV file")
        # If comp_name_to_material_name_map is None,
        # use the component names as the material
        mats_list = (
            [comp_name_to_material_name_map[e[1]] for e in comps_info]
            if comp_name_to_material_name_map
            else [e[1] for e in comps_info]
        )
        with open(intm_materials_csv_file_path, "w") as f:
            for i, mat_name in enumerate(mats_list):
                f.write(mat_name)
                if i != len(mats_list) - 1:
                    f.write("\n")

    if use_cached_files and intm_merged_brep_file.exists():
        bluemira_print(
            f"Skipping `merge_brep_geometries`, using '{intm_merged_brep_file}'"
        )
    else:
        bluemira_print("Running `merge_brep_geometries`")
        merge_brep_geometries(
            intm_brep_file,
            intm_merged_brep_file,
            dist_tolerance=merge_dist_tolerance,
            enable_logging=enable_ext_debug_logging,
        )

    bluemira_print("Running `facet_brep_to_dagmc`")
    facet_brep_to_dagmc(
        intm_merged_brep_file,
        output_h5m_file=intm_dagmc_file,
        materials_csv_file=intm_materials_csv_file_path,
        lin_deflection_tol=lin_deflection_tol,
        tol_is_absolute=lin_deflection_is_absolute,
        ang_deflection_tol=angular_deflection_tol,
        enable_logging=enable_ext_debug_logging,
    )

    try:
        _run_check_or_make_watertight(
            intm_dagmc_file,
            output_dagmc_model_path,
            check_or_make="make" if run_make_watertight else "check",
        )
    except CalledProcessError as e:
        bluemira_critical(
            f"Process failed to run, dagmc model saved to {intm_dagmc_file}"
        )
        bluemira_critical(f"stdout:\n{e.stdout}")
        bluemira_critical(f"stderr:\n{e.stderr}")
        raise

    if save_vtk_model:
        bluemira_print("Running `dagmc_to_vtk`, converting model to VTK")
        dagmc_to_vtk(output_dagmc_model_path, output_vtk_file_path)

    if clean_up_cached:
        bluemira_print("Cleaning up intermediate files")

        # Clean up intermediate files
        for file in [
            intm_materials_csv_file_path,
            intm_brep_file,
            intm_merged_brep_file,
            intm_dagmc_file,
        ]:
            # if make_watertight not run, do not delete the file
            if not run_make_watertight and file == intm_dagmc_file:
                continue

            bluemira_debug(f"Cleaning up: {file}")
            if file.exists():
                file.unlink()

    return mats_list
