# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Example showing how to run a DAGMC model OpenMC.
"""

# %%
import json
from pathlib import Path

import openmc

from bluemira.base.file import get_bluemira_root
from bluemira.base.look_and_feel import bluemira_print
from bluemira.codes.openmc import DAGMCSolver
from bluemira.codes.openmc.sources import make_tokamak_source
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.materials.cache import establish_material_cache, get_cached_material

par = Path(__file__).parent
# %% [markdown]
# # Running a DAGMC model in OpenMC

# %%
# OptimisedReactor can be exported by running examples/design/optimised_reactor.ex.py
# Change this to the name of your DAGMC model
dag_model_path = par / "OptimisedReactor.h5m"
meta_data_path = par / "OptimisedReactor.meta.json"
eq_data_path = par / "OptimisedReactor.eq.json"

# %%


# Fill me in eg "/a/path/to/"
cross_section_folder = ""

if not cross_section_folder:
    raise ValueError("Please fill in the path to your cross_section xml file")

cross_section_xml = Path(cross_section_folder, "cross_sections.xml")
# %%
establish_material_cache([
    Path(get_bluemira_root(), "examples", "design", "design_materials.py")
    .resolve()
    .as_posix(),
    "matproplib",
])

omc_output_path = par / "omc"
# Ensure OpenMC output directory exists
omc_output_path.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Running the DAGMC model in OpenMC

# %%

params = {
    "R_0": {"value": 9, "unit": "m"},
    "profile_rho_ped": {"value": 0.94, "unit": "dimensionless"},
    "P_fus": {"value": 2, "unit": "GW"},
    "n_profile_alpha": {"value": 1.0, "unit": "dimensionless"},
    "n_e_core": {"value": 1e20, "unit": "1/m³"},
    "n_e_ped": {"value": 5e19, "unit": "1/m³"},
    "n_e_sep": {"value": 3e19, "unit": "1/m³"},
    "T_profile_alpha": {"value": 1.45, "unit": "dimensionless"},
    "T_profile_beta": {"value": 2.0, "unit": "dimensionless"},
    "T_e_core": {"value": 3.8e-15, "unit": "J"},
    "T_e_ped": {"value": 8.8e-16, "unit": "J"},
    "T_e_sep": {"value": 1.6e-17, "unit": "J"},
    "T_ie_ratio": {"value": 1.0, "unit": "dimensionless"},
    "n_i_fuel": {"value": 6e19, "unit": "1/m³"},
    "n_e": {"value": 7e19, "unit": "1/m³"},
    "shaf_shift": {"value": 0.6, "unit": "m"},
}

build_config = {
    "cross_section_xml": cross_section_xml.as_posix(),
    "photon_transport": True,
    "electron_treatment": "ttb",
    "export_dagmc_model": True,
    "dagmc_export_dir": par,
    "neutronics_output_path": par,
    "particles": 10000,
    "batches": 2,
    "rel_max_lost_particles": 0.01,
    "max_lost_particles": 10,
    "converter_config": {
        "converter_type": "fast_ctd",
        "imprint_geometry": True,
        "imprint_per_compound": True,
        "minimum_include_volume": 0.001,
        "fix_step_to_brep_geometry": False,
        "merge_dist_tolerance": 0.001,
        "lin_deflection_tol": 0.001,
        "lin_deflection_is_absolute": False,
        "angular_deflection_tol": 0.5,
        "run_make_watertight": True,
        "save_vtk_model": True,
        "enable_ext_debug_logging": False,
        "use_cached_files": False,
        "clean_up_cached": False,
    },
}

# load model materials
with open(meta_data_path) as meta_file:
    bom = json.load(meta_file)["bom"]

materials = [
    get_cached_material(mat_name).convert(
        "openmc", {"temperature": 301, "pressure": 101325}
    )
    for mat_name in bom
]


# create tally function
def dagmc_tallys(
    material_list,  # noqa: ARG001
    model: openmc.Geometry,
    mesh_shape: tuple[float, ...] = (100, 100, 100),
):
    """Create tallys for openmc"""
    # mesh that covers the geometry
    mesh = openmc.RegularMesh.from_domain(model, dimension=mesh_shape)
    mesh_filter = openmc.MeshFilter(mesh)

    # name, scores, filters
    return [
        ("heating", "heating", None),
        ("heating_on_mesh", "heating", [mesh_filter]),
        ("TBR", "(n,Xt)", None),
        ("tbr_on_mesh", "(n,Xt)", [mesh_filter]),
        ("flux_on_mesh", "flux", [mesh_filter]),
    ]


# load DAG model
solver = DAGMCSolver(
    params,
    build_config,
    Equilibrium.from_eqdsk(eq_data_path, from_cocos="bluemira"),
    source=make_tokamak_source,
    dagmc_model_path=dag_model_path,
    materials=materials,
    tally_function=dagmc_tallys,
)

openmc_results, derived_results = solver.execute(run_mode="run")


# %% [markdown]
# ## Extracting the OpenMC results
#
# This section extracts the results from the OpenMC simulation, including the
# total breeding ratio (TBR) and the heating deposited in the reactor.
# %%

bluemira_print(f"The reactor has a TBR of {derived_results.TBR}")
bluemira_print(f"Standard deviation on the TBR is {openmc_results.tbr_err}")

heating_cell_tally = openmc_results.statepoint.get_tally(name="heating")
bluemira_print(
    f"The heating of {heating_cell_tally.mean.sum() / 1e6} MeV "
    "per source particle is deposited"
)
bluemira_print(
    "Standard deviation on the heating tally is"
    f" {heating_cell_tally.std_dev.sum() / 1e3} keV"
)
