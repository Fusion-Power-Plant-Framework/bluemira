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
import vtk
from openmc_plasma_source import tokamak_source
from vtkmodules.util import numpy_support

from bluemira.base.file import get_bluemira_root
from bluemira.base.look_and_feel import bluemira_print
from bluemira.materials.cache import establish_material_cache, get_cached_material

par = Path(__file__).parent
# %% [markdown]
# # Running a DAGMC model in OpenMC

# %%
n_batches = 5
particles_per_batch = 10000
# there are other parameters that can be set, see the tokamak_source below
source_major_radius = 900  # cm, the reactor major radius
source_minor_radius = 292.258  # cm, the reactor minor radius

# OptimisedReactor can be exported by running examples/design/optimised_reactor.ex.py
# Change this to the name of your DAGMC model
dag_model_path = par / "EUDEMO.h5m"
meta_data_path = par / "EUDEMO.meta.json"

# %%
establish_material_cache([
    Path(get_bluemira_root()) / "data" / "materials" / "materials.json",
    Path(get_bluemira_root()) / "data" / "materials" / "mixtures.json",
])
omc_output_path = par / "omc"
# Ensure OpenMC output directory exists
omc_output_path.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Running the DAGMC model in OpenMC

# %%
dagmc_univ = openmc.DAGMCUniverse(
    filename=dag_model_path.as_posix(),
    auto_geom_ids=True,
).bounded_universe()

# load model materials
with open(meta_data_path) as meta_file:
    bom = json.load(meta_file)["bom"]
openmc_mats = [
    get_cached_material(mat_name).to_openmc_material(temperature=294) for mat_name in bom
]

# load DAG model
geometry = openmc.Geometry(dagmc_univ)

my_source = tokamak_source(
    elongation=1.557,
    ion_density_centre=1.09e20,
    ion_density_pedestal=1.09e20,
    ion_density_peaking_factor=1,
    ion_density_separatrix=3e19,
    ion_temperature_centre=45.9e3,
    ion_temperature_pedestal=6.09e3,
    ion_temperature_separatrix=0.1e3,
    ion_temperature_peaking_factor=8.06,
    ion_temperature_beta=6,
    major_radius=source_major_radius,
    minor_radius=source_minor_radius,
    pedestal_radius=0.8 * source_minor_radius,
    mode="H",
    shafranov_factor=0.44789,
    triangularity=0.270,
    fuel={"D": 0.5, "T": 0.5},
)

settings = openmc.Settings()
settings.batches = n_batches
settings.particles = 10000
settings.inactive = 0
settings.run_mode = "fixed source"
settings.source = my_source
settings.output = {"path": omc_output_path.as_posix()}

# TALLIES
# based on https://github.com/fusion-energy/magnetic_fusion_openmc_dagmc_paramak_example

# record the heat deposited in entire geometry
heating_cell_tally = openmc.Tally(name="heating")
heating_cell_tally.scores = ["heating"]

# record the total TBR
tbr_cell_tally = openmc.Tally(name="tbr")
tbr_cell_tally.scores = ["(n,Xt)"]

# mesh that covers the geometry
mesh = openmc.RegularMesh.from_domain(geometry, dimension=(100, 100, 100))
mesh_filter = openmc.MeshFilter(mesh)

# mesh tally using the previously created mesh and records heating on the mesh
heating_mesh_tally = openmc.Tally(name="heating_on_mesh")
heating_mesh_tally.filters = [mesh_filter]
heating_mesh_tally.scores = ["heating"]

# mesh tally using the previously created mesh and records TBR on the mesh
tbr_mesh_tally = openmc.Tally(name="tbr_on_mesh")
tbr_mesh_tally.filters = [mesh_filter]
tbr_mesh_tally.scores = ["(n,Xt)"]

flux_mesh_tally = openmc.Tally(name="flux_on_mesh")
flux_mesh_tally.filters = [mesh_filter]
flux_mesh_tally.scores = ["flux"]

tallies = openmc.Tallies([
    tbr_cell_tally,
    tbr_mesh_tally,
    heating_cell_tally,
    heating_mesh_tally,
    flux_mesh_tally,
])

model = openmc.Model(
    geometry=geometry,
    tallies=tallies,
    settings=settings,
)
# For some reason, have to set the materials after the model is created
# (this is a bug in OpenMC)
model.materials = openmc_mats
model.export_to_model_xml()
model.run()

# %% [markdown]
# ## Extracting the OpenMC results
#
# This section extracts the results from the OpenMC simulation, including the
# total breeding ratio (TBR) and the heating deposited in the reactor.


# %%
def numpy_to_vtk(data, output_name, scaling=(1, 1, 1)):
    """Convert a numpy array to a VTK image data file."""
    data_type = vtk.VTK_FLOAT
    shape = data.shape

    flat_data_array = data.flatten()
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=flat_data_array, deep=True, array_type=data_type
    )
    vtk_data.SetName(output_name)

    half_x = int(0.5 * scaling[0] * (shape[0] - 1))
    half_y = int(0.5 * scaling[1] * (shape[1] - 1))
    half_z = int(0.5 * scaling[2] * (shape[2] - 1))

    img = vtk.vtkImageData()
    img.GetPointData().SetScalars(vtk_data)
    img.SetSpacing(scaling[0], scaling[1], scaling[2])
    img.SetDimensions(shape[0], shape[1], shape[2])
    img.SetOrigin(-half_x, -half_y, -half_z)

    # Save the VTK file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(f"{output_name}.vti")
    writer.SetInputData(img)
    writer.Write()


sp_path = omc_output_path / f"statepoint.{n_batches}.h5"
sp = openmc.StatePoint(sp_path.as_posix())

tbr_cell_tally = sp.get_tally(name="tbr")
tbr_mesh_tally = sp.get_tally(name="tbr_on_mesh")
heating_cell_tally = sp.get_tally(name="heating")
heating_mesh_tally = sp.get_tally(name="heating_on_mesh")
flux_mesh_tally = sp.get_tally(name="flux_on_mesh")

bluemira_print(f"The reactor has a TBR of {tbr_cell_tally.mean.sum()}")
bluemira_print(f"Standard deviation on the TBR is {tbr_cell_tally.std_dev.sum()}")

bluemira_print(
    f"The heating of {heating_cell_tally.mean.sum() / 1e6} MeV "
    "per source particle is deposited"
)
bluemira_print(
    f"Standard deviation on the heating tally is {heating_cell_tally.std_dev.sum()}"
)

mesh = tbr_mesh_tally.find_filter(openmc.MeshFilter).mesh
mesh.write_data_to_vtk(
    filename="tbr_mesh_mean.vtk",
    datasets={"mean": tbr_mesh_tally.mean},
)

model_w = dagmc_univ.bounding_box.width

heating_mesh_mean = heating_mesh_tally.mean.reshape(100, 100, 100)
flux_mesh_mean = flux_mesh_tally.mean.reshape(100, 100, 100)
scaling = tuple(
    round(t / c) for c, t in zip(heating_mesh_mean.shape, model_w, strict=False)
)
# If you want to view only the +ve x half of the tally, uncomment the next line
# heating_mesh_mean[:, :, heating_mesh_mean.shape[2] // 2 : -1] = 0
numpy_to_vtk(heating_mesh_mean, "heating_mesh_mean", scaling)
numpy_to_vtk(flux_mesh_mean, "flux_mesh_mean", scaling)
