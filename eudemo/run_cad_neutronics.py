# %%
from pathlib import Path

import openmc
import vtk
from matplotlib import pyplot as plt
from openmc_plasma_source import TokamakSource
from openmc_plasma_source.plotting import scatter_tokamak_source
from openmc_source_plotter import (
    plot_source_direction,
    plot_source_energy,
    plot_source_position,
)
from vtkmodules.util import numpy_support


# %%
def numpyToVTK(data, output_file, scaling=(1, 1, 1)):
    data_type = vtk.VTK_FLOAT
    shape = data.shape

    flat_data_array = data.flatten()
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=flat_data_array, deep=True, array_type=data_type
    )
    vtk_data.SetName("Flux")

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
    writer.SetFileName(output_file)
    writer.SetInputData(img)
    writer.Write()


fn = "dagmc_models/EUDEMO-wt.h5m"


# %%
dagmc_univ = openmc.DAGMCUniverse(filename=fn, auto_geom_ids=True).bounded_universe()
geometry = openmc.Geometry(dagmc_univ)

# %%
my_plasma = TokamakSource(
    mode="H",
    ion_density_centre=1.09e20,
    ion_density_peaking_factor=1.508,
    ion_density_pedestal=1.09e20,
    ion_density_separatrix=3e19,
    ion_temperature_centre=45.9,
    ion_temperature_peaking_factor=8.06,
    ion_temperature_pedestal=6.09,
    ion_temperature_separatrix=0.1,
    ion_temperature_beta=6,
    major_radius=9.204 * 100,
    minor_radius=2.9690322580645163 * 100,
    pedestal_radius=0.8 * 2.9690322580645163 * 100,
    elongation=1.6812300445035215,
    shafranov_factor=0.5525451657419864 * 100,
    triangularity=0.3336744697351377,
)
# %%
# source plots
plot = True
if plot:
    ax = scatter_tokamak_source(my_plasma, quantity="ion_temperature")
    ax.set_title("Source particle profile with ion temperatures")
    ax.set_xlabel("X [cm]")
    ax.set_ylabel("Z [cm]")
    plt.colorbar(ax.collections[0], label="Ion Temperature [keV]")
    plt.show(block=True)

    f = plot_source_energy(my_plasma.sources)
    f.update_layout(title="Source particle energy profile", showlegend=False)
    f.show()
    # f = plot_source_direction(my_plasma.sources)
    # f.show()
    f = plot_source_position(my_plasma.sources, n_samples=5000)
    f.update_layout(
        title={
            "text": "Source particle location in x-y",
            "yref": "paper",
            "y": 0.95,
            "yanchor": "bottom",
        },
        showlegend=False,
        margin={"t": 10, "l": 10, "b": 10, "r": 10},
        scene_camera={
            "eye": {"x": 0, "y": 0, "z": 1},
            "up": {"x": 1, "y": 0, "z": 1},
            "projection": {"type": "orthographic"},
        },
        scene={"xaxis_title": "X [cm]", "yaxis_title": "Y [cm]", "zaxis_title": ""},
    )
    f.show()

# %%
settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.batches = 10
settings.particles = 10000
settings.source = my_plasma.sources
settings.output = {"path": (Path(__file__).parent / "omc_run_output").as_posix()}

mats = []
for i in range(1, 9):
    iron = openmc.Material(name=f"mat_{i}")
    iron.add_nuclide("Fe54", 0.0564555822608)
    iron.add_nuclide("Fe56", 0.919015287728)
    iron.add_nuclide("Fe57", 0.0216036861685)
    iron.add_nuclide("Fe58", 0.00292544384231)
    iron.set_density("g/cm3", 7.874)
    mats.append(iron)
materials = openmc.Materials(mats)

mesh = openmc.RegularMesh.from_domain(geometry, dimension=(100, 100, 100))

mesh_filter = openmc.MeshFilter(mesh)

material_filter = openmc.MaterialFilter(mats)

flux_tally = openmc.Tally(name="all comps flux")
flux_tally.filters = [mesh_filter, material_filter]
flux_tally.scores = ["flux"]

# %%
model = openmc.Model(
    materials=materials, geometry=geometry, tallies=[flux_tally], settings=settings
)
model.export_to_model_xml()

# %%
# sp_filename = model.run()
sp_filename = "omc_run_output/statepoint.10.h5"
# %%
with openmc.StatePoint(sp_filename) as sp:
    flux_tally = sp.get_tally(name="all comps flux")

flux_mean = flux_tally.mean.reshape(100, 100, 100, 8)
fm_n = flux_mean.sum(axis=-1)
plt.figure(figsize=(10, 10))
# plt.imshow(flux_mean[50, :, :], origin="lower")
plt.imshow(fm_n.sum(axis=2), origin="lower")
# for i in range(flux_mean.shape[-1]):
#     fm_n = flux_mean[:, :, :, i]
#     plt.figure(figsize=(10, 10))
#     # plt.imshow(flux_mean[50, :, :], origin="lower")
#     plt.imshow(fm_n.sum(axis=2), origin="lower")

# %%
fm_n_save = fm_n[:, :50, :]
a = dagmc_univ.bounding_box.width
a[1] /= 2
scaling = tuple(round(t / c) for c, t in zip(fm_n_save.shape, a, strict=False))
numpyToVTK(fm_n_save, "flux_tot.vti", scaling)
# for i in range(flux_mean.shape[-1]):
#     fm_n = flux_mean[:, :, :, i]
#     scaling = tuple(
#         round(t / c) for c, t in zip(fm_n.shape, dagmc_univ.bounding_box.width)
#     )
#     numpyToVTK(fm_n, f"flux_{i}.vti", scaling)

# %%
