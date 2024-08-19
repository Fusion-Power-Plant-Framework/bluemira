# from cad_to_dagmc import CadToDagmc

# my_model = CadToDagmc()
# # my_model.add_stp_file(
# #     filename="Divertor.step",
# #     material_tags=["Divertor"] * 48,
# # )
# my_model.add_stp_file(
#     filename="Blanket.step",
#     material_tags=["blanket"] * 80,
# )

# my_model.export_dagmc_h5m_file(
#     max_mesh_size=300,
#     min_mesh_size=200,
#     mesh_algorithm=1,
#     implicit_complement_material_tag="air",
# )
# %%
import math
from os import write
from pathlib import Path

import openmc
from matplotlib import pyplot as plt
from openmc_plasma_source import TokamakSource
from openmc_plasma_source.plotting import plot_tokamak_source_3D
from openmc_source_plotter import (
    plot_source_direction,
    plot_source_energy,
    plot_source_position,
)

# %%
dagmc_univ = openmc.DAGMCUniverse(filename="dagmc_models/mac.h5m", auto_geom_ids=True)

graveyard = openmc.Sphere(r=1500, boundary_type="vacuum")

cad_cell = openmc.Cell(region=-graveyard, fill=dagmc_univ)

root = openmc.Universe()
root.add_cells([cad_cell])
geometry = openmc.Geometry(root)

# %%
my_plasma = TokamakSource(
    mode="H",
    ion_density_centre=1.09e20,
    ion_density_peaking_factor=1,
    ion_density_pedestal=1.09e20,
    ion_density_separatrix=3e19,
    ion_temperature_centre=45.9,
    ion_temperature_peaking_factor=8.06,
    ion_temperature_pedestal=6.09,
    ion_temperature_separatrix=0.1,
    ion_temperature_beta=6,
    major_radius=9.204,
    minor_radius=0.5,
    pedestal_radius=0.8 * 0.5,
    elongation=0.5,
    shafranov_factor=0.1,
    triangularity=0.3336744697351377,
)

# source plots
plot = False
if plot:
    ax = plot_tokamak_source_3D(my_plasma)
    plt.show(block=True)

    f = plot_source_energy(my_plasma.sources)
    f.show()
    f = plot_source_direction(my_plasma.sources)
    f.show()
    f = plot_source_position(my_plasma.sources)
    f.show()

# %%
settings = openmc.Settings()
settings.run_mode = "fixed source"
settings.batches = 10
settings.particles = 10000
settings.source = my_plasma.sources
settings.rel_max_lost_particles = 0.8
settings.max_write_lost_particles = int(1e9)
settings.sourcepoint = {"write": False}
settings.output = {"path": (Path(__file__).parent / "omc_run_output").as_posix()}

iron = openmc.Material(name="test_mat_A")
iron.add_nuclide("Fe54", 0.0564555822608)
iron.add_nuclide("Fe56", 0.919015287728)
iron.add_nuclide("Fe57", 0.0216036861685)
iron.add_nuclide("Fe58", 0.00292544384231)
iron.set_density("g/cm3", 7.874)

# water = openmc.Material(name="water")
# water.add_nuclide("H1", 2.0, "ao")
# water.add_nuclide("O16", 1.0, "ao")
# water.set_density("g/cc", 1.0)
# water.add_s_alpha_beta("c_H_in_H2O")
# water.id = 41

materials = openmc.Materials([iron])

mesh = openmc.RegularMesh.from_domain(geometry, dimension=(100, 100, 100))

mesh_filter = openmc.MeshFilter(mesh)

material_filter = openmc.MaterialFilter([iron])

flux_tally = openmc.Tally(name="blanket flux")
flux_tally.filters = [mesh_filter, material_filter]
flux_tally.scores = ["flux"]

# %%
model = openmc.Model(
    materials=materials, geometry=geometry, tallies=[flux_tally], settings=settings
)
model.export_to_model_xml()

# %%
sp_filename = model.run()

# %%
with openmc.StatePoint(sp_filename) as sp:
    flux_tally = sp.get_tally(name="blanket flux")

flux_mean = flux_tally.mean.reshape(*mesh.dimension)
plt.figure(figsize=(10, 10))
plt.imshow(flux_mean[50, :, :], origin="lower")
# plt.imshow(flux_mean.sum(axis=2), origin="lower")


# %%
