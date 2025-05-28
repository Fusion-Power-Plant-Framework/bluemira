import math
from pathlib import Path

import openmc

omc_output_path = Path(__file__).parent / "omc"
dag_fn = "OptimisedReactor.h5m"

# %% SOURCE
major_radius = 894  # cm
minor_radius = 288  # cm
radius = openmc.stats.Discrete(
    [major_radius - minor_radius, major_radius + minor_radius], [1, 1]
)
z_values = openmc.stats.Discrete([-minor_radius, minor_radius], [1, 1])
angle = openmc.stats.Uniform(a=0.0, b=math.radians(360))
my_source = openmc.IndependentSource(
    space=openmc.stats.CylindricalIndependent(
        r=radius, phi=angle, z=z_values, origin=(0.0, 0.0, 0.0)
    ),
    angle=openmc.stats.Isotropic(),
    energy=openmc.stats.muir(e0=14080000.0, m_rat=5.0, kt=20000.0),
)

# specifies the simulation computational intensity
settings = openmc.Settings()
settings.batches = 10
settings.particles = 10000
settings.inactive = 0
settings.run_mode = "fixed source"
settings.source = my_source
settings.output = {"path": omc_output_path.as_posix()}


# %% MATERIALS

mat_names = ["SS316-LN", "Toroidal_Field_Coil_2015"]
mats = []
for n in mat_names:
    iron = openmc.Material(name=n)
    iron.add_nuclide("Fe54", 0.0564555822608)
    iron.add_nuclide("Fe56", 0.919015287728)
    iron.add_nuclide("Fe57", 0.0216036861685)
    iron.add_nuclide("Fe58", 0.00292544384231)
    iron.set_density("g/cm3", 7.874)
    mats.append(iron)
air_mat = openmc.Material(name="air")
air_mat.add_element("N", 0.7)
air_mat.add_element("O", 0.3)
air_mat.set_density("g/cm3", 0.001)
mats.append(air_mat)
materials = openmc.Materials(mats)

# %% DAG GEOMETRY

dagmc_univ = openmc.DAGMCUniverse(
    filename=dag_fn,
    auto_geom_ids=True,
).bounded_universe()
geometry = openmc.Geometry(dagmc_univ)

# %% TALLIES

# adds a tally to record the heat deposited in entire geometry
heating_cell_tally = openmc.Tally(name="heating")
heating_cell_tally.scores = ["heating"]

# adds a tally to record the total TBR
tbr_cell_tally = openmc.Tally(name="tbr")
tbr_cell_tally.scores = ["(n,Xt)"]

# creates a mesh that covers the geometry
mesh = openmc.RegularMesh.from_domain(geometry, dimension=(100, 100, 100))
mesh_filter = openmc.MeshFilter(mesh)

# makes a mesh tally using the previously created mesh and records heating on the mesh
heating_mesh_tally = openmc.Tally(name="heating_on_mesh")
heating_mesh_tally.filters = [mesh_filter]
heating_mesh_tally.scores = ["heating"]

# makes a mesh tally using the previously created mesh and records TBR on the mesh
tbr_mesh_tally = openmc.Tally(name="tbr_on_mesh")
tbr_mesh_tally.filters = [mesh_filter]
tbr_mesh_tally.scores = ["(n,Xt)"]

# groups the two tallies
tallies = openmc.Tallies([
    tbr_cell_tally,
    tbr_mesh_tally,
    heating_cell_tally,
    heating_mesh_tally,
])

# %% MODEL
model = openmc.Model(
    materials=materials, geometry=geometry, tallies=tallies, settings=settings
)
model.export_to_model_xml()

# %%
sp_filename = model.run()
print(f"Run completed. Results saved to {sp_filename}")
