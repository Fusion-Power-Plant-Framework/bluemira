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
from pathlib import Path

import openmc
import vtk
from vtkmodules.util import numpy_support

from bluemira.base.look_and_feel import bluemira_print
from bluemira.codes.openmc.params import OpenMCNeutronicsSolverParams
from bluemira.codes.openmc.sources import create_ring_source
from bluemira.materials.cache import establish_material_cache, get_cached_material

par = Path(__file__).parent
# %% [markdown]
# # Running a DAGMC model in OpenMC

# %%
n_batches = 5
particles_per_batch = 10000
ring_source_major_radius = 900  # cm, the reactor major radius
# OptimisedReactor can be exported by running examples/design/optimised_reactor.ex.py
# Change this to the name of your DAGMC model
dag_model_path = par / "OptimisedReactor.h5m"
meta_data_path = par / "OptimisedReactor.meta.json"

# # %%
# establish_material_cache([
#     Path(get_bluemira_root()) / "data" / "materials" / "materials.json",
#     Path(get_bluemira_root()) / "data" / "materials" / "mixtures.json",
# ])
omc_output_path = par / "omc"
# Ensure OpenMC output directory exists
omc_output_path.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Running the DAGMC model in OpenMC

# %%
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
Optimised reactor example, showing how to export a reactor
to a DAGMC model and run it in OpenMC.
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.parameter_frame import Parameter
from bluemira.codes.openmc.solver import OpenMCDAGMCNeutronicsSolver
from bluemira.equilibria.coils import Coil, CoilSet
from bluemira.equilibria.diagnostics import PicardDiagnostic, PicardDiagnosticOptions
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.optimisation.constraints import (
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
)
from bluemira.equilibria.optimisation.problem import (
    UnconstrainedTikhonovCurrentGradientCOP,
)
from bluemira.equilibria.profiles import CustomProfile
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.equilibria.solve import PicardIterator


def _lcfs_parameterisation(R_0, A):
    return JohnerLCFS({
        "r_0": {"value": R_0},
        "z_0": {"value": 0.0},
        "a": {"value": R_0 / A},
        "kappa_u": {"value": 1.6},
        "kappa_l": {"value": 1.9},
        "delta_u": {"value": 0.4},
        "delta_l": {"value": 0.4},
        "phi_u_neg": {"value": 0.0},
        "phi_u_pos": {"value": 0.0},
        "phi_l_neg": {"value": 45.0},
        "phi_l_pos": {"value": 30.0},
    })


def ref_eq(R_0, A) -> Equilibrium:  # noqa: D103
    x = [5.4, 14.0, 17.75, 17.75, 14.0, 7.0, 2.77, 2.77, 2.77, 2.77, 2.77]
    z = [9.26, 7.9, 2.5, -2.5, -7.9, -10.5, 7.07, 4.08, -0.4, -4.88, -7.86]
    dx = [0.6, 0.7, 0.5, 0.5, 0.7, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4]
    dz = [0.6, 0.7, 0.5, 0.5, 0.7, 1.0, 2.99 / 2, 2.99 / 2, 5.97 / 2, 2.99 / 2, 2.99 / 2]

    coils = []
    j = 1
    for i, (xi, zi, dxi, dzi) in enumerate(zip(x, z, dx, dz, strict=False)):
        if j > 6:  # noqa: PLR2004
            j = 1
        ctype = "PF" if i < 6 else "CS"  # noqa: PLR2004
        coil = Coil(
            xi,
            zi,
            current=0,
            dx=dxi,
            dz=dzi,
            ctype=ctype,
            name=f"{ctype}_{j}",
        )
        coils.append(coil)
        j += 1

    coilset = CoilSet(*coils)
    coilset.assign_material("CS", j_max=16.5e6, b_max=12.5)
    coilset.assign_material("PF", j_max=12.5e6, b_max=11.0)

    cs = coilset.get_coiltype("CS")
    cs.fix_sizes()
    cs.discretisation = 0.3

    B_0 = 4.8901  # T
    I_p = 19.07e6  # A

    grid = Grid(3.0, 13.0, -10.0, 10.0, 65, 65)

    profiles = CustomProfile(
        np.array([
            86856,
            86506,
            84731,
            80784,
            74159,
            64576,
            52030,
            36918,
            20314,
            4807,
            0.0,
        ]),
        -np.array([
            0.125,
            0.124,
            0.122,
            0.116,
            0.106,
            0.093,
            0.074,
            0.053,
            0.029,
            0.007,
            0.0,
        ]),
        R_0=R_0,
        B_0=B_0,
        I_p=I_p,
    )

    eq = Equilibrium(coilset, grid, profiles, psi=None)

    lcfs = (
        _lcfs_parameterisation(R_0, A).create_shape().discretise(byedges=True, ndiscr=50)
    )

    x_bdry, z_bdry = lcfs.x, lcfs.z
    arg_inner = np.argmin(x_bdry)

    isoflux = IsofluxConstraint(
        x_bdry,
        z_bdry,
        x_bdry[arg_inner],
        z_bdry[arg_inner],
        tolerance=0.5,  # Difficult to choose...
        constraint_value=0.0,  # Difficult to choose...
    )

    xp_idx = np.argmin(z_bdry)
    x_point = FieldNullConstraint(
        x_bdry[xp_idx],
        z_bdry[xp_idx],
        tolerance=1e-4,  # [T]
    )
    current_opt_problem = UnconstrainedTikhonovCurrentGradientCOP(
        coilset, eq, MagneticConstraintSet([isoflux, x_point]), gamma=1e-7
    )
    diagnostic_plotting = PicardDiagnosticOptions(plot=PicardDiagnostic.EQ)
    program = PicardIterator(
        eq,
        current_opt_problem,
        fixed_coils=True,
        relaxation=0.2,
        diagnostic_plotting=diagnostic_plotting,
    )
    program()

    plt.show()

    return eq


eq = ref_eq(9, 3)

"Toroidal_Field_Coil_2015"
"Homogenised_HCPB_2015_v3_BZ"

establish_material_cache(["eurofusion_materials.library", "matproplib"])

openmc_mats = [
    get_cached_material(mat_name).convert(
        "openmc", op_cond={"temperature": 294, "pressure": 101325}
    )
    for mat_name in ["Toroidal_Field_Coil_2015", "Homogenised_HCPB_2015_v3_BZ"]
]

params = OpenMCNeutronicsSolverParams(
    R_0=Parameter(name="R_0", value=9, unit="m"),
    profile_rho_ped=Parameter(
        name="profile_rho_ped",
        value=0.94,
        unit="dimensionless",
        source="Input",
        long_name="Pedestal location in normalized radius",
    ),
    P_fus=Parameter(
        name="P_fus",
        value=2000,
        unit="megawatt",
        source="Input",
        long_name="Total fusion power",
    ),
    n_profile_alpha=Parameter(
        name="n_profile_alpha", value=1.0, unit="dimensionless", source="Input"
    ),
    n_e_core=Parameter(name="n_e_core", value=1.5e20, unit="1/m^3"),
    n_e_ped=Parameter(name="n_e_ped", value=8e19, unit="1/m^3"),
    n_e_sep=Parameter(name="n_e_sep", value=3e19, unit="1/m^3"),
    T_profile_alpha=Parameter(name="T_profile_alpha", value=1.45, unit="dimensionless"),
    T_profile_beta=Parameter(name="T_profile_beta", value=2.0, unit="dimensionless"),
    T_e_core=Parameter(name="T_e_core", value=20, unit="kiloelectron_volt"),
    T_e_ped=Parameter(name="T_e_ped", value=5.5, unit="kiloelectron_volt"),
    T_e_sep=Parameter(name="T_e_sep", value=0.1, unit="kiloelectron_volt"),
    T_ie_ratio=Parameter(name="T_ie_ratio", value=1, unit="dimensionless"),
    n_i_fuel=Parameter(name="n_i_fuel", value=8e19, unit="1/m^3"),
    n_e=Parameter(name="n_e", value=1e20, unit="1/m^3"),
    shaf_shift=Parameter(name="shaf_shift", value=0.5, unit="meter"),
)

solver = OpenMCDAGMCNeutronicsSolver(
    params,
    {
        "particles": 10000,
        "cross_section_xml": Path(
            "eudemo/config/cross_section_data/cross_sections.xml"
        ).resolve(),
    },
    eq,
    lambda *args: (create_ring_source(ring_source_major_radius, 0), 1, 1),
    dag_model_path,
    openmc_mats,
)
solver.execute("run")
import pdb

pdb.set_trace()
dagmc_univ = openmc.DAGMCUniverse(
    filename=dag_model_path.as_posix(),
    auto_geom_ids=True,
).bounded_universe()


# # load model materials
# with open(meta_data_path) as meta_file:
#     bom = json.load(meta_file)["bom"]

# load DAG model
geometry = openmc.Geometry(dagmc_univ)

# my_source =

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
