import os
import shutil
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate
from scipy.interpolate import interp1d

import bluemira.equilibria.fem_fixed_boundary.equilibrium as equilibrium
import bluemira.equilibria.fem_fixed_boundary.utilities as utilities
import bluemira.geometry.tools as geotools
from bluemira.base.components import PhysicalComponent
from bluemira.base.constants import MU_0
from bluemira.base.file import get_bluemira_path, get_bluemira_root
from bluemira.base.logs import set_log_level
from bluemira.codes import transport_code_solver
from bluemira.equilibria.fem_fixed_boundary.equilibrium import (
    solve_transport_fixed_boundary,
)
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemGradShafranovFixedBoundary,
    refine_mesh,
)
from bluemira.equilibria.fem_fixed_boundary.file import save_fixed_boundary_to_file
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.geometry.coordinates import Coordinates
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf

set_log_level("NOTSET")

SOLVER_MODULE_REF = "bluemira.codes.plasmod.api"
RUN_SUBPROCESS_REF = "bluemira.codes.interface.run_subprocess"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PARAMS_FILE = os.path.join(DATA_DIR, "params.json")

if plasmod_binary := shutil.which("plasmod"):
    PLASMOD_PATH = os.path.dirname(plasmod_binary)
else:
    PLASMOD_PATH = os.path.join(os.path.dirname(get_bluemira_root()), "plasmod/bin")
binary = os.path.join(PLASMOD_PATH, "plasmod")

plasmod_build_config = {
    "problem_settings": {},
    "mode": "read",
    "binary": binary,
    "directory": get_bluemira_path("", subfolder="generated_data"),
}

plasmod_solver = transport_code_solver(
    params={},
    build_config=plasmod_build_config,
    module="PLASMOD",
)

plasmod_solver.execute("read")

outputs = plasmod_solver.plasmod_outputs()

I_p = 0.20501396465e08
R_0 = 0.898300000e01
B_0 = 0.531000000e01
amin = outputs.amin
shif = outputs.shif
dprof = outputs.dprof
kprof = outputs.kprof
qprof = outputs.qprof
volprof = outputs.volprof

g2 = outputs.g2
g3 = outputs.g3

ffprime = outputs.ffprime
press = outputs.press
pprime = outputs.pprime

psi = outputs.psi

theta = np.linspace(0, 2 * np.pi, 201)
rPLASMOD_sep = R_0 + shif[-1] + amin * (np.cos(theta) - dprof[-1] * np.sin(theta) ** 2)
zPLASMOD_sep = amin * kprof[-1] * np.sin(theta)


points = Coordinates({"x": rPLASMOD_sep, "z": zPLASMOD_sep})
Plasmod_sep = geotools.interpolate_bspline(points)
Plasmod_sep_surf = geotools.BluemiraFace(Plasmod_sep)

xPsiPlasmod = np.sqrt(psi / psi[-1])

c_vol = Plasmod_sep_surf.area * 2 * np.pi * Plasmod_sep_surf.center_of_mass[0]

from bluemira.base.logs import set_log_level

set_log_level("DEBUG")
from bluemira.base.look_and_feel import bluemira_error

try:
    np.testing.assert_almost_equal(c_vol, volprof[-1])
except AssertionError as e:
    bluemira_error(f"Assertion error: {e}")


_pprime = -pprime / (-2 * np.pi * R_0 * 1e-6)
_ffprime = -ffprime / (-2 * np.pi / MU_0 / R_0 * 1e-6)

plasma = PhysicalComponent("Plasma", shape=Plasmod_sep_surf)

plasma.shape.mesh_options = {"lcar": 0.3, "physical_group": "plasma"}
plasma.shape.boundary[0].mesh_options = {"lcar": 0.3, "physical_group": "lcfs"}

directory = get_bluemira_path("", subfolder="generated_data")
meshfiles = [os.path.join(directory, p) for p in ["Mesh.geo_unrolled", "Mesh.msh"]]
meshing.Mesh(meshfile=meshfiles)(plasma)


msh_to_xdmf("Mesh.msh", dimensions=(0, 2), directory=directory)

mesh, boundaries, subdomains, labels = import_mesh(
    "Mesh",
    directory=directory,
    subdomains=True,
)

# check mesh refinement
if True:
    coarse_mesh = mesh

    refined_mesh = refine_mesh(mesh, np.array((9, 0, 0)), num_levels=2, distance=1.5)

    import meshio

    # Convert the mesh to a meshio.Mesh object
    points = refined_mesh.coordinates()
    cells = {"triangle": refined_mesh.cells()}
    meshio_mesh = meshio.Mesh(points, cells)

    # Save the refined mesh in .msh format using meshio
    meshio.write("refined_mesh.msh", meshio_mesh, file_format="gmsh22")

    mesh = refined_mesh


f_pprime = interp1d(xPsiPlasmod, pprime, fill_value="extrapolate")
f_ffprime = interp1d(xPsiPlasmod, ffprime, fill_value="extrapolate")


start_time = datetime.now()
print("\n Solving GSE for a give pprime and ffprime")
gs_solver = FemGradShafranovFixedBoundary(
    f_pprime,
    f_ffprime,
    I_p,
    R_0,
    B_0,
    max_iter=30,
    iter_err_max=1e-3,
    p_order=2,
    relaxation=0,
)
gs_solver.set_mesh(mesh)
gs_solver.set_profiles(f_pprime, f_ffprime, I_p, R_0, B_0)

gs_solver.solve(plot=False)
print(f"\n GSE solving time = {datetime.now() - start_time}")

mesh_points = mesh.coordinates()
c_psi = np.array([gs_solver.psi(p) for p in mesh_points])

utilities.plot_scalar_field(mesh_points[:, 0], mesh_points[:, 1], c_psi)


x1D, flux_surfaces = utilities.get_flux_surfaces_from_mesh(
    mesh, gs_solver.psi_norm_2d, x_1d=xPsiPlasmod
)


import dolfin

start_time = datetime.now()
print(f"\n Start equilibrium.calc_metric_coefficients")

dpsi_dx = gs_solver.psi.dx(0)
dpsi_dz = gs_solver.psi.dx(1)

w = dolfin.VectorFunctionSpace(gs_solver.mesh, "CG", 1)
grad_psi_2D_func = dolfin.project(dolfin.as_vector((dpsi_dx, dpsi_dz)), w)

x1D, V, g1, g2, g3 = equilibrium.calc_metric_coefficients(
    flux_surfaces,
    grad_psi_2D_func,
    x1D,
    gs_solver.psi_ax,
)
print(
    f"\n equilibrium.calc_metric_coefficients solving time = {datetime.now() - start_time}"
)

q = interp1d(xPsiPlasmod, outputs.qprof, fill_value="extrapolate")
q = q(x1D)

p = interp1d(xPsiPlasmod, outputs.press, fill_value="extrapolate")
p = p(x1D)

Psi_ax = gs_solver.psi_ax
Psi_b = gs_solver.psi_b

Ip, Phi1D, Psi1D, pprime_psi1D_data, F, FFprime = equilibrium.calc_curr_dens_profiles(
    x1D, p, q, g2, g3, V, 0, B_0, R_0, Psi_ax, Psi_b
)

psi_plasmod = outputs.psi[-1] - outputs.psi

fig, axs = plt.subplots(3, 3)
axs[0, 0].plot(xPsiPlasmod, outputs.g2)
axs[0, 0].plot(x1D, g2)
axs[0, 0].set_title("g2")
axs[0, 1].plot(xPsiPlasmod, outputs.g3)
axs[0, 1].plot(x1D, g3)
axs[0, 1].set_title("g3")
axs[1, 0].plot(xPsiPlasmod, outputs.volprof)
axs[1, 0].plot(x1D, V)
axs[1, 0].set_title("V")
axs[1, 1].plot(xPsiPlasmod, outputs.pprime)
axs[1, 1].plot(x1D, pprime_psi1D_data)
axs[1, 1].set_title("pprime")
axs[1, 2].plot(xPsiPlasmod, outputs.ffprime)
axs[1, 2].plot(x1D, FFprime)
axs[1, 2].set_title("FFprime")
axs[0, 2].plot(x1D, F)
axs[0, 2].set_title("F")
axs[2, 0].plot(xPsiPlasmod, psi_plasmod)
axs[2, 0].plot(x1D, Psi1D)
axs[2, 0].set_title("Psi1D")
axs[2, 1].plot(xPsiPlasmod, outputs.phi)
axs[2, 1].plot(x1D, Phi1D)
axs[2, 1].set_title("Phi1D")
plt.show()

x_axis, z_axis = utilities.find_magnetic_axis(gs_solver.psi, mesh=mesh)

radius_plasmod = np.linspace(0, outputs.amin, xPsiPlasmod.size)
vprime_plasmod = np.gradient(outputs.volprof, radius_plasmod)

radius_bluemira = np.linspace(0, R_0 + outputs.amin - x_axis, x1D.size)
vprime_bluemira = np.gradient(V, radius_bluemira)

plt.plot(radius_plasmod, outputs.vprime)
plt.plot(radius_plasmod, vprime_plasmod)
plt.plot(radius_bluemira, vprime_bluemira)
plt.show()


Ip, Phi1D, Psi1D, pprime_psi1D_data, F, FFprime = equilibrium.calc_curr_dens_profiles(
    xPsiPlasmod,
    outputs.press,
    outputs.qprof,
    outputs.g2,
    outputs.g3,
    outputs.volprof,
    0,
    B_0,
    R_0,
    psi_plasmod[0],
    psi_plasmod[-1],
)

fig, axs = plt.subplots(3, 3)

axs[0, 0].plot(xPsiPlasmod, outputs.volprof)
axs[0, 0].set_title("V")
axs[0, 1].plot(xPsiPlasmod, outputs.g2)
axs[0, 1].set_title("g2")
axs[0, 2].plot(xPsiPlasmod, outputs.g3)
axs[0, 2].set_title("g3")

axs[1, 0].plot(xPsiPlasmod, outputs.pprime)
axs[1, 0].plot(xPsiPlasmod, pprime_psi1D_data)
axs[1, 0].set_title("pprime")
axs[1, 1].plot(xPsiPlasmod, outputs.ffprime)
axs[1, 1].plot(xPsiPlasmod, FFprime)
axs[1, 1].set_title("FFprime")

axs[1, 2].plot(xPsiPlasmod, psi_plasmod)
axs[1, 2].plot(xPsiPlasmod, Psi1D)
axs[1, 2].set_title("Psi1D")
axs[2, 0].plot(xPsiPlasmod, outputs.phi)
axs[2, 0].plot(xPsiPlasmod, Phi1D)
axs[2, 0].set_title("Phi1D")

plt.show()
