import os

import dolfin
import matplotlib.pyplot as plt
import numpy as np

import bluemira.geometry.tools as tools
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.file import get_bluemira_path
from bluemira.geometry.face import BluemiraFace
from bluemira.magnetostatics.finite_element_2d import (
    Bz_coil_axis,
    FemMagnetostatic2d,
    ScalarSubFunc,
)
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf

# Geometry parameters
r_enclo = 100
lcar_enclo = 1.0

rc = 5
drc = 0.025
lcar_coil = 0.05

# Geometry creation
poly_coil = tools.make_polygon(
    [[rc - drc, rc + drc, rc + drc, rc - drc], [0, 0, 0, 0], [-drc, -drc, +drc, +drc]],
    closed=True,
    label="poly_enclo",
)

poly_coil.mesh_options = {"lcar": lcar_coil, "physical_group": "poly_coil"}
coil = BluemiraFace(poly_coil)
coil.mesh_options = {"lcar": lcar_coil, "physical_group": "coil"}

poly_enclo = tools.make_polygon(
    [[0, r_enclo, r_enclo, 0], [0, 0, 0, 0], [-r_enclo, -r_enclo, r_enclo, r_enclo]],
    closed=True,
    label="poly_enclo",
)

poly_enclo.mesh_options = {"lcar": lcar_enclo, "physical_group": "poly_enclo"}
enclosure = BluemiraFace([poly_enclo, poly_coil])
enclosure.mesh_options = {"lcar": lcar_enclo, "physical_group": "enclo"}

c_universe = Component(name="universe")
c_enclo = PhysicalComponent(name="enclosure", shape=enclosure, parent=c_universe)
c_coil = PhysicalComponent(name="coil", shape=coil, parent=c_universe)

# Mesh

directory = get_bluemira_path("", subfolder="generated_data")
meshfiles = [os.path.join(directory, p) for p in ["Mesh.geo_unrolled", "Mesh.msh"]]

meshing.Mesh(meshfile=meshfiles)(c_universe, dim=2)

msh_to_xdmf("Mesh.msh", dimensions=(0, 2), directory=directory)

mesh, boundaries, subdomains, labels = import_mesh(
    "Mesh",
    directory=directory,
    subdomains=True,
)
dolfin.plot(mesh)
plt.show()

# EM solver setup
em_solver = FemMagnetostatic2d(2)
em_solver.set_mesh(mesh, boundaries)

Ic = 1e6
jc = Ic / coil.area
markers = [labels["coil"]]
functions = [jc]
jtot = ScalarSubFunc(functions, markers, subdomains)

f_space = dolfin.FunctionSpace(mesh, "DG", 0)
f = dolfin.Function(f_space)
f.interpolate(jtot)

# Solve
em_solver.define_g(jtot)
em_solver.solve()
em_solver.calculate_b()

# Comparison between the obtained B with the filament theoretical value on the z axis
z_points_axis = np.linspace(0, r_enclo, 200)
r_points_axis = np.zeros(z_points_axis.shape)
Bz_axis = np.array(
    [em_solver.B(x) for x in np.array([r_points_axis, z_points_axis]).T]
).T[1]
B_teo = np.array([Bz_coil_axis(rc, 0, z, Ic) for z in z_points_axis])

fig, ax = plt.subplots()
ax.plot(z_points_axis, Bz_axis, label="B_calc")
ax.plot(z_points_axis, B_teo, label="B_teo")
plt.xlabel("r (m)")
plt.ylabel("B (T)")
plt.legend()
plt.show()
