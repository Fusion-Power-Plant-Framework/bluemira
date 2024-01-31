# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Compare the magnetic field on the axis of a coil with a very small cross-section
calculated with the fem module and the analytic solution as limit of the
Biot-Savart law.
"""

from pathlib import Path

import gmsh
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from mpi4py import MPI

from bluemira.base.components import Component, PhysicalComponent
from bluemira.geometry import tools
from bluemira.geometry.face import BluemiraFace, BluemiraWire
from bluemira.magnetostatics import greens
from bluemira.magnetostatics.fem_utils import (
    Association,
    create_j_function,
    model_to_mesh,
)
from bluemira.magnetostatics.finite_element_2d import (
    Bz_coil_axis,
    FemMagnetostatic2d,
)
from bluemira.mesh import meshing

DATA_DIR = Path(__file__).parent

model_rank = MPI.COMM_WORLD.rank
mesh_comm = MPI.COMM_WORLD

r_enclo = 30
lcar_enclo = 2
lcar_axis = lcar_enclo / 20

rc = 5
drc = 0.01
lcar_coil = 0.01

poly_coil = tools.make_polygon(
    [
        [rc - drc, rc + drc, rc + drc, rc - drc],
        [-drc, -drc, +drc, +drc],
        [0, 0, 0, 0],
    ],
    closed=True,
    label="poly_enclo",
)

poly_coil.mesh_options = {"lcar": lcar_coil, "physical_group": "poly_coil"}
coil = BluemiraFace(poly_coil)
coil.mesh_options = {"lcar": lcar_coil, "physical_group": "coil"}

poly_axis = tools.make_polygon([[0, 0, 0], [-r_enclo, 0, r_enclo], [0, 0, 0]])
poly_axis.mesh_options = {"lcar": lcar_axis, "physical_group": "poly_axis"}

poly_ext = tools.make_polygon(
    [
        [0, r_enclo, r_enclo, 0],
        [r_enclo, r_enclo, -r_enclo, -r_enclo],
        [0, 0, 0, 0],
    ],
    label="poly_ext",
)
poly_ext.mesh_options = {"lcar": lcar_enclo, "physical_group": "poly_ext"}

poly_enclo = BluemiraWire([poly_axis, poly_ext], "poly_enclo")
poly_enclo.mesh_options = {"lcar": lcar_enclo, "physical_group": "poly_enclo"}

r_enclo1 = 150
lcar_enclo1 = 10

poly_ext1 = tools.make_polygon(
    [
        [0, r_enclo, r_enclo, 0, 0, r_enclo1, r_enclo1, 0],
        [r_enclo, r_enclo, -r_enclo, -r_enclo, -r_enclo1, -r_enclo1, r_enclo1, r_enclo1],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    label="poly_ext1",
    closed=True,
)
poly_ext1.mesh_options = {"lcar": lcar_enclo1, "physical_group": "poly_ext1"}
poly_enclo1 = BluemiraWire([poly_ext1], "poly_enclo1")
poly_enclo1.mesh_options = {"lcar": lcar_enclo1, "physical_group": "poly_enclo1"}

enclosure = BluemiraFace([poly_enclo, poly_coil])
enclosure.mesh_options = {"lcar": lcar_enclo, "physical_group": "enclo"}

enclosure1 = BluemiraFace([poly_enclo1])
enclosure1.mesh_options = {"lcar": lcar_enclo1, "physical_group": "enclo1"}

c_universe = Component(name="universe")
c_enclo = PhysicalComponent(name="enclosure", shape=enclosure, parent=c_universe)
c_enclo1 = PhysicalComponent(name="enclosure1", shape=enclosure1, parent=c_universe)
c_coil = PhysicalComponent(name="coil", shape=coil, parent=c_universe)

meshfiles = [Path(DATA_DIR, p).as_posix() for p in ["Mesh.geo_unrolled", "Mesh.msh"]]
m = meshing.Mesh(meshfile=meshfiles)
m(c_universe, dim=2)

(mesh, ct, ft), labels = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)

gmsh.write("Mesh.msh")
gmsh.finalize()

em_solver = FemMagnetostatic2d(2)
em_solver.set_mesh(mesh, ct)

current = 1e6
coil_tag = labels["coil"][1]
j_functions = [Association(1, coil_tag, current)]
jtot = create_j_function(mesh, ct, j_functions)

em_solver.define_g(jtot)
em_solver.solve()
B = em_solver.calculate_b()

# Comparison of the theoretical and calculated magnetic field (B).
# Note: The comparison is conducted along the z-axis, where an
# analytical expression is available. However, due to challenges
# in calculating the gradient of dPsi/dx along the axis for CG
# element, the points are translated by a value of deltax.
deltax = 0.25
z_points_axis = np.linspace(0, r_enclo / 2, 200)
r_points_axis = np.zeros(z_points_axis.shape) + deltax
points = np.array([r_points_axis, z_points_axis, 0 * z_points_axis]).T

Bz_axis = B(points)
Bz_axis = Bz_axis[:, 1]
z_points = points[:, 1]
B_teo = np.array([Bz_coil_axis(rc, 0, z, current) for z in z_points])

ax: Axes
_, ax = plt.subplots()
ax.plot(z_points, Bz_axis, label="B_calc")
ax.plot(z_points, B_teo, label="B_teo")
ax.set_xlabel("r (m)")
ax.set_ylabel("B (T)")
ax.legend()
plt.show()

_, ax = plt.subplots()
ax.plot(z_points, Bz_axis - B_teo, label="B_calc - B_teo")
ax.set_xlabel("r (m)")
ax.set_ylabel("error (T)")
ax.legend()
plt.show()

# I just set an absolute tolerance for the comparison (since the magnetic field
# goes to zero, the comparison cannot be made on the basis of a relative
# tolerance). An allclose comparison was out of discussion considering the
# necessary accuracy.
np.testing.assert_allclose(Bz_axis, B_teo, atol=2.5e-4)


z_offset = 100 * drc

points_x = np.linspace(0, r_enclo, 200)
points_z = np.zeros(z_points_axis.shape) + z_offset

new_points = np.array([points_x, points_z, 0 * points_z]).T
new_points = new_points[1:]

B_fem = em_solver.calculate_b()(new_points)
Bx_fem = B_fem.T[0]
Bz_fem = B_fem.T[1]

g_psi, g_bx, g_bz = greens.greens_all(rc, 0, new_points[:, 0], new_points[:, 1])
g_psi *= current
g_bx *= current
g_bz *= current

_, ax = plt.subplots()
ax.plot(new_points[:, 0], Bx_fem, label="Bx_fem")
ax.plot(new_points[:, 0], g_bx, label="Green Bx")
ax.set_xlabel("r (m)")
ax.set_ylabel("Bx (T)")
ax.legend()
plt.show()

_, ax = plt.subplots()
ax.plot(new_points[:, 0], Bz_fem, label="Bz_fem")
ax.plot(new_points[:, 0], g_bz, label="Green Bz")
ax.set_xlabel("r (m)")
ax.set_ylabel("Bz (T)")
ax.legend()
plt.show()

_, ax = plt.subplots()
ax.plot(new_points[:, 0], Bx_fem - g_bx, label="Bx_calc - GreenBx")
ax.plot(new_points[:, 0], Bz_fem - g_bz, label="Bz_calc - GreenBz")
ax.legend()
ax.set_xlabel("r (m)")
ax.set_ylabel("error (T)")
plt.show()

np.testing.assert_allclose(Bx_fem, g_bx, atol=3e-4)
np.testing.assert_allclose(Bz_fem, g_bz, atol=6e-4)
