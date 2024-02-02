# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path

import gmsh
import numpy as np

from bluemira.base.components import Component, PhysicalComponent
from bluemira.geometry.face import BluemiraFace, BluemiraWire
from bluemira.geometry.tools import make_polygon
from bluemira.magnetostatics import greens
from bluemira.magnetostatics.biot_savart import Bz_coil_axis
from bluemira.magnetostatics.fem_utils import (
    Association,
    create_j_function,
    model_to_mesh,
)
from bluemira.magnetostatics.finite_element_2d import FemMagnetostatic2d
from bluemira.mesh import meshing

DATA_DIR = Path(Path(__file__).parent, "test_generated_data")


class TestGetNormal:
    def test_simple_thin_coil(self, tmp_path):
        """
        Compare the magnetic field on the axis of a coil with a very small
        cross-section calculated with the fem module and the analytic
        solution as limit of the Biot-Savart law.
        """
        ri = 0.01  # Inner radius of copper wire
        rc = 5  # Outer radius of copper wire
        r_enclo_int = 25  # Radius of domain
        r_enclo_ext = 250
        i_wire = 1e6  # wire's current
        gdim = 2  # Geometric dimension of the mesh

        # Define geometry for wire cylinder
        nwire = 20  # number of wire divisions
        lwire = 0.1  # mesh characteristic length for each segment

        nenclo = 20  # number of external enclosure divisions
        lenclo = 0.5  # mesh characteristic length for each segment

        lcar_axis = 0.1  # axis characteristic length

        # enclosure
        theta_encl = np.linspace(np.pi / 2, -np.pi / 2, nenclo)
        r_encl = r_enclo_int * np.cos(theta_encl)
        z_encl = r_enclo_int * np.sin(theta_encl)

        # adding (0,0) to improve mesh quality
        enclosure_points = np.array([
            [0, 0, 0],
            *[[r_encl[ii], z_encl[ii], 0] for ii in range(r_encl.size)],
        ])

        nenclo_ext = 40
        lenclo_ext = 20
        # external enclosure
        theta_encl_ext = np.linspace(np.pi / 2, -np.pi / 2, nenclo_ext)
        r_encl_ext = r_enclo_ext * np.cos(theta_encl_ext)
        z_encl_ext = r_enclo_ext * np.sin(theta_encl_ext)

        enclosure_points_ext1 = np.array([
            *[[r_encl_ext[ii], z_encl_ext[ii], 0] for ii in range(r_encl_ext.size)]
        ])
        enclosure_points_ext2 = enclosure_points[1:][::-1]
        poly_enclo_ext = make_polygon(
            np.concatenate((enclosure_points_ext1, enclosure_points_ext2)), closed=True
        )
        poly_enclo_ext.mesh_options = {
            "lcar": lenclo_ext,
            "physical_group": "poly_enclo_ext",
        }

        enclosure_ext = BluemiraFace([poly_enclo_ext])
        enclosure_ext.mesh_options.physical_group = "enclo_ext"

        poly_enclo1 = make_polygon(enclosure_points[0:2])
        poly_enclo1.mesh_options = {"lcar": lcar_axis, "physical_group": "poly_enclo1"}

        poly_enclo2 = make_polygon(enclosure_points[1:])
        poly_enclo2.mesh_options = {"lcar": lenclo, "physical_group": "poly_enclo2"}

        poly_enclo3 = make_polygon(np.array([enclosure_points[-1], enclosure_points[0]]))
        poly_enclo3.mesh_options = {"lcar": lcar_axis, "physical_group": "poly_enclo3"}

        poly_enclo = BluemiraWire([poly_enclo1, poly_enclo2, poly_enclo3])
        poly_enclo.close("poly_enclo")
        poly_enclo.mesh_options = {"lcar": lenclo, "physical_group": "poly_enclo"}

        # coil
        theta_coil = np.linspace(0, 2 * np.pi, nwire)
        r_coil = rc + ri * np.cos(theta_coil[:-1])
        z_coil = ri * np.sin(theta_coil)

        coil_points = [[r_coil[ii], z_coil[ii], 0] for ii in range(r_coil.size)]

        poly_coil = make_polygon(coil_points, closed=True)
        lcar_coil = np.ones([poly_coil.vertexes.shape[1], 1]) * lwire
        poly_coil.mesh_options = {"lcar": lwire, "physical_group": "poly_coil"}

        coil = BluemiraFace([poly_coil])
        coil.mesh_options.physical_group = "coil"

        enclosure = BluemiraFace([poly_enclo, poly_coil])
        enclosure.mesh_options.physical_group = "enclo"

        c_universe = Component(name="universe")
        c_enclo_ext = PhysicalComponent(
            name="enclosure_Ext", shape=enclosure_ext, parent=c_universe
        )
        c_enclo = PhysicalComponent(name="enclosure", shape=enclosure, parent=c_universe)
        c_coil = PhysicalComponent(name="coil", shape=coil, parent=c_universe)

        # Create the mesh (by default, mesh is stored in the file Mesh.msh")
        meshfiles = [
            Path(tmp_path, p).as_posix() for p in ["Mesh.geo_unrolled", "Mesh.msh"]
        ]

        meshing.Mesh(meshfile=meshfiles)(c_universe, dim=2)

        (mesh, ct, ft), labels = model_to_mesh(gmsh.model, gdim=2)
        gmsh.write("Mesh.msh")
        gmsh.finalize()

        em_solver = FemMagnetostatic2d(2)
        em_solver.set_mesh(mesh, ct)

        # Define source term (coil current distribution) for the fem problem
        coil_tag = labels["coil"][1]
        jtot = create_j_function(mesh, ct, [Association(1, coil_tag, i_wire)])

        # solve the em problem and calculate the magnetic field B
        em_solver.define_g(jtot)
        em_solver.solve()

        # Check along the z axis (analytical solution)
        r_offset = 2 * lcar_axis

        z_points_axis = np.linspace(0, r_enclo_int, 200)
        r_points_axis = np.zeros(z_points_axis.shape) + r_offset
        b_points = np.array([r_points_axis, z_points_axis, 0 * z_points_axis]).T

        Bz_axis = em_solver.calculate_b()(b_points)
        Bz_axis = Bz_axis[:, 1]
        bz_points = b_points[:, 1]
        B_z_teo = np.array([Bz_coil_axis(rc, 0, z, i_wire) for z in bz_points])

        np.testing.assert_allclose(Bz_axis, B_z_teo, atol=2e-4)

        # Check along a radial path at z_offset (solution from green function)
        z_offset = 100 * ri

        points_x = np.linspace(r_offset, r_enclo_int, 200)
        points_z = np.zeros(z_points_axis.shape) + z_offset

        new_points = np.array([points_x, points_z, 0 * points_z]).T
        new_points = new_points[1:]

        B_fem = em_solver.calculate_b()(new_points)
        Bx_fem = B_fem.T[0]
        Bz_fem = B_fem.T[1]

        g_psi, g_bx, g_bz = greens.greens_all(rc, 0, new_points[:, 0], new_points[:, 1])
        g_psi *= i_wire
        g_bx *= i_wire
        g_bz *= i_wire

        np.testing.assert_allclose(Bx_fem, g_bx, atol=2e-4)
        np.testing.assert_allclose(Bz_fem, g_bz, atol=2e-4)
