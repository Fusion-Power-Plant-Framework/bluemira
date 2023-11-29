# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path

import dolfin
import numpy as np

from bluemira.base.components import Component, PhysicalComponent
from bluemira.geometry import tools
from bluemira.geometry.face import BluemiraFace
from bluemira.magnetostatics.finite_element_2d import (
    Bz_coil_axis,
    FemMagnetostatic2d,
    ScalarSubFunc,
)
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf


class TestGetNormal:
    def test_simple_thin_coil(self, tmp_path):
        """
        Compare the magnetic field on the axis of a coil with a very small cross-section
        calculated with the fem module and the analytic solution as limit of the
        Biot-Savart law.
        """

        r_enclo = 100
        lcar_enclo = 0.5

        rc = 5
        drc = 0.01
        lcar_coil = 0.01

        poly_coil = tools.make_polygon(
            [
                [rc - drc, rc + drc, rc + drc, rc - drc],
                [0, 0, 0, 0],
                [-drc, -drc, +drc, +drc],
            ],
            closed=True,
            label="poly_enclo",
        )

        poly_coil.mesh_options = {"lcar": lcar_coil, "physical_group": "poly_coil"}
        coil = BluemiraFace(poly_coil)
        coil.mesh_options = {"lcar": lcar_coil, "physical_group": "coil"}

        poly_enclo = tools.make_polygon(
            [
                [0, r_enclo, r_enclo, 0],
                [0, 0, 0, 0],
                [-r_enclo, -r_enclo, r_enclo, r_enclo],
            ],
            closed=True,
            label="poly_enclo",
        )

        poly_enclo.mesh_options = {"lcar": lcar_enclo, "physical_group": "poly_enclo"}
        enclosure = BluemiraFace([poly_enclo, poly_coil])
        enclosure.mesh_options = {"lcar": lcar_enclo, "physical_group": "enclo"}

        c_universe = Component(name="universe")
        c_enclo = PhysicalComponent(name="enclosure", shape=enclosure, parent=c_universe)
        c_coil = PhysicalComponent(name="coil", shape=coil, parent=c_universe)

        meshfiles = [
            Path(tmp_path, p).as_posix() for p in ["Mesh.geo_unrolled", "Mesh.msh"]
        ]
        m = meshing.Mesh(meshfile=meshfiles)
        m(c_universe, dim=2)

        msh_to_xdmf("Mesh.msh", dimensions=(0, 2), directory=tmp_path)

        mesh, boundaries, subdomains, labels = import_mesh(
            "Mesh",
            directory=tmp_path,
            subdomains=True,
        )

        dolfin.plot(mesh)

        em_solver = FemMagnetostatic2d(3)
        em_solver.set_mesh(mesh, boundaries)

        current = 1e6
        jc = current / coil.area
        markers = [labels["coil"]]
        functions = [jc]
        jtot = ScalarSubFunc(functions, markers, subdomains)

        em_solver.define_g(jtot)
        em_solver.solve()
        em_solver.calculate_b()

        z_points_axis = np.linspace(0, r_enclo, 200)
        r_points_axis = np.zeros(z_points_axis.shape)

        Bz_axis = np.array(
            [em_solver.B(x) for x in np.array([r_points_axis, z_points_axis]).T]
        ).T[1]

        B_teo = np.array([Bz_coil_axis(rc, 0, z, current) for z in z_points_axis])

        # I just set an absolute tolerance for the comparison (since the magnetic field
        # goes to zero, the comparison cannot be made on the basis of a relative
        # tolerance). An allclose comparison was out of discussion considering the
        # necessary accuracy.
        np.testing.assert_allclose(Bz_axis, B_teo, atol=1e-4)
