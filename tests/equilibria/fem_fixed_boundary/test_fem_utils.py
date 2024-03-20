# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path

import gmsh
import numpy as np
from dolfinx.fem import Constant, FunctionSpace
from mpi4py import MPI
from petsc4py import PETSc

from bluemira.base.components import PhysicalComponent
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_polygon
from bluemira.magnetostatics.fem_utils import (
    BluemiraFemFunction,
    closest_point_in_mesh,
    integrate_f,
    model_to_mesh,
)
from bluemira.mesh.meshing import Mesh


def create_test_mesh(dl: float = 2, lcar: float = 0.1):
    wire = make_polygon(
        np.array([[0, 0, 0], [dl, 0, 0], [dl, dl, 0], [0, dl, 0]]), closed=True
    )
    wire.mesh_options.lcar = lcar
    wire.mesh_options.physical_group = "wire"
    face = BluemiraFace([wire])
    face.mesh_options.lcar = lcar
    face.mesh_options.physical_group = "face"

    c_universe = PhysicalComponent(name="all", shape=face)

    data_dir = Path(__file__).parent
    model_rank = MPI.COMM_WORLD.rank
    mesh_comm = MPI.COMM_WORLD

    meshfiles = [Path(data_dir, p).as_posix() for p in ["Mesh.geo_unrolled", "Mesh.msh"]]
    m = Mesh(meshfile=meshfiles)
    m(c_universe, dim=2)

    (mesh, ct, ft), labels = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=2)

    gmsh.write("Mesh.msh")
    gmsh.finalize()

    return (mesh, ct, ft), labels


class TestFemUtils:
    def setup_method(self):
        self.l = 3
        self.lcar = 0.1
        (self.mesh, self.ct, self.ft), self.labels = create_test_mesh(self.l, self.lcar)

    def test_closest_point_in_mesh(self):
        single_point = np.array([-1, -1, 0])
        closest_point = np.array([0, 0, 0])
        np.testing.assert_allclose(
            closest_point_in_mesh(self.mesh, single_point), closest_point
        )

        points = np.array([[0, 0, 0], [-1, -1, 0], [-1, self.l, 0]])
        closest_points = np.array([[0, 0, 0], [0, 0, 0], [0, self.l, 0]])
        np.testing.assert_allclose(
            closest_point_in_mesh(self.mesh, points), closest_points
        )

    def test_integrate_f(self):
        func = Constant(self.mesh, PETSc.ScalarType(1))
        area = integrate_f(func, self.mesh, self.ct, 1)
        assert np.allclose(area, self.l**2)

        def expr(x):
            return x[0]

        V = FunctionSpace(self.mesh, ("Lagrange", 1))  # noqa: N806
        dofs_points = V.tabulate_dof_coordinates()
        func = BluemiraFemFunction(V)
        func.x.array[:] = np.array([expr(x) for x in dofs_points])
        area = integrate_f(func, self.mesh, self.ct, 1)
        assert np.allclose(area, 0.5 * self.l**2 * self.l)
