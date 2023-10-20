# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Solver for a 2D magnetostatic problem with cylindrical symmetry
"""
from typing import Optional, Tuple

import dolfinx.fem
import numpy as np
from dolfinx.fem import (
    Expression,
    FunctionSpace,
    VectorFunctionSpace,
    dirichletbc,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import Mesh, locate_entities_boundary
from petsc4py.PETSc import ScalarType
from ufl import (
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    as_vector,
    dot,
    dx,
    grad,
)

from bluemira.base.constants import MU_0
from bluemira.magnetostatics.fem_utils import BluemiraFemFunction


def Bz_coil_axis(
    r: float,
    z: Optional[float] = 0,
    pz: Optional[float] = 0,
    current: Optional[float] = 1,
) -> float:
    """
    Calculate the theoretical vertical magnetic field of a filament coil
    (of radius r and centred in (0, z)) on a point on the coil axis at
    a distance pz from the axis origin.

    Parameters
    ----------
    r:
        Coil radius [m]
    z:
        Vertical position of the coil centroid [m]
    pz:
        Vertical position of the point on the axis on which the magnetic field
        shall be calculated [m]
    current:
        Current of the coil [A]

    Returns
    -------
    Vertical magnetic field on the axis [T]

    Notes
    -----
    \t:math:`\\dfrac{1}{2}\\dfrac{\\mu_{0}Ir^2}{(r^{2}+(pz-z)^{2})^{3/2}}`
    """
    return 0.5 * MU_0 * current * r**2 / (r**2 + (pz - z) ** 2) ** 1.5


class FemMagnetostatic2d:
    """
    A 2D magnetostic solver for 2D planar and axisymmetric problems.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        mesh of the FEM model
    cell_tags:
        mesh cell tags
    face_tags:
        mesh face tags
    eltype: Tuple (default ("CG",1))
        tuple with the specification of the element family and the degree of the element
    """

    # TODO: check if the problem can be solved with any element type or if "eltype"
    #       should be hardcoded and removed from the signature.
    def __init__(
        self, mesh: Mesh, cell_tags=None, face_tags=None, eltype: Tuple = ("CG", 1)
    ):
        self.mesh = mesh
        self.cell_tags = cell_tags
        self.face_tags = face_tags
        self._eltype = eltype
        self.V = FunctionSpace(self.mesh, self._eltype)
        self.psi = BluemiraFemFunction(self.V)

        self.u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        x = SpatialCoordinate(self.mesh)
        self.a = (
            1 / (2.0 * np.pi * MU_0) * (1 / x[0] * dot(grad(self.u), grad(self.v))) * dx
        )

        self.g = BluemiraFemFunction(self.V)
        self.L = self.g * self.v * dx

    def define_g(self, g: BluemiraFemFunction):
        """
        Define g, the right hand side function of the Poisson problem

        Parameters
        ----------
        g:
            Right hand side function of the Poisson problem
        """
        self.g = g
        self.L = self.g * self.v * dx

    def solve(
        self,
        dirichlet_bcs: Optional[Tuple[int, BluemiraFemFunction]] = None,
    ):
        """
        Solve the defined static electromagnetic problem.

        Parameters
        ----------
        dirichlet_bcs : Tuple[int, BluemiraFemFunction] (default None)
            Dirichlet boundary conditions given as a tuple of (marker,dirichlet_function)

        Warning
        -------
        User-defined boundary conditions still not implemented

        TODO: check if it is possible just to have the same topological output without
            increase too much the computational time.
        """
        if dirichlet_bcs is None:
            tdim = self.mesh.topology.dim
            facets = locate_entities_boundary(
                self.mesh, tdim - 1, lambda x: np.full(x.shape[1], True)
            )
            dofs = locate_dofs_topological(self.V, tdim - 1, facets)
            dirichlet_bcs = [dirichletbc(ScalarType(0), dofs, self.V)]

        problem = LinearProblem(
            self.a,
            self.L,
            u=self.psi,
            bcs=dirichlet_bcs,
            petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )
        self.psi = problem.solve()

        return self.psi

    def compute_B(self, eltype: Optional[Tuple] = None):
        """
        Compute the magnetic field interpolating the result in a Function Space
        with eltype elements.
        """
        base_eltype = ("DG", 0)

        if eltype is None:
            eltype = base_eltype

        W0 = VectorFunctionSpace(self.mesh, base_eltype)
        B0 = BluemiraFemFunction(W0)

        x = SpatialCoordinate(self.mesh)

        r = x[0]

        B_expr = Expression(
            as_vector(
                (
                    -self.psi.dx(1) / (2 * np.pi * r),
                    self.psi.dx(0) / (2 * np.pi * r),
                )
            ),
            W0.element.interpolation_points(),
        )

        B0.interpolate(B_expr)

        if eltype is not None:
            W = VectorFunctionSpace(self.mesh, eltype)
            B = BluemiraFemFunction(W)
            B.interpolate(B0)
        else:
            B = B0

        return B
