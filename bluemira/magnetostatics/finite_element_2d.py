# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Solver for a 2D magnetostatic problem with cylindrical symmetry
"""

from typing import Iterable, Optional, Union

import dolfinx.fem
import numpy as np
from dolfinx.fem import (
    Expression,
    dirichletbc,
    functionspace,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import locate_entities_boundary
from petsc4py.PETSc import ScalarType
from ufl import (
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    as_vector,
    dot,
    ds,
    dx,
    grad,
)

from bluemira.base.constants import MU_0
from bluemira.magnetostatics.fem_utils import BluemiraFemFunction


def Bz_coil_axis(r: float, z: float = 0, pz: float = 0, current: float = 1) -> float:
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
    A 2D magnetostatic solver. The solver is thought as support for the fem fixed
    boundary module and it is limited to axisymmetric magnetostatic problem
    with toroidal current sources. The Maxwell equations, as function of the poloidal
    magnetic flux (:math:`\\Psi`), are then reduced to the form ([Zohm]_, page 25):

    .. math::
        r^2 \\nabla\\cdot\\left(\\frac{\\nabla\\Psi}{r^2}\\right) = 2
        \\pi r \\mu_0 J_{\\Phi}

    whose weak formulation is defined as ([Villone]_):

    .. math::
        \\int_{D_p} {\\frac{1}{r}}{\\nabla}{\\Psi}{\\cdot}{\\nabla} v \\,dr\\,dz = 2
        \\pi \\mu_0 \\int_{D_p} J_{\\Phi} v \\,dr\\,dz

    where :math:`v` is the basis element function of the defined functional subspace
    :math:`V`.

    .. [Zohm] H. Zohm, Magnetohydrodynamic Stability of Tokamaks, Wiley-VCH, Germany,
       2015
    .. [Villone] VILLONE, F. et al. Plasma Phys. Control. Fusion 55 (2013) 095008,
       https://doi.org/10.1088/0741-3335/55/9/095008

    Parameters
    ----------
    p_order:
        Order of the approximating polynomial basis functions
    """

    def __init__(self, p_order: int = 2):
        self.p_order = p_order
        self.mesh = None
        self.V = None
        self.g = None
        self.boundaries = None
        self.problem = None

        self.psi = None

    def set_mesh(
        self,
        mesh: Union[dolfinx.mesh.Mesh, str],
        boundaries: Optional[Union[dolfinx.mesh.Mesh, str]] = None,
        dirichlet_bc_function: Optional[
            Union[dolfinx.fem.Expression, BluemiraFemFunction]
        ] = None,
        dirichlet_marker: Optional[int] = None,
        neumann_bc_function: Optional[
            Union[dolfinx.fem.Expression, BluemiraFemFunction]
        ] = None,
    ):
        """
        Set the mesh for the solver

        Parameters
        ----------
        mesh:
            Filename of the xml file with the mesh definition or a dolfin mesh
        boundaries:
            Filename of the xml file with the boundaries definition or a MeshFunction
            that defines the boundaries
        dirichlet_bc_function:
            Dirichlet boundary condition function
        dirichlet_marker:
            Identification number for the dirichlet boundary
        neumann_bc_function:
            Neumann boundary condition function

        """
        # check whether mesh is a filename or a mesh, then load it or use it
        self.mesh = dolfinx.mesh.Mesh(mesh) if isinstance(mesh, str) else mesh

        # define boundaries
        if boundaries is None:
            tdim = self.mesh.topology.dim
            self.boundaries = locate_entities_boundary(
                self.mesh, tdim - 1, lambda x: np.full(x.shape[1], True)
            )
        else:
            self.boundaries = boundaries

        # define the function space and bilinear forms
        # the Continuos Galerkin function space has been chosen as suitable for the
        # solution of the magnetostatic weak formulation in a Soblev Space H1(D)
        self.V = functionspace(self.mesh, ("P", self.p_order))

        # define trial and test functions
        u = TrialFunction(self.V)
        self.v = TestFunction(self.V)

        self.a = (
            1
            / (2.0 * np.pi * MU_0)
            * (1 / SpatialCoordinate(self.mesh)[0] * dot(grad(u), grad(self.v)))
            * dx
        )

        # initialize solution
        self.psi = BluemiraFemFunction(self.V)
        # self.psi.set_allow_extrapolation(True)

        # initialize g to zero
        self.g = BluemiraFemFunction(self.V)

    def define_g(self, g: Union[dolfinx.fem.Expression, BluemiraFemFunction]):
        """
        Define g, the right hand side function of the Poisson problem

        Parameters
        ----------
        g:
            Right hand side function of the Poisson problem
        """
        self.g = g

    def setup_problem(
        self,
        dirichlet_bc_function: Optional[
            Union[dolfinx.fem.Expression, BluemiraFemFunction]
        ] = None,
        dirichlet_marker: Optional[int] = None,
        neumann_bc_function: Optional[
            Union[dolfinx.fem.Expression, BluemiraFemFunction]
        ] = None,
    ):
        """
        Define Dirichlet boundary conditions and setup problem

        Parameters
        ----------
        dirichlet_bc_function:
            Dirichlet boundary condition function
        dirichlet_marker:
            Identification number for the dirichlet boundary

        """
        # # define the right hand side
        # self.L = self.g * self.v * dx # - neumann_bc_function * self.v * ds

        # # define the Dirichlet boundary conditions
        if dirichlet_bc_function is None:
            tdim = self.mesh.topology.dim
            facets = locate_entities_boundary(
                self.mesh, tdim - 1, lambda x: np.full(x.shape[1], True)
            )
            dofs = locate_dofs_topological(self.V, tdim - 1, facets)
            bcs = [dirichletbc(ScalarType(0), dofs, self.V)]
        else:
            # TODO: we should pass directly the BCs, not the functions since
            # dolfinx wants functions and dofs.
            bcs = (
                dirichlet_bc_function
                if isinstance(dirichlet_bc_function, Iterable)
                else [dirichlet_bc_function]
            )

        if neumann_bc_function is None:
            neumann_bc = 0
            # neumann_bc_function = dolfinx.fem.Expression(
            #     Constant(self.mesh, ScalarType(0)),
            #     self.V.element.interpolation_points(),
            # )
        else:
            raise NotImplementedError
            neumann_bc_function * self.v * ds

        # solve the system taking into account the boundary conditions
        self.L = self.g * self.v * dx - neumann_bc  # - neumann_bc_function * self.v * ds

        self.problem = LinearProblem(
            self.a,
            self.L,
            u=self.psi,
            bcs=bcs,
            # petsc_options={"ksp_type": "preonly", "pc_type": "lu"},
        )

    def solve(self) -> BluemiraFemFunction:
        if self.problem is None:
            self.setup_problem()

        self.psi = self.problem.solve()

        return self.psi

    def calculate_b(self) -> BluemiraFemFunction:
        """
        Calculates the magnetic field intensity from psi

        Note: code from Fenics_tutorial (
        https://link.springer.com/book/10.1007/978-3-319-52462-7), pag. 104
        """
        # new function space for mapping B as vector
        W = functionspace(self.mesh, ("DG", 0, (self.mesh.geometry.dim,)))  # noqa: N806
        B = BluemiraFemFunction(W)
        x_0 = SpatialCoordinate(self.mesh)[0]
        B_expr = Expression(
            as_vector((
                -self.psi.dx(1) / (2 * np.pi * x_0),
                self.psi.dx(0) / (2 * np.pi * x_0),
            )),
            W.element.interpolation_points(),
        )
        B.interpolate(B_expr)

        return B
