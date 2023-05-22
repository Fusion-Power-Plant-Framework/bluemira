from typing import List, Tuple

import dolfinx.fem
import numpy as np
from dolfinx.fem import (
    Expression,
    Function,
    FunctionSpace,
    VectorFunctionSpace,
    assemble_scalar,
    dirichletbc,
    form,
    locate_dofs_topological,
)
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import Mesh, locate_entities_boundary
from petsc4py.PETSc import ScalarType
from ufl import (
    Measure,
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    as_vector,
    dot,
    dx,
    grad,
)


class FemMagnetostatic2D:
    """
    A 2D magnetostic solver for 2D planar and axisymmetric problems.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
        mesh of the FEM model
    boundaries:
        boundaries mesh tags
    eltype: Tuple (default ("CG",1))
        tuple with the specification of the element family and the degree of the element
    symmetry: str (default "cyl")
        specification of the problem symmetry. "cart" for planar problems and "cyl" for
        axisymmetric problems.
    """

    # TODO: check if the problem can be solved with any eltype or the field should
    #   be hardcoded and removed from the signature.
    def __init__(
        self, mesh: Mesh, boundaries, eltype: Tuple = ("CG", 1), symmetry: str = "cyl"
    ):
        self.mesh = mesh
        self.boundaries = boundaries
        self._eltype = eltype
        self._V = FunctionSpace(self.mesh, self._eltype)
        self._symmetry = symmetry
        self.__implemented_symmetry = ["cart", "cyl"]

    @property
    def symmetry(self):
        return self._symmetry

    @symmetry.setter
    def symmetry(self, value: str):
        if value in self.__implemented_symmetry:
            self._symmetry = "value"
        else:
            raise ValueError(
                f"{value} is not an available model symmetry ({self.__implemented_symmetry})"
            )

    def solve(self, J, dirichlet_bcs: Tuple[int, dolfinx.fem.Function] = None):
        """
        Solve the em problem.

        Parameters
        ----------
        J : dolfinx.fem.Function
            current density function in the mesh region
        dirichlet_bcs : Tuple[int, dolfinx.fem.Function] (default None)
            Dirichlet boundary conditions given as a tuple of (marker,dirichlet_function)

        Notes
        -----
        if symmetry is "cyl", self.psi is updated
        if symmetry is "cart", self.A_z is updated

        TODO: check if it is possible just to have the same topological output without
            increase too much the computational time.
        """
        tdim = self.mesh.topology.dim
        facets = locate_entities_boundary(
            self.mesh, tdim - 1, lambda x: np.full(x.shape[1], True)
        )
        dofs = locate_dofs_topological(self._V, tdim - 1, facets)
        if dirichlet_bcs is None:
            bc = dirichletbc(ScalarType(0), dofs, self._V)
        else:
            # TODO: implements user-defined boundary conditions
            raise ValueError("User-defined boundary conditions still not implemented")
        u = TrialFunction(self._V)
        v = TestFunction(self._V)
        mu_0 = 4 * np.pi * 1e-7
        if self._symmetry == "cart":
            a = 1 / mu_0 * dot(grad(u), grad(v)) * dx
            L = J * v * dx
        if self._symmetry == "cyl":
            x = SpatialCoordinate(self.mesh)
            a = 1 / (2.0 * np.pi * mu_0) * (1 / x[0] * dot(grad(u), grad(v))) * dx
            L = J * v * dx
        u_h = Function(self._V)
        problem = LinearProblem(a, L, u=u_h, bcs=[bc])
        # TODO: check petsc_options
        #   problem = LinearProblem(a, L, u=u_h, bcs=[bc],
        #       petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
        u_h = problem.solve()

        if self.symmetry == "cart":
            self.A_z = u_h
        if self.symmetry == "cyl":
            self.Psi = u_h

    def compute_B(self, eltype: Tuple = ("DG", 0)):
        """
        Compute the magnetic field interpolating the result in a Function Space
        with eltype elements.
        """
        W0 = VectorFunctionSpace(self.mesh, ("DG", 0))
        B0 = Function(W0)

        if self.symmetry == "cart":
            B_expr = Expression(
                as_vector((self.A_z.dx(1), -self.A_z.dx(0))),
                W0.element.interpolation_points(),
            )
        if self.symmetry == "cyl":
            epsilon = 1e-8
            x = SpatialCoordinate(self.mesh)

            # TODO: create a user expression to consider the epsilon
            #  correction only when x[0] = 0
            r = abs(x[0] - epsilon)
            B_expr = Expression(
                as_vector(
                    (
                        -self.Psi.dx(1) / (2 * np.pi * r),
                        self.Psi.dx(0) / (2 * np.pi * r),
                    )
                ),
                W0.element.interpolation_points(),
            )

        B0.interpolate(B_expr)

        if eltype is not None:
            W = VectorFunctionSpace(self.mesh, eltype)
            B = Function(W)
            B.interpolate(B0)
        else:
            B = B0

        return B


def create_j_function(mesh: dolfinx.mesh.Mesh, boundaries, values: List[Tuple]):
    """

    Parameters
    ----------
    mesh: dolfinx.mesh.Mesh
        mesh of the FEM model
    boundaries:
        boundaries mesh tags
    values: List[Tuple[float, int]]
        list of association (current, boundaries_tag)

    Returns
    -------
    dolfinx.fem.Function:
        a dolfinx function with the values of the density current to be applied
        at each cell

    TODO: this function works only for constant values, but it should be
        modified to account for more complex current functions.
    """
    dx = Measure("dx", subdomain_data=boundaries, domain=mesh)

    Q = FunctionSpace(mesh, ("DG", 0))
    wire_tags = np.unique(boundaries.values)
    J = Function(Q)
    # As we only set some values in J, initialize all as 0
    J.x.array[:] = 0
    for v, tag in values:
        if tag in wire_tags:
            area = assemble_scalar(form(1 * dx(tag)))
            cells = boundaries.find(tag)
            J.x.array[cells] = np.full_like(cells, v / area, dtype=ScalarType)
    return J
