# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
Bluemira module for the solution of a 2D magnetostatic problem with cylindrical symmetry
using fenics FEM solver
"""

import dolfin


class FemMagnetostatic2d:
    """
    2D magnetostic solver

    Parameters
    ----------
    mesh : dolfin.mesh or string
           the filename of the xml file with the mesh definition
           or a dolfin mesh
    boundaries : dolfin.MeshFunction or string
                 the filename of the xml file with the boundaries definition
                 or a MeshFunction that defines the boundaries
    p : int
        the order of the approximating polynomial basis functions

    """

    def __init__(self, mesh, boundaries=None, p=3):
        # ======================================================================
        # define the geometry
        if isinstance(
            mesh, str
        ):  # check wether mesh is a filename or a mesh, then load it or use it
            self.mesh = dolfin.Mesh(mesh)  # define the mesh
        else:
            self.mesh = mesh  # use the mesh

        # ======================================================================
        # define boundaries
        if boundaries is None:  # Dirichlet B.C. are defined
            self.boundaries = dolfin.MeshFunction(
                "size_t", mesh, mesh.topology().dim() - 1
            )  # initialize the MeshFunction
        elif isinstance(boundaries, str):
            # check wether boundaries is a filename or a MeshFunction,
            # then load it or use it
            self.boundaries = dolfin.MeshFunction(
                "size_t", self.mesh, boundaries
            )  # define the boundaries
        else:
            self.boundaries = boundaries

        # ======================================================================
        # define the function space and bilinear forms
        self.V = dolfin.FunctionSpace(self.mesh, "CG", p)  # the solution function space

        # define trial and test functions
        self.u = dolfin.TrialFunction(self.V)
        self.v = dolfin.TestFunction(self.V)

        # Define r
        r = dolfin.Expression("x[0]", degree=p)

        self.a = (
            1
            / (2.0 * dolfin.pi * 4 * dolfin.pi * 1e-7)
            * (1 / r * dolfin.dot(dolfin.grad(self.u), dolfin.grad(self.v)))
            * dolfin.dx
        )

        # initialize solution
        self.psi = dolfin.Function(self.V)

    def solve(
        self,
        g,
        dirichlet_bc_function=None,
        dirichlet_marker=None,
        neumann_bc_function=None,
    ):
        """
        Solve the weak formulation maxwell equation given a right hand side g,
        Dirichlet and Neumann boundary conditions.

        Parameters
        ----------
        g : dolfin.Expression or dolfin.Function
            the right hand side function of the Poisson problem
        dirichlet_bc_function : dolfin.Expression o dolfin.Function
                              the Dirichlet boundary condition function
        neumann_bc_function : dolfin.Expression or dolfin.Function
                            the Neumann boundary condition function
        dirichlet_marker : int
                           the identification number for the dirichlet boundary

        Returns
        -------
        psi : dolfin function
            the solution of the magnetostatic problem
        """
        if neumann_bc_function is None:
            neumann_bc_function = dolfin.Expression("0.0", degree=2)

        # define the right hand side
        self.L = g * self.v * dolfin.dx - neumann_bc_function * self.v * dolfin.ds

        # define the Dirichlet boundary conditions
        if dirichlet_bc_function is None:
            dirichlet_bc_function = dolfin.Expression("0.0", degree=2)
            dirichlet_bc = dolfin.DirichletBC(
                self.V, dirichlet_bc_function, "on_boundary"
            )
        else:
            dirichlet_bc = dolfin.DirichletBC(
                self.V, dirichlet_bc_function, self.boundaries, dirichlet_marker
            )  # dirichlet_marker is the identification of Dirichlet BC in the mesh
        bcs = [dirichlet_bc]

        # solve the system taking into account the boundary conditions
        dolfin.solve(self.a == self.L, self.psi, bcs)

        self.__calculate_b()

        # return the solution
        return self.psi

    def __calculate_b(self):
        """Calculates the magnetic field intensity from psi"""
        w = dolfin.VectorFunctionSpace(
            self.mesh, "P", 1
        )  # new function space for mapping B as vector

        r = dolfin.Expression("x[0]", degree=1)

        # calculate derivatives
        Bx = -self.psi.dx(1) / (2 * dolfin.pi * r)
        Bz = self.psi.dx(0) / (2 * dolfin.pi * r)

        self.B = dolfin.project(
            dolfin.as_vector((Bx, Bz)), w
        )  # project B as vector to new function space
