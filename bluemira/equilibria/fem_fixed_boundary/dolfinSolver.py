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
Grad-Shafranov solver
"""
import dolfin
import numpy

import bluemira.equilibria.fem_fixed_boundary.tools as tools


class GradShafranovLagrange:
    """
    A solver for the Grad-Shafranov equation:
    # Todo: add GS equation
    where any right hand side, as function of psi, can be prescribed.

    Lagrange interpolants of order p are used for the unknown quantity.

    Parameters
    ----------
    g : float, function, dolfin.Expression, dolfin.Function
        the right hand side function of the Poisson problem

    mesh : dolfin.mesh, str
           a dolfin mesh object or the filename of the xml file with the
           mesh definition

    boundaries : dolfin.MeshFunction or string
                 the filename of the xml file with the boundaries definition
                 or a MeshFunction that defines the boundaries

    p : int
        the order of the approximating polynomial basis functions

    dirichlet_boundary : dolfin.Expression o dolfin.Function
                          the Dirichlet boundary condition function

    dirichlet_marker : int
                       the identification number for the dirichlet boundary

    neumann_boundary : dolfin.Expression or dolfin.Function
                        the Neumann boundary condition function

    tol : float64
          the error goal to stop the iteration process
    """

    def __init__(
        self,
        g,
        mesh,
        p=3,
        boundaries=None,
        dirichlet_boundary=None,
        dirichlet_marker=None,
        neumann_boundary=None,
        tol=1e-6,
    ):
        self.tol = tol

        # check if mesh is a filename or a mesh, then load it or use it
        if isinstance(mesh, str):
            self.mesh = dolfin.Mesh(mesh)
        else:
            self.mesh = mesh

        # define boundaries
        if boundaries is None:
            # Dirichlet B.C. are defined
            self.boundaries = dolfin.MeshFunction(
                "size_t", mesh, mesh.topology().dim() - 1
            )
        # check if boundaries is a filename or a MeshFunction, then load it or use it
        elif isinstance(boundaries, str):
            self.boundaries = dolfin.MeshFunction("size_t", self.mesh, boundaries)
        else:
            self.boundaries = boundaries

        # define the function space and bilinear forms
        # the solution function space
        self.V = dolfin.FunctionSpace(self.mesh, "CG", p)
        # define trial and test functions
        self.u = dolfin.TrialFunction(self.V)
        self.v = dolfin.TestFunction(self.V)
        # Define r
        r = dolfin.Expression("x[0]", degree=p)

        # define the left hand side
        self.a = (
            1
            / (2.0 * dolfin.pi * 4 * dolfin.pi * 1e-7)
            * (1 / r * dolfin.dot(dolfin.grad(self.u), dolfin.grad(self.v)))
            * dolfin.dx
        )

        # initialize solution
        self.psi = dolfin.Function(self.V)

        if neumann_boundary is None:
            self.neumann_boundary = dolfin.Expression("0.0", degree=2)

        self.g = g

        # define the Dirichlet boundary conditions
        if dirichlet_boundary is None:
            self.dirichlet_boundary = dolfin.Expression("0.0", degree=2)
            self.dirichlet_boundary = dolfin.DirichletBC(
                self.V, self.dirichlet_boundary, "on_boundary"
            )
        else:
            self.dirichlet_boundary = dolfin.DirichletBC(
                self.V, dirichlet_boundary, self.boundaries, dirichlet_marker
            )
        self._bcs = [self.dirichlet_boundary]

    @property
    def g(self):
        """
        Right hand side, as function of psi
        Todo: add a better description and, probably, find a better name
        """
        return self._g

    @g.setter
    def g(self, value):
        """
        Right hand side, as function of psi
        Todo: add a better description and, probably, find a better name
        """
        self._g = self.check_g(value)

    def check_g(self, value):
        """Check the value for g and, if valid, return a dolfin function"""
        if isinstance(value, (int, float)) or callable(value):
            return tools.func_to_dolfinFunction(value, self.V)
        elif isinstance(value, (dolfin.Expression, dolfin.Function)):
            return value
        else:
            raise ValueError(
                f"{value} of type{type(value)} is neither an int, float, callable, dolfin.Expression, "
                f"or dolfin.Function"
            )

    # WIP to speed up the GS solver
    # def update_g(self, value):
    #     g = self.check_g(value)
    #     mesh = self.V.mesh()
    #     points = mesh.coordinates()
    #     d2v = self.V.dofmap().tabulate_local_to_global_dofs()
    #     data = numpy.array([g(p) for p in points])
    #     new_data = [data[d2v[i]] for i in range(mesh.num_vertices())]
    #     self._g.vector().set_local(new_data)

    @property
    def psi(self):
        """
        Magnetic flux
        """
        return self._psi

    @psi.setter
    def psi(self, value):
        """
        Magnetic flux
        """
        self._psi = value

    @property
    def psi_max(self):
        """
        Maximum value of the magnetic flux
        """
        return self.psi.vector().max()

    @property
    def dirichlet_boundary_function(self):
        """
        dolfin.Expression or dolfin.Function the Dirichlet boundary condition function
        """
        return self._dirichlet_boundary_function

    @dirichlet_boundary_function.setter
    def dirichlet_boundary_function(self, value):
        """
        dolfin.Expression or dolfin.Function the Dirichlet boundary condition function
        """
        self._dirichlet_boundary_function = value

    @property
    def dirichlet_markers(self):
        """
        The identification number (int) for the dirichlet boundary
        """
        return self._dirichlet_markers

    @dirichlet_markers.setter
    def dirichlet_markers(self, value):
        """
        The identification number (int) for the dirichlet boundary
        """
        self._dirichlet_markers = value

    @property
    def neumann_boundary_function(self):
        """
        dolfin.Expression or dolfin.Function the Neumann boundary condition function
        """
        return self._neumann_boundary_function

    @neumann_boundary_function.setter
    def neumann_boundary_function(self, value):
        """
        dolfin.Expression or dolfin.Function the Neumann boundary condition function
        """
        self._neumann_boundary_function = value

    @property
    def tol(self):
        """The error goal (float) to stop the iteration process"""
        return self._tol

    @tol.setter
    def tol(self, value):
        """The error goal (float) to stop the iteration process"""
        self._tol = value

    def solve(self):
        """
        Solve the Grad-Shafranov equation given a right hand side g, Dirichlet and
        Neumann boundary conditions and convergence tolerance error.
        """
        # define the right hand side
        self.L = self.g * self.v * dolfin.dx - self.neumann_boundary * self.v * dolfin.ds

        # solve the system taking into account the boundary conditions
        dolfin.solve(self.a == self.L, self.psi, self._bcs)

        self._calculate_B()

        # dx = dolfin.Measure("dx", domain=self.mesh)
        # print(f"total current: {dolfin.assemble(g*dx)}")

        # return the solution
        return self.psi

    def _calculate_B(self):  # noqa(N802)
        """Postprocessing function to calculate the magnetic field"""
        # new function space for mapping B as vector
        w = dolfin.VectorFunctionSpace(self.mesh, "P", 1)

        r = dolfin.Expression("x[0]", degree=1)

        # calculate derivatives
        Bx = -self.psi.dx(1) / (2 * dolfin.pi * r)
        Bz = self.psi.dx(0) / (2 * dolfin.pi * r)

        # project B as vector to new function space
        self.B = dolfin.project(dolfin.as_vector((Bx, Bz)), w)
