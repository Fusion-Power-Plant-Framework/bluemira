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
and toroidal current source using fenics FEM solver
"""
from typing import Callable, Iterable, Optional, Union

import dolfin
import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.constants import MU_0
from bluemira.base.look_and_feel import bluemira_print_flush
from bluemira.equilibria.fem_fixed_boundary.utilities import (
    ScalarSubFunc,
    plot_scalar_field,
)


class FemMagnetostatic2d:
    """
    A 2D magnetostic solver. The solver is thought as support for the fem fixed
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
    mesh : Union[dolfin.Mesh, str]
        Filename of the xml file with the mesh definition or a dolfin mesh
    boundaries : Union[dolfin.Mesh, str]
        Filename of the xml file with the boundaries definition or a MeshFunction that
        defines the boundaries
    p_order : int
        Order of the approximating polynomial basis functions
    """  # noqa (W505)

    def __init__(
        self,
        mesh: Union[dolfin.Mesh, str],
        boundaries: Union[dolfin.Mesh, str] = None,
        p_order: int = 3,
    ):

        self.p_order = p_order

        # check whether mesh is a filename or a mesh, then load it or use it
        self.mesh = dolfin.Mesh(mesh) if isinstance(mesh, str) else mesh

        # define boundaries
        if boundaries is None:
            # initialize the MeshFunction
            self.boundaries = dolfin.MeshFunction(
                "size_t", mesh, mesh.topology().dim() - 1
            )
        elif isinstance(boundaries, str):
            # check wether boundaries is a filename or a MeshFunction,
            # then load it or use it
            self.boundaries = dolfin.MeshFunction(
                "size_t", self.mesh, boundaries
            )  # define the boundaries
        else:
            self.boundaries = boundaries

        # define the function space and bilinear forms
        # the Continuos Galerkin function space has been chosen as suitable for the
        # solution of the magnetostatic weak formulation in a Soblev Space H1(D)
        self.V = dolfin.FunctionSpace(self.mesh, "CG", self.p_order)

        # define trial and test functions
        self.u = dolfin.TrialFunction(self.V)
        self.v = dolfin.TestFunction(self.V)

        # Define r
        r = dolfin.Expression("x[0]", degree=self.p_order)

        self.a = (
            1
            / (2.0 * dolfin.pi * MU_0)
            * (1 / r * dolfin.dot(dolfin.grad(self.u), dolfin.grad(self.v)))
            * dolfin.dx
        )

        # initialize solution
        self.psi = dolfin.Function(self.V)
        self.psi.set_allow_extrapolation(True)

    def solve(
        self,
        g: Union[dolfin.Expression, dolfin.Function],
        dirichlet_bc_function: Union[dolfin.Expression, dolfin.Function] = None,
        dirichlet_marker: int = None,
        neumann_bc_function: Union[dolfin.Expression, dolfin.Function] = None,
    ) -> dolfin.Function:
        """
        Solve the weak formulation maxwell equation given a right hand side g,
        Dirichlet and Neumann boundary conditions.

        Parameters
        ----------
        g : Union[dolfin.Expression, dolfin.Function]
            Right hand side function of the Poisson problem
        dirichlet_bc_function : Union[dolfin.Expression, dolfin.Function]
            Dirichlet boundary condition function
        dirichlet_marker : int
            Identification number for the dirichlet boundary
        neumann_bc_function : Union[dolfin.Expression, dolfin.Function]
            Neumann boundary condition function

        Returns
        -------
        psi : dolfin.Function
            Poloidal magnetic flux as solution of the magnetostatic problem
        """
        if neumann_bc_function is None:
            neumann_bc_function = dolfin.Expression("0.0", degree=self.p_order)

        # define the right hand side
        self.L = g * self.v * dolfin.dx - neumann_bc_function * self.v * dolfin.ds

        # define the Dirichlet boundary conditions
        if dirichlet_bc_function is None:
            dirichlet_bc_function = dolfin.Expression("0.0", degree=self.p_order)
            dirichlet_bc = dolfin.DirichletBC(
                self.V, dirichlet_bc_function, "on_boundary"
            )
        else:
            dirichlet_bc = dolfin.DirichletBC(
                self.V, dirichlet_bc_function, self.boundaries, dirichlet_marker
            )
        self.bcs = [dirichlet_bc]

        # solve the system taking into account the boundary conditions
        dolfin.solve(
            self.a == self.L,
            self.psi,
            self.bcs,
            solver_parameters={"linear_solver": "default"},
        )

        # return the solution
        return self.psi

    def calculate_b(self):
        """
        Calculates the magnetic field intensity from psi

        Note: code from Fenics_tutorial (
        https://link.springer.com/book/10.1007/978-3-319-52462-7), pag. 104
        """
        # new function space for mapping B as vector
        w = dolfin.VectorFunctionSpace(self.mesh, "CG", 1)

        r = dolfin.Expression("x[0]", degree=1)

        # calculate derivatives
        Bx = -self.psi.dx(1) / (2 * dolfin.pi * r)
        Bz = self.psi.dx(0) / (2 * dolfin.pi * r)

        # project B as vector to new function space
        self.B = dolfin.project(dolfin.as_vector((Bx, Bz)), w)

        return self.B


class FemGradShafranovFixedBoundary(FemMagnetostatic2d):
    """
    A 2D fem Grad Shafranov solver. The solver is thought as support for the fem fixed
    boundary module.

    Parameters
    ----------
    mesh : Union[dolfin.Mesh, str]
        Filename of the xml file with the mesh definition or a dolfin mesh
    boundaries : Union[dolfin.Mesh, str]
        Filename of the xml file with the boundaries definition or a MeshFunction that
        defines the boundaries
    p_order : int
        Order of the approximating polynomial basis functions
    """  # noqa (W505)

    def __init__(
        self,
        mesh: Union[dolfin.Mesh, str],
        boundaries: Union[dolfin.Mesh, str] = None,
        p_order: int = 3,
    ):
        super().__init__(mesh, boundaries, p_order)
        self.k = 1

    @property
    def psi_ax(self) -> np.ndarray:
        """Poloidal flux on the magnetic axis"""
        return np.max(self.psi.vector()[:])

    @property
    def psi_b(self) -> float:
        """Poloidal flux on the boundary"""
        return 0.0  # np.min(self.psi.vector()[:])

    @property
    def psi_norm_2d(self) -> Callable:
        """Normalized flux function in 2-D"""

        def myfunc(x):
            return np.sqrt(
                np.abs((self.psi(x) - self.psi_ax) / (self.psi_b - self.psi_ax))
            )

        return myfunc

    def _create_g_func(
        self,
        pprime: Union[Callable, float],
        ffprime: Union[Callable, float],
        curr_target: Optional[float] = None,
    ) -> Callable:
        """
        Return the density current function given pprime and ffprime.

        Parameters
        ----------
        pprime: Union[callable, float]
            pprime as function of psi_norm (1-D function)
        ffprime: Union[callable, float]
            ffprime as function of psi_norm (1-D function)
        curr_target: Optional[float]
            Target current (also used to initialize the solution in case self.psi is
            still 0 and pprime and ffprime are, then, not defined) [A]
            If None, the plasma current is calculated and not constrained

        Returns
        -------
        g: callable
            Source current to solve the magnetostatic problem
        """
        dx = dolfin.Measure("dx", domain=self.mesh)
        area = dolfin.assemble(dolfin.Constant(1) * dx())

        j_target = curr_target / area if curr_target else 1.0

        def g(x):
            if self.psi_ax == 0:
                return j_target
            else:
                r = x[0]
                x_psi = self.psi_norm_2d(x)

                a = r * (pprime(x_psi) if callable(pprime) else pprime)
                b = 1 / MU_0 / r * (ffprime(x_psi) if callable(ffprime) else ffprime)

                return self.k * (a + b)

        return g

    def _create_g(
        self,
        pprime: Union[Callable, float],
        ffprime: Union[Callable, float],
        curr_target: Optional[float] = None,
    ) -> Callable:
        """
        Return the density current DOLFIN function given pprime and ffprime.

        Parameters
        ----------
        pprime: Union[callable, float]
            pprime as function of psi_norm (1-D function)
        ffprime: Union[callable, float]
            ffprime as function of psi_norm (1-D function)
        curr_target: float
            Target current (also used to initialize the solution in case self.psi is
            still 0 and pprime and ffprime are, then, not defined)

        Returns
        -------
        g: callable
            Source current to solve the magnetostatic problem
        """
        myfunc = self._create_g_func(pprime, ffprime, curr_target)
        func = ScalarSubFunc(myfunc)
        return func

    def _calculate_curr_tot(self):
        """Calculate the total current into the domain"""
        dx = dolfin.Measure("dx", domain=self.mesh)
        return dolfin.assemble(self.g * dx())

    def _update_curr(self, curr_target: float):
        self.k = 1
        if curr_target:
            self.k = curr_target / self._calculate_curr_tot()

    def _plot_current_iteration(self, points: Iterable, i_iter: int):
        curr_data = np.array([self.g_func(p) for p in points])
        self._plot_array(
            points, curr_data, f"J current at iteration {i_iter}", contour=False
        )

    def _plot_array(
        self, points: np.ndarray, array: np.ndarray, title: str, contour: bool = True
    ):
        plot_scalar_field(
            points[:, 0],
            points[:, 1],
            array,
            levels=20,
            axis=None,
            tofill=True,
            contour=contour,
        )
        plt.title(title)
        plt.show()

    def solve(
        self,
        pprime: Union[callable, float],
        ffprime: Union[callable, float],
        curr_target: Optional[float] = None,
        dirichlet_bc_function: Optional[
            Union[dolfin.Expression, dolfin.Function]
        ] = None,
        dirichlet_marker: Optional[int] = None,
        neumann_bc_function: Optional[Union[dolfin.Expression, dolfin.Function]] = None,
        iter_err_max: float = 1e-5,
        max_iter: int = 10,
        relaxation: float = 0.0,
        plot: bool = False,
    ) -> dolfin.Function:
        """
        Solve the G-S problem given pprime and ffprime.

        Parameters
        ----------
        pprime: Union[callable, float]
            pprime as function of psi_norm (1-D function)
        ffprime: Union[callable, float]
            ffprime as function of psi_norm (1-D function)
        curr_target: Optional[float]
            Target total plasma current [A]
            If None, plasma current is calculated and not constrained
        dirichlet_bc_function : Optional[Union[dolfin.Expression, dolfin.Function]]
            Dirichlet boundary condition function. Defaults to a Dirichlet boundary
            condition of 0 on the plasma boundary.
        dirichlet_marker : int
            Identification number for the dirichlet boundary
        neumann_bc_function : Optional[Union[dolfin.Expression, dolfin.Function]]
            Neumann boundary condition function. Defaults to a Neumann boundary
            condition of 0 on the plasma boundary.
        iter_err_max: float
            Convergence criterion value
        max_iter: int
            Maximum number of iterations
        relaxation: float
            Relaxation factor for the Picard iteration procedure
        plot: bool
            Whether or not to plot

        Returns
        -------
        psi: dolfin.Function
            dolfin.Function for psi
        """
        points = self.mesh.coordinates()

        if plot:
            self.g_func = self._create_g_func(pprime, ffprime, curr_target)
            self._plot_current_iteration(points, 0)

        self.g = self._create_g(pprime, ffprime, curr_target)

        super().solve(
            self.g, dirichlet_bc_function, dirichlet_marker, neumann_bc_function
        )
        self._update_curr(curr_target)

        eps = 1.0
        for i in range(1, max_iter + 1):
            prev_psi = self.psi.vector()[:]

            if plot:
                self._plot_current_iteration(points, i)

            prev = np.array([self.psi_norm_2d(p) for p in points])

            if plot:
                self._plot_array(
                    points, prev, f"Normalized magnetic coordinate at iteration {i}"
                )

            super().solve(
                self.g, dirichlet_bc_function, dirichlet_marker, neumann_bc_function
            )

            new = np.array([self.psi_norm_2d(p) for p in points])
            diff = new - prev

            if plot:
                self._plot_array(points, diff, f"G-S error at iteration {i}")

            eps = np.linalg.norm(diff, ord=2) / np.linalg.norm(new, ord=2)

            bluemira_print_flush(
                f"iter = {i} eps = {eps:.3E} psi_ax : {self.psi_ax:.2f}"
            )

            # Update psi in-place (Fenics handles this with the below syntax)
            self.psi.vector()[:] = (1 - relaxation) * self.psi.vector()[
                :
            ] + relaxation * prev_psi

            self._update_curr(curr_target)

            if eps > iter_err_max:
                break

        return self.psi
