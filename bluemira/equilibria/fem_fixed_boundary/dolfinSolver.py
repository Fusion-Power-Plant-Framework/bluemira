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

# Grad-Shafranov solver
# =====================

# The main class for solving the Grad-Shafranov equation

# Description
# -----------
# This class implements a general solver for the Grad-Shafranov equation where any
# right hand side, as function of (P(\psi),f(\psi)), can be prescribed.
# """

import numpy
import dolfin

class GradShafranovLagrange:
    def __init__(self, mesh, boundaries=None, p=3):
        # """
        # Grad-Shafranov solver implementation as a class.
        
        # DESCRIPTION
        # -----------
        # Solves the Grad-Shafranov equation:
        
        # Lagrange interpolants of order p are used for the unknown quantity.
        
        # INPUTS
        # ------
        # mesh : dolfin.mesh or string
        #        the filename of the xml file with the mesh definition
        #        or a dolfin mesh
                       
        # boundaries : dolfin.MeshFunction or string
        #              the filename of the xml file with the boundaries definition
        #              or a MeshFunction that defines the boundaries
                              
        # p : int
        #     the order of the approximating polynomial basis functions
            
        # """        

        """
        Grad-Shafranov solver implementation as a class.
        
        :param constrList: list of constraints of the type 
                [(P1, angle1),(P2, angle2), ...].
                Points must to have a "list" rappresentation, i.e. the x and y 
                coordinates must be accessible using P[0] and P[1].
                Note: FreeCAD.Base.Vector fits the requirements
        :type constrList: list(tuple)
        
        :param degree: option to consider angle as degree, defaults to False
        :type degree: boolean    
        
        :return: center coordinates and major and minor axis length
        :rtype: tuple
        
        Example:
            constrainList = [(P1,angle1), (P2,), (P3,)]
        """

        #======================================================================
        # define the geometry
        if isinstance(mesh, str): # check wether mesh is a filename or a mesh, then load it or use it
            self.mesh = dolfin.Mesh(mesh) # define the mesh
        else:
            self.mesh = mesh # use the mesh
        
        #======================================================================
        # define boundaries        
        if boundaries is None:                                                        # Dirichlet B.C. are defined
            self.boundaries = dolfin.MeshFunction("size_t", mesh, mesh.topology().dim()-1) # initialize the MeshFunction       
        elif isinstance(boundaries, str): # check wether boundaries is a filename or a MeshFunction, then load it or use it
            self.boundaries = dolfin.MeshFunction("size_t", self.mesh, boundaries) # define the boundaries
        else:
            self.boundaries = boundaries
        
        #======================================================================
        # define the function space and bilinear forms

        # the solution function space
        self.V = dolfin.FunctionSpace(self.mesh, 'CG', p)
        
        # define trial and test functions
        self.u = dolfin.TrialFunction(self.V)
        self.v = dolfin.TestFunction(self.V)               

        # Define r
        r = dolfin.Expression('x[0]', degree=p)
        
        self.a = 1/(2.*dolfin.pi*4*dolfin.pi*1e-7)*(1/r*dolfin.dot(dolfin.grad(self.u),dolfin.grad(self.v)))*dolfin.dx
        
        # initialize solution
        self.psi = dolfin.Function(self.V)

    @property
    def g(self):
        return self._g

    @g.setter
    def g(self, value):
        self._g = value

    @property
    def psi(self):
        return self._psi

    @psi.setter
    def psi(self, value):
        self._psi = value

    @property
    def psi_max(self):
        return self.psi.vector().max()

    def solve(self, g, dirichletBCFunction=None, dirichlet_marker=None,
              neumannBCFunction=None):
        """
        Solve the Grad-Shafranov equation given a right hand side g, Dirichlet and 
        Neumann boundary conditions and convergence tolerance error.
        
        INPUTS
        ------
        g : dolfin.Expression or dolfin.Function
            the right hand side function of the Poisson problem
        
        dirichletBCFunction : dolfin.Expression o dolfin.Function
                              the Dirichlet boundary condition function
        
        neumannBCFunction : dolfin.Expression or dolfin.Function
                            the Neumann boundary condition function
         
        tol : float64 
              the error goal to stop the iteration process
        
        dirichlet_marker : int
                           the identification number for the dirichlet boundary
                           
        OUTPUTS
        -------
        psi : the solution of the Grad-Shafranov problem
        
        """        
        if neumannBCFunction is None:
            neumannBCFunction = dolfin.Expression('0.0', degree=2)
        
        # define the right hand side         
        self.L = g*self.v*dolfin.dx - neumannBCFunction*self.v*dolfin.ds
        
        # define the Dirichlet boundary conditions
        if dirichletBCFunction is None:
            dirichletBCFunction = dolfin.Expression('0.0', degree=2)
            dirichletBC = dolfin.DirichletBC(self.V, dirichletBCFunction, 'on_boundary')
        else:
            dirichletBC = dolfin.DirichletBC(self.V, dirichletBCFunction, self.boundaries, dirichlet_marker) # dirichlet_marker is the identification of Dirichlet BC in the mesh
        bcs = [dirichletBC] 

        # solve the system taking into account the boundary conditions
        dolfin.solve(self.a == self.L, self.psi, bcs)
        self.__calculateB()

        dx = dolfin.Measure('dx', domain=self.mesh)
        print(f"total current: {dolfin.assemble(g*dx)}")

        # return the solution
        return self.psi

    # def __calculateB(self):
    #     # from https://scicomp.stackexchange.com/questions/32844/electromagnetism-fem-fenics-interpolation-leakage-effect
        
    #     # POSTPROCESSING
    #     #W = dolfin.VectorFunctionSpace(self.mesh, 'P', 1) # new function space for mapping B as vector
        
    #     r = dolfin.Expression('x[0]', degree = 1)
        
    #     # calculate derivatives
    #     Bx = -self.psi.dx(1)/(2*dolfin.pi*r)
    #     By = self.psi.dx(0)/(2*dolfin.pi*r)
        
    #     #B = dolfin.project( dolfin.as_vector(( Bx, By )), W ) # project B as vector to new function space
    #     B_abs = numpy.power( Bx**2 + By**2, 0.5 ) # compute length of vector
        
    #     # # plot B vectors
    #     # dolfin.plot(B)
    #     # plt.show()
        
    #     # define new function space as Discontinuous Galerkin
    #     abs_B = dolfin.FunctionSpace(self.mesh, 'DG', 0)
    #     f = B_abs # obtained solution is "source" for solving another PDE
        
    #     # make new weak formulation
    #     w_h = dolfin.TrialFunction(abs_B)
    #     v = dolfin.TestFunction(abs_B)
        
    #     a = w_h*v*dolfin.dx
    #     L = f*v*dolfin.dx
        
    #     w_h = dolfin.Function(abs_B)
    #     dolfin.solve(a == L, w_h)
        
    #     # # plot the solution
    #     # dolfin.plot(w_h)
    #     # plt.show()
    #     self.B = w_h
        
    def __calculateB(self):
        # POSTPROCESSING
        W = dolfin.VectorFunctionSpace(self.mesh, 'P', 1) # new function space for mapping B as vector
        
        r = dolfin.Expression('x[0]', degree = 1)
        
        # calculate derivatives
        Bx = -self.psi.dx(1)/(2*dolfin.pi*r)
        Bz = self.psi.dx(0)/(2*dolfin.pi*r)
        
        self.B = dolfin.project( dolfin.as_vector(( Bx, Bz )), W ) # project B as vector to new function space
        
        
