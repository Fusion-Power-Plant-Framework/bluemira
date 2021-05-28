#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# __doc__ = """

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
    
        #

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
        
        self.V = dolfin.FunctionSpace(self.mesh,'CG',p) # the solution function space
        
        # define trial and test functions
        self.u = dolfin.TrialFunction(self.V)
        self.v = dolfin.TestFunction(self.V)               

        # Define r
        r = dolfin.Expression('x[0]', degree = p)
        
        self.a = 1/(2.*dolfin.pi*4*dolfin.pi*1e-7)*(1/r*dolfin.dot(dolfin.grad(self.u),dolfin.grad(self.v)))*dolfin.dx
        
        # initialize solution
        self.psi = dolfin.Function(self.V)

    def solve(self, g, dirichletBCFunction=None, dirichlet_marker=None, 
              neumannBCFunction=None):
        """Solve the Grad-Shafranov equation given a right hand side g, Dirichlet and
        Neumann boundary conditions and convergence tolerance error.
        
        Parameters
        ----------
        
        g: dolfin.Expression, dolfin.Function
            the right hand side function of the Poisson problem
        
        dirichletBCFunction: dolfin.Expression, dolfin.Function
            the Dirichlet boundary condition function
        
        neumannBCFunction: dolfin.Expression, dolfin.Function
            the Neumann boundary condition function
        
        tol: float64
            the error goal to stop the iteration process
        
        dirichlet_marker: int
            the identification number for the dirichlet boundary
        
        Returns
        -------
        psi:
            the solution of the Grad-Shafranov problem

        """
        if neumannBCFunction is None:
            neumannBCFunction = dolfin.Expression('0.0', degree = 2)
        
        # define the right hand side         
        self.L = g*self.v*dolfin.dx - neumannBCFunction*self.v*dolfin.ds
        
        # define the Dirichlet boundary conditions
        if dirichletBCFunction is None:
            dirichletBCFunction = dolfin.Expression('0.0', degree = 2)
            dirichletBC = dolfin.DirichletBC(self.V, dirichletBCFunction, 'on_boundary')
        else:
            dirichletBC = dolfin.DirichletBC(self.V, dirichletBCFunction, self.boundaries, dirichlet_marker) # dirichlet_marker is the identification of Dirichlet BC in the mesh
        bcs = [dirichletBC] 

        # solve the system taking into account the boundary conditions
        dolfin.solve(self.a == self.L, self.psi, bcs)
        
        self.__calculateB()
        
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
        Bx = -self.psi.dx(2)/(2*dolfin.pi*r)
        By = Bx*0
        Bz = self.psi.dx(0)/(2*dolfin.pi*r)
        
        self.B = dolfin.project( dolfin.as_vector(( Bx, By, Bz )), W ) # project B as vector to new function space
        
        