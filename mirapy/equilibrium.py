# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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

import anytree
import numpy
import freecad, Part
from FreeCAD import Base

# import mirapy
from . import core
from . import meshing
from . import msh2xdmf
from . import dolfinSolver
from . import emag

import os
import matplotlib.pyplot as plt
import dolfin

import scipy
import scipy.interpolate
from scipy.integrate import quad    

import numpy
from numpy import dot, transpose, eye, array
from numpy.linalg import inv

class equilibrium(object):
    """
    Equilibrium class
    """
    def __init__(self):
        self.__solvers = {}
        self.__default_filter = lambda node: (hasattr(node, 'filaments')
                                              and node.filaments is not None)
        
    def __call__(self, machine, constraints=None):
        self.__machine = machine
        self.__constraints = constraints

        if 'fixed_boundary' not in self.__solvers:
            self.plasma_fixed_boundary()

        if constraints is not None:
            error = self.__constraints(self.__machine)
            print(error)
        
    @property
    def machine(self):
        return self.__machine
    

    def plasma_fixed_boundary(self, createmesh = True, meshfile = "Mesh.msh", 
                              meshdir = ".", Pax = None, Pax_lcar = 0.1, 
                              maxiter = 100, tol = 1e-6, p = 5, solver = None):
    
        
        plasma = self.machine.get_Plasma()
        
        if plasma is None:
            raise ValueError("No plasma has been found")

        if solver is None:
            print("Solver is None")
            if ((not hasattr(plasma.shape, 'physicalGroups'))
                    or plasma.shape.physicalGroups is None):
                plasma.shape.physicalGroups = {1: "sol", 2: "plasma"}
            else:
                if not 1 in plasma.shape.physicalGroups:
                    plasma.shape.physicalGroups[1] = "sol"
        
                if not 2 in plasma.shape.physicalGroups:
                    plasma.shape.physicalGroups[2] = "plasma"
            
            mesh_dim = 2
    
            if plasma.J is None:
                raise ValueError('Plamsa Jp must to be defined')
            
            fullmeshfile = os.path.join(meshdir, meshfile)
            
            print(fullmeshfile)
            
            if createmesh:
                #### Mesh Generation ####
                mesh = meshing.Mesh("plasma")
                mesh.meshfile = fullmeshfile
                if Pax is not None:
                    # P0lcar = plasma.shape.lcar/2.
                    mesh.embed = [(Pax, Pax_lcar)]
                mesh(plasma)
    
            # Run the conversion
            msh2xdmf.msh2xdmf(meshfile, dim = mesh_dim)
            
            # Run the import
            prefix, _ = os.path.splitext(fullmeshfile)
            
            mesh, boundaries, subdomains, labels = \
                msh2xdmf.import_mesh_from_xdmf(
                    prefix = prefix,
                    dim = mesh_dim,
                    directory = meshdir,
                    subdomains = True,
                )
          
            solver = dolfinSolver.GradShafranovLagrange(mesh, p=p)

        # Calculate plasma geometrical parameters            
        plasma.calculatePlasmaParameters(solver.mesh)
        
        eps = 1.0e10        # error measure ||u-u_k||
        i = 0               # iteration counter
        while eps > tol and i < maxiter:
            prev = solver.psi.compute_vertex_values()
            i += 1
            plasma.psi = solver.psi
            g = plasma.J_to_dolfinFunction(solver.V)
            solver.solve(g)
            diff = plasma.psi.compute_vertex_values() - prev
            eps = numpy.linalg.norm(diff, ord=numpy.Inf)
            print('iter = {} eps = {}'.format(i, eps))
            plasma.dolfinUpdate(solver.V)

        self.__solvers['fixed_boundary'] = solver        
        plasma.updateFilaments(solver.V)
    
    def get_solver(self, label = None):
        if label is None:
            return self.__solvers
        elif label in self.__solvers.keys():
            return self.__solvers[label]
        else:
            return None

    def remove_solver(self, label = None):
        return self.__solvers.pop(label, None)
    
    def tikhonov_solution(self, epsilon=1e-5, maxiter=1000):
        """
        Solves the constrained linear problem using tikhanov regularization
        """

        constraint_matrix = []
        constraint_rhs = []
        
        constraint_matrix, control_components = \
            self.__constraints._create_constraint_matrix(self.__machine)
        # Constraint matrix
        A = constraint_matrix        
        # Number of controls (length of x)
        ncontrols = A.shape[1]

        delta = 1e22
        iteration = 0
        while delta > epsilon and iteration < maxiter:
            constraint_rhs = self.__constraints._create_constraint_rhs(
                self.__machine)
    
            if constraint_matrix.shape[0] != constraint_rhs.shape[0]:
                raise ValueError("Error on constraint matrix and rhs")
                    
            b = numpy.reshape(constraint_rhs, (-1,))
        
            # Solve by Tikhonov regularisation
            # minimise || Ax - b ||^2 + ||gamma x ||^2
            #
            # x = (A^T A + gamma^2 I)^{-1}A^T b
            
            # Calculate current change
            current_change =  dot(inv(dot(transpose(A), A) 
                                      + self.gamma**2 * eye(ncontrols)
                                      ), dot(transpose(A),b)
                                  )
            
            # adjust coils' currents        
            for i in range(len(control_components)):
                c = control_components[i]
                c.filaments.Itot = c.filaments.Itot + current_change[i]
            
            delta = sum(abs(current_change))
            iteration += 1
            print("iteration: {}, error: {}".format(iteration, delta))

        constraint_rhs = self.__constraints._create_constraint_rhs(
            self.__machine)
        
        return self.__constraints._error()       

class constrain(object):
    """
    Adjust coil currents using constraints. To use this class,
    first create an instance by specifying the constraints

    >>> controlsystem = constrain(xpoints = [(1.0, 1.1), (1.0,-1.0)])

    controlsystem will now attempt to create x-points at
    (R,Z) = (1.0, 1.1) and (1.0, -1.1) in any Equilibrium

    >>> controlsystem(machine)

    where machine is an Equilibrium object which is modified by
    the call.
    
    The constraints which can be set are:

    xpoints - A list of X-point [[R1,Z1,0.], [R2, Z2, 0.]] locations 
             (type = 3D numpy matrix)
    
    isoflux - A list of isoflux points [[R1,Z1,0.], [R2, Z2, 0.]]
             (at least 2 points, type = 3D numpy matrix)
    
    psivals - A list of ( 3Dpoints, psi) values
              (3Dpoints is a 3D numpy matrix)

    At least one constraint must be included
    
    gamma - A scalar, minimises the magnitude of the coil currents
    """

    def __init__(self, xpoints=None, isoflux=None, psivals=None,
                 bxpoints=None, bypoints=None, bzpoints=None, epsilon=1e-5):
        """
        Create an instance, specifying the constraints to apply
        """
        self.xpoints = xpoints
        self.bxpoints = bxpoints
        self.bypoints = bypoints
        self.bzpoints = bzpoints
        self.isoflux = isoflux
        self.psivals = psivals
        
        self._constraint_matrix = []
        self._constraint_rhs = []
        
        self._epsilon = epsilon
        
    def _error(self):
        return scipy.linalg.norm(self._constraint_rhs)
    
    def _create_constraint_matrix(self, machine, nodefilter=None):
        """
        Apply constraints to Equilibrium machine
        """

        constraint_matrix = []
        if nodefilter is None:
            nodefilter = lambda node: (node.is_leaf
                                       and hasattr(node, "filaments")
                                       and node.filaments is not None
                                       and isinstance(node.filaments,
                                                      emag.FilamentItemGreen)
                                       and not node.filaments.fixed_current
                                       )

        control_components = machine.select_nodes(nodefilter)
        
        ###### apply xpoint constrains
        if not self.xpoints is None:             
            G = machine.calculateB(self.xpoints, total = False, green = True,
                                   nodefilter=nodefilter)
            constraint_matrix.append(numpy.vstack(G.T))

        ###### apply bpoint constrains
        if not self.bxpoints is None:
            for point, _ in self.bxpoints:             
                G = machine.calculateB(point, total = False, green = True,
                                       nodefilter=nodefilter)
                constraint_matrix.append(numpy.vstack(G.T[0,:]))

        ###### apply bxpoint constrains
        if not self.bypoints is None:
            for point, _ in self.bxpoints:             
                G = machine.calculateB(point, total = False, green = True,
                                       nodefilter=nodefilter)
                constraint_matrix.append(numpy.vstack(G.T[1,:]))

        ###### apply bzpoint constrains
        if not self.bzpoints is None:
            for point, _ in self.bzpoints:             
                G = machine.calculateB(point, total = False, green = True,
                                       nodefilter=nodefilter)
                constraint_matrix.append(numpy.vstack(G.T[2,:]))

        ###### Constrain points to have the same flux
        if not self.isoflux is None:
            G = machine.calculatePsi(self.isoflux, total = False, green = True,
                                     nodefilter=nodefilter)
            constraint_matrix.append(-(G.T[1:] - G.T[:-1]))

        ###### Constrain the value of psi
        if not self.psivals is None:
            for points, fixed_psi in self.psivals:
                G = machine.calculatePsi(points, total = False, green = True,
                                         nodefilter=nodefilter)
                constraint_matrix.append(numpy.vstack(G.T))

        if len(constraint_matrix) == 0:
            raise ValueError("No variable currents given")

        constraint_matrix = numpy.vstack(constraint_matrix)
        
        return constraint_matrix, control_components
    
    def _create_constraint_rhs(self, machine, nodefilter=None):
        constraint_rhs = []
        if nodefilter is None:
            nodefilter = lambda node: (node.is_leaf
                                       and hasattr(node, "filaments")
                                       and node.filaments is not None
                                       and isinstance(node.filaments,
                                                      emag.FilamentItemGreen)
                                       and node.filaments.fixed_current
                                       )

        ###### apply xpoint constrains
        if not self.xpoints is None:
            B = machine.calculateB(self.xpoints, nodefilter=nodefilter)
            constraint_rhs.append(-B.T.flatten())

        ###### apply bxpoint constrains
        if not self.bxpoints is None:
            for point, fixed_b in self.bxpoints:
                B = machine.calculateB(point, nodefilter=nodefilter)                
                constraint_rhs.append(fixed_b - B.T[0,:].flatten())

        ###### apply bypoint constrains
        if not self.bypoints is None:
            for point, fixed_b in self.bypoints:
                B = machine.calculateB(point, nodefilter=nodefilter)                
                constraint_rhs.append(fixed_b - B.T[1,:].flatten())

        ###### apply bzpoint constrains
        if not self.bzpoints is None:
            for point, fixed_b in self.bzpoints:
                B = machine.calculateB(point, nodefilter=nodefilter)                
                constraint_rhs.append(fixed_b - B.T[2,:].flatten())

        ###### Constrain points to have the same flux
        if not self.isoflux is None:
            Psi = machine.calculatePsi(self.isoflux, nodefilter=nodefilter)            
            constraint_rhs.append(Psi[1:] - Psi[:-1])

        ###### Constrain the value of psi
        if not self.psivals is None:
            for points, fixed_psi in self.psivals:
                psi = machine.calculatePsi(points, nodefilter=nodefilter)               
                constraint_rhs.append(fixed_psi - psi)

        if len(constraint_rhs) == 0:
            raise ValueError("No constraints given")

        constraint_rhs = numpy.concatenate(constraint_rhs)
        
        return constraint_rhs



# class constrain(object):
#     """
#     Adjust coil currents using constraints. To use this class,
#     first create an instance by specifying the constraints

#     >>> controlsystem = constrain(xpoints = [(1.0, 1.1), (1.0,-1.0)])

#     controlsystem will now attempt to create x-points at
#     (R,Z) = (1.0, 1.1) and (1.0, -1.1) in any Equilibrium

#     >>> controlsystem(machine)

#     where machine is an Equilibrium object which is modified by
#     the call.
    
#     The constraints which can be set are:

#     xpoints - A list of X-point [[R1,Z1,0.], [R2, Z2, 0.]] locations 
#              (type = 3D numpy matrix)
    
#     isoflux - A list of isoflux points [[R1,Z1,0.], [R2, Z2, 0.]]
#              (at least 2 points, type = 3D numpy matrix)
    
#     psivals - A list of ( 3Dpoints, psi) values
#               (3Dpoints is a 3D numpy matrix)

#     At least one constraint must be included
    
#     gamma - A scalar, minimises the magnitude of the coil currents
#     """

#     def __init__(self, xpoints=None, gamma=1e-12, isoflux=None,
#                  psivals=None, bpoints=None, bxpoints=None, bzpoints=None):
#         """
#         Create an instance, specifying the constraints to apply
#         """
#         self.xpoints = xpoints
#         self.bpoints = bpoints
#         self.bxpoints = bxpoints
#         self.bzpoints = bzpoints
#         self.gamma = gamma
#         self.isoflux = isoflux
#         self.psivals = psivals
        
#         self._constraint_matrix = []
#         self._constraint_rhs = []
        
#         self._epsilon = 1e-5
        
#     def __call__(self, machine):
#         """
#         Apply constraints to machine
#         """

#         constraint_matrix = []
#         constraint_rhs = []
        
#         constraint_matrix, control_components = self._create_constraint_matrix(machine)
#         # Constraint matrix
#         A = constraint_matrix        
#         # Number of controls (length of x)
#         ncontrols = A.shape[1]

#         delta = 1e22
#         iteration = 0
#         while delta > self._epsilon and iteration < 1000:
#             constraint_rhs = self._create_constraint_rhs(machine)
    
#             if constraint_matrix.shape[0] != constraint_rhs.shape[0]:
#                 raise ValueError("Error on constraint matrix and rhs")
                    
#             b = numpy.reshape(constraint_rhs, (-1,))
        
#             # Solve by Tikhonov regularisation
#             # minimise || Ax - b ||^2 + ||gamma x ||^2
#             #
#             # x = (A^T A + gamma^2 I)^{-1}A^T b
            
#             # Calculate current change
#             current_change =  dot(inv(dot(transpose(A), A)
#                                       + self.gamma**2 * eye(ncontrols)), 
#                                     dot(transpose(A),b))
            
#             # adjust coils' currents        
#             for i in range(len(control_components)):
#                 c = control_components[i]
#                 c.filaments.Itot = c.filaments.Itot + current_change[i]
            
#             delta = sum(abs(current_change))
#             iteration += 1
#             print("iteration: {}, error: {}".format(iteration, delta))

#         constraint_rhs = self._create_constraint_rhs(machine)

#         self._constraint_rhs = constraint_rhs
#         self._constraint_matrix = constraint_matrix
        
#         return self._error()     
    
#     def _error(self):
#         return scipy.linalg.norm(self._constraint_rhs)
    
#     def _create_constraint_matrix(self, machine):
#         """
#         Apply constraints to Equilibrium machine
#         """

#         constraint_matrix = []

#         filter_ = lambda node: (hasattr(node, 'filaments') and 
#                                 (not node.filaments is None) and 
#                                 (not node.filaments.fixed_current))

#         prev_nodefilter = machine.nodefilter
        
#         machine.nodefilter = filter_
        
#         control_components = machine.selected_nodes()
        
#         ###### apply xpoint constrains
#         if not self.xpoints is None:             
#             G = machine.calculateB(self.xpoints, total = False, green = True)
#             constraint_matrix.append(numpy.vstack(G.T[0:2,:]))

#         ###### apply bpoint constrains
#         if not self.bpoints is None:
#             for point, _ in self.bpoints:             
#                 G = machine.calculateB(point, total = False, green = True)
#                 constraint_matrix.append(numpy.vstack(G.T[0:2,:]))

#         ###### apply bxpoint constrains
#         if not self.bxpoints is None:
#             for point, _ in self.bxpoints:             
#                 G = machine.calculateB(point, total = False, green = True)
#                 constraint_matrix.append(numpy.vstack(G.T[0,:]))

#         ###### apply bzpoint constrains
#         if not self.bzpoints is None:
#             for point, _ in self.bzpoints:             
#                 G = machine.calculateB(point, total = False, green = True)
#                 constraint_matrix.append(numpy.vstack(G.T[1,:]))

#         ###### Constrain points to have the same flux
#         if not self.isoflux is None:
#             G = machine.calculatePsi(self.isoflux, total = False, green = True)
#             constraint_matrix.append(-(G.T[1:] - G.T[:-1]))

#         ###### Constrain the value of psi
#         if not self.psivals is None:
#             for points, fixed_psi in self.psivals:
#                 G = machine.calculatePsi(points, total = False, green = True)
#                 constraint_matrix.append(numpy.vstack(G.T))

#         if len(constraint_matrix) == 0:
#             raise ValueError("No variable currents given")

#         constraint_matrix = numpy.vstack(constraint_matrix)

#         machine.nodefilter = prev_nodefilter
        
#         return constraint_matrix, control_components
    
#     def _create_constraint_rhs(self, machine):
#         constraint_rhs = []   

#         ###### apply xpoint constrains
#         if not self.xpoints is None:
#             B = machine.calculateB(self.xpoints)
#             constraint_rhs.append(-B.T[0:2,:].flatten())

#         ###### apply bpoint constrains
#         if not self.bpoints is None:
#             for point, fixed_b in self.bpoints:
#                 B = machine.calculateB(point)                
#                 constraint_rhs.append(fixed_b[0:2] - B.T[0:2,:].flatten())

#         ###### apply bpoint constrains
#         if not self.bxpoints is None:
#             for point, fixed_b in self.bxpoints:
#                 B = machine.calculateB(point)                
#                 constraint_rhs.append(fixed_b - B.T[0,:].flatten())

#         ###### apply bpoint constrains
#         if not self.bzpoints is None:
#             for point, fixed_b in self.bzpoints:
#                 B = machine.calculateB(point)                
#                 constraint_rhs.append(fixed_b - B.T[1,:].flatten())

#         ###### Constrain points to have the same flux
#         if not self.isoflux is None:
#             Psi = machine.calculatePsi(self.isoflux)            
#             constraint_rhs.append(Psi[1:] - Psi[:-1])

#         ###### Constrain the value of psi
#         if not self.psivals is None:
#             for points, fixed_psi in self.psivals:
#                 psi = machine.calculatePsi(points)                
#                 constraint_rhs.append( fixed_psi - psi)

#         if len(constraint_rhs) == 0:
#             raise ValueError("No constraints given")

#         constraint_rhs = numpy.concatenate(constraint_rhs)
        
#         return constraint_rhs
        