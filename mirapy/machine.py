#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""

from . import core
from . import meshing
from . import msh2xdmf
from . import dolfinSolver
from . import femhelper
from . import emag
from . import const

import anytree, numpy, freecad, Part
from FreeCAD import Base
import matplotlib.pyplot as plt
import os
import dolfin

class Machine():
    """ """
    
    def __init__(self, nsect:int, root:core.Component):
        self.__tokamak = root
        self.__nsect = nsect
        self.__default_filter = lambda node: node.is_leaf #select all leaves
        self.set_to_default()

    def set_to_default(self):
        """ """
        self.__filter = self.__default_filter
            
    @property
    def tokamak(self):
        """ """
        return self.__tokamak
    
    @tokamak.setter
    def tokamak(self, value):
        """

        Parameters
        ----------
        value :
            

        Returns
        -------

        """
        print("Warning: Machine.tokamak is going to be changed."
              " This operation calls set_to_default function.")
        self.__tokamak = value
        self.set_to_default()
    
    @property
    def nsect(self):
        """ """
        return self.__nsect
    
    @property
    def nodefilter(self):
        """ """
        return self.__filter

    @nodefilter.setter
    def nodefilter(self, value):
        """

        Parameters
        ----------
        value :
            

        Returns
        -------

        """
        self.__filter = value
    
    def select_nodes(self, nodefilter):
        """

        Parameters
        ----------
        nodefilter :
            

        Returns
        -------

        """
        nodes = anytree.search.findall(self.__tokamak, 
                                   filter_= nodefilter)
        return nodes

    def selected_nodes(self):
        """ """
        return self.select_nodes(self.__filter)
    
    def __getClass(self, cls_name):
        filter_ = lambda node: isinstance(node, cls_name)
        nodes = self.select_nodes(filter_)
        return nodes

    ### Magnetic functions
    def get_Plasma(self):
        """ """
        plasma = self.__getClass(core.Plasma)
        if len(plasma) != 1:
            print("Warning: found {} plasma(s). None is"
                  " returned.".format(len(plasma)))
            return None
        return plasma[0]

    def get_TFCoils(self):
        """ """
        return self.__getClass(core.TFCoil)

    def get_PFCoils(self):
        """ """
        return self.__getClass(core.PFCoil)

    def get_Coils(self):
        """ """
        return self.__getClass(core.Coil)

    def ripple(self, targetpoints, ndiscr=3):
        """

        Parameters
        ----------
        targetpoints :
            
        ndiscr :
             (Default value = 3)

        Returns
        -------

        """
        filter_ = lambda node: isinstance(node, core.TFCoil)
        nodes = self.select_nodes(filter_)
        ripple = numpy.zeros(targetpoints.shape[0])
        points = [Part.Point(Base.Vector(p)) for p in targetpoints]
        for i in range(len(points)):
            p = points[i]
            B = numpy.zeros((ndiscr, 3))

            toroidalPoints = []
            angles = numpy.linspace(0.0, 360.0/self.__nsect, num=ndiscr)

            for alpha in angles:
                newp = p.copy()
                bp = Base.Placement(Base.Vector(), Base.Vector(0,0,1), alpha)
                newp.rotate(bp)
                toroidalPoints.append(newp)
                
            points3D = numpy.array([[v.X, v.Y, v.Z] for v in toroidalPoints])
            for n in nodes:
                B += n.filaments.calculateB(points3D)
            
            rotateB = [Part.Point(Base.Vector(v)) for v in B]
            
            for j in range(len(rotateB)):
                rotateB[j].rotate(Base.Placement(Base.Vector(), Base.Vector(0,0,1), -angles[j]))
           
            rotateB = numpy.array([[v.X, v.Y, v.Z] for v in rotateB])
            By = rotateB[:,1] 
            Bmax = max(By)
            Bmin = min(By)
            ripple[i] = abs((Bmax - Bmin)/(Bmax + Bmin))
        
        ripple = numpy.array([ripple]).T
        result = numpy.append(targetpoints, ripple, axis=1)  
            
        return result
    
    def ripple_sol(self, ndiscr = 3, ndiscrsol = 50):
        """

        Parameters
        ----------
        ndiscr :
             (Default value = 3)
        ndiscrsol :
             (Default value = 50)

        Returns
        -------

        """
        plasma = self.get_Plasma()
        # if isinstance(plasma.shape, mirapy.geo.Shape):
        #     targetpoints = plasma.shape.getSingleWire().discretize(ndiscrsol)
        # else:
        targetpoints = plasma.shape.getSingleWires()[0].discretize(ndiscrsol)
        targetpoints = numpy.array(targetpoints)
        result = self.ripple(targetpoints, ndiscr)
        return result

    def calculateB(self, targetpoints, total=True, green=False, nodefilter=None):
        """

        Parameters
        ----------
        targetpoints :
            
        total :
             (Default value = True)
        green :
             (Default value = False)
        nodefilter :
             (Default value = None)

        Returns
        -------

        """
        if nodefilter is None:
            nodefilter = self.__filter
        nodes = self.select_nodes(nodefilter)
        B = numpy.asarray([n.filaments.calculateB(targetpoints, green) 
                           for n in nodes if (hasattr(n, "filaments")
                                              and not n.filaments is None)
                           ])
        if total:
            B = sum(B)
        return B

    def calculateA(self, targetpoints, total=True, green=False, nodefilter=None):
        """

        Parameters
        ----------
        targetpoints :
            
        total :
             (Default value = True)
        green :
             (Default value = False)
        nodefilter :
             (Default value = None)

        Returns
        -------

        """
        if nodefilter is None:
            nodefilter = self.__filter
        nodes = self.select_nodes(nodefilter)
        A = numpy.asarray([n.filaments.calculateA(targetpoints, green) 
                           for n in nodes if not n.filaments is None])
        if total:
            A = sum(A)
        return A

    def calculatePsi(self, targetpoints, total = True, green = False,
                     nodefilter = None):
        """

        Parameters
        ----------
        targetpoints :
            
        total :
             (Default value = True)
        green :
             (Default value = False)
        nodefilter :
             (Default value = None)

        Returns
        -------

        """
        
        if nodefilter is None:
            nodefilter = lambda node: (node.is_leaf
                                       and hasattr(node, "filaments")
                                       and isinstance(node.filaments,
                                                      emag.FilamentItemGreen)
                                      )
        nodes = self.select_nodes(nodefilter)
        
        Psi = numpy.asarray([n.filaments.calculatePsi(targetpoints, green) 
                             for n in nodes])
        if total:
            Psi = sum(Psi)
        return Psi
    
    def calculateFz(self, component, nodefilter = None):
        """

        Parameters
        ----------
        component :
            
        nodefilter :
             (Default value = None)

        Returns
        -------

        """
        if nodefilter is None:
            nodefilter = lambda node: (hasattr(node, "filaments")
                                       and isinstance(node.filaments,
                                                      emag.FilamentItemGreen)
                                       )
        if component in self.get_PFCoils():
            x = component.filaments.Rc
            y = x*0.
            z = component.filaments.Zc

            targetpoints = numpy.array([x, y, z]).T
            Br = self.calculateB(targetpoints, nodefilter=nodefilter)[:,0]
            Fz = -component.filaments.Ifil*Br*2*numpy.pi*component.filaments.Rc
            return sum(Fz)
        else:
            raise ValueError("Component must be a PFCoil")
    
    def calculateFzSep(self, components, nodefilter = None):
        """

        Parameters
        ----------
        components :
            
        nodefilter :
             (Default value = None)

        Returns
        -------

        """
        if nodefilter is None:
            nodefilter = lambda node: isinstance(node, core.PFCoil)

        Fz = numpy.array([self.calculateFz(c) for c in components])
        Fzup = numpy.zeros(Fz.shape)
        Fzdown = numpy.zeros(Fz.shape)
        Fzup[0] = Fz[0]
        Fzdown[0] = Fz[-1]
        for i in range(len(Fz)-1):
            Fzup[i+1] = Fzup[i] + Fz[i+1]
            Fzdown[i+1] = Fzdown[i] + Fz[-1-(i+1)] 
        
        Fzupmax = max(Fzup)
        Fzdownmin = min(Fzdown)
        
        FzSep = (abs(Fzupmax) + abs(Fzdownmin))/2.
        
        return (Fz, Fzup, Fzdown, Fzupmax, Fzdownmin, FzSep)
    
    def calculateBetap(self):
        """ """
        plasma = self.get_Plasma()
        if plasma is None:
            print("A plasma must to be specified in order to calculate Betap")
            return None

        if not hasattr(plasma, 'mesh'):
            if (not hasattr(plasma.shape, 'physicalGroups') or
                    plasma.shape.physicalGroups is None):
                plasma.shape.physicalGroups = {1: "sol", 2: "plasma"}
            else:
                if not 1 in plasma.shape.physicalGroups:
                    plasma.shape.physicalGroups[1] = "sol"
        
                if not 2 in plasma.shape.physicalGroups:
                    plasma.shape.physicalGroups[2] = "plasma"
            
            meshfile = "plasma.msh"
            meshdir = os.getcwd()
            mesh_dim = 2
            
            mesh = meshing.Mesh("plasma")
            mesh.meshfile = os.path.join(meshdir, meshfile)
            mesh(plasma)
            
            # Run the conversion
            msh2xdmf.msh2xdmf(meshfile, dim= mesh_dim, directory = meshdir)
            
            # Run the import
            prefix, _ = os.path.splitext(meshfile)
            
            mesh, boundaries, subdomains, labels = msh2xdmf.import_mesh_from_xdmf(
                prefix = prefix,
                dim = mesh_dim,
                directory = meshdir,
                subdomains = True,
                )
            plasma.mesh = mesh
        
        mesh = plasma.mesh
        # Define r and z
        r = dolfin.Expression('x[0]', degree = 2)
        
        # Define subdomain integration measure
        dx = dolfin.Measure('dx', domain = mesh)
        ds = dolfin.Measure('ds', domain = mesh)

        Ip = plasma.Itot
        def myfunc(x):
            """

            Parameters
            ----------
            x :
                

            Returns
            -------

            """
            psi = plasma.psi(x)
            psinorm = (plasma.psi(x) - plasma.psiax)/(plasma.psib - plasma.psiax)
            return plasma.p(psinorm)
        
        p = femhelper.func_to_dolfinFunction(myfunc, 
                    dolfin.FunctionSpace(mesh, 'Lagrange', 2))
        
        pavg = dolfin.assemble(p*r*dx())/dolfin.assemble(r*dx())
        Bpavg = const.MU0*Ip/plasma.lp
        betap = 2*const.MU0*pavg/Bpavg**2
        
        return betap
    
    def plot2D(self, axis = None, show: bool = False, ndiscr: int = 100, 
               *argv, **kwargs):

        """2D plot of the machine's tokamak. Only leaves are plotted.

        Parameters
        ----------
        axis : matplotlib.pyplot.axis
            matplotlib.pyplot.axis. \
            Defaults to None.
        show : bool
            if True, plot is shown. (Default value = False)
        ndiscr : int
            number of points for the discretization. \
            (Default value = 100)
        *argv :
            not used.
        **kwargs :
            dictionary used to specify other plot properties.

        Returns
        -------
        axis : matplotlib.pyplot.axis
            plot axis.

        """

        axis = self.tokamak.plot2D(axis, show, ndiscr)
        return axis

    def plot2Dfilaments(self, axis = None, show = False):
        """Plot filaments.
        
        .. todo:
            * add *argv and **kwargs in order to be compatible with the other\
            plot functions.
            * move this method to Filament.

        Parameters
        ----------
        axis :
            (Default value = None)
        show :
            (Default value = False)

        Returns
        -------
        axis :

        """

        leaves = self.tokamak.leaves
        for l in leaves:
            if hasattr(l, 'filaments') and l.filaments:
                axis = l.filaments.plot2D(axis = axis, show = False)

        if show:
            plt.show()
            
        return axis
       