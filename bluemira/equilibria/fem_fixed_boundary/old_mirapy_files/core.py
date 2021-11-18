#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:30:45 2020

@author: ivan
"""
import math
from mirapy import const

import anytree
from anytree import NodeMixin, RenderTree
import freecad
import Part

import numpy as np
import mirapy
from mirapy import plotting, geohelper, meshing, emag, femhelper

import gmsh
import matplotlib.pyplot as plt

import scipy.interpolate

import os
import numpy
import dolfin

import logging
module_logger = logging.getLogger(__name__)

class Shape():
    def __init__(self, objs, label = "", lcar = 0.1, mesh_dim = 2):
        self.objs = None
        if not hasattr(objs, '__len__'):
            objs = [objs]
        if all(isinstance(o, (Part.Wire, Part.Face, Shape)) for o in objs):
            self.objs = objs
        else:
            print(objs)
            raise ValueError("Only Part.Wire, Part.Face and shape can be used as objs.")
        self.label = label
        self.mesh_dim = mesh_dim
        self.lcar = lcar

    @property
    def Length(self):
        return sum([o.Length for o in self.objs])

    # @property
    # def Area(self):
    #     area = 0.
    #     try:
    #         area = sum([o.Area for o in self.objs])
    #     except:
    #         print("Area calculation failed")
    #     return area
    
    @property
    def Wires(self):
        wires = []
        for o in self.objs:
            wires += o.Wires
        return wires
    
    def getSingleWire(self):
        return Part.Wire(self.Wires)

    def getCenter(self):
        return self.getSingleWire().BoundBox.Center

    def getCurves(self):
        output = []
        for o in self.objs:
            if isinstance(o, Shape):
                out = o.getCurves()
                if out:
                    output += out
            elif isinstance(o, Part.Wire):
                out = geohelper.convert_Wire_to_Curves(o)
                if out:
                    output += out                
        return output

    def search(self, label):
        output = []
        if self.label == label:
            output.append(self)
        for o in self.objs:
            if isinstance(o, Shape):
                output += o.search(label)
        return output

    def plot2D(self, axis = None, show = False, ndiscr = 100, *argv, **kwargs):
        axis = axis
        for o in self.objs:
            if isinstance(o, Part.Wire):
                axis = plotting.plotWire2D(o, axis, show, ndiscr, **kwargs)
            if isinstance(o, Part.Face):
                axis = plotting.plotFace2D(o, axis, show, ndiscr, **kwargs)
            if isinstance(o, Shape):
                axis = o.plot2D(axis, show, ndiscr, **kwargs)                            
        return axis

    def __repr__(self):
        new = []
        new.append("({}:".format(type(self).__name__))
        new.append(" {}".format(self.label))
        new.append(" {}".format(self.objs))
        new.append(" {}".format(self.lcar))
        new.append(")")
        return ", ".join(new)     


class Component(NodeMixin):
    def __init__(self, name = "", shape = None, mesh_dim = 3, parent=None, children=None, 
                 filaments = None, *argsv, **kwargs):
        
        #unique name
        self.name = name
        #geometrical component
        if shape:
            self.shape = shape
        self.mesh_dim = mesh_dim
        #parent
        self.parent = parent
        #children
        if children:
            self.children = children
        
        if kwargs:
            for k,v in kwargs.items():
                self.__dict__[k] = v

        self.kwargs_plot2D_options = {}

        self.repr_dict = ['name', '_shape', 'mesh_dim']

        self.filaments = filaments

    @property
    def shape(self):
        if hasattr(self, '_shape'):
            return self._shape
        return None
    
    @shape.setter
    def shape(self, value):
        if not isinstance(value, Shape):
            raise ValueError("Component shape must be a Shape")
        self._shape = value

    def search_shape(self, name):
        objs = []
        if self.shape:
            o = self.shape.search(name)
            if o:
                objs += o
        for l in self.children:
            o = l.search_shape(name)
            if o:
                objs += o
        return objs

    def get_component(self, name):
        c = anytree.search.findall_by_attr(self, name)
        if not c:
            return c
        return c[0]

    def plot2D(self, axis = None, show = False, ndiscr = 100, *argv, **kwargs):
        leaves = self.leaves
        for l in leaves:
            if l.shape:        
                if not kwargs:
                    kwargs = self.kwargs_plot2D_options
                    axis = l.shape.plot2D(axis, False, ndiscr, *argv, **kwargs)
        if show:
            plt.show()        
        return axis

    def plot2Dfilaments(self, axis = None, show = False):

        if self.filaments:
            axis = self.filaments.plot2D(axis = axis, show = False)
        else:
            for node in self.children:
                axis = node.plot2Dfilaments(axis = axis, show = False)      

        if show:
            plt.show()
            
        return axis


        
    def __repr__(self):
        new = []
        new.append("({}:".format(type(self).__name__))
        [new.append("{}: {}".format(k,v)) for k,v in self.__dict__.items() if k in self.repr_dict]
        new.append(")")
        return ", ".join(new)
    

class Mesh():
    def __init__(self, modelname = "Mesh", terminal = 1, mesh_dim = 2, 
                 meshfile = "Mesh.geo_unrolled", embed = [], physicalGroups = []):
        
        self.modelname = modelname
        self.terminal = terminal
        self.mesh_dim = mesh_dim
        self.meshfile = meshfile
        # embed is an array of tuple (point, lcar)
        self.embed = embed
        self.physicalGroups = physicalGroups

    def __call__(self, obj, clean = True):
        meshing._freecadGmsh._initialize_mesh(self.terminal, self.modelname)
        self.__mesh_obj(obj)
        self.__embed_points()
        self.__check_physical_groups(obj)
        meshing._freecadGmsh._generate_mesh(self.mesh_dim)  
        meshing._freecadGmsh._save_mesh(self.meshfile)
        meshing._freecadGmsh._finalize_mesh()
        self.__clean_mesh_obj(obj, clean)

    def __mesh_obj(self, obj):
        mesh_ref = min(obj.mesh_dim, self.mesh_dim)
        if isinstance(obj, Shape):
            self.__mesh_shape(obj, mesh_ref)  
        if isinstance(obj, Component):
            self.__mesh_component(obj, mesh_ref)        
            
    def __clean_mesh_obj(self, obj, clean = True):
        if isinstance(obj, Shape):
            self.__clean_mesh_shape(obj, clean)
        if isinstance(obj, Component):
            self.__clean_mesh_component(obj, clean)  

    def __mesh_shape(self, obj, mesh_dim = 3):
        if not hasattr(obj, "ismeshed") or not obj.ismeshed:
#            print("__mesh_shape name = {}".format(obj.label))
            mesh_ref = min(obj.mesh_dim, mesh_dim, self.mesh_dim)
            curveloops = []
            obj.mesh_dict = []
            for o in obj.objs:
                if isinstance(o, Part.Wire):
                    obj.mesh_dict += meshing._freecadGmsh._convert_wire_to_gmsh(
                        o, obj.lcar, mesh_ref)
                if isinstance(o, Shape):
                    self.__mesh_shape(o, mesh_ref)          
                    # fragment only made on points and curves
                    # if fragment in made also on surface, the construction points
                    # internal to a surface will be considered as "Point in Surface"
                    # (like Embedded points)
                    # try: all_ent, oo, oov = meshing._freecadGmsh._fragment([0,1,2])
                    all_ent, oo, oov = meshing._freecadGmsh._fragment([0,1])
        
                    o.mesh_dict = meshing._freecadGmsh._map_mesh_dict(o.mesh_dict, 
                                                              all_ent, oov)

                    if mesh_ref > 1:
                        try:
                            curvetag = [t for k,t in o.mesh_dict if k == 1]
                            looptag = gmsh.model.occ.addCurveLoop(curvetag)
                            gmsh.model.occ.synchronize()
                            obj.mesh_dict += [(-1, looptag)]
                        except:
                            obj.mesh_dict += o.mesh_dict

            if mesh_ref > 1:
                    try:
                        curvetag = [t for k,t in obj.mesh_dict if k == 1]
                        looptag = gmsh.model.occ.addCurveLoop(curvetag)
                        gmsh.model.occ.synchronize()
                        obj.mesh_dict += [(-1, looptag)]
                    except:
                        pass
                    
            obj.ismeshed = True        

    def __mesh_component(self, obj, mesh_dim = 3):
        #only leaves are meshed
        leaves = obj.leaves
        mesh_ref = min(obj.mesh_dim, mesh_dim)
        for l in leaves:
            l_mesh_ref = min(mesh_ref, l.mesh_dim)
            self.__mesh_shape(l.shape, l_mesh_ref)

            if l_mesh_ref > 1:
                try:
                    looptag = [t for k,t in l.shape.mesh_dict if k == -1]
                    l.shape.mesh_dict += [(2, gmsh.model.occ.addPlaneSurface(looptag))]
                    gmsh.model.occ.synchronize()
                except:
                    print("WARNING: surface creation for %s failed", l.name)
                    
    def __clean_mesh_shape(self, obj, clean = True):
        if hasattr(obj, 'ismeshed'):
            delattr(obj, 'ismeshed')
            try:
                if clean:
                    delattr(obj, 'mesh_dict')
            except:
                print("Warning: shape meshed without mesh_dict.TO CHECK!")
            
            for o in obj.objs:
                if isinstance(o, Shape):
                    self.__clean_mesh_shape(o, clean)
            
    def __clean_mesh_component(self, obj, clean = True):
        leaves = obj.leaves
        for l in leaves:
            self.__clean_mesh_shape(l.shape, clean)
    
                
    def get_mesh_dict(self, obj, dim = None):
        mesh_dict = []
        if hasattr(obj, "mesh_dict"):
            mesh_dict += obj.mesh_dict
        if isinstance(obj, Shape):
            for o in obj.objs:
                mesh_dict += self.get_mesh_dict(o)
        if isinstance(obj, Component):
            if obj.shape:
                mesh_dict += self.get_mesh_dict(obj.shape)
            if not obj.is_leaf:
                for l in obj.leaves:
                    mesh_dict += self.get_mesh_dict(l)
        if not dim is None:
            mesh_dict = [v for k,v in mesh_dict if k == dim]
        return mesh_dict

    def __check_physical_groups(self, obj):
        if hasattr(obj, "physicalGroups"):
            for dim,name in obj.physicalGroups.items():
                meshing._freecadGmsh._add_physical_group(obj, dim, name)
        if isinstance(obj, Shape):
            for i in obj.objs:
                self.__check_physical_groups(i)
        if isinstance(obj, Component):
            if obj.shape:
                self.__check_physical_groups(obj.shape)
            if not obj.is_leaf:
                print("This condition should not happen. Only leaves are meshed!")
                [self.__check_physical_groups(l) for l in obj.leaves]
    
    def __embed_points(self):
        if self.embed:
            for p,l in self.embed:
                meshing._freecadGmsh._embed_point(p,l)


class FilamentItemGreen():
    
    def __init__(self, name = None, Rc = [], Zc = [], Ifil = [], Itot = 0.,
                 fixed_current = False):
        
        self.name = name
        self.fixed_current = fixed_current
        Rc = np.asarray(Rc)
        Zc = np.asarray(Zc)
        Ifil = np.asarray(Ifil)

        if len(Rc) == len(Zc):
            self.Rc = Rc
            self.Zc = Zc

            nfil = len(Rc)

            if Ifil.size == 0:
                self.Itot = Itot
            elif len(Ifil) == nfil:    
                self.Ifil = Ifil
            else:
                print("Unexpected error: Rc, Zc and Ifil have not the same length")
                raise
        else:
            print("Unexpected error: Rc and Zc have not the same length")
            raise

    @property
    def Itot(self):
        return sum(self.Ifil)

    @Itot.setter
    def Itot(self, Itot):
        nfil = len(self.Rc)
        if nfil > 0:
            self.Ifil = np.ones(nfil)*Itot/nfil
        else:
            print("Warning: no filaments")
            self.Ifil = np.asarray([])

    def calculateBr(self, targetpoints, green = False):
        if len(targetpoints.shape) == 1:
            targetpoints = numpy.array([targetpoints])
        R = targetpoints[:,0]
        Z = targetpoints[:,1]
        if green:
            Ifil = None
        else:
            Ifil = self.Ifil
        Br = emag.Greens.calculateBr(self.Rc, self.Zc, R, Z, Ifil)
        return Br

    def calculateBz(self, targetpoints, green = False):
        if len(targetpoints.shape) == 1:
            targetpoints = numpy.array([targetpoints])
        R = targetpoints[:,0]
        Z = targetpoints[:,1]
        if green:
            Ifil = None
        else:
            Ifil = self.Ifil
        Bz = emag.Greens.calculateBz(self.Rc, self.Zc, R, Z, Ifil)
        return Bz

    def calculateB(self, targetpoints, green = False):        
        if len(targetpoints.shape) == 1:
            targetpoints = numpy.array([targetpoints])
        R = targetpoints[:,0]
        Z = targetpoints[:,1]
        if green:
            Ifil = None
        else:
            Ifil = self.Ifil

        B = emag.Greens.calculateB(self.Rc, self.Zc, R, Z, Ifil)
        return B

    def calculatePsi(self, targetpoints, green = False):
        if len(targetpoints.shape) == 1:
            targetpoints = numpy.array([targetpoints])
        R = targetpoints[:,0]
        Z = targetpoints[:,1]
        if green:
            Ifil = None
        else:
            Ifil = self.Ifil
        Psi = emag.Greens.calculatePsi(self.Rc, self.Zc, R, Z, Ifil)
        return Psi

    def plot2D(self, axis = None, show = False):
        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot()
            
        plt.plot(self.Rc, self.Zc, 's', marker='o', color='red')
        
        plt.gca().set_aspect("equal")
        
        if show:
            plt.show()
            
        return axis

    def __repr__(self):
        return "{" + str(self.name) +": "+ str(self.Itot) + ", " + str(self.Rc.shape) + "}"        

class FilamentItemBiot():
    def __init__(self, name = None, filshape = [], Ifil = [], 
                 ndiscr = 100, dcable = 0.01, fixed_current = False):
        
        self.name = name
        self.__ndiscr = ndiscr
        self.dcable = dcable        
        self.filshape = filshape        
        self.fixed_current = fixed_current
        self.Ifil = Ifil
        
    @property
    def filpoints(self):
        return self.__filpoints

    @filpoints.setter
    def filpoints(self,value):
        self.__filpoints  = value
        self.__r = np.array([(p[1:] + p[0:-1])/2. for p in self.__filpoints])
        self.__dl = np.array([(p[1:] - p[0:-1]) for p in self.__filpoints])        
    
    @property
    def r(self):
        return self.__r
    
    @property
    def dl(self):
        return self.__dl
    
    @property
    def ndiscr(self):
        return self.__ndiscr
    
    @ndiscr.setter
    def ndiscr(self, value):
        self.__ndiscr = value
        try:
            self.filpoints = np.array([np.array(c.discretize(self.__ndiscr)) for c in self.__filshape])
        except:
            print("no ndiscr")
            pass
    
    @property
    def Idl(self):
        return np.array([np.multiply(d,I) for d,I in zip(self.__dl,self.__Ifil)])
    
    @property
    def Itot(self):
        Itot = 0.
        if self.Ifil:
            Itot = sum(np.array(self.__Ifil))
        return Itot

    @Itot.setter
    def Itot(self, Itot):
        nfil = len(self.__filshape)
        self.__Ifil = []
        if nfil > 0:
            self.__Ifil = np.ones(nfil)*Itot/nfil
        else:
            print("Warning: no filaments")

    @property
    def Ifil(self):
        return self.__Ifil

    @Ifil.setter
    def Ifil(self, Ifil):
        nfil = len(self.filshape)        
        if nfil:        
            if len(Ifil) == nfil:    
                self.__Ifil = Ifil
            else:
                print("warning: filshape and Ifil have not the same length")
                print("Ifil is set to []")
                self.__Ifil = []
        else:
            self.__Ifil = []

    @property
    def filshape(self):
        return self.__filshape
    
    @filshape.setter
    def filshape(self, filshape):
        self.__filshape = filshape
        self.__filpoints = np.array([])
        self.__r = np.array([])
        self.__dl = np.array([])
        if not filshape is None:
            
            if not hasattr(filshape, "__len__"):
                self.__filshape = [filshape]
                        
            self.ndiscr = self.__ndiscr

    def calculateB(self, targetpoints, green = False):
        """
        Calculate B at targetpoints. Option "green" not used (only
        left for compatibility with other methods).

        Args:
            targetpoints (3D numpy.array): target points
            green (bool): NOT USED. Defaults to False.

        Returns:
            TYPE: B at target points.

        """
        from mirapy.emag import BiotSavart
        
        if len(targetpoints.shape) == 1:
            targetpoints = numpy.array([targetpoints])
        
        r = np.vstack(self.__r)
        Idl = np.vstack(self.Idl)

        return BiotSavart.calculateB(targetpoints, r, Idl)

    def calculatePsi(self, targetpoints, green = False):
        """
        Calculate Psi at targetpoints. Option "green" not used (only
        left for compatibility with other methods).

        Args:
            targetpoints (3D numpy.array): target points
            green (bool): NOT USED. Defaults to False.

        Returns:
            TYPE: B at target points.

        WARNING: this method is still not implemente and returns always 0.

        """
        if len(targetpoints.shape) == 1:
            targetpoints = numpy.array([targetpoints])
        return numpy.zeros(len(targetpoints))


    def selfForce(self):
        B = self.calculateB(np.vstack(self.r))
        F = np.cross(np.vstack(self.Idl), B)
        return F
    
    def plot(self, axis = None, show = False):

        if axis is None:
            fig = plt.figure()
            axis = plt.axes(projection="3d")
            
        if not self.filshape is None:
            for p in self.__filpoints:
                axis.plot(p[:,0], p[:,1], p[:,2], 'black')
                axis.scatter3D(p[:,0], p[:,1], p[:,2], marker = 'o', c = 'red')
                
        if show:
            plt.show()
            
        return axis
    
    def plot2D(self, axis = None, show = False):

        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot()
            
        if not self.filshape is None:
            for p in self.__filpoints:
                axis.plot(p[:,0], p[:,1], c = 'red')
        
        plt.gca().set_aspect("equal")
        
        if show:
            plt.show()
            
        return axis    
    
    def __repr__(self):
        return "{" + str(self.name) +": "+ str(self.Itot) + ", " + str(self.__r.shape) + "}"  


class FilamentTFCoil(FilamentItemBiot):
    def __init__(self, name = None, filshape = [], Ifil = [], ndiscr = 100, 
                 dcable = 0.01, fixed_current = False, nsect = 1):

        self.nsect = nsect
        
        super().__init__(name = name, filshape = filshape, Ifil = Ifil, 
                     ndiscr = ndiscr, dcable = dcable, 
                     fixed_current = fixed_current)
        
    @property
    def nsect(self):
        return self.__nsect
    
    @nsect.setter
    def nsect(self,value):
        self.__nsect = value
        self.__angleTF = np.linspace(0., 360., num = self.__nsect + 1)   

    @property
    def angleTF(self):
        return self.__angleTF    

    @FilamentItemBiot.filshape.setter
    def filshape(self, filshape):
        FilamentItemBiot.filshape.fset(self, filshape)       
        self.filpoints = np.vstack(np.array([geohelper.rotate_points(points, self.angleTF[:-1], 'y', order = 1) 
                                   for points in self._FilamentItemBiot__filpoints]))        
 
    @FilamentItemBiot.Ifil.setter
    def Ifil(self, value):
        FilamentItemBiot.Ifil.fset(self, value)
        self._FilamentItemBiot__Ifil = self._FilamentItemBiot__Ifil * self.nsect

    def plot2D(self, axis = None, show = False, allsectors = False):

        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot()
            
        if not self.filshape is None:
            if allsectors:
                for p in self.filpoints:
                    axis.plot(p[:,0], p[:,1], c = 'red')
            else:
                p = self.filpoints[0]
                axis.plot(p[:,0], p[:,1], c = 'red')                
        
        plt.gca().set_aspect("equal")
        
        if show:
            plt.show()
            
        return axis 

    def __repr__(self):
        new = []
        new.append("({}:".format(type(self).__name__))
        new.append(" {}".format(self.name))
        new.append(" {}".format(self.Itot))
        new.append(" {}".format(self.r.shape))
        new.append(" {}".format(self.Idl.shape))
        new.append(" {}".format(len(self.filshape)))        
        new.append(")")
        return ", ".join(new)  

class Coil(Component):
    @classmethod
    def cast(cls, c: Component):
        assert isinstance(c, Component)
        c.__class__ = cls  # now mymethod() is available
        if not hasattr(c, 'filaments'):
            c.filaments = None
        assert isinstance(c, Coil)
        return c    

class PFCoil(Coil):
    @classmethod
    def cast(cls, c: Component):
        assert isinstance(c, Component)
        if isinstance(c, Coil):
            c.__class__ = cls
        elif isinstance(c, Component):
            Coil.cast(c)
            c.__class__ = cls            
        assert isinstance(c, PFCoil)
        return c

class TFCoil(Coil):
    @classmethod
    def cast(cls, c: Component):
        assert isinstance(c, Component)
        if isinstance(c, Coil):
            c.__class__ = cls
        elif isinstance(c, Component):
            Coil.cast(c)
            c.__class__ = cls            
        assert isinstance(c, TFCoil)
        return c

class Plasma(Coil):
    @classmethod
    def cast(cls, c: Component):
        assert isinstance(c, Component)
        if isinstance(c, Coil):
            c.__class__ = cls
        elif isinstance(c, Component):
            Coil.cast(c)
            c.__class__ = cls            
        assert isinstance(c, Plasma)
        
        # plasma pressure and pprime as function of normalized Psi
        c._p = None
        c.__pprime = None
        # f function of normalized Psi
        c._f = None
        c._ffprime = None
        # plasma temperature as function of normalized Psi
        c._temperature = None
        # plasma current density
        c._J = None
        return c

    @property    
    def J(self):
        return self._J
    
    @J.setter
    def J(self, function):
        self._J = function

    @property    
    def p(self):
        return self._pressure
    
    @p.setter
    def p(self, function):
        #pressure = scipy.interpolate.UnivariateSpline(data[:, 0], data[:, 1], ext = 0)
        self._pprime = function.derivate()
        self._p = function

    @property    
    def pprime(self):
        return self._pprime
    
    @pprime.setter
    def pprime(self, function, integration_points=20):
        print("Plasma pprime")
        psinorm = numpy.linspace(0., 1., integration_points)
        data = numpy.zeros(len(psinorm))
        for i in range(len(data)):
            data[i] = function.integrate(psinorm[i], 1.0)
        self._p = scipy.interpolate.UnivariateSpline(psinorm, data, ext = 0)    
        self._pprime = function

    @property    
    def f(self):
        return self._f
    
    @f.setter
    def f(self, function):
        self._ffprime = function * function.derivate()
        self._f = function

    @property    
    def ffprime(self):
        return self._ffprime
    
    @ffprime.setter
    def ffprime(self, function, integration_points=20):
        print("Plasma ffprime")
        psinorm = numpy.linspace(0., 1., integration_points)
        data = numpy.zeros(len(psinorm))
        for i in range(len(data)):
            data[i] = math.sqrt(0.5*function.integrate(psinorm[i], 1.0))
        self._f = scipy.interpolate.UnivariateSpline(psinorm, data, ext = 0)
        self._ffprime = function
    
    @property    
    def temperature(self):
        return self._temperature
    
    @temperature.setter
    def temperature(self, function):
        #temperature = scipy.interpolate.UnivariateSpline(data[:, 0], data[:, 1], ext = 0)
        self._temperature = function
        
    def updateFilaments(self, V = None, p = 3):
        if V is None:
            meshfile = "plasma.msh"
            meshdir = os.getcwd()
            mesh_dim = 2
            
            mesh = mirapy.core.Mesh("plasma")
            mesh.meshfile = os.path.join(meshdir, meshfile)
            mesh(self)
            
            # Run the conversion
            mirapy.msh2xdmf.msh2xdmf(meshfile, dim= mesh_dim, directory = meshdir)
            
            # Run the import
            prefix, _ = os.path.splitext(meshfile)
            
            mesh, boundaries, subdomains, labels = mirapy.msh2xdmf.import_mesh_from_xdmf(
                prefix = prefix,
                dim = mesh_dim,
                directory = meshdir,
                subdomains = True,
                )
            
            V = dolfin.FunctionSpace(self.mesh, 'CG', p)
        else:
            mesh = V.mesh()
                
        pfil = numpy.array([dolfin.Cell(mesh,i).midpoint().array() 
                                for i in range(mesh.num_cells())])
        
        dx = dolfin.Measure('dx', domain=mesh) #, subdomain_data = subdomains)
        J = self.J_to_dolfinFunction(V)
        cfil = dolfin.assemble(dolfin.TestFunction(
            dolfin.FunctionSpace(mesh, "DG", 0))*J*dx)[:]
        
        self.filaments = mirapy.core.FilamentItemGreen(name = "PlasmaFil", 
                                Rc = pfil[:,0], Zc = pfil[:,1], Ifil = cfil, 
                                fixed_current = True)

    def calculatePlasmaParameters(self, mesh = None):

        if (not hasattr(self.shape, 'physicalGroups')) or self.shape.physicalGroups is None:
            self.shape.physicalGroups = {1: "external", 2: "plasma"}
        else:
            if not 1 in self.shape.physicalGroups:
                self.shape.physicalGroups[1] = "external"
    
            if not 2 in self.shape.physicalGroups:
                self.shape.physicalGroups[2] = "plasma"
        
        if mesh is None:
            meshfile = "plasma.msh"
            meshdir = os.getcwd()
            mesh_dim = 2
            
            mesh = mirapy.core.Mesh("plasma")
            mesh.meshfile = os.path.join(meshdir, meshfile)
            mesh(self)
            
            # Run the conversion
            mirapy.msh2xdmf.msh2xdmf(meshfile, dim= mesh_dim, directory = meshdir)
            
            # Run the import
            prefix, _ = os.path.splitext(meshfile)
            
            mesh, boundaries, subdomains, labels = mirapy.msh2xdmf.import_mesh_from_xdmf(
                prefix = prefix,
                dim = mesh_dim,
                directory = meshdir,
                subdomains = True,
                )
                
        # Define r and z
        r = dolfin.Expression('x[0]', degree = 2)
        
        # Define subdomain integration measure
        dx = dolfin.Measure('dx', domain = mesh)
        ds = dolfin.Measure('ds', domain = mesh)

        # Calculate plasma geometrical parameters
        self.lp = dolfin.assemble(dolfin.Constant(1)*ds())
        print("Plasma poloidal length calculated by dolfin: {}".format(self.lp))
        plasma_length_Freecad = self.shape.getSingleWire().Length
        print("Plasma poloidal length calculated by FreeCAD: {}".format(plasma_length_Freecad))
            
        self.Ap = dolfin.assemble(dolfin.Constant(1)*dx())
        print("Plasma area calculated by dolfin: {}".format(self.Ap))
        
        plasma_area_Freecad = Part.Face(self.shape.getSingleWire()).Area
        print("Plasma area calculated by FreeCAD: {}".format(plasma_area_Freecad))

        self.Sp = 2*dolfin.pi*dolfin.assemble(r*ds())
        print("Plasma surface calculated by dolfin: {}".format(self.Sp))            
        
        self.Vp = 2*dolfin.pi*dolfin.assemble(r*dx())
        print("Plasma volume calculated by dolfin: {}".format(self.Vp)) 

    def J_to_dolfinFunction(self,J, V):
        f = dolfin.Function(V)
        p = V.ufl_element().degree()
        mesh = V.mesh()
        points  = mesh.coordinates()
        # psi = u.compute_vertex_values()
        # psi = psi[:,numpy.newaxis]
        # x = numpy.concatenate((points,psi), 1)
        # print(points)
        data = numpy.array([J(point) for point in points])
        # print("data = {}".format(data))
        
        if p > 1:
            # generate a 1-degree function space
            V1 = dolfin.FunctionSpace(mesh,'CG',1)
            f1 = dolfin.Function(V1)
            d2v = dolfin.dof_to_vertex_map(V1)        
            new_data = [data[d2v[i]] for i in range(mesh.num_vertices())]
            f1.vector().set_local(new_data)
            f = dolfin.interpolate(f1,V)
        else:
            d2v = dolfin.dof_to_vertex_map(V)        
            new_data = [data[d2v[i]] for i in range(mesh.num_vertices())]
            f.vector().set_local(new_data)      
        return f

    def dolfinUpdate(self, V, u):
        pass


class PlasmaSolovev(Plasma):
    @classmethod
    def cast(cls, c: Component, A1, A2):
        assert isinstance(c, Component)
        if isinstance(c, Coil) or isinstance(c, Component):
            Plasma.cast(c)
            c.__class__ = cls
        elif isinstance(c, Plasma):
            c.__class__ = cls
        assert isinstance(c, PlasmaSolovev)
        
        c.A1 = A1
        c.A2 = A2
        c.psib = 0
        c.psiax = 100
        
        cls.pprime_function(c)
        cls.ffprime_function(c)
        cls.J_function(c)
        return c

    @classmethod
    def pprime_function(self, c):
        def myfunc(x):
            return -c.A1/const.MU0
        c.pprime = myfunc

    @classmethod
    def ffprime_function(self, c):
        def myfunc(x):
            return c.A2
        c.ffprime = myfunc
    
    @classmethod
    def J_function(self, c):
        def myfunc(x):
            # if len(x) == 1:
            #     x = numpy.array([x])
            # r = x[:,0]
            # z = x[:,1]
            # psi = x[:,2]
            r = x[0]
            return -1/const.MU0*(-const.MU0*r*c.pprime(c.psi(x)) - 1/r*c.ffprime(c.psi(x)))
        c.J = myfunc

    @Plasma.pprime.setter
    def pprime(self, function, integration_points=20):
        print("PlasmaSolovev pprime")
        psinorm = numpy.linspace(0., 1., integration_points)
        data = numpy.zeros(len(psinorm))
        for i in range(len(data)):
            data[i] = function(psinorm[i])*(psinorm[i] - 1.0)
        self._p = scipy.interpolate.UnivariateSpline(psinorm, data, ext = 0)    
        self._pprime = function

    @Plasma.ffprime.setter
    def ffprime(self, function, integration_points=20):
        print("PlasmaSolovev ffprime")
        psinorm = numpy.linspace(0., 1., integration_points)
        data = numpy.zeros(len(psinorm))
        for i in range(len(data)):
            data[i] = math.sqrt(0.5 * function(psinorm[i]) * (psinorm[i]**2 - 1.0))
        self._f = scipy.interpolate.UnivariateSpline(psinorm, data, ext = 0)
        self._ffprime = function
    

class PlasmaFreeGS(Plasma):
    @classmethod
    def cast(cls, c: Component, Itot, J0 = 1, psib = 0, psiax = 1, 
             k = 0, a1 = 2, a2 = 1, R0 = 1, b0 = 0.5):
        assert isinstance(c, Component)
        if isinstance(c, Coil) or isinstance(c, Component):
            Plasma.cast(c)
            c.__class__ = cls
        elif isinstance(c, Plasma):
            c.__class__ = cls
        assert isinstance(c, PlasmaFreeGS)
        
        c.Itot = Itot
        c.J0 = J0
        c.psib = psib
        c.psiax = psiax
        c.k = k
        c.a1 = a1
        c.a2 = a2
        c.R0 = R0
        c.b0 = b0
        
        cls.pprime_function(c)
        cls.ffprime_function(c)
        cls.J_function(c)
        
        return c

    @classmethod
    def pprime_function(self, c):
        def myfunc(x):
            k = c.k
            a1 = c.a1
            a2 = c.a2
            R0 = c.R0
            b0 = c.b0
            return k*(b0/R0)*(1-numpy.clip(x, 0., 1.)**a1)**a2
        c.pprime = myfunc

    @classmethod
    def ffprime_function(self, c):
        def myfunc(x):
            k = c.k
            a1 = c.a1
            a2 = c.a2
            R0 = c.R0
            b0 = c.b0
            return k*((1-b0)*R0)*(1-numpy.clip(x, 0., 1.)**a1)**a2 
        c.ffprime = myfunc
    
    @classmethod
    def J_function(self, c):
        def myfunc(x):
            # if len(x.shape) == 1:
            #     x = numpy.array([x])
            # r = x[:,0]
            # z = x[:,1]
            # psi = x[:,2]
            r = x[0]
            # print(x)
            psinorm = (c.psi(x) - c.psiax)/(c.psib - c.psiax)
            return c.J0 + (r*c.pprime(psinorm) + 1/r*c.ffprime(psinorm))
        c.J = myfunc

    @Plasma.pprime.setter
    def pprime(self, function, integration_points=20):
        print("PlasmaFreeGS pprime")
        psinorm = numpy.linspace(0., 1., integration_points)
        data = numpy.zeros(len(psinorm))
        for i in range(len(data)):
            data[i] = function(psinorm[i])*(psinorm[i] - 1.0)
        self._p = scipy.interpolate.UnivariateSpline(psinorm, data, ext = 0)    
        self._pprime = function

    @Plasma.ffprime.setter
    def ffprime(self, function, integration_points=20):
        print("PlasmaFreeGS ffprime")
        psinorm = numpy.linspace(0., 1., integration_points)
        data = numpy.zeros(len(psinorm))
        for i in range(len(data)):
            data[i] = math.sqrt(0.5 * function(psinorm[i]) * (psinorm[i]**2 - 1.0))
        self._f = scipy.interpolate.UnivariateSpline(psinorm, data, ext = 0)
        self._ffprime = function

    def dolfinUpdate(self, V):
        print("PlasmaFreeGS update")
        mesh = V.mesh()
        points = mesh.coordinates()
        psi = self.psi.compute_vertex_values()
        self.psiax = psi.max()
        indmax = numpy.argmax(psi)
        Pax = points[indmax]
        self.R0 = Pax[0]
        self.J0 = 0
        self.k = 1
        
        # Define subdomain integration measure
        dx = dolfin.Measure('dx', domain = mesh)

        # Calculate plasma geometrical parameters
        J = self.J_to_dolfinFunction(V)
        self.k = self.Itot/dolfin.assemble(J*dx())
        