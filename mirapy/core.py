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

from __future__ import annotations

# import mathematical lib
import numpy
import scipy
import math

# import tree lib
import anytree
from anytree import NodeMixin, RenderTree

# import mirapy modules
from . import geo
from . import plotting
from . import emag
from . import meshing
from . import msh2xdmf
from . import const
# from . import plotting

# import freecad lib
import freecad
import Part
from FreeCAD import Base

# import fenics/dolfin lib
import dolfin

# import matplotlib
import matplotlib.pyplot as plt

# import input/output lib
import os

from typing import Union

import logging
module_logger = logging.getLogger(__name__)


class Component(NodeMixin):
    """A component class. It is based on a tree structure."""

    # Dictionary for __repr__ function
    repr_dict = ['name', '_shape', 'mesh_dim']

    def __init__(
                self,
                name: str = "",
                shape: Union[geo.Shape, geo.Shape2D] = None,
                parent: Component = None,
                children: Component = None,
                *argsv, **kwargs
            ):
        """
        Constructor for the Component class.
        """
        self._shape_inputs = {}
        self.name = name
        # geometrical component
        if shape:
            self.shape = shape
        # parent
        self.parent = parent
        # children
        if children:
            self.children = children

        # # Not used anymore
        # # kwargs is used in case other properties must to be added
        # if kwargs:
        #     for k, v in kwargs.items():
        #         self.__dict__[k] = v

        # plot options
        self.kwargs_plot2D_options = {'ndiscr': 100,
                                      'plane': 'xy',
                                      'poptions': {},
                                      'woptions': {'color': 'black',
                                                   'linewidth': '2'},
                                      'foptions': {'color': 'red'}}

    @classmethod
    def _check_shape(cls, shape):
        """Check on shape type
        Return True if the shape is a geo.Shape or geo.Shape2D instance.
        False otherwise."""
        if shape is None:
            return True
        elif isinstance(shape, (geo.Shape, geo.Shape2D)):
            return True
        else:
            return False
            # raise ValueError('Only geo.Shape and geo.Shape2D'
            #                  ' are accepted as shape.')

    @property
    def shape(self):
        """Component's geometrical shape.
        It can also be a function that return a valid shape according
        to the _check_shape function"""
        if hasattr(self, '_shape'):
            return self._shape
        return None

    @shape.setter
    def shape(self, value):
        # if value is None
        if value is None:
            self._shape = None
            self._shape_function = None
            self._shape_inputs = None
        # if value is a geo shape
        elif self.__class__._check_shape(value):
            self._shape = value
            self._shape_function = None
            self._shape_inputs = None
        # if value is a tuple (function, *args) that creates the filaments
        elif type(value) is dict:

            function = value['function']

            if 'args' in value.keys():
                args = value['args']
            else:
                args = []

            if 'kwargs' in value.keys():
                kwargs = value['kwargs']
            else:
                kwargs = []

            if callable(function):
                shape = function(*args, **kwargs)
                if self.__class__._check_shape(shape):
                    self._shape = shape
                else:
                    raise ValueError("Shape function must return a"
                                     " Shape or Shape2D object.")
                self._shape_function = function
                self._shape_inputs = {'args': args, 'kwargs': kwargs}
            else:
                raise ValueError("Shape function is not callable.")
        else:
            raise ValueError("Shape value is not valid.")

    def recalculate_shape(self, *args, **kwargs):
        """
        Recalculate the component's shape in case a shape_function
        has been defined.

        Parameters
        ----------
        
        value:
            arguments for the shape function.

        Returns
        -------
        
        None. The component's shape is internally modified

        """
        if hasattr(self, "_shape_function"):
            value = {'function': self._shape_function,
                     'args': args,
                     'kwargs': kwargs}
            self.shape(value)
            return True
        else:
            print("No shape function has been defined")
            return False

    def search_shape(self, name: str):
        """
        This function search shapes with the specified name. \
        The search is made for the Component and the respective children, \
        recursively.

        Parameters
        ----------
        
        name: str
            name of the shape.

        Returns
        -------
        
        objs: (list[Shape])
            list of any shape with the searched name.

        """
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

    def get_component(self, name: str, first: bool = True):
        """
        Find the components with the specfied name.

        .. note::
            this function is just a wrapper of the \
            anytree.search.findall_by_attr function.

        Parameters
        ----------
        
        name: str
            component's name to search.
        first: (bool, optional)
            if True, only the first element is returned.

        Returns
        -------
        
        Component:
            the first component of the search.

        """
        c = anytree.search.findall_by_attr(self, name)
        if not c:
            return c
        if first:
            return c[0]
        return c

    def set_plot2D_options(self, **kwargs):
        """Set plotting options for the component"""
        self.kwargs_plot2D_options = kwargs

    def set_ndiscr(self, value):
        """Set the number of discretizetion points used for the plotting.

        Parameters
        ----------
        
        value: int
            number of discretization points
        
        """

        self.kwargs_plot2D_options['ndiscr'] = value

    def set_plane(self, plane):
        """Set the visualization plane for the plotting. The component shape
        is projected in the specified plane.

        .. note::
            Check plotting.plot2D for available plane options

        Parameters
        ----------
        
        plane: (Union[str, Base.Placement]):
            
        """
        self.kwargs_plot2D_options['plane'] = plane

    def set_poptions(self, **kwargs):
        """Set plotting options for points"""
        self.kwargs_plot2D_options['poptions'] = kwargs

    def set_woptions(self, **kwargs):
        """Set plotting options for wires"""
        self.kwargs_plot2D_options['woptions'] = kwargs

    def set_foptions(self, **kwargs):
        """Set plotting options for face"""
        self.kwargs_plot2D_options['foptions'] = kwargs

    def set_wcolor(self, color):
        """Set wires color"""
        self.kwargs_plot2D_options['woptions'] = {
            **self.kwargs_plot2D_options['woptions'],
            **{'color': color}
            }

    def set_fcolor(self, color):
        """Set filling color for face"""
        self.kwargs_plot2D_options['foptions'] = {
            **self.kwargs_plot2D_options['foptions'],
            **{'color': color}
            }

    def plot2D(
            self,
            axis=None,
            show: bool = False,
            ndiscr: int = 100,
            # plane: Union[str, Base.Placement] = 'xy',
            # woptions: dict = {'color': 'black', 'linewidth': '2'},
            # foptions: dict = {'color': 'red'},
    ):
        """2D plot of the Component. If the component has children, this\
            function plots only the leaves attached to the component.

        .. note::
            Set object's properties in kwargs_plot2D_options to customize
            the object visualization

        Parameters
        ----------
        
        axis: matplotlib.pyplot.axis, optional
            matplotlib.pyplot.axis. (Default = None)
        show: bool, optional
            if True, plot is shown. (Default = False)
        ndiscr: int, optional
            number of points for the discretization. (Default = 100)

        Returns
        -------
        axis: matplotlib.pyplot.axis
            plot axis.

        """

        leaves = self.leaves
        for l in leaves:
            if l.shape:
                kwargs = l.kwargs_plot2D_options
                axis, _ = plotting.plot2D(l.shape,
                                          axis=axis,
                                          show=False,
                                          **kwargs
                                          )
        if show:
            plt.show()
        return axis

    def __repr__(self):
        new = []
        new.append("({}:".format(type(self).__name__))
        [new.append("{}: {}".format(k, v))
         for k, v in self.__dict__.items() if k in self.repr_dict]
        new.append(")")
        return ", ".join(new)


class Coil(Component):
    """Class to define a coil"""

    repr_dict = Component.repr_dict + ['filaments', 'Itot']

    def __init__(self,
                 name: str = "",
                 shape: Union[geo.Shape, geo.Shape2D] = None,
                 parent: Component = None,
                 children: Component = None,
                 filaments: Union[emag.FilamentItemBiot,
                                  emag.FilamentItemGreen,
                                  dict] = None,
                 *argsv,
                 **kwargs
                 ):

        super().__init__(name, shape, parent, children, *argsv, **kwargs)
        self._filaments = None
        self._filaments_function = None
        self._filaments_inputs = None

        self.filaments = filaments

    @classmethod
    def _check_filaments(cls, filaments):
        """Check on filaments' type"""
        if filaments is None:
            return True
        elif isinstance(filaments, (emag.FilamentItemBiot,
                                    emag.FilamentItemGreen)):
            return True
        else:
            return False
            # raise ValueError('Only FilamentItemBiot and FilamentItemGreen'
            #                  ' are accepted as filaments.')

    @property
    def filaments(self):
        """Component's geometrical shape"""
        if hasattr(self, '_filaments'):
            return self._filaments
        return None

    @filaments.setter
    def filaments(self, value):
        # if value is None
        if value is None:
            self._filaments = None
            self._filaments_function = None
            self._filaments_inputs = None
        # if value is a filament
        elif self.__class__._check_filaments(value):
            self._filaments = value
            self._filaments_function = None
            self._filaments_inputs = None
        # if value is a tuple (function, *args) that creates the filaments
        elif type(value) is dict:

            function = value['function']

            if 'args' in value.keys():
                args = value['args']
            else:
                args = []

            if 'kwargs' in value.keys():
                kwargs = value['kwargs']
            else:
                kwargs = []

            if callable(function):
                filaments = function(*args, **kwargs)
                if self.__class__._check_filaments(filaments):
                    self._filaments = filaments
                else:
                    raise ValueError("Current function must return a"
                        " FilamentItemBiot or FilamentItemGreen object.")
                self._filaments_function = function
                self._filaments_inputs = {'args': args, 'kwargs': kwargs}
            else:
                raise ValueError("Current function is not callable.")
        else:
            raise ValueError("Current value is not valid.")

    def recalculate_filaments(self, *args, **kwargs):
        """
        Recalculate the component's current shape in case a current_function
        has been defined.

        Parameters
        ----------
        value:
            arguments for the current function.

        Returns
        -------
        
        bool:
            True if the component's shape is internally modified.

        """

        if hasattr(self, "_filaments_function"):
            value = {'function': self._filaments_function,
                     'args': args,
                     'kwargs': kwargs}
            self.filaments = value
            return True
        else:
            print("No current function has been defined")
            return False

    @classmethod
    def cast(cls, c: Component):
        """Static method to convert a component in a coil.
        Filaments property is set to None."""
        assert isinstance(c, Component)
        c.__class__ = cls  # now mymethod() is available
        c.current = None
        assert isinstance(c, Coil)
        return c

    @property
    def Itot(self):
        if self.filaments is None:
            return None
        return self.filaments.Itot

    @Itot.setter
    def Itot(self, value):
        if self.filaments is None:
            pass
        self.filaments.Itot = value

    def set_fixed_current(self, value: bool):
        if self.filaments is not None:
            self.filaments.fixed_current = value

    def __repr__(self):
        new = []
        new.append("({}:".format(type(self).__name__))
        [new.append("{}: {}".format(k, v))
         for k, v in self.__dict__.items() if k in self.repr_dict]
        new.append(")")
        return ", ".join(new)


class PFCoil(Coil):

    @classmethod
    def _check_filaments(cls, filaments):
        if filaments is None:
            return True
        return isinstance(filaments, emag.FilamentItemGreen)

    @classmethod
    def cast(cls, c: Component):
        assert isinstance(c, Component)
        if isinstance(c, Coil):
            if PFCoil._check_filaments(c.filaments):
                c.__class__ = cls
            else:
                raise ValueError("Wrong filaments type")
        elif isinstance(c, Component):
            Coil.cast(c)
            c.__class__ = cls
        assert isinstance(c, PFCoil)
        return c


class TFCoil(Coil):

    @classmethod
    def _check_filaments(cls, filaments):
        if filaments is None:
            return True
        return isinstance(filaments, emag.FilamentItemBiot)

    @classmethod
    def cast(cls, c: Component):
        assert isinstance(c, Component)
        if isinstance(c, Coil):
            if TFCoil._check_filaments(c.filaments):
                c.__class__ = cls
            else:
                raise ValueError("Wrong filaments type")
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
        c.set_pprime(None)
        # f function of normalized Psi
        c._f = None
        c.set_ffprime(None)
        # plasma temperature as function of normalized Psi
        c._temperature = None
        # plasma current density
        c._J = None
        # plasma R-axis
        c.Rax = None
        # plasma Btor at R-axis
        c.Bt = None
        return c

    @property    
    def J(self):
        return self._J
    
    @J.setter
    def J(self, function):
        self._J = function

    @property
    def fvac(self):
        return self.Rax * self.Bt

    @property    
    def p(self):
        return self._p
    
    @p.setter
    def p(self, function):
        #pressure = scipy.interpolate.UnivariateSpline(data[:, 0], data[:, 1], ext = 0)
        self._pprime = function.derivate()
        self._p = function

    @property    
    def pprime(self):
        return self._pprime
    
    def set_pprime(self, function, integration_points=20):
        if callable(function):
            # print("Plasma pprime")
            self._pprime = function
            psinorm = numpy.linspace(0., 1., integration_points)
            data = numpy.zeros(len(psinorm))
            for i in range(len(data)):
                data[i] = scipy.integrate.quad(function, psinorm[i], 1.0)[0]
                data[i] *= (self.psiax - self.psib)
            self._p = scipy.interpolate.UnivariateSpline(psinorm, data, ext = 0)
        else:
            print("{} not callable".format(function))

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
    
    def set_ffprime(self, function, integration_points=20):
        if callable(function):
            # print("Plasma ffprime")        
            psinorm = numpy.linspace(0., 1., integration_points)
            data = numpy.zeros(len(psinorm))
            for i in range(len(data)):
                data[i] = math.sqrt(2*scipy.integrate.quad(function,
                                                           psinorm[i],
                                                           1.0
                                                           )[0]
                                    )
                data[i] *= (self.psiax - self.psib)
                data[i] += self.fvac
            self._f = scipy.interpolate.UnivariateSpline(psinorm, data, ext=0)
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
            
            mesh = meshing.Mesh("plasma")
            mesh.meshfile = os.path.join(meshdir, meshfile)
            mesh(self)
            
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
            
            V = dolfin.FunctionSpace(self.mesh, 'CG', p)
        else:
            mesh = V.mesh()
                
        pfil = numpy.array([dolfin.Cell(mesh,i).midpoint().array() 
                                for i in range(mesh.num_cells())])
        
        dx = dolfin.Measure('dx', domain=mesh) #, subdomain_data = subdomains)
        J = self.J_to_dolfinFunction(V)
        cfil = dolfin.assemble(dolfin.TestFunction(
            dolfin.FunctionSpace(mesh, "DG", 0))*J*dx)[:]
        
        self.filaments = emag.FilamentItemGreen(name = "PlasmaFil", 
                                Rc = pfil[:,0], Zc = pfil[:,2], Ifil = cfil, 
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
            
            mesh = meshing.Mesh("plasma")
            mesh.meshfile = os.path.join(meshdir, meshfile)
            mesh(self)
            
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
                
        # Define r and z
        r = dolfin.Expression('x[0]', degree = 2)
        
        # Define subdomain integration measure
        dx = dolfin.Measure('dx', domain = mesh)
        ds = dolfin.Measure('ds', domain = mesh)

        # Calculate plasma geometrical parameters
        self.lp = dolfin.assemble(dolfin.Constant(1)*ds())
        print("Plasma poloidal length calculated by dolfin: {}".format(self.lp))
        plasma_length_Freecad = self.shape.Length
        print("Plasma poloidal length calculated by FreeCAD: {}".format(plasma_length_Freecad))
            
        self.Ap = dolfin.assemble(dolfin.Constant(1)*dx())
        print("Plasma area calculated by dolfin: {}".format(self.Ap))
        
        plasma_area_Freecad = self.shape.Area
        print("Plasma area calculated by FreeCAD: {}".format(plasma_area_Freecad))

        self.Sp = 2*dolfin.pi*dolfin.assemble(r*ds())
        print("Plasma surface calculated by dolfin: {}".format(self.Sp))            

        filpoints_ = self.shape.Wires[0].discretize(1000)
        filpoints_ = numpy.array([[p[0], p[1], p[2]] for p in filpoints_])
        r_ = (filpoints_[1:] + filpoints_[0:-1])/2.
        dl_ = (filpoints_[1:] - filpoints_[0:-1])
        normdl_ = numpy.array([numpy.linalg.norm(l) for l in dl_])

        plasma_surf_Freecad = 2. * numpy.pi * sum(r_[:,0]*normdl_)
        print("Plasma lateral surface calculated by FreeCAD: {}".format(plasma_surf_Freecad))    
 
        self.Vp = 2*dolfin.pi*dolfin.assemble(r*dx())
        print("Plasma volume calculated by dolfin: {}".format(self.Vp)) 

        plasma_volume_Freecad = self.shape.Area * (
            2. * numpy.pi * self.shape.face.CenterOfMass[0])
        print("Plasma volume calculated by FreeCAD: {}".format(plasma_volume_Freecad))

    def J_to_dolfinFunction(self, V):
        f = dolfin.Function(V)
        p = V.ufl_element().degree()
        mesh = V.mesh()
        points  = mesh.coordinates()
        # psi = u.compute_vertex_values()
        # psi = psi[:,numpy.newaxis]
        # x = numpy.concatenate((points,psi), 1)
        # print(points)
        data = numpy.array([self.J(point) for point in points])
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
        
        c.pprime_function()
        c.ffprime_function()
        c.J_function()
        return c

    def pprime_function(self):
        def myfunc(x):
            return -self.A1/const.MU0
        self.set_pprime(myfunc)

    def ffprime_function(self):
        def myfunc(x):
            return self.A2
        self.set_ffprime(myfunc)
    
    def J_function(self):
        def myfunc(x):
            r = x[0]
            return -1/const.MU0*(-const.MU0*r*self.pprime(self.psi(x))
                                 - 1/r*self.ffprime(self.psi(x)))
        self.J = myfunc

class PlasmaFreeGS(Plasma):
    @classmethod
    def cast(cls, c: Component, Itot, J0 = 1., psib = 0., psiax = 1., 
              k = 0., a1 = 2., a2 = 1., Rax = 1., b0 = 0.5, Bt = 1.):
        assert isinstance(c, Component)
        if isinstance(c, Coil) or isinstance(c, Component):
            Plasma.cast(c)
            c.__class__ = cls
        elif isinstance(c, Plasma):
            c.__class__ = cls
        assert isinstance(c, PlasmaFreeGS)
        
        c.Ip = Itot
        c.J0 = J0
        c.psib = psib
        c.psiax = psiax
        c.k = k
        c.a1 = a1
        c.a2 = a2
        c.Rax = Rax
        c.b0 = b0
        c.Bt = Bt
        
        c.pprime_function()
        c.ffprime_function()
        c.J_function()
        
        return c

    def pprime_function(self, integration_points=20):
        def myfunc(x):
            k = self.k
            a1 = self.a1
            a2 = self.a2
            Rax = self.Rax
            b0 = self.b0
            return k*(b0/Rax)*(1-numpy.clip(x, 0., 1.)**a1)**a2
        self.set_pprime(myfunc)

    def ffprime_function(self, integration_points=20):
        def myfunc(x):
            k = self.k
            a1 = self.a1
            a2 = self.a2
            Rax = self.Rax
            b0 = self.b0
            return k*((1-b0)*Rax)*(1-numpy.clip(x, 0., 1.)**a1)**a2 
        self.set_ffprime(myfunc)
    
    def J_function(self):
        def myfunc(x):
            r = x[0]
            psinorm = (self.psi(x) - self.psiax)/(self.psib - self.psiax)
            return self.J0 + (r*self.pprime(psinorm) \
                              + 1/r*self.ffprime(psinorm))
        self.J = myfunc

    def dolfinUpdate(self, V):
        print("PlasmaFreeGS update")
        mesh = V.mesh()
        points = mesh.coordinates()
        psi = self.psi.compute_vertex_values()
        
        # plotting.plot_scalar_field(points[:,0], points[:,1], psi)
        
        self.psiax = psi.max()
        indmax = numpy.argmax(psi)
        Pax = points[indmax]
        self.Rax = Pax[0]
        self.J0 = 0
        self.k = 1

        # Update pprime and ffprime for J calculation
        self.pprime_function()
        self.ffprime_function()

        # Define subdomain integration measure
        dx = dolfin.Measure('dx', domain = mesh)

        # Calculate plasma geometrical parameters
        J = self.J_to_dolfinFunction(V)
        self.k = self.Ip/dolfin.assemble(J*dx())
        
        # Update pprime and ffprime since self.k is changed
        self.pprime_function()
        self.ffprime_function()        
        