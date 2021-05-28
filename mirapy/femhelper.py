#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""

import dolfin
import numpy


# From version 2019.1, use subclass UserExpression:
class dolfin_expr(dolfin.UserExpression):
    def __init__(self, function, **kwargs):
        # Call superclass constructor with keyword arguments to properly
        # set up the instance:
        super().__init__(**kwargs)
        # Perform custom setup tasks for the subclass after that:
        self._function = function

    def eval(self, values, x):
        f = self._function
        values[0] = f(x)

    def value_shape(self):
        return ()


class dolfin_psinorm(dolfin.UserExpression):
    def __init__(self, psi, psiax, psib, **kwargs):
        # Call superclass constructor with keyword arguments to properly
        # set up the instance:
        super().__init__(**kwargs)
        # Perform custom setup tasks for the subclass after that:
        self._psi = psi
        self._psiax = psiax
        self._psib = psib

    def eval(self, values, x):
        psi = self._psi
        psiax = self._psiax
        psib = self._psib
        values[0] = (psi(x) - psib)/(psiax - psib)

    def value_shape(self):
        return ()


class dolfin_gf(dolfin.UserExpression):
    def __init__(self, g, f, **kwargs):
        # Call superclass constructor with keyword arguments to properly
        # set up the instance:
        super().__init__(**kwargs)
        # Perform custom setup tasks for the subclass after that:
        self._f = f
        self._g = g

    def eval(self, values, x):
        f = self._f
        g = self._g
        values[0] = g(f(x))

    def value_shape(self):
        return ()


class dpdpsi_expr(dolfin.UserExpression):
    def __init__(self, p, psi, psiax, psib, **kwargs):
        # Call superclass constructor with keyword arguments to properly
        # set up the instance:
        super().__init__(**kwargs)
        # Perform custom setup tasks for the subclass after that:
        self._p = p
        self._psi = psi
        self._psiax = psiax
        self._psib = psib

    def eval(self, values, x):
        p = self._p
        psi = self._psi
        psiax = self._psiax
        psib = self._psib
        p1 = p.derivative()
        psi_norm = dolfin_psinorm(psi, psiax, psib)
        values[0] = p1(psi_norm(x))/(psiax - psib)

    def value_shape(self):
        return ()


class Jplasma_expr(dolfin.UserExpression):
    def __init__(self, dpdpsi, alpha=1, **kwargs):
        # Call superclass constructor with keyword arguments to properly
        # set up the instance:
        super().__init__(**kwargs)
        # Perform custom setup tasks for the subclass after that:
        self._dpdpsi = dpdpsi
        self._alpha = alpha

    def eval(self, values, x):
        dpdpsi = self._dpdpsi
        alpha = self._alpha
        values[0] = 2*dolfin.pi*x[0]*dpdpsi(x)*alpha

    def value_shape(self):
        return ()


def errornormperc(mesh, f1, f2):
    dx = dolfin.Measure('dx', domain=mesh)
    return dolfin.assemble(dolfin.dot(f1-f2, f1-f2)*dx)/dolfin.assemble(dolfin.dot(f1,f1)*dx)

def Solovev(mesh, Ip, betap):
    # Define r and z
    r = dolfin.Expression('x[0]', degree = 2)
    
    # Define subdomain integration measure
    dx = dolfin.Measure('dx', domain = mesh)
    ds = dolfin.Measure('ds', domain = mesh)

    # Calculate plasma geometrical parameters
    lp = dolfin.assemble(dolfin.Constant(1)*ds())
    print("Plasma poloidal length calculated by dolfin: {}".format(lp))
      
    Ap = dolfin.assemble(dolfin.Constant(1)*dx())
    print("Plasma area calculated by dolfin: {}".format(Ap))
    
    Sp = 2*dolfin.pi*dolfin.assemble(r*ds())
    print("Plasma surface calculated by dolfin: {}".format(Sp))            
    
    Vp = 2*dolfin.pi*dolfin.assemble(r*dx())
    print("Plasma volume calculated by dolfin: {}".format(Vp))
    
def func_to_dolfinFunction(func, V):
    f = dolfin.Function(V)
    p = V.ufl_element().degree()
    mesh = V.mesh()
    points  = mesh.coordinates()
    # psi = u.compute_vertex_values()
    # psi = psi[:,numpy.newaxis]
    # x = numpy.concatenate((points,psi), 1)
    # print(points)
    data = numpy.array([func(point) for point in points])
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

def convertFunctionToDolfinExpression(func):
    # In version 2019.1, you want to subclass UserExpression:
    class e(dolfin.UserExpression):
        def __init__(self, function, **kwargs):
            # Call superclass constructor with keyword arguments to properly
            # set up the instance:
            super().__init__(**kwargs)
            # Perform custom setup tasks for the subclass after that:
            self._function = function
            
        def eval(self, values, x):
            p = self._function
            values[0] = p(x)[0]
    
        def value_shape(self):
            return ()
        
    # NOTE: UserExpressions are always interpolated in finite element spaces during
    # assembly of forms.  You can control which space by passing a degree
    # keyword argument.  The default value, if nothing is passed, is degree=2.
    # For the given regularized delta function, it will always be an approximation,
    # since the formula is non-polynomial.  (An alternative would be to define
    # it directly in UFL, using `x = SpatialCoordinate(mesh)`.)
    eInstance = e(func, degree=1)
    
    return eInstance