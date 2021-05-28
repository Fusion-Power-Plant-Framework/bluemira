#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ivan
"""
import freecad
import Part
from FreeCAD import Base

import math
import numpy

import mirapy.algebra as algebra
import mirapy.geo as geo
import mirapy.core as core
import mirapy.machine as machine
import mirapy.plotting as plotting
import mirapy.emag as emag    

def set_comp_placement(comp, plane):
    """

    Parameters
    ----------
    comp :
        
    plane :
        

    Returns
    -------

    """
    if not isinstance(plane, Base.Placement):
        raise ValueError("plane must be a Base.Placement")
    if not isinstance(comp, core.Component):
        raise ValueError("Comp must be a core.Component")

    if comp.shape:
        comp.shape.Placement = plane
    if hasattr(comp, "filaments"):
        try:
            comp.filaments.Placement = plane
        except:
            pass
    for c in comp.children:
        set_comp_placement(c, plane)

def create_pf_cs_filaments(CSPFcomps, nx, ny, Itot):
    """

    Parameters
    ----------
    CSPFcomps :
        
    nx :
        
    ny :
        
    Itot :
        

    Returns
    -------

    """
    if not hasattr(CSPFcomps,'__len__'):
        CSPFcomps = [CSPFcomps]

    if not hasattr(nx,'__len__'):
        nx = [nx]

    if not hasattr(ny,'__len__'):
        ny = [ny]

    if not hasattr(Itot,'__len__'):
        Itot = [Itot]
    
    if len(CSPFcomps) != len(nx) != len(ny) != len(Itot):
        raise ValueError("CSPFcomps, nx, ny, and Itot must have"
                         " the same length")

    for i in range(len(CSPFcomps)):
        coil = CSPFcomps[i]
        if isinstance(coil.shape, geo.Shape):
            w = coil.shape.getSingleWire()
        elif isinstance(coil.shape, geo.Shape2D):
            w = coil.shape.getSingleWires()[0]
        else:
            raise ValueError("This should not happen.")
        coil.filaments = emag.Utils.create_rectangular_filaments_from_shape(
            w, Itot[i], nx[i], ny[i])


def create_tf_filaments(tfcoilin, offset, nsect, Itf, ndiscr = 200,
                        dcable = 0.01):
    """Fuction to add filaments to the TF coil component. Nsect current
    filaments are added to the TF component. Each filament is placed
    at the center of the TF coil at a radial distance equal to "offset".
    The tfcoil component is directly modified by the function.

    Parameters
    ----------
    tfcoilin : Part.Wire
        TFcoil internal shape.
    offset : float
        distance between the inner shape of the TF coil and the current
        filament.
    nsect : int
        number of sectors.
    Itf : float
        current in a single filament. The total current is nsect * Itf
    ndiscr :
         (Default value = 200)
    dcable :
         (Default value = 0.01)

    Returns
    -------

    
    """

    wfil = tfcoilin.makeOffset2D(offset)
    
    return emag.FilamentTFCoil(filshape = wfil, 
                               Ifil = [Itf], 
                               ndiscr = ndiscr, 
                               dcable = dcable, 
                               fixed_current = False, 
                               nsect = nsect)


class SimpleSingleNull(machine.Machine):
    """ """
    
    def __init__(self, nsect, data=None, *args, **kwargs):
        if data is None:
            data = {}
            #Reactor parameters
            data['R0']          = 8.938;            # (m)  Major radius
            data['A']           = 3.1;              # (-)   Aspect ratio
            data['nsectors']    = 16                # (-)  Numero di settori toroidali
            data['Bt']          = 4.960             # (T)  Toroidal magnetic field at R0
            
            #Plasma parameters
            data['Ip']          = 19.6e6            # (A)  Total plasma current
            data['kappaXU']     = 1.68;             # (-)   Upper plasma elongation at the X point
            data['kappaXL']     = 1.88;             # (-)   Lower plasma elongation at the X point
            data['deltaXU']     = 0.50;             # (-)   Upper plasma triangularity at the X point
            data['deltaXL']     = 0.50;             # (-)   Lower plasma triangularity at the X point
            data['psiPU']       = 0.0/180*math.pi;       # (-)   Plasma shape angle outboard upper
            data['psiMU']       = math.pi;       # (-)   Plasma shape angle inboard upper
            data['psiPL']       = 30.0/180*math.pi;      # (-)   Plasma shape angle outboard lower
            data['psiML']       = 120.0/180*math.pi;      # (-)   Plasma shape angle inboard lower
    
            #PFcoils parameters
            data['rzPF1'] = [5.4, 9.26, 1.2, 1.2]
            data['rzPF2'] = [14., 7.9, 1.4, 1.4]
            data['rzPF3'] = [17.75, 2.5, 1., 1.]
            data['rzPF4'] = [17.75, -2.5, 1., 1.]
            data['rzPF5'] = [14., -7.9, 1.4, 1.4]
            data['rzPF6'] = [7., -10.5, 2., 2.]
            data['nx_PF'] = [1, 1, 1, 1, 1, 1] #number of filament columns for each PF
            data['ny_PF'] = [1, 1, 1, 1, 1, 1] #number of filament rows for each PF
            data['I_PF'] = [12.49e6, -3.81e6, -7.67e6, -1.65e6, -9.79e6, 18.71e6]
            data['I_PF'] = [14.26e6, 6.08e6, -9.21e6, 11.72e6, -8.06e6, 21.6e6]
    
            #CScoils parameters
            data['rzCS1'] = [2.77, 7.07, 0.8, 2.98]
            data['rzCS2'] = [2.77, 4.08, 0.8, 2.98]
            data['rzCS3'] = [2.77, -0.4, 0.8, 5.95]
            data['rzCS4'] = [2.77, -4.88, 0.8, 2.98]
            data['rzCS5'] = [2.77, -7.86, 0.8, 2.98]
            data['nx_CS'] = [5, 5, 5, 5, 5] #number of filament columns for each PF
            data['ny_CS'] = [10, 10, 20, 10, 10] #number of filament rows for each PF
            data['I_CS'] = [16.11e6, 4.78e6, -3.27e6, 0.77e6, 22.01e6]
            data['I_CS'] = [29.71e6, 29.86e6, 59.72e6, 29.86e6, 29.86e6]

        tokamak = core.Component("SimpleSingleNull")
        plasma = self.create_plasma(**data)
        plasma.parent = tokamak

        pfcoils = core.Component("PFCoils", parent=tokamak)

        PFs = []
        
        for i in range(6):
            i = i+1
            pf = self.create_pfcoil(*data['rzPF'+str(i)], "PF"+str(i))
            pf.parent = pfcoils
            PFs.append(pf)

        create_pf_cs_filaments(PFs, data['nx_PF'], data['ny_PF'], data['I_PF'])

        cscoils = core.Component("CSCoils", parent=tokamak)

        CSs = []
        for i in range(5):
            i = i+1
            cs = self.create_pfcoil(*data['rzCS'+str(i)], "CS"+str(i))
            cs.parent = cscoils
            CSs.append(cs)

        create_pf_cs_filaments(CSs, data['nx_CS'], data['ny_CS'], data['I_CS'])

        super().__init__(nsect, tokamak)
    
    def create_plasma(self, R0, A, kappaXU, kappaXL, deltaXU, deltaXL,
                      psiPU, psiMU, psiPL, psiML, lcar=0.25, Ip=19.6e6,
                      Bt=4.96, *args, **kwargs):
        """

        Parameters
        ----------
        R0 :
            
        A :
            
        kappaXU :
            
        kappaXL :
            
        deltaXU :
            
        deltaXL :
            
        psiPU :
            
        psiMU :
            
        psiPL :
            
        psiML :
            
        lcar :
             (Default value = 0.25)
        Ip :
             (Default value = 19.6e6)
        Bt :
             (Default value = 4.96)
        *args :
            
        **kwargs :
            

        Returns
        -------

        """

        a = R0/A
        XU = Base.Vector(R0-a*deltaXU,a*kappaXU,0)
        XL = Base.Vector(R0-a*deltaXL,-a*kappaXL,0)
        Po = Base.Vector(R0+a,0,0)
        Pi = Base.Vector(R0-a,0,0)
    
        C1 = algebra.bezier2pointstangent(XL, psiPL, Po, math.pi/2)
        C2 = algebra.arcofellipse2pointstangent(Po,math.pi/2,XU,psiPU) 
        C3 = algebra.arcofellipse2pointstangent(XU,psiMU,Pi,math.pi/2)
        C4 = algebra.bezier2pointstangent(Pi,3*math.pi/2.,XL,psiML + math.pi)
    
        wire = Part.Wire(Part.Shape([C1,C2,C3,C4]).Edges)
        shape = geo.Shape(wire, "Plasma")
    
        tuples = {"XU": (XU, None),
                  "XL": (XL, None),
                  "Po": (Po, None),
                  "Pi": (Pi, None),
                  "XUp": (XU, numpy.rad2deg(psiPU)),
                  "XUm": (XU, numpy.rad2deg(psiMU)),
                  "XLp": (XL, numpy.rad2deg(psiPL)),
                  "XLm": (XL, numpy.rad2deg(psiML)),
                  "Pop": (Po, 90.),
                  "Pom": (Po, 270.),
                  "Pip": (Pi, 90.),
                  "Pim": (Pi, 270.),              
                  }
    
        geoConstrDict = {}
        
        for k,v in tuples.items():
            geoConstrDict[k] = geo.geoConstraint(v[0], angle=v[1],
                                                     lscale=1,label=k)
        
        plasma = core.Component("Plasma", geo.Shape2D(shape))
        plasma.shape.allshapes[0].lcar = lcar
        plasma.geoConstr = geoConstrDict
        core.Plasma.cast(plasma)
        plasma.calculatePlasmaParameters()
        J0 = Ip/plasma.Ap
        core.PlasmaFreeGS.cast(plasma, Ip, J0 = J0)
        
        return plasma
    
    def create_pfcoil(self,X,Y,w,h,label="",lcar=0.1):
        """

        Parameters
        ----------
        X :
            
        Y :
            
        w :
            
        h :
            
        label :
             (Default value = "")
        lcar :
             (Default value = 0.1)

        Returns
        -------

        """
        
        P1 = Base.Vector(X-w/2.,Y-h/2.,0.)
        P2 = Base.Vector(X+w/2.,Y-h/2.,0.)
        P3 = Base.Vector(X+w/2.,Y+h/2.,0.)
        P4 = Base.Vector(X-w/2.,Y+h/2.,0.)
        
        line1 = Part.LineSegment(P1,P2)
        line2 = Part.LineSegment(P2,P3)
        line3 = Part.LineSegment(P3,P4)
        line4 = Part.LineSegment(P4,P1)
        
        shape = geo.Shape(Part.Wire(Part.Shape([line1,line2,line3,line4]).Edges),
                          lcar=lcar)
        
        points = {"P1": P1,
                  "P2": P2,
                  "P3": P3,
                  "P4": P4}
        
        geoConstrDict = {}
        
        for k,v in points.items():
            geoConstrDict[k] = geo.geoConstraint(v, angle = None, lscale = 1,
                                                 label = k)
        
        pf = core.PFCoil(label, geo.Shape2D(shape))
        pf.geoConstr = geoConstrDict
            
        return pf