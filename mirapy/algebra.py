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

import math
import numpy
from scipy.optimize import fsolve
from scipy.spatial.transform import Rotation as R

import freecad
import Part
from FreeCAD import Base

import logging
module_logger = logging.getLogger(__name__)

def ellipseFit(constrList:list, degree=False):
    """ellipseFit: fit an (non tilted) ellipse on the xy-plane with points and
    tangents constraints.
    
    .. math::
    
       ax^2 + by^2 + cx + dy +1 = 0

    Parameters
    ----------
    constrList : list(tuple)
        list of constraints of the type
        [(P1, angle1),(P2, angle2), ...].
        Points must have a "list" rappresentation, i.e. the x and y
        coordinates must be accessible using P[0] and P[1].
        Note: FreeCAD.Base.Vector fits the requirements
    degree : boolean
        if True angle are given in degrees. Defaults to False (radiants)    

    Returns
    -------
    type
        tuple) center coordinates and major and minor axis length
        (xC,yC,rhoX,rhoY)
        
    Examples
    --------

    >>> constrainList = [(P1,angle1), (P2,), (P3,)]
    """
    A = [];
    b = [];
    
    for c in constrList:
        xP = c[0][0] #x component
        yP = c[0][1] #y component
        
        A.append([xP**2, yP**2, xP, yP])
        b.append(-1.)
    
        if len(c) > 1:
            ang = c[1]            
            if ang != None:
                if degree:
                    ang = math.radians(ang)
                b.append(0.)
                if ang == math.pi/2 or ang == 3.*math.pi/2.:
                    A.append([0., 2.*yP, 0., 1.])
                else:
                    m = math.tan(ang)
                    A.append([2.*xP, 2.*m*yP, 1., m])
    
    output = None
    try:
        X = numpy.linalg.solve(A,b)
        xC = -X[2]/X[0]/2.
        yC = -X[3]/X[1]/2.
        rhoX = math.sqrt(xC**2 + X[1]/X[0]*yC**2 -1./X[0])
        rhoY = math.sqrt(rhoX**2*X[0]/X[1])
        output = (xC,yC,rhoX,rhoY)
    except:
        module_logger.debug("No ellipse has been found for the \
                            given constraints.")
    return output


def circle2P(P1, P2, C0, r:[int,float] = 0):
    """circle2P: fit a circle on the xy-plane passing for P1 and P2
    and having radius r. An inital guess of the center C (C0) is
    neseccary to choose between the two possible solutions.
    Points (i.e. P1, P2, and C0) must have a "list" rappresentation,
    i.e. the x and y coordinates must be accessible using P[0] and P[1].

    Parameters
    ----------
    P1 : point (as numpy.array, list, etc.)
        constraint point
    P2 : point (as numpy.array, list, etc.)
        constraint point
    C0 : point (as numpy.array, list, etc.)
        initial center guess
    r : double
        circle radius. If r<=0, r is calculated as P1-C0 distance.

    Returns
    -------
    tuple: (Cx, Cy, r)
    
    """
    
    if r<=0:
        r = math.sqrt((P1[0] - C0[0])**2 + (P1[1] - C0[1])**2)
    
    def equations(p):
        """

        Parameters
        ----------
        p :
            

        Returns
        -------

        """
        a,b,c = p
        eq = numpy.empty(3)
        eq[0] = P1[0]*a + P1[1]*b + c + (P1[0]*P1[0] + P1[1]*P1[1])
        eq[1] = P2[0]*a + P2[1]*b + c + (P2[0]*P2[0] + P2[1]*P2[1])
        eq[2] = a*a + b*b - 4*c - 4*r*r
        return eq
    
    pGuess = numpy.array([-2*C0[0],-2*C0[1], C0[0]*C0[0] + C0[1]*C0[1] - r*r])
    a, b, c =  fsolve(equations, pGuess)

    return (-a/2.0,-b/2.0,r)


def linesIntersection(P1,angle1,P2,angle2):
    """linesIntersection: find the intersection of two lines defined
    on the xy plane by point and angle.

    Parameters
    ----------
    P1 :
        first point - numpy.array (x,y,z)
        
    angle1 :
        tangent angle @P1 [rad]
        
    P2 :
        second point - numpy.array (x,y,z)
        
    angle2 :
        tangent angle @P2 [rad]
        

    Returns
    -------

    
    """
    
    if math.isclose(math.tan(angle1), math.tan(angle2), abs_tol=1e-8):
        return None
    
    if (P1[0] == P2[0]) and (P1[1] == P2[1]):
        import copy
        return copy.deepcopy(P1)
    
    M = []
    v = []
    
    angle = angle1;
    P = P1
    if angle == math.pi/2 or angle == 3*math.pi/2:
        M.append([1,0])
        v.append([P[0]])
    else:
        m = math.tan(angle)
        M.append([m,-1])
        v.append([m*P[0] - P[1]])

    angle = angle2;
    P = P2
    if angle == math.pi/2 or angle == 3*math.pi/2:
        M.append([1,0])
        v.append([P[0]])
    else:
        m = math.tan(angle)
        M.append([m,-1])
        v.append([m*P[0] - P[1]])
    
    X = numpy.linalg.solve(M,v)
    Pi = numpy.array([X[0][0], X[1][0], 0])
    
    return Pi


def rotate_points(points3D, angle, raxis, order = 0):
    """rotate_points: function that rotate a set of 3D points.

    Parameters
    ----------
    points3D : 3D numpy.array
        set of points.
    angle : float
        angle in radiants.
    raxis :
        'x', 'y' or 'z'.
    order :
        Defaults to 0.

    Returns
    -------
    rotated points: array
        

    """
    # no need of a points3D copy since numpy.dot doesn't modify points3D
    r = R.from_euler(raxis, angle, degrees=True)
    return numpy.stack(numpy.array(numpy.dot(points3D, r.as_matrix())),
                       axis=order)


def bezier2pointstangent(P1, alpha1, P2, alpha2):
    """

    Parameters
    ----------
    P1 :
        
    alpha1 :
        
    P2 :
        
    alpha2 :
        

    Returns
    -------

    """
    curve = Part.BezierCurve()
    Pint = linesIntersection(P1, alpha1, P2, alpha2)
    pointslist = (P1, Pint, P2)
    curve.setPoles(pointslist)
    return curve


def arcofellipse2pointstangent(P1, alpha1, P2, alpha2, reverse=False):   
    """

    Parameters
    ----------
    P1 :
        
    alpha1 :
        
    P2 :
        
    alpha2 :
        
    reverse :
         (Default value = False)

    Returns
    -------

    """
    constrList = [(P1,alpha1),(P2,alpha2)]
    params = ellipseFit(constrList)
    center = Base.Vector(params[0],params[1],0)
    
    z1 = P1 - center
    z2 = P2 - center
    
    if params[2]>params[3]:
        MajorRadius = Base.Vector(params[2], 0., 0.)
        MinorRadius = Base.Vector(0., params[3], 0.)
    else:
        MajorRadius = Base.Vector(0., params[3], 0.)
        MinorRadius = Base.Vector(-params[2], 0., 0.)
    
    s1 = center + MajorRadius
    s2 = center + MinorRadius
    
    curve = Part.Ellipse(s1, s2, center)
    if reverse:
        curve.reverse()
    
    curve = Part.ArcOfEllipse(curve,curve.parameter(P1),
                              curve.parameter(P2))
    return curve


def pointat(P = None, distance = 1.0, angle = 0.0, vector = None):
    """

    Parameters
    ----------
    P :
         (Default value = None)
    distance :
         (Default value = 1.0)
    angle :
         (Default value = 0.0)
    vector :
         (Default value = None)

    Returns
    -------

    """
    
    if not P:
        P = Base.Vector(0.0, 0.0, 0.0) 
    
    P1 = None
    
    if vector == None:
        P1 = P + Base.Vector(math.cos(angle), 
                              math.sin(angle), 0).multiply(distance)
    elif isinstance(vector, Base.Vector):
        P1 = P + vector.multiply(distance)     
    
    return P1