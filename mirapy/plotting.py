#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# import from freecad
import freecad
import Part
from FreeCAD import Base

# import numerical lib
import numpy

# import graphical lib
import matplotlib.pyplot as plt
# import matplotlib.tri as tri

# import mirapy lib
from . import geo
from . import core

from typing import Union

DEFAULT = {}
DEFAULT['poptions'] = {'s': 30, 'facecolors': 'blue', 'edgecolors': 'black'}
DEFAULT['woptions'] = {'color': 'black', 'linewidth': '2'}
DEFAULT['foptions'] = {'color': 'red'}

def discretizeByEdges(w: Part.Wire, ndiscr: int):
    """Discretize a wire taking into account the edges of which it consists of.

    Parameters
    ----------
    w : Part.Wire
        wire to be discretized.
    ndiscr : int
        number of points for the whole wire discretization.

    Returns
    -------
    output : list(Base.Vector)
        list of Base.Vector points.

    """
    # discretization points array
    output = []
    # a dl is calculated for the discretization of the different edges
    dl = w.Length/float(ndiscr)
    # edges are discretised taking into account their orientation
    # Note: this is a tricky part in Freecad. Reversed wires need a
    # reverse operation for the generated points and the list of generated
    # points for each edge.
    for e in w.OrderedEdges:
        pointse = e.discretize(Distance=dl)
        # if edge orientation is reversed, the generated list of points
        # must be reversed
        if e.Orientation == "Reversed":
            pointse.reverse()
        output += pointse
    # if wire orientation is reversed, output must be reversed
    if w.Orientation == "Reversed":
        output.reverse()
    return output


def plotPoint2D(points,
                 axis=None,
                 show: bool = False,
                 ndiscr: int = 100,
                 plane: Union[str, Base.Placement] = 'xy',
                 poptions: dict = DEFAULT['poptions'],
                 *args,
                 **kwargs,
                 ):
    """
    Plots a 2D point
    
    Parameters
    ----------
    points : list of points
    axis : matplotlib axis
         (Default value = None)
    show: bool
         (Default value = False)
    ndiscr: int
         (Default value = 100)
    plane: Union[str, Base.Placement]
         (Default value = 'xy')
    poptions: dict
         (Default value = DEFAULT['poptions'])
    *args :
        
    **kwargs :
        

    Returns
    -------

    """
        
    if not poptions:
        return axis, [] 

    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot()

    if plane == 'xy':
        # Base.Placement(origin, axis, angle)
        plane = Base.Placement(Base.Vector(), Base.Vector(1., 0., 0.), 0.)
    elif plane == 'xz':
        # Base.Placement(origin, axis, angle)
        plane = Base.Placement(Base.Vector(), Base.Vector(1., 0., 0.), -90.)
    elif plane == 'yz':
        # Base.Placement(origin, axis, angle)
        plane = Base.Placement(Base.Vector(), Base.Vector(0., 1., 0.), 90.)
    elif isinstance(plane, Base.Placement):
        # nothing to do
        pass    

    if hasattr(points, '__len__'):
        if isinstance(points, numpy.ndarray):
            if len(points.shape)==1:
                points = [points]
        elif isinstance(points, Base.Vector):
                points = [points]
        
    for p in points:
        
        if isinstance(p, numpy.ndarray):
            p = Part.Vertex(Base.Vector(p))
        elif isinstance(p, Base.Vector):
            p = Part.Vertex(p)
            print(p)

        if not isinstance(p, Part.Vertex):
            print(p)
            raise ValueError("Point must be a numpy.ndarray, a Base.Vector, or"
                             " a Part.Vertex instance.")

        # make a copy of the point
        p = p.copy()
        p.Placement = p.Placement.multiply(plane)

        axis.scatter(p.X, p.Y, **poptions)

    plt.gca().set_aspect("equal")
    if show:
        plt.show()

    return axis, numpy.array(points)


def plotWire2D(
        wire,
        axis=None,
        show: bool = False,
        ndiscr: int = 100,
        plane: Union[str, Base.Placement] = 'xy',
        poptions: dict = DEFAULT['poptions'],
        woptions: dict = DEFAULT['woptions'],
        *args,
        **kwargs,
        ):
    """Plot a Part.Wire or list(Part.Wire) on a 2D plane.

    Parameters
    ----------
    wire : Part.Wire
        the wire to be plotted.
    axis : matplot.axis
        matplotlib axis for plotting.\
        Defaults to None.
    show: bool
         matplotlib option. (Default value = False)
    ndiscr: int
        number of discretization points (Default value = 100)
    plane: Union[str, Base.Placement]
         (Default value = 'xy')
    poptions: dict
         (Default value = DEFAULT['poptions'])
    woptions: dict
         (Default value = DEFAULT['woptions'])
    *args :
        
    **kwargs :
        

    Returns
    -------
    axis : matplot.axis
        axis used for multiplots.

    Raises
    ------
    ValueError
        in case the obj is not a part.Wire or \
        the used plane option is not available.

    """

    # Note: only Part.Wire objects are allowed as input for this function.
    # However, any object that can be discretized by means of the function
    # "discretizeByEdges" would be suitable. In case the function can be
    # extended to outher objects changing the "discretizeByEdges" function.
    if not woptions and not poptions:
        return axis    

    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot()

    if plane == 'xy':
        # Base.Placement(origin, axis, angle)
        plane = Base.Placement(Base.Vector(), Base.Vector(1., 0., 0.), 0.)
    elif plane == 'xz':
        # Base.Placement(origin, axis, angle)
        plane = Base.Placement(Base.Vector(), Base.Vector(1., 0., 0.), -90.)
    elif plane == 'yz':
        # Base.Placement(origin, axis, angle)
        plane = Base.Placement(Base.Vector(), Base.Vector(0., 1., 0.), 90.)
    elif isinstance(plane, Base.Placement):
        # nothing to do
        pass 

    if not hasattr(wire, '__len__'):
        wire = [wire]
    
    for w in wire:

        if not isinstance(w, Part.Wire):
            raise ValueError("wire must be a Part.Wire")

        # make a copy of the wire
        w = w.copy()
        w.Placement = w.Placement.multiply(plane)

        pointsw = discretizeByEdges(w, ndiscr)

        for p in pointsw:
            x = [p[0] for p in pointsw]
            y = [p[1] for p in pointsw]
            z = [p[2] for p in pointsw]

        axis.plot(x, y, **woptions)

        axis, _ = plotPoint2D(w.Vertexes, axis=axis, poptions=poptions)

    plt.gca().set_aspect("equal")
    if show:
        plt.show()

    return axis, numpy.array(pointsw)


def plotFace2D(
        faces,
        axis=None,
        show: bool = False,
        ndiscr: int = 100,
        plane: Union[str, Base.Placement] = 'xy',
        poptions: dict = DEFAULT['poptions'],
        woptions: dict = DEFAULT['woptions'],
        foptions: dict = DEFAULT['foptions'],
        *args,
        **kwargs,
        ):
    """

    Parameters
    ----------
    faces :
        
    axis :
         (Default value = None)
    show: bool
         (Default value = False)
    ndiscr: int
         (Default value = 100)
    plane: Union[str, Base.Placement]
         (Default value = 'xy')
    poptions: dict
         (Default value = DEFAULT['poptions'])
    woptions: dict
         (Default value = DEFAULT['woptions'])
    foptions: dict
         (Default value = DEFAULT['foptions'])
    *args :
        
    **kwargs :
        

    Returns
    -------

    """

    if not foptions and not woptions and not poptions:
        return axis

    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot()

    if plane == 'xy':
        # Base.Placement(origin, axis, angle)
        plane = Base.Placement(Base.Vector(), Base.Vector(1., 0., 0.), 0.)
    elif plane == 'xz':
        # Base.Placement(origin, axis, angle)
        plane = Base.Placement(Base.Vector(), Base.Vector(1., 0., 0.), -90.)
    elif plane == 'yz':
        # Base.Placement(origin, axis, angle)
        plane = Base.Placement(Base.Vector(), Base.Vector(0., 1., 0.), 90.)
    elif isinstance(plane, Base.Placement):
        # nothing to do
        pass 

    if not hasattr(faces, '__len__'):
        faces = [faces]

    if not all(isinstance(f, Part.Face) for f in faces):
        raise ValueError("faces must be Part.Face objects")
    
    for f in faces:

        x = []
        y = []
        z = []

        # make a copy of face since we are going to change placement
        f = f.copy()
        # change face placement
        f.Placement = f.Placement.multiply(plane)
        # set plane to the default
        plane = 'xy'
            
        for w in f.Wires:
            # Since the face's placement has been already changed, it is
            # necessery to specify the 'xy' plane when plotting the wires
            axis, pointsw = plotWire2D(w, axis, show, ndiscr, plane,
                                       poptions=poptions,
                                       woptions=woptions)                
            x += [p[0] for p in pointsw] + [None]
            y += [p[1] for p in pointsw] + [None]      
            z += [p[2] for p in pointsw] + [None]      

        x = x[:-1]
        y = y[:-1]
        if foptions:
            plt.fill(x, y, **foptions)
        
    plt.gca().set_aspect("equal")
    if show:
        plt.show()

    pointsf = numpy.array([[x1, y1, z1] for x1, y1, z1 in zip(x, y, z)])

    return axis, pointsf


def plotgeoConstr(
        geoConstr: geo.geoConstraint,
        axis=None,
        show: bool = False,
        plane: Union[str, Base.Placement] = 'xy',
        poptions: dict = DEFAULT['poptions'],
        woptions: dict = DEFAULT['woptions'],
        *args,
        **kwargs,
):
    """Plot a geoConstraint on a 2D plane.

    Parameters
    ----------
    geoConstr : geo.geoConstr
        the wire to be plotted.
    axis : matplot.axis
        matplotlib axis for plotting.\
        Defaults to None.
    show : bool
        matplotlib option. (Default value = False)
    ndiscr : int
        number of discretization points. Defaults to 100.
    plane : Union[str, Base.Placement]
        plane on which the \
        plot has to be performed. (Default value = 'xy')
    poptions : dict
        plot matplotlib options for points
    woptions : dict
        plot matplotlib options for lines
    *args :
    **kwargs :

    Returns
    -------
    axis : matplot.axis
        axis used for multiplots.

    Raises
    ------
    ValueError
        in case the obj is not a part.Wire or \
        the used plane option is not available.

    """

    delta_label = [0.1, 0.1]

    if not isinstance(geoConstr, geo.geoConstraint):
        raise ValueError("geoConstr must be a geo.geoConstraint")
    
    if woptions is None:
        woptions = {'color': 'black', 'linewidth': '2', 'linestyle': 'dashed'}

    pointsw = []
    
    if not hasattr(geoConstr, '__len__'):
        geoConstr = [geoConstr]

    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot()

    if plane == 'xy':
        # Base.Placement(origin, axis, angle)
        plane = Base.Placement(Base.Vector(), Base.Vector(1., 0., 0.), 0.)
    elif plane == 'xz':
        # Base.Placement(origin, axis, angle)
        plane = Base.Placement(Base.Vector(), Base.Vector(1., 0., 0.), -90.)
    elif plane == 'yz':
        # Base.Placement(origin, axis, angle)
        plane = Base.Placement(Base.Vector(), Base.Vector(0., 1., 0.), 90.)
    elif isinstance(plane, Base.Placement):
        # nothing to do
        pass

    for gc in geoConstr:
        # make a copy of the wire
        p = Part.Vertex(gc.point)
        p.Placement = p.Placement.multiply(plane)
        p = p.Point
        
        if poptions:
            axis.scatter(p[0], p[1], **poptions)
        
        pointsw = numpy.array(p)

        if gc.line is not None:
            w = gc.line
            axis, pointsw = plotWire2D(w, axis, show, 2, plane,
                                       poptions={}, woptions=woptions)

        if gc.label is not None and gc.angle is None:
            axis.annotate(gc.label,
                          (p[0] + delta_label[0], p[1] + delta_label[1]),
                          fontsize=12,
                          )

    plt.gca().set_aspect("equal")
    if show:
        plt.show()

    return axis, numpy.array(pointsw)

def plot2D(
        obj,
        axis=None,
        show: bool = False,
        ndiscr: int = 100,
        plane: Union[str, Base.Placement] = 'xy',
        poptions: dict = DEFAULT['poptions'],
        woptions: dict = DEFAULT['woptions'],
        foptions: dict = DEFAULT['foptions'],
        *argv,
        **kwargs,
):
    """Plotting function for 1D/2D objects

    Parameters
    ----------
    obj :
        object to be plotted.
    axis :
        matplotlib axis. (Default value = None)
    show : bool
        if True the plot is shown. (Default value = False)
    ndiscr : int
        Number of discretization points for 1D\
        objects. Defaults to 100.
    plane : Union[str, Base.Placement]
        plane on which the plot \
        is made. (Default value = 'xy').
    poptions : dict
        matplotlib options for 0D entities. \
        (Default value = DEFAULT['poptions']).
    woptions : dict
        matplotlib options for 1D entities.\
        (Default value = DEFAULT['woptions']).
    foptions : dict
        matplotlib options for 0D entities.\
        (Default value = DEFAULT['foptions']).
    *argv :
        not used.
    **kwargs :
        used for recursion
        
    Returns
    -------
    
        None.

    """
    pointsw = []
    newkwargs = {'poptions':poptions,
                 'woptions':woptions,
                 'foptions':foptions,}

    # plot point
    if isinstance(obj, (numpy.ndarray, Base.Vector, Part.Vertex)):
        axis, _ = plotPoint2D(obj, axis=axis, show=show, plane=plane,
                             **newkwargs)
    # plot geoConstraint
    elif isinstance(obj, geo.geoConstraint):
        axis, _ = plotgeoConstr(obj, axis=axis, show=show, plane=plane,
                             **newkwargs)
    # plot Wire
    elif isinstance(obj, Part.Wire):
        axis, _ = plotWire2D(obj, axis=axis, show=show, plane=plane,
                             ndiscr=ndiscr, **newkwargs)
    # plot Face
    elif isinstance(obj, Part.Face):
        axis, _ = plotFace2D(obj, axis=axis, show=show, plane=plane,
                             ndiscr=ndiscr, **newkwargs)
    # plot Shape
    elif isinstance(obj, geo.Shape):
        for o in obj.boundary:
            if isinstance(o, Part.Wire):
                # print("plot wire")
                axis, pointsw = plotWire2D(o, axis=axis, show=show, plane=plane,
                             ndiscr=ndiscr, **newkwargs)
            if isinstance(o, geo.Shape):
                # print("plot shape - call to plotWire2D")
                axis, pointsw = plot2D(o, axis=axis, show=show, plane=plane,
                                       ndiscr=ndiscr, **newkwargs)
    # plot Shape2D
    elif isinstance(obj, geo.Shape2D):
        face = obj.face
        axis, pointsw = plotFace2D(face, axis, show, ndiscr, plane,
                                   **newkwargs)
    # plot Component
    elif isinstance(obj, core.Component):
        leaves = obj.leaves
        for l in leaves:
            if l.shape:
                axis, pointsw = plot2D(l.shape, axis, False, ndiscr, plane,
                                       **newkwargs)
    else:
        print("Object type not supported for plot2D")

    return axis, pointsw


def plot_scalar_field(x, y, z, levels = 20, axis = None,
                      show=False, tofill=True):
    """

    Parameters
    ----------
    x :
        
    y :
        
    z :
        
    levels :
         (Default value = 20)
    axis :
         (Default value = None)
    show :
         (Default value = False)
    tofill :
         (Default value = True)

    Returns
    -------

    """
    cntr = None
    cntrf = None
    
    if axis is None:
        fig = plt.figure()
        axis = fig.add_subplot()
        
    # # ----------
    # # Tricontour
    # # ----------
    # # Directly supply the unordered, irregularly spaced coordinates
    # # to tricontour.
    
    cntr = axis.tricontour(x, y, z, levels = levels, linewidths=0.5, colors='k')
    if tofill:
        cntrf = axis.tricontourf(x, y, z, levels = levels, cmap="RdBu_r")
        plt.gcf().colorbar(cntrf, ax=axis)

    plt.gca().set_aspect("equal")

    if show:
        plt.show()

    return axis, cntr, cntrf

# def plot_scalar_function(func, xi, yi, dx, dy, ngridx=100, ngridy=100, 
#                          levels=20, axis=None, tofill=True):
    
#     x = np.linspace(xi, xi + dx, ngridx)
#     y = np.linspace(yi, yi + dy, ngridy)
#     xv, yv = np.meshgrid(x, y)
#     points = np.vstack([xv.ravel(), yv.ravel()]).T
#     field = func(points)
#     axis, cntr, cntrf = plot_scalar_field(points[:,0], points[:,1], 
#                                               field, ngridx, ngridy, 
#                                               levels, axis, tofill)
#     return axis, cntr, cntrf, points, field


# def find_contour(x, y, data, level, axis = None):
#     import matplotlib.pyplot as plt
#     from skimage import measure
    
#     # Find contours at a constant value of "level"
#     contours = measure.find_contours(data, level)
    
#     # Display the image and plot all contours found
#     if axis is None:
#         fig = plt.figure()
#         axis = fig.add_subplot()
        
#     axis.imshow(data, cmap=plt.cm.plasma)
    
#     for contour in contours:
#         axis.plot(contour[:, 1], contour[:, 0], linewidth=2)
    
#     return contours, axis
