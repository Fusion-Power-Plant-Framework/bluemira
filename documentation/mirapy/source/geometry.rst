========
Geometry
========

mirapy.geo
==========

The geometrical core of mirapy is based on FreeCAD package.

.. note:: Currently only a minor part of the FreeCAD potentiality has been
    used leaving the possibility to evaluate also other free geometrical codes.
    However, FreeCAD seems to offer a lot of useful functionality and a GUI
    that could be extended to integrate a mirapy interface.


Even if the FreeCAD geometrical core (i.e. Part and Base modules) would
already implement all the necessary geometrical structures and functions for
an easy definition of a tokamak geometry, it has been found limited for what
concerns the interface with external mesh tools and FEM solvers.
In particular, without calling the GUI, it is
not possible to label and propagate FreeCAD geometrical entities
making extremely complex the set up of a mesh and the identification of
boundaries and subdomains.
To overcome this limitation, the geometrical module of mirapy implements
two main classes:

* Shape: a container of freecad Part.Wire objects (1D objects)
* Shape2D: a container of freecad Part.Face objects (2D objects) defined
    usign Shape istances.

Both classes can be used recursively becoming container of Shape and Shape2D
objects, respectively.
Such kind of implementation allows the management of physical groups and
identifications of particular shapes that can be used easily when creating
complex structures and to obtain suitable "domains" for FEM models
implementation.

A basic example for the creation of a Shape and Shape2D object is reported
in the following:

.. literalinclude:: _static/demos/geo/simple_shape_creation.py

A simple application for the creation of a plasma-like shape:

.. literalinclude:: _static/demos/geo/plasma_shape_example.py

.. figure:: _static/images/demos/geo/plasma_shape_creation.png
   :width: 200
