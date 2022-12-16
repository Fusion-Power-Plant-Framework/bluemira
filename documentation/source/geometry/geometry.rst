geometry
========

The ``geometry`` module contains the core functionality to represent objects used for a
``bluemira`` analysis. The geometrical core of bluemira is based on FreeCAD package. A basic api has been
implemented to interface with main FreeCAD objects and functions.

.. note:: Currently only a minor part of the FreeCAD potentiality has been
    implemented in the respective api.

The geometrical module of bluemira implements the following main classes:

* :py:class:`bluemira.geometry.wire.BluemiraGeo`: geometry abstract class
* :py:class:`bluemira.geometry.wire.BluemiraWire`: a container of FreeCAD Part.Wire
  objects
* :py:class:`bluemira.geometry.face.BluemiraFace`: a container of FreeCAD Part.Face
  objects
* :py:class:`bluemira.geometry.shell.BluemiraShell`: a container of FreeCAD Part.Shell
  objects
* :py:class:`bluemira.geometry.solid.BluemiraSolid`: a container of FreeCAD Part.Solid
  objects

Main functions to create or manipulate geometry objects are implemented within
:py:class:`bluemira.geometry.tool`. A lot of the functionality can be explored in our
examples. Further discussion on the background and basics are linked below.

.. toctree::
   :maxdepth: 4
   :caption: Contents:

   overview
   parameterisation
