
============
FreeCAD
============

The geometrical model of MIRApy is based on the FreeCAD library (v.0.19).
The FreeCAD source code is managed with git, and is public, open and available
under the LGPL license. It can be copied, downloaded, read, analysed,
redistributed and modified by anyone.

FreeCAD implements most of the data structures and functions necessary
for the creation of a tokamak geometry. In order to decouple the code
from the FreeCAD GUI, a limited number of FreeCAD modules are used.
In particular ``Part``, ``FreeCAD.Base``, and ``Draft``.

To import FreeCAD in your code use the commands (comment lines related to
modules not used in your code):

.. code-block:: python

    import freecad
    from FreeCAD import Base
    import Part
    import Draft

A complete description of topological data and geometrical objects available
in FreeCAD can be found in TopologicalData_

.. _TopologicalData: https://wiki.freecadweb.org/Topological_data_scripting


.. todo::
    complete this description adding information about the mostly used
    FreeCAD classes in mirapy.
