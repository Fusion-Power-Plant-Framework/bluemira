========
Mesh
========

mirapy.meshing
==============

The mesh module of mirapy mainly interface the output of the geometrical
module (base on FreeCAD entities) with gmsh.
Thanks to the implementation of mirapy.geo.Shape and mirapy.geo.Shape,
it is possible to mesh the different components considering shape
primitives, easily changing the mesh size, embedding points, etc.

.. note:: At present, the module is limited to 2D mesh. It could be
    easily extended to 3D mesh implementing an adeguate mirapy.geo.Shape3D.

A basic example for the creation of a mesh is reported in the following:

.. literalinclude:: _static/demos/mesh/plasma_mesh_example.py

.. figure:: _static/images/demos/mesh/plasma_mesh_example.png
   :width: 200

   Example of mesh for a typical plasma shape with embedded points
   that change the mesh size.
   
.. warning::
    By default, when physical groups are defined, Gmsh only saves
    the entities that belong to physical groups.
    
    http://onelab.info/pipermail/gmsh/2020/013685.html
    
.. todo::
    check how to change this default option
