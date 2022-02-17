Meshing
=======

The mesh core of bluemira is based on the open source 3D finite element mesh
generator gmsh. A basic api has been implemented to interface with geometry
objects and functions.

.. note:: Currently only a minor part of the gmsh potentiality has been
    implemented in the respective api.

The meshing module of bluemira implements the following main classes:

* :py:class:`bluemira.mesh.meshing.Mesh`
* :py:class:`bluemira.mesh.meshing._FreeCADApi`
