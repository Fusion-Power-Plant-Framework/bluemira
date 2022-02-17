Meshing
=======

The mesh core of bluemira is based on the open source 3D finite element mesh
generator gmsh. A basic api has been implemented to interface with geometry
objects and functions.

.. note:: Currently only a minor part of the gmsh potentiality has been
    implemented in the respective api.

The meshing module of bluemira implements the following main classe:

* :py:class:`bluemira.mesh.meshing.Mesh`
* :py:class:`bluemira.mesh.meshign.Meshable`

Meshable objects
================
All objects that inherit from Meshable are provided by a mesh_options dictionary
in which the following properties can be specified:

* lcar: characteristic mesh length size associated to the vertexes of the geometric
  object

* physical_group: label to group the model entities


Geometry definition and Mesh assignment
=======================================

.. code-block:: python

        poly = tools.make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], closed=True, label="poly"
        )

        poly.mesh_options = {"lcar": lcar, "physical_group": "poly"}

        m = meshing.Mesh()
        m(poly)
