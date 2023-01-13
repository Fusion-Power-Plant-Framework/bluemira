Meshing
=======

The mesh core of bluemira is based on the open source 3D finite element mesh
generator gmsh_. A basic api has been implemented to interface with geometry
objects and functions.

.. note:: Currently only a minor part of the gmsh potentiality has been
    implemented in the respective api.

.. warning:: Only 1D and 2D mesh operations are implemented. Mesh of 3D objects will
   raise and error.

The meshing module of bluemira implements the following main classes:

* :py:class:`bluemira.mesh.meshing.Meshable`: base class from which meshable objects
  inherit
* :py:class:`bluemira.mesh.meshing.Mesh`: active class that performs the mesh operation

Meshable objects
----------------
All objects that inherit from :py:class:`Meshable` are provided by a mesh_options dictionary
in which the following properties can be specified:

* lcar: characteristic mesh length size associated to the vertexes of the geometric
  object

* physical_group: label to group the model entities


Geometry definition and Mesh assignment
---------------------------------------
All :py:class:`BluemiraGeo` objects inherit from :py:class:`Meshable`. After creating a geo object,
`mesh_options` must to be specified (no default values are used). The easiest way is to
use a simple dictionary with `lcar` and `physical_group` keys.

.. code-block:: python
        from bluemira.geometry.tools import make_polygon
        from bluemira.mesh.meshing import Mesh

        poly = make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], closed=True, label="poly"
        )

        poly.mesh_options = {"lcar": 0.1, "physical_group": "poly"}

        m = Mesh()
        m(poly)


The previous code results in the generation of a mesh file, `Mesh.msh` by default, in
which the mesh is stored, and a gmsh file, `Mesh.geo_unrolled` by default, for
checking purpose.

.. important::

    Only objects that have a `physical_group` are exported into the `Mesh.msh` file (see
    gmsh_ for more information).

fenics import
-------------
Once the mesh has been generated, it can be imported in a PDEs solver. Fenics_ solver,
is integrated into bluemira.

.. code-block:: python
    from bluemira.mesh.tools import import_mesh

    mesh, boundaries, subdomains, labels = import_mesh(
        "Mesh",
        directory=".",
        subdomains=True,
    )
    print(mesh.coordinates())


.. _Fenics: https://fenicsproject.org/
.. _gmsh: https://gmsh.info
