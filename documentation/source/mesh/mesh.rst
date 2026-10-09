Meshing
=======

``bluemira`` uses the open-source finite element mesh generator Gmsh_ to generate meshes
from geometry.

The main meshing classes are:

* :py:class:`bluemira.mesh.meshing.Mesh`
* :py:class:`bluemira.mesh.meshing.Meshable`
* :py:class:`bluemira.mesh.meshing.MeshOptions`
* :py:class:`bluemira.mesh.meshing.MeshSettings`
* :py:class:`bluemira.mesh.meshing.GmeshSession`

Both 2D and 3D meshes are supported.

.. note::
    ``bluemira`` currently exposes a subset of the options available in Gmsh.


Mesh options
------------

All :py:class:`BluemiraGeo` objects have `mesh_options` that can be used to
control the local mesh.

The currently available options are:

`lcar`\
Characteristic mesh size associated with the geometry.

`physical_group`\
Name used to identify the geometry in the generated mesh.

For example:

.. code-block:: python

    from bluemira.geometry.tools import make_polygon

    poly = make_polygon(
        [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], closed=True, label="poly"
    )

    poly.mesh_options = {"lcar": 0.1, "physical_group": "poly"}


Gmsh session
------------

A Gmsh session must be active before generating a mesh.

For scripts, the recommended approach is to use :py:class:`bluemira.mesh.meshing.GmshSession`
as a context manager:

.. code-block:: python

    from bluemira.mesh import meshing

    with meshing.GmshSession():
        meshing.Mesh()

The session is automatically cleaned up when the context is exited.

For interactive or notebook use, the session can also be controlled explicitly:

.. code-block:: python

    from bluemira.mesh import meshing

    session = meshing.GmshSession()
    session.initialize()

    meshing.Mesh()

    session.finalize()


Generating a 2D mesh
--------------------

A simple 2D mesh can be generated from a face:

.. code-block:: python

    from bluemira.geometry.face import BluemiraFace
    from bluemira.geometry.tools import make_polygon
    from bluemira.mesh import meshing

    poly = make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], closed=True, label="poly"
        )

    poly.mesh_options = {"lcar": 0.1, "physical_group": "poly"}

    surface = BluemiraFace(poly)

    surface.mesh_options = {"lcar": 0.2, "physical_group": "surface"}

    with meshing.GmshSession():
        meshing.Mesh()(surface, dim=2)


Generating a 3D mesh
--------------------

The same interface can be used for 3D geometry:

.. code-block:: python

    from bluemira.geometry.face import BluemiraFace
    from bluemira.geometry.tools import extrude_shape, make_circle
    from bluemira.mesh import meshing

    circle = make_circle(radius=5, center=[0, 0, 0])
    cylinder = extrude_shape(BluemiraFace(circle), vec=[0, 0, 10])

    cylinder.mesh_options = {"lcar": 1.0, "physical_group": "cylinder"}

    with meshing.GmshSession():
        meshing.Mesh()(cylinder, dim=3)


Global mesh settings
--------------------

Global mesh generation settings can be supplied using :py:class:`bluemira.mesh.meshing.MeshSettings`.

The currently available settings include:

* `algorithm_2d` - 2D meshing algorithm.
* `algorithm_3d` - 3D meshing algorithm.
* `element_order` - Geometric accuracy and interpolation precision of mesh elements.
* `mesh_size_min` - Minimum mesh element size.
* `mesh_size_max` - Maximum mesh element size.
* `optimise` - Improve the quality of elements.

For example:

.. code-block:: python

    from bluemira.mesh import meshing

    settings = meshing.MeshSettings(
        element_order=2,
        mesh_size_min=0.05,
        mesh_size_max=0.5,
        optimise=True,
    )

    with meshing.GmshSession():
        meshing.Mesh(settings=settings)


Output files
------------

By default, :py:class:`bluemira.mesh.meshing.Mesh` writes:

`Mesh.geo_unrolled`\
Gmsh model file.

`Mesh.msh`\
Generated mesh.

Alternative output paths may be supplied:

.. code-block:: python

    from bluemira.mesh import meshing

    with meshing.GmshSession():
        mesh = meshing.Mesh(
            meshfile=["output/Mesh.geo_unrolled", "output/Mesh.msh"]
        )

.. important::
    Geometry that needs to be identified in downstream finite element workflows
    should be assigned a `physical_group`.


Importing generated meshes
--------------------------

A generated `.msh` file can be converted and imported using the utilities in
:mod:`bluemira.mesh.tools`.

.. code-block:: python

    from bluemira.mesh.tools import import_mesh, msh_to_xdmf

    msh_to_xdmf("Mesh.msh", dimensions=(0, 2))

    mesh, boundaries, subdomains, labels = import_mesh("Mesh", subdomains=True)


.. _Gmsh: https://gmsh.info/
