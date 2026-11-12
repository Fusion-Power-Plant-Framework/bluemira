# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Examples of using the Bluemira mesh module to generate 2D and 3D meshes.
"""

# %% [markdown]
# # Meshing Example
# ## Introduction
#
# Bluemira uses the open-source mesh generator Gmsh to generate finite-element
# meshes from Bluemira geometry.
#
# This tutorial demonstrates:
#
# - Generating 2D meshes from faces and component trees.
# - Setting local mesh options on geometry.
# - Configuring global mesh generation settings.
# - Selecting element types and meshing strategies.
# - Generating 3D meshes from solid geometry.
# - Converting Gmsh meshes into DOLFINx meshes.
# - Exporting and importing meshes using XDMF.
#
# ## Imports

# %%
from pathlib import Path

import gmsh

from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.file import get_bluemira_path
from bluemira.base.logs import set_log_level
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.geometry import tools
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.wire import BluemiraWire
from bluemira.magnetostatics.fem_utils import model_to_mesh, plot_mesh
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf, plot_dolfinx_mesh

set_log_level("DEBUG")


# %% [markdown]
# ## 1. Creating a 2D mesh
#
# We begin by creating a simple 2D geometry consisting of a plasma boundary
# and a coil with a casing.
#
# Each geometry object can be assigned local mesh options. The `lcar` option
# specifies the characteristic mesh size, while `physical_group` assigns a
# name to the corresponding region in the generated mesh.

# %%
p = JohnerLCFS()
lcfs = p.create_shape(label="LCFS")

poly1 = tools.make_polygon(
    [[0, 1, 1], [0, 0, 0], [0, 0, 1]], closed=False, label="poly1"
)
poly2 = tools.make_polygon(
    [[1, 0, 0], [0, 0, 0], [1, 1, 0]], closed=False, label="poly2"
)
poly3 = tools.make_polygon(
    [[0.25, 0.75, 0.75], [0, 0, 0], [0.25, 0.25, 0.75]], closed=False, label="poly3"
)
poly4 = tools.make_polygon(
    [[0.75, 0.25, 0.25], [0, 0, 0], [0.75, 0.75, 0.25]], closed=False, label="poly4"
)
poly_out = BluemiraWire([poly1, poly2], label="poly_out")
poly_in = BluemiraWire([poly3, poly4], label="poly_in")
coil_out = BluemiraFace([poly_out, poly_in], label="coil_out")
coil_in = BluemiraFace([poly_in], label="coil_in")


# %% [markdown]
# ### Local mesh options
#
# The mesh size can be set independently for each geometry object. This allows
# finer meshes to be generated in regions where greater resolution is needed.
#
# The `physical_group` option assigns a name to the geometry. These names are
# useful when identifying regions in downstream finite-element calculations.

# %%
lcfs.mesh_options = {"lcar": 0.75, "physical_group": "LCFS"}
face = BluemiraFace(lcfs, label="plasma_surface")
face.mesh_options = {"lcar": 0.5, "physical_group": "surface"}

poly1.mesh_options = {"lcar": 0.25, "physical_group": "poly1"}
poly2.mesh_options = {"lcar": 0.25, "physical_group": "poly2"}
poly3.mesh_options = {"lcar": 0.75, "physical_group": "poly3"}
poly4.mesh_options = {"lcar": 0.75, "physical_group": "poly4"}
coil_out.mesh_options = {"lcar": 1, "physical_group": "coil"}
coil_in.mesh_options = {"lcar": 0.3, "physical_group": "coil"}

# %% [markdown]
# ### Component trees
#
# When several geometries need to be meshed together, they can be organised
# into a component tree. This makes it easier to manage complex models and
# assign mesh options to individual physical components.

# %%
c_all = Component(name="all")
c_plasma = PhysicalComponent(name="plasma", shape=face, parent=c_all)
c_coil = Component(name="coil", parent=c_all)
c_coil_in = PhysicalComponent(name="coil_in", shape=coil_in, parent=c_coil)
c_coil_out = PhysicalComponent(name="coil_out", shape=coil_out, parent=c_coil)

# %% [markdown]
# ### Generate the mesh
#
# A Gmsh session must be active before generating a mesh. The recommended
# approach for scripts and notebooks is to use `GmshSession` as a context
# manager. The session is automatically finalised when the context is exited.
#
# `Mesh` generates the mesh and writes the Gmsh geometry and mesh files.

# %%
directory = Path(get_bluemira_path("", subfolder="generated_data/")).joinpath(
    "examples", "mesh_tutorial"
)
directory.mkdir(parents=True, exist_ok=True)

meshfiles = [
    (directory / "Mesh.geo_unrolled").as_posix(),
    (directory / "Mesh.msh").as_posix(),
]

with meshing.GmshSession():
    gmsh_mesh = meshing.Mesh(meshfile=meshfiles)
    entities = gmsh_mesh(c_all, dim=2)

print(entities)


# %% [markdown]
# ## 2. Choosing element types and meshing strategies
#
# Bluemira supports requesting different element types and meshing strategies
# through geometry mesh options. Gmsh does not guarantee to successfully
# produce a mesh adhering to these settings, and will fallback to an unstructured
# tetrahedra mesh if unsuccessful.
#
# For example, a quadrilateral mesh can be requested using
# `element_type="quad"`. A structured meshing strategy can also be requested
# for geometries that support it.
#
# The following example uses a simple rectangular face to demonstrate these
# options independently of the more complicated geometry above.

# %%
square = BluemiraFace(
    tools.make_polygon(
        [
            [0, 0, 0],
            [4, 0, 0],
            [4, 4, 0],
            [0, 4, 0],
        ],
        closed=True,
    ),
    label="square",
)

square.mesh_options = {
    "lcar": 0.5,
    "physical_group": "square",
    "element_type": "quad",
    "mesh_strategy": "structured",
}

with meshing.GmshSession():
    meshing.Mesh()(square, dim=2)
    (quad_mesh, quad_cell_tags, quad_facet_tags), quad_labels = model_to_mesh(
        gmsh.model,
        gdim=2,
    )

plot_mesh(quad_mesh, quad_cell_tags)

# %% [markdown]
# The `element_type` option can also be set to `"tri"` to request triangular
# elements. The following example uses the same geometry with triangular
# elements and the default meshing strategy.

# %%
square.mesh_options = {
    "lcar": 0.5,
    "physical_group": "square",
    "element_type": "tri",
}

with meshing.GmshSession():
    meshing.Mesh()(square, dim=2)
    (tri_mesh, tri_cell_tags, tri_facet_tags), tri_labels = model_to_mesh(
        gmsh.model,
        gdim=2,
    )

plot_mesh(tri_mesh, tri_cell_tags)


# %% [markdown]
# ## 3. Global mesh settings
#
# Local mesh options control the mesh associated with individual geometry
# objects. Global settings can be supplied through `MeshSettings` and apply
# to the mesh generation process.
#
# The available settings include:
#
# - `algorithm_2d`: Gmsh algorithm used for 2D meshing.
# - `algorithm_3d`: Gmsh algorithm used for 3D meshing.
# - `element_order`: element interpolation order.
# - `mesh_size_min`: minimum mesh element size.
# - `mesh_size_max`: maximum mesh element size.
# - `optimise`: whether to optimise the generated mesh.
#
# Here we request a second-order mesh and constrain the global mesh-size range.

# %%
settings = meshing.MeshSettings(
    element_order=2,
    mesh_size_min=0.1,
    mesh_size_max=0.2,
    optimise=True,
)

with meshing.GmshSession():
    meshing.Mesh(settings=settings)(square, dim=2)
    (second_order_mesh, second_order_cell_tags, _), second_order_labels = model_to_mesh(
        gmsh.model, gdim=2
    )

plot_mesh(second_order_mesh, second_order_cell_tags)


# %% [markdown]
# ## 4. Physical groups and mesh tags
#
# Physical groups identify geometric regions in the generated mesh. They are
# particularly important when a mesh is used in a finite-element calculation,
# where different regions may have different material properties or boundary
# conditions.
#
# The `model_to_mesh` utility converts the current Gmsh model into a DOLFINx
# mesh and returns cell and facet tags, along with a mapping of physical group
# names to identifiers.

# %%
print(second_order_labels)
print(second_order_cell_tags.values)


# %% [markdown]
# ## 5. Converting a mesh to XDMF
#
# A generated Gmsh mesh can also be exported to XDMF and subsequently imported
# into DOLFINx.
#
# This is useful when the mesh needs to be stored and reused independently of
# the Gmsh model.

# %%
msh_to_xdmf(
    "Mesh.msh",
    dimensions=(0, 2),
    directory=directory,
)

mesh, boundaries, subdomains, labels = import_mesh(
    "Mesh",
    directory=directory,
    subdomains=True,
)

print(mesh.geometry.x)

# %% [markdown]
# The imported mesh can be visualised using the DOLFINx plotting utility.

# %%
plot_dolfinx_mesh(mesh)


# %% [markdown]
# ## 6. Generating a 3D mesh
#
# The same meshing interface can be used to generate three-dimensional meshes.
# In this example, a square face is extruded to create a solid volume.
#
# The mesh is generated with `dim=3`, and the resulting Gmsh model is converted
# directly into a DOLFINx mesh using `model_to_mesh`.

# %%
square_3d = BluemiraFace(
    tools.make_polygon(
        [
            [0, 0, 0],
            [4, 0, 0],
            [4, 0, 4],
            [0, 0, 4],
        ],
        closed=True,
    ),
    label="square",
)

cube = tools.extrude_shape(square_3d, vec=[0, 4, 0])
cube.mesh_options = {
    "lcar": 0.5,
    "physical_group": "cube",
}

with meshing.GmshSession():
    meshing.Mesh()(cube, dim=3)
    (volume_mesh, volume_cell_tags, volume_facet_tags), volume_labels = model_to_mesh(
        gmsh.model, gdim=3
    )

plot_mesh(volume_mesh, volume_cell_tags)

# %% [markdown]
# ## 7. Structured hexahedral meshes
#
# For suitable geometries, structured meshing can be used to generate
# hexahedral elements.
#
# Here, the surface mesh is requested to use quadrilaterals, while the volume
# mesh is requested to use hexahedra. The geometry and meshing strategy must
# be compatible with structured meshing for this to succeed.

# %%
square_hex = BluemiraFace(
    tools.make_polygon(
        [
            [0, 0, 0],
            [4, 0, 0],
            [4, 0, 4],
            [0, 0, 4],
        ],
        closed=True,
    ),
    label="square",
)

square_hex.mesh_options = {
    "lcar": 0.5,
    "physical_group": "square",
    "element_type": "quad",
}

cube_hex = tools.extrude_shape(square_hex, vec=[0, 4, 0])
cube_hex.mesh_options = {
    "lcar": 0.5,
    "physical_group": "cube",
    "element_type": "hex",
    "mesh_strategy": "structured",
}

with meshing.GmshSession():
    meshing.Mesh()(cube_hex, dim=3)
    (hex_mesh, hex_cell_tags, hex_facet_tags), hex_labels = model_to_mesh(
        gmsh.model,
        gdim=3,
    )

plot_mesh(hex_mesh, hex_cell_tags)


# %% [markdown]
# ## 8. 3D geometry from a swept winding pack
#
# More complex 3D geometry can also be meshed. The following example creates
# a winding-pack cross-section and sweeps it along a Princeton-D centreline.
#
# This demonstrates how the same meshing interface can be used with geometry
# generated by Bluemira's geometry tools.

# %%

tf_centreline = PrincetonD({
    "x1": {"value": 4, "fixed": True},
    "x2": {"value": 15, "fixed": True},
    "dz": {"value": 0, "fixed": True},
}).create_shape()

half_width = 0.5
half_depth = 0.5

cross_section = tools.make_polygon(
    {
        "x": [3.5, 4.5, 4.5, 3.5],
        "y": [-half_depth, -half_depth, half_depth, half_depth],
        "z": 0,
    },
    closed=True,
)

tf_winding_pack = tools.sweep_shape(
    cross_section,
    tf_centreline,
)

tf_winding_pack.mesh_options = {
    "lcar": 0.8,
    "physical_group": "tf_winding_pack",
}

with meshing.GmshSession():
    meshing.Mesh()(tf_winding_pack, dim=3)
    (tf_mesh, tf_cell_tags, tf_facet_tags), tf_labels = model_to_mesh(
        gmsh.model,
        gdim=3,
    )

plot_mesh(tf_mesh, tf_cell_tags)


# %% [markdown]
# ## 9. Output files
#
# By default, `meshing.Mesh` writes:
#
# - `Mesh.geo_unrolled`: the Gmsh geometry model.
# - `Mesh.msh`: the generated mesh.
#
# Alternative output paths can be supplied using the `meshfile` argument.
#
# For example:
#
# ```python
# with meshing.GmshSession():
#     meshing.Mesh(
#         meshfile=[
#             "output/Geometry.geo_unrolled",
#             "output/Geometry.msh",
#         ]
#     )(cube, dim=3)
# ```
#
# Geometry that needs to be identified in downstream finite-element
# calculations should be assigned a `physical_group`.
