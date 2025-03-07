from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from rich.progress import track

from bluemira.geometry.base import BluemiraGeoT
from bluemira.geometry.imprint_solids import ImprintableSolid, imprint_solids

try:
    from pymoab import core, types

    pymoab_available = True
except ImportError:
    pymoab_available = False

import Mesh
import MeshPart

import Part  # isort: skip


class MoabCore:
    def __init__(self):
        if not pymoab_available:
            raise ImportError("PyMOAB is required to convert CAD to DAGMC.")
        self.core = core.Core()

        self.tags_surf_sense = self.core.tag_get_handle(
            "GEOM_SENSE_2",
            2,
            types.MB_TYPE_HANDLE,
            types.MB_TAG_SPARSE,
            create_if_missing=True,
        )
        self.tags_category = self.core.tag_get_handle(
            types.CATEGORY_TAG_NAME,
            types.CATEGORY_TAG_SIZE,
            types.MB_TYPE_OPAQUE,
            types.MB_TAG_SPARSE,
            create_if_missing=True,
        )
        self.tags_name = self.core.tag_get_handle(
            types.NAME_TAG_NAME,
            types.NAME_TAG_SIZE,
            types.MB_TYPE_OPAQUE,
            types.MB_TAG_SPARSE,
            create_if_missing=True,
        )
        self.tags_geom_dimension = self.core.tag_get_handle(
            types.GEOM_DIMENSION_TAG_NAME,
            1,
            types.MB_TYPE_INTEGER,
            types.MB_TAG_DENSE,
            create_if_missing=True,
        )
        self.tags_global_id = self.core.tag_get_handle(types.GLOBAL_ID_TAG_NAME)

    def add_entity(self, entity_set, entity):
        self.core.add_entity(entity_set, entity)

    def add_entities(self, entity_set, entities: list):
        self.core.add_entities(entity_set, entities)

    def create_meshset(self):
        return self.core.create_meshset()

    def create_vertices(self, vertices):
        return self.core.create_vertices(vertices)

    def create_element(self, element_type, vertex_indices):
        return self.core.create_element(element_type, vertex_indices)

    def define_parent_child(self, parent_set, child_set):
        self.core.add_parent_child(parent_set, child_set)

    def tag_global_id(self, entity_set, id_no):
        self.core.tag_set_data(self.tags_global_id, entity_set, id_no)

    def tag_geom_dimension(self, entity_set, dim):
        self.core.tag_set_data(self.tags_geom_dimension, entity_set, dim)

    def tag_category(self, entity_set, category):
        self.core.tag_set_data(self.tags_category, entity_set, category)

    def tag_name_material(self, entity_set, name):
        self.core.tag_set_data(self.tags_name, entity_set, f"mat:{name}")

    def tag_surf_sense(self, entity_set, sense_data):
        self.core.tag_set_data(self.tags_surf_sense, entity_set, sense_data)


class MoabMeshable:
    def __init__(self, id: int, label: str, occ_faces: list[Any]):
        self.id = id
        self.label = label
        self.occ_faces = occ_faces
        self.volume_set: Any | None = None
        self.group_set: Any | None = None
        self._sets_established = False

    def establish_sets(self, mbc: MoabCore):
        group_set = mbc.create_meshset()
        volm_set = mbc.create_meshset()

        mbc.tag_global_id(volm_set, self.id)

        mbc.tag_category(group_set, "Group")
        mbc.tag_category(volm_set, "Volume")

        mbc.tag_geom_dimension(group_set, 4)
        mbc.tag_geom_dimension(volm_set, 3)

        mbc.tag_name_material(group_set, self.label)

        mbc.add_entity(group_set, volm_set)

        self.volume_set = volm_set
        self.group_set = group_set

        self._sets_established = True

    @classmethod
    def from_imprintable(cls, imp: ImprintableSolid):
        return cls(id(imp.occ_solid), imp.label, imp.imprinted_faces)


class MoabMesher:
    def __init__(
        self,
    ):
        self.mbc = MoabCore()

        self.meshables: set[MoabMeshable] = set()
        self.face_to_meshables: dict[Any, set[MoabMeshable]] = {}
        self.processed_faces = set()

        self.meshed = False

    def add_imprintable(self, imprintable: ImprintableSolid):
        mbl = MoabMeshable.from_imprintable(imprintable)
        self.meshables.add(mbl)
        for face in mbl.occ_faces:
            self.face_to_meshables.setdefault(face, set()).add(mbl)

    def add_solid(self, solid: BluemiraGeoT, label: str | None = None):
        self.add_imprintable(
            ImprintableSolid.from_bluemira_solid(label or solid.label, solid)
        )

    def add_imprintables(self, imprintables: Iterable[ImprintableSolid]):
        for imp in imprintables:
            self.add_imprintable(imp)

    def _establish_meshable_sets(self):
        for mbl in self.meshables:
            mbl.establish_sets(self.mbc)

    def mesh_to_file(
        self,
        faceting_tolerance=0.001,
    ):
        if self.meshed:
            return

        self._establish_meshable_sets()

        for mbl in self.meshables:
            meshable_volm_set = mbl.volume_set

            for face in mbl.occ_faces:
                # create the surface set

                # if it's been created before,
                # create a new one but don't tag it
                # (only define the parent-child relationship and inverse sense)
                if face in self.processed_faces:
                    surf_set = self.mbc.create_meshset()
                else:
                    surf_set = self.mbc.create_meshset()
                    self.mbc.tag_global_id(surf_set, id(face))
                    self.mbc.tag_category(surf_set, "Surface")
                    self.mbc.tag_geom_dimension(surf_set, 2)
                self.processed_faces.add(face)

                self.mbc.define_parent_child(meshable_volm_set, surf_set)

                # define the surface sense
                linked_meshables = self.face_to_meshables[face]

                other_mbl = None
                for linked_mbl in linked_meshables:
                    if linked_mbl != mbl:
                        other_mbl = linked_mbl
                        break

                sense_data_volm_sets = [mbl.volume_set]
                if other_mbl:
                    sense_data_volm_sets.append(other_mbl.volume_set)
                else:
                    sense_data_volm_sets.append(0)
                sense_data_volm_sets = np.array(sense_data_volm_sets, dtype="uint64")

                self.mbc.tag_surf_sense(surf_set, sense_data_volm_sets)

                # mesh the face and associate it with the surface set
                """             AllowQuad=0)

Args:
    Shape (required, topology) - TopoShape to create mesh of.
    LinearDeflection (required, float)
    AngularDeflection (optional, float)
    Segments (optional, boolean)
    GroupColors (optional, list of (Red, Green, Blue) tuples)
    MaxLength (required, float)
    MaxArea (required, float)
    LocalLength (required, float)
    Deflection (required, float)
    MinLength (required, float)
    Fineness (required, integer)
    SecondOrder (optional, integer boolean)
    Optimize (optional, integer boolean)
    AllowQuad (optional, integer boolean)
    GrowthRate (optional, float)
    SegPerEdge (optional, float)
    SegPerRadius (optional, float)"""

                msh: Mesh.Mesh = MeshPart.meshFromShape(
                    Part.__fromPythonOCC__(face),
                    LinearDeflection=0.1,
                    AngularDeflection=0.5,
                    Segments=True,
                    MaxLength=0.1,
                    AllowQuad=0,
                )
                mesh_pts = [(p.x, p.y, p.z) for p in msh.Points]

                # This creates the vertices in moab (globally)
                # Use this to map the local vertex index to the global vertex index
                verts = self.mbc.create_vertices(mesh_pts)
                self.mbc.add_entity(surf_set, verts)

                # for facet in msh.Facets:
                #     tri = (
                #         verts[facet.PointIndices[0]],
                #         verts[facet.PointIndices[1]],
                #         verts[facet.PointIndices[2]],
                #     )

                #     moab_triangle = self.mbc.create_element(types.MBTRI, tri)
                #     self.mbc.add_entity(surf_set, moab_triangle)

                moab_triangles = [
                    self.mbc.create_element(
                        types.MBTRI,
                        (
                            verts[facet.PointIndices[0]],
                            verts[facet.PointIndices[1]],
                            verts[facet.PointIndices[2]],
                        ),
                    )
                    for facet in msh.Facets
                ]
                self.mbc.add_entities(surf_set, moab_triangles)
        all_sets = self.mbc.core.get_entities_by_handle(0)

        file_set = self.mbc.core.create_meshset()

        self.mbc.add_entities(file_set, all_sets)

        self.mbc.core.write_file("dagmc.h5m")
        self.mbc.core.write_file("dagmc.vtk")

        self.meshed = True


def save_cad_to_dagmc_model(
    shapes: Iterable[BluemiraGeoT],
    names: list[str],
    filename: Path,
    *,
    faceting_tolerance=0.001,
):
    """Converts the shapes with their associated names to a dagmc file using PyMOAB."""
    imprinted_solids = []

    # do a per compound imprint for now.
    # In the future, one should extract all solids then do the imprint on _all_ of them
    for shape in track(shapes):
        if isinstance(shape, BluemiraCompound):
            imprinted_solids.extend(imprint_solids(shape.solids))
        else:
            imprinted_solids.append(shape)

    # for each solid, extract the face
    # look up the face to see if it exists
    # if it does, associate the face with the solid (this one)
    # if it doesn't, create the face and associate it with the solid
    # it's mesh (faceting)


if __name__ == "__main__":
    from bluemira.geometry.base import BluemiraGeoT
    from bluemira.geometry.compound import BluemiraCompound
    from bluemira.geometry.face import BluemiraFace
    from bluemira.geometry.tools import (
        extrude_shape,
        make_polygon,
    )

    # Create a box
    box_a = BluemiraFace(
        make_polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])
    )
    box_a = extrude_shape(box_a, [0, 0, 1])
    box_b = deepcopy(box_a)
    box_b.translate([-0.6, -0.6, 1])
    box_c = deepcopy(box_a)
    box_c.translate([0.6, 0.6, 1])

    pre_imps = [box_a, box_b, box_c]
    # show_cad(pre_imps)
    imps = imprint_solids(pre_imps)

    mesher = MoabMesher()
    mesher.add_imprintables(imps)
    mesher.mesh_to_file()
