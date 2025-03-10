from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods
from OCC.Extend.TopologyUtils import TopologyExplorer
from rich.progress import track

from bluemira.display.displayer import show_cad
from bluemira.geometry.base import BluemiraGeoT
from bluemira.geometry.imprint_solids import ImprintableSolid, imprint_solids

try:
    from pymoab import core, types

    pymoab_available = True
except ImportError:
    pymoab_available = False


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
        #   result = MBI()->tag_get_handle("FACETING_TOL", 1, moab::MB_TYPE_DOUBLE,
        #                          faceting_tol_tag,
        #                          moab::MB_TAG_SPARSE | moab::MB_TAG_CREAT);
        self.tags_faceting_tol = self.core.tag_get_handle(
            "FACETING_TOL",
            1,
            types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE | types.MB_TAG_CREAT,
        )
        self.tags_global_id = self.core.tag_get_handle(types.GLOBAL_ID_TAG_NAME)

        self._dim_global_ids = {
            0: 0,  # vertex
            1: 0,  # curve
            2: 0,  # surface
            3: 0,  # volume
            4: 0,  # group
        }
        self._dim_categories = {
            0: "Vertex\0",
            1: "Curve\0",
            2: "Surface\0",
            3: "Volume\0",
            4: "Group\0",
        }

    def create_tagged_entity_set(self, dim: int):
        ms = self.core.create_meshset(
            types.MESHSET_ORDERED if dim == 1 else types.MESHSET_SET
        )

        dim_id = self._dim_global_ids[dim] + 1
        self._dim_global_ids[dim] = dim_id

        self.core.tag_set_data(self.tags_global_id, ms, dim_id)
        self.core.tag_set_data(self.tags_category, ms, self._dim_categories[dim])
        if dim <= 3:
            self.core.tag_set_data(self.tags_geom_dimension, ms, dim)

        return ms

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

    def tag_faceting_tol(self, entity_set, tol):
        self.core.tag_set_data(self.tags_faceting_tol, entity_set, tol)


class MoabMeshable:
    def __init__(self, id: int, label: str, occ_faces: list[Any]):
        self.id = id
        self.label = label
        self.occ_faces = occ_faces
        self.volume_set: Any | None = None
        self.group_set: Any | None = None
        self._sets_established = False

    @classmethod
    def from_imprintable(cls, imp: ImprintableSolid):
        return cls(id(imp.occ_solid), imp.label, imp.imprinted_faces)

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


class MoabMesher:
    def __init__(
        self,
    ):
        self.mbc = MoabCore()

        self._dbg_imps = []
        self.meshables: set[MoabMeshable] = set()
        self.face_to_meshables: dict[Any, set[MoabMeshable]] = {}
        self.processed_faces = set()

        self.label_to_volume_sets: dict[str, set[Any]] = {}
        self.face_to_surf_set = {}
        self.edge_to_curve_set = {}
        self.vertex_to_vertex_set = {}

        self.vertex_to_node = {}

        self.meshed = False

        self.bt = BRep_Tool()

    def add_imprintable(self, imprintable: ImprintableSolid):
        self._dbg_imps.append(imprintable)
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

    def _init_meshable_sets(self):
        for mbl in self.meshables:
            mbl.establish_sets(self.mbc)

    def _find_or_add_curve_set(self, edge):
        if edge in self.edge_to_curve_set:
            return self.edge_to_curve_set[edge], True
        curve_set = self.mbc.create_tagged_entity_set(1)
        self.edge_to_curve_set[edge] = curve_set
        return curve_set, False

    def _find_or_add_vertex_set(self, vertex):
        if vertex in self.vertex_to_vertex_set:
            return self.vertex_to_vertex_set[vertex], True
        vertex_set = self.mbc.create_tagged_entity_set(0)
        self.vertex_to_vertex_set[vertex] = vertex_set
        return vertex_set, False

    def _find_or_add_vertex_node(self, vertex: tuple[float, float, float]):
        # keeps vertex mapping stable
        vertex = tuple(np.round(vertex, 6))
        if vertex in self.vertex_to_node:
            return self.vertex_to_node[vertex]
        node_id = self.mbc.create_vertices([vertex])[0]
        self.vertex_to_node[vertex] = node_id
        return node_id

    def _find_or_add_vertex_nodes(self, vertices):
        return [self._find_or_add_vertex_node(vertex) for vertex in vertices]

    def _establish_volume_sets(self):
        for mbl in self.meshables:
            volm_set = self.mbc.create_tagged_entity_set(3)
            mbl.volume_set = volm_set
            self.label_to_volume_sets.setdefault(mbl.label, set()).add(volm_set)

    def _establish_surface_sets(self):
        for face in self.face_to_meshables:
            # shouldn't be possible, given we're iterating over the keys
            if face in self.face_to_surf_set:
                continue
            # TODO: create enum of the magic numbers
            surf_set = self.mbc.create_tagged_entity_set(2)
            self.face_to_surf_set[face] = surf_set

    def _populate_surface_curve_sets(
        self, face, surf_set, triangulation, location, nodes
    ):
        ex = TopologyExplorer(face)
        for edge in ex.edges():
            curve_set, existed = self._find_or_add_curve_set(edge)
            if not existed:
                edge_mesh_vert_idxs = (
                    self.bt.PolygonOnTriangulation(edge, triangulation, location)
                    .Nodes()
                    .to_numpy_array()
                )

                # remove the last index if it's closed (circular)
                closed = edge_mesh_vert_idxs[0] == edge_mesh_vert_idxs[-1]
                if closed:
                    edge_mesh_vert_idxs = edge_mesh_vert_idxs[:-1]

                # subtract 1 to make it 0-based index
                curve_nodes = [nodes[i - 1] for i in edge_mesh_vert_idxs]
                curve_edges = [
                    self.mbc.create_element(
                        types.MBEDGE, (curve_nodes[i], curve_nodes[i + 1])
                    )
                    for i in range(len(curve_nodes) - 1)
                ]
                if closed:
                    curve_edges.append(
                        self.mbc.create_element(
                            types.MBEDGE, (curve_nodes[-1], curve_nodes[0])
                        )
                    )

                self.mbc.add_entities(curve_set, curve_edges)
                self.mbc.add_entities(curve_set, curve_nodes)

                self._populate_curve_vertex_sets(edge, curve_set)

            self.mbc.define_parent_child(surf_set, curve_set)

    def _populate_curve_vertex_sets(self, edge, curve_set):
        ex = TopologyExplorer(edge)
        for edge_vertex in ex.vertices():
            vertex_set, existed = self._find_or_add_vertex_set(edge_vertex)
            if not existed:
                vtx = self.bt.Pnt(topods.Vertex(edge_vertex)).Coord()
                node = self._find_or_add_vertex_node(vtx)
                self.mbc.add_entity(vertex_set, node)
            self.mbc.define_parent_child(curve_set, vertex_set)

    def _process_surfaces(self, faceting_tolerance):
        self._establish_surface_sets()

        for face, surf_set in self.face_to_surf_set.items():
            BRepMesh_IncrementalMesh(face, faceting_tolerance)

            location = TopLoc_Location()
            triangulation = self.bt.Triangulation(face, location)

            vertices = []
            for i in range(1, triangulation.NbNodes() + 1):
                transform = location.Transformation()
                vert_node_xyz = triangulation.Node(i).XYZ()
                transform.Transforms(vert_node_xyz)
                vertices.append(vert_node_xyz.Coord())

            tris = triangulation.Triangles()
            tris_vert_idxs = [
                tris.Value(i).Get() for i in range(1, triangulation.NbTriangles() + 1)
            ]

            nodes = self._find_or_add_vertex_nodes(vertices)
            moab_triangles = [
                self.mbc.create_element(
                    types.MBTRI,
                    (
                        nodes[tri_vert_idxs[0] - 1],
                        nodes[tri_vert_idxs[1] - 1],
                        nodes[tri_vert_idxs[2] - 1],
                    ),
                )
                for tri_vert_idxs in tris_vert_idxs
            ]

            self.mbc.add_entities(surf_set, nodes)
            self.mbc.add_entities(surf_set, moab_triangles)

            self._populate_surface_curve_sets(
                face, surf_set, triangulation, location, nodes
            )

    def _process_volumes(self):
        self._establish_volume_sets()

        for mbl in self.meshables:
            meshable_volm_set = mbl.volume_set

            for face in mbl.occ_faces:
                surf_set = self.face_to_surf_set[face]

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

                self.mbc.define_parent_child(meshable_volm_set, surf_set)

    def _process_groups(self):
        for label, volm_sets in self.label_to_volume_sets.items():
            group_set = self.mbc.create_tagged_entity_set(4)
            self.mbc.tag_name_material(group_set, label)
            self.mbc.add_entities(group_set, volm_sets)

    def _create_file_set(self, faceting_tolerance):
        file_set = self.mbc.create_meshset(types.MBENTITYSET)
        self.mbc.tag_faceting_tol(file_set, faceting_tolerance)

        all_sets = self.mbc.core.get_entities_by_handle(0)
        self.mbc.add_entities(file_set, all_sets.to_array())

    def perform(self, faceting_tolerance=0.1):
        if self.meshed:
            return

        self._process_surfaces(faceting_tolerance)
        self._process_volumes()
        self._process_groups()
        self._create_file_set(faceting_tolerance)

        self.meshed = True

    def to_file(self, file_name, *, include_vtk=False):
        if not self.meshed:
            raise RuntimeError("Meshing has not been performed yet.")

        self.mbc.core.write_file(f"{file_name}.h5m")
        if include_vtk:
            self.mbc.core.write_file(f"{file_name}.vtk")


def save_cad_to_dagmc_model(
    shapes: Iterable[BluemiraGeoT],
    names: list[str],
    filename: Path,
    *,
    faceting_tolerance=0.1,
):
    """Converts the shapes with their associated names to a dagmc file using PyMOAB."""
    mesher = MoabMesher()

    # do a per compound imprint for now.
    # In the future, one should extract all solids then do the imprint on _all_ of them
    for shape in track(shapes):
        if isinstance(shape, BluemiraCompound):
            imps = imprint_solids(shape.solids)
            mesher.add_imprintables(imps)
        else:
            mesher.add_solid(shape)

    mesher.perform(faceting_tolerance)
    mesher.to_file(filename)


if __name__ == "__main__":
    from bluemira.geometry.base import BluemiraGeoT
    from bluemira.geometry.compound import BluemiraCompound
    from bluemira.geometry.face import BluemiraFace
    from bluemira.geometry.tools import (
        extrude_shape,
        make_circle,
        make_polygon,
    )

    # Create a box
    box_a = BluemiraFace(
        make_polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]])
    )
    box_a = extrude_shape(box_a, [0, 0, 1])
    box_b = deepcopy(box_a)
    box_b.translate([-0.6, 1, -0.6])
    box_c = deepcopy(box_a)
    box_c.translate([0.6, 1, 0.6])
    circ = BluemiraFace(make_circle(0.5, (0, 0, 1)), label="circ")
    circ = extrude_shape(circ, [0, 0, 1])

    pre_imps = [circ, box_a, box_b, box_c]
    show_cad(pre_imps)
    imps = imprint_solids(pre_imps)

    mesher = MoabMesher()
    mesher.add_imprintables(imps)
    mesher.perform()
