from collections.abc import Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any

import Mesh
import Part
import numpy as np
import openmc
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepMesh import (
    BRepMesh_IncrementalMesh,
)

# from OCC.Core.Poly import Poly_MeshPurpose
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods
from OCC.Extend.TopologyUtils import TopologyExplorer
from rich.progress import track

from bluemira.base.look_and_feel import bluemira_print
from bluemira.geometry.base import BluemiraGeoT
from bluemira.geometry.compound import BluemiraCompound
from bluemira.geometry.imprint_solids import ImprintableSolid, imprint_solids
from bluemira.mesh.shimwell import vertices_to_h5m

import MeshPart  # isort:skip

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
        self.tags_faceting_tol = self.core.tag_get_handle(
            "FACETING_TOL",
            1,
            types.MB_TYPE_DOUBLE,
            types.MB_TAG_SPARSE | types.MB_TAG_CREAT,
            create_if_missing=True,
        )
        self.tags_global_id = self.core.tag_get_handle(types.GLOBAL_ID_TAG_NAME)

        self._dim_global_ids = {
            0: 0,  # vertex
            1: 0,  # curve
            2: 0,  # surface
            3: 0,  # volume
            4: 0,  # group
        }
        self._global_id_cnt = 0
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

        return ms, dim_id

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

    def tag_name_material(self, entity_set, name):
        self.core.tag_set_data(self.tags_name, entity_set, f"mat:{name}")

    def tag_surf_sense(self, entity_set, sense_data):
        self.core.tag_set_data(self.tags_surf_sense, entity_set, sense_data)

    def tag_faceting_tol(self, entity_set, tol):
        self.core.tag_set_data(self.tags_faceting_tol, entity_set, tol)


class MoabMeshable:
    def __init__(self, id, label: str, occ_faces: list[Any]):
        self.id = id
        self.label = label
        self.occ_faces = occ_faces
        self.volume_set: Any | None = None
        self.volume_set_global_id: int | None = None
        self.group_set: Any | None = None
        self.group_set_global_id: int | None = None

    @classmethod
    def from_imprintable(cls, imp: ImprintableSolid):
        return cls(id(imp.occ_solid), imp.label, imp.imprinted_faces)


class MoabMesher:
    def __init__(
        self,
    ):
        self.mbc = MoabCore()

        self._dbg_imps: list[ImprintableSolid] = []
        self.scale = 100  # m to cm

        self.meshables: list[MoabMeshable] = []
        self.label_to_volume_sets: dict[str, set[Any]] = {}

        self.face_to_meshables: dict[Any, set[MoabMeshable]] = {}

        self.face_to_surf_set = {}
        self.edge_to_curve_set = {}
        self.vertex_to_vertex_set = {}

        self.vertex_to_node = {}

        self.meshed = False

        self.bt = BRep_Tool()

    def add_imprintable(self, imprintable: ImprintableSolid):
        self._dbg_imps.append(imprintable)
        self.meshables.append(MoabMeshable.from_imprintable(imprintable))

    def add_solid(self, solid: BluemiraGeoT, label: str | None = None):
        self.add_imprintable(
            ImprintableSolid.from_bluemira_solid(label or solid.label, solid)
        )

    def add_imprintables(self, imprintables: Iterable[ImprintableSolid]):
        for imp in imprintables:
            self.add_imprintable(imp)

    def _find_or_add_curve_set(self, edge):
        if edge in self.edge_to_curve_set:
            return self.edge_to_curve_set[edge], True
        curve_set = self.mbc.create_tagged_entity_set(1)[0]
        self.edge_to_curve_set[edge] = curve_set
        return curve_set, False

    def _find_or_add_vertex_set(self, vertex):
        if vertex in self.vertex_to_vertex_set:
            return self.vertex_to_vertex_set[vertex], True
        vertex_set = self.mbc.create_tagged_entity_set(0)[0]
        self.vertex_to_vertex_set[vertex] = vertex_set
        return vertex_set, False

    def _find_or_add_vertex_node(self, vertex: tuple[float, float, float]):
        # keeps vertex mapping stable
        vertex = tuple(np.round(vertex, 6) * self.scale)
        # for some horrible reason, the vertex is stored in the wrong order
        # vertex = (
        #     -vertex[2],
        #     vertex[1],
        #     vertex[0],
        # )
        if vertex in self.vertex_to_node:
            return self.vertex_to_node[vertex]
        node_id = self.mbc.create_vertices([vertex])[0]
        self.vertex_to_node[vertex] = node_id
        return node_id

    def _find_or_add_vertex_nodes(self, vertices):
        return [self._find_or_add_vertex_node(vertex) for vertex in vertices]

    def _establish_sets(self):
        for mbl in self.meshables:
            mbl.group_set, mbl.group_set_global_id = self.mbc.create_tagged_entity_set(4)
            self.mbc.tag_name_material(mbl.group_set, mbl.label)
            mbl.volume_set, mbl.volume_set_global_id = self.mbc.create_tagged_entity_set(
                3
            )

            for face in mbl.occ_faces:
                self.face_to_meshables.setdefault(face, set()).add(mbl)
                if face in self.face_to_surf_set:
                    continue
                surf_set = self.mbc.create_tagged_entity_set(2)[0]
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

    def _process_surfaces(self, faceting_tolerance, *, mesh=False):
        bluemira_print("MoabMesher: Processing surfaces.")

        for face, surf_set in track(self.face_to_surf_set.items()):
            if mesh:
                msh: Mesh.Mesh = MeshPart.meshFromShape(
                    Part.__fromPythonOCC__(face),
                    LinearDeflection=0.1,
                    AngularDeflection=0.5,
                    Segments=True,
                    MaxLength=0.1,
                    AllowQuad=0,
                )
                mesh_pts = [(p.x, p.y, p.z) for p in msh.Points]

                nodes = self._find_or_add_vertex_nodes(mesh_pts)
                moab_triangles = [
                    self.mbc.create_element(
                        types.MBTRI,
                        (
                            nodes[facet.PointIndices[0]],
                            nodes[facet.PointIndices[1]],
                            nodes[facet.PointIndices[2]],
                        ),
                    )
                    for facet in msh.Facets
                ]
            else:
                BRepMesh_IncrementalMesh(face, faceting_tolerance, False, 0.5, True)

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
                    tris.Value(i).Get()
                    for i in range(1, triangulation.NbTriangles() + 1)
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

            # self._populate_surface_curve_sets(
            #     face, surf_set, triangulation, location, nodes
            # )

    def _process_volumes(self):
        bluemira_print("MoabMesher: Processing volumes.")

        for mbl in self.meshables:
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

                self.mbc.define_parent_child(mbl.volume_set, surf_set)
            self.mbc.add_entity(mbl.group_set, mbl.volume_set)

    def _create_file_set(self, faceting_tolerance):
        file_set = self.mbc.core.create_meshset(types.MBENTITYSET)
        self.mbc.tag_faceting_tol(file_set, faceting_tolerance)

        all_sets = self.mbc.core.get_entities_by_handle(0)
        self.mbc.add_entities(file_set, all_sets.to_array())

        """
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


        """

    def p_new(
        self,
        filename,
        faceting_tolerance,
        implicit_complement_material_tag=None,
    ):
        print("p_new")
        vert_to_vert_idx_map = {}
        face_to_id_map = {}
        vertices: list[tuple[float, float, float]] = []
        triangles_by_solid_by_face: dict[int, dict[int, tuple[int, int, int]]] = {}
        mats = []

        volm_id = 0

        def _get_or_add_vert(vert) -> int:
            vert = tuple(np.round(vert, 6) * self.scale)
            vert = (
                -vert[2],
                vert[1],
                vert[0],
            )
            if vert in vert_to_vert_idx_map:
                return vert_to_vert_idx_map[vert]
            idx = len(vertices)
            vertices.append(list(vert))
            vert_to_vert_idx_map[vert] = idx
            return idx

        def _face_to_id(face):
            if face in face_to_id_map:
                return face_to_id_map[face]
            face_to_id_map[face] = len(face_to_id_map) + 1
            return face_to_id_map[face]

        for mbl in self.meshables:
            tris_by_face = {}
            for f in mbl.occ_faces:
                face_verts_to_verts = []

                msh: Mesh.Mesh = MeshPart.meshFromShape(
                    Part.__fromPythonOCC__(f),
                    LinearDeflection=faceting_tolerance,
                    AngularDeflection=0.5,
                    Segments=False,
                    MaxLength=0.1,
                    AllowQuad=0,
                )
                mesh_pts = [_get_or_add_vert((p.x, p.y, p.z)) for p in msh.Points]
                face_verts_to_verts.append(mesh_pts)
                tri_idxs = [
                    (
                        mesh_pts[facet.PointIndices[0]],
                        mesh_pts[facet.PointIndices[1]],
                        mesh_pts[facet.PointIndices[2]],
                    )
                    for facet in msh.Facets
                ]

                # aMeshParams = IMeshTools_Parameters()
                # aMeshParams.Deflection = faceting_tolerance
                # aMeshParams.MeshAlgo = IMeshTools_MeshAlgoType_Delabella

                # BRepMesh_IncrementalMesh(f, aMeshParams)
                # # BRepMesh_IncrementalMesh(f, faceting_tolerance, False, 0.5, True)

                # location = TopLoc_Location()
                # triangulation = self.bt.Triangulation(f, location)
                # for i in range(1, triangulation.NbNodes() + 1):
                #     transform = location.Transformation()
                #     vert_node_xyz = triangulation.Node(i).XYZ()
                #     transform.Transforms(vert_node_xyz)
                #     vert_idx = _get_or_add_vert(vert_node_xyz.Coord())
                #     face_verts_to_verts.append(vert_idx)

                # tris = triangulation.Triangles()
                # tris_vert_idxs = [
                #     tris.Value(i).Get()
                #     for i in range(1, triangulation.NbTriangles() + 1)
                # ]

                # tri_idxs = [
                #     (
                #         face_verts_to_verts[tri_vert_idxs[0] - 1],
                #         face_verts_to_verts[tri_vert_idxs[1] - 1],
                #         face_verts_to_verts[tri_vert_idxs[2] - 1],
                #     )
                #     for tri_vert_idxs in tris_vert_idxs
                # ]

                fid = _face_to_id(f)
                tris_by_face[fid] = tri_idxs

            vid = volm_id + 1
            volm_id = vid

            triangles_by_solid_by_face[vid] = tris_by_face
            mats.append(mbl.label)
        print("vertices_to_h5m")
        vertices_to_h5m(
            vertices,
            triangles_by_solid_by_face,
            mats,
            faceting_tolerance=faceting_tolerance,
            filename=filename,
            implicit_complement_material_tag=implicit_complement_material_tag,
        )
        self.meshed = True

    def perform(self, faceting_tolerance=0.1, mesh=False):
        if self.meshed:
            return
        bluemira_print("MoabMesher: Meshing begin.")

        self._establish_sets()

        meshable_labels = {mbl.label for mbl in self.meshables}
        bluemira_print(
            f"Meshing {len(self.meshables)} volumes, "
            f"with {len(self.face_to_surf_set)} surfaces\n"
            f"Labels: {meshable_labels}\n"
            f"faceting_tolerance: {faceting_tolerance}"
        )

        self._process_surfaces(faceting_tolerance, mesh=mesh)
        self._process_volumes()
        self._create_file_set(faceting_tolerance)

        bluemira_print("MoabMesher: Meshing complete.")

        self.meshed = True

    def write_file(self, file_name, *, include_vtk=False):
        if not self.meshed:
            raise RuntimeError("Meshing has not been performed yet.")

        bluemira_print(f"MoabMesher: Writing file to {file_name}.h5m")
        self.mbc.core.write_file(f"{file_name}.h5m")
        if include_vtk:
            bluemira_print(f"MoabMesher: Writing file to {file_name}.vtk")
            self.mbc.core.write_file(f"{file_name}.vtk")

        print(self._get_volumes_and_materials_from_h5m(f"{file_name}.h5m"))

    def _get_volumes_and_materials_from_h5m(self, filename: str) -> dict:
        """Reads in a DAGMC h5m file and uses PyMoab to find the volume ids with
        their associated material tags.

        Arguments:
            filename: the filename of the DAGMC h5m file

        Returns
        -------
            A dictionary of volume ids and material tags
        """
        mbcore = core.Core()
        mbcore.load_file(filename)
        category_tag = mbcore.tag_get_handle(types.CATEGORY_TAG_NAME)
        group_category = ["Group"]
        group_ents = mbcore.get_entities_by_type_and_tag(
            0, types.MBENTITYSET, category_tag, group_category
        )
        name_tag = mbcore.tag_get_handle(types.NAME_TAG_NAME)
        id_tag = mbcore.tag_get_handle(types.GLOBAL_ID_TAG_NAME)
        vol_mat = {}
        for group_ent in group_ents:
            group_name = mbcore.tag_get_data(name_tag, group_ent)[0][0]
            # confirm that this is a material!
            if group_name.startswith("mat:"):
                vols = mbcore.get_entities_by_type(group_ent, types.MBENTITYSET)
                for vol in vols:
                    id = mbcore.tag_get_data(id_tag, vol)[0][0].item()
                    vol_mat[id] = group_name
        return vol_mat


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
    for shape, name in zip(shapes, names, strict=True):
        if isinstance(shape, BluemiraCompound):
            solids = shape.solids
            imps = imprint_solids(solids, [name] * len(solids))
            mesher.add_imprintables(imps)
        else:
            mesher.add_solid(shape, name)

    # mesher.p_new(filename, faceting_tolerance)
    mesher.perform(faceting_tolerance)
    mesher.write_file(filename, include_vtk=True)


if __name__ == "__main__":
    from bluemira.geometry.base import BluemiraGeoT
    from bluemira.geometry.face import BluemiraFace
    from bluemira.geometry.tools import (
        extrude_shape,
        make_circle,
        make_polygon,
    )

    # Create a box
    box_a = BluemiraFace(
        make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 0]], closed=True
        ),
    )
    box_a = extrude_shape(box_a, [0, 0, 1])
    box_b = deepcopy(box_a)
    box_b.translate([-0.6, 1, -0.6])
    box_c = deepcopy(box_a)
    box_c.translate([0.6, 1, 0.6])
    circ = BluemiraFace(make_circle(0.1, (0.3, 0.3, 1)), label="circ")
    circ = extrude_shape(circ, [0, 0, 0.5])

    pre_imps = [box_a, circ]
    names = ["box_a", "circ"]
    # show_cad(pre_imps)
    imps = imprint_solids(pre_imps, names)

    filename = "dagmc"

    mesher = MoabMesher()
    mesher.add_imprintables(imps)
    # mesher.p_new(filename, 0.001, "air")
    mesher.perform(0.001, mesh=True)
    mesher.write_file(filename, include_vtk=True)
    print(mesher._get_volumes_and_materials_from_h5m(f"{filename}.h5m"))

    mats = [openmc.Material(name=f"{n}") for n in names]
    for mat in mats:
        mat.add_nuclide("Fe56", 1)
        mat.set_density("g/cm3", 1)

    names.append("air")
    air_mat = openmc.Material(name="air")
    air_mat.add_element("N", 0.7)
    air_mat.add_element("O", 0.3)
    air_mat.set_density("g/cm3", 0.001)
    mats.append(air_mat)

    my_materials = openmc.Materials(mats)

    mat_filters = [openmc.MaterialFilter(mat) for mat in mats]

    dag_univ = openmc.DAGMCUniverse(f"{filename}.h5m").bounded_universe()
    geometry = openmc.Geometry(dag_univ)

    my_source = openmc.IndependentSource()

    center_of_geometry = (
        (dag_univ.bounding_box[0][0] + dag_univ.bounding_box[1][0]) / 2,
        (dag_univ.bounding_box[0][1] + dag_univ.bounding_box[1][1]) / 2,
        (dag_univ.bounding_box[0][2] + dag_univ.bounding_box[1][2]) / 2,
    )
    # sets the location of the source which is not on a vertex
    center_of_geometry_nudged = (
        center_of_geometry[0] + 0.1,
        center_of_geometry[1] + 0.1,
        center_of_geometry[2] + 0.1,
    )

    my_source.space = openmc.stats.Point(center_of_geometry_nudged)
    # sets the direction to isotropic
    my_source.angle = openmc.stats.Isotropic()
    # sets the energy distribution to 100% 14MeV neutrons
    my_source.energy = openmc.stats.Discrete([14e6], [1])

    # specifies the simulation computational intensity
    my_settings = openmc.Settings()
    my_settings.batches = 10
    my_settings.particles = 10000
    my_settings.inactive = 0
    my_settings.run_mode = "fixed source"
    my_settings.source = my_source
    my_settings.photon_transport = False
    # my_settings.output = {"path": (Path(__file__).parent / "omc_run_output").as_posix()}

    # cell_tally = openmc.Tally(name="flux")
    # cell_tally.scores = ["flux"]

    # # groups the two tallies
    # tallies = openmc.Tallies([cell_tally])

    tallies = []
    for i, m_f in enumerate(mat_filters):
        tally = openmc.Tally(name=f"{names[i]}_flux_tally")
        tally.filters = [m_f]
        tally.scores = ["flux"]
        tallies.append(tally)

    my_tallies = openmc.Tallies(tallies)

    # my_settings = openmc.Settings()
    # my_settings.batches = 2
    # my_settings.inactive = 0
    # my_settings.particles = 5000
    # my_settings.run_mode = "fixed source"

    # # Create a DT point source
    # my_source = openmc.Source()
    # my_source.space = openmc.stats.Point((0, 0, 0))
    # my_source.angle = openmc.stats.Isotropic()
    # my_source.energy = openmc.stats.Discrete([2e6], [1])
    # my_settings.source = my_source

    # universe = openmc.DAGMCUniverse(f"{filename}.h5m").bounded_universe()
    # geometry = openmc.Geometry(universe)

    model = openmc.Model(geometry, my_materials, my_settings, my_tallies)

    output_file_from_cad = model.run()
    with openmc.StatePoint(output_file_from_cad) as sp_from_cad:
        results = [sp_from_cad.get_tally(name=f"{n}_flux_tally") for n in names]

    for n, res in zip(names, results, strict=True):
        print(f"{n} mean {res.mean} std dev {res.std_dev}")
