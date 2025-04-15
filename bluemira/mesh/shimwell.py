import numpy as np
from pymoab import core, types
from rich.progress import track


def define_moab_core_and_tags() -> tuple[core.Core, dict]:
    """Creates a MOAB Core instance which can be built up by adding sets of
    triangles to the instance

    Returns
    -------
        (pymoab Core): A pymoab.core.Core() instance
        (pymoab tag_handle): A pymoab.core.tag_get_handle() instance
    """
    # create pymoab instance
    moab_core = core.Core()

    tags = dict()

    sense_tag_name = "GEOM_SENSE_2"
    sense_tag_size = 2
    tags["surf_sense"] = moab_core.tag_get_handle(
        sense_tag_name,
        sense_tag_size,
        types.MB_TYPE_HANDLE,
        types.MB_TAG_SPARSE,
        create_if_missing=True,
    )

    tags["category"] = moab_core.tag_get_handle(
        types.CATEGORY_TAG_NAME,
        types.CATEGORY_TAG_SIZE,
        types.MB_TYPE_OPAQUE,
        types.MB_TAG_SPARSE,
        create_if_missing=True,
    )

    tags["name"] = moab_core.tag_get_handle(
        types.NAME_TAG_NAME,
        types.NAME_TAG_SIZE,
        types.MB_TYPE_OPAQUE,
        types.MB_TAG_SPARSE,
        create_if_missing=True,
    )

    tags["geom_dimension"] = moab_core.tag_get_handle(
        types.GEOM_DIMENSION_TAG_NAME,
        1,
        types.MB_TYPE_INTEGER,
        types.MB_TAG_DENSE,
        create_if_missing=True,
    )

    # Global ID is a default tag, just need the name to retrieve
    tags["global_id"] = moab_core.tag_get_handle(types.GLOBAL_ID_TAG_NAME)

    return moab_core, tags


def vertices_to_h5m(
    vertices: list[tuple[float, float, float]],
    triangles_by_solid_by_face: list[list[tuple[int, int, int]]],
    material_tags: list[str],
    faceting_tolerance: float,
    filename: str = "dagmc.h5m",
    implicit_complement_material_tag: str | None = None,
):
    """Converts vertices and triangle sets into a tagged h5m file compatible
    with DAGMC enabled neutronics simulations

    Args:
        vertices:
        triangles:
        material_tags:
        h5m_filename:
        implicit_complement_material_tag:
    """
    if len(material_tags) != len(triangles_by_solid_by_face):
        msg = f"The number of material_tags provided is {len(material_tags)} and the number of sets of triangles is {len(triangles_by_solid_by_face)}. You must provide one material_tag for every triangle set"
        raise ValueError(msg)

    # limited attribute checking to see if user passed in a list of CadQuery vectors
    if (
        hasattr(vertices[0], "x")
        and hasattr(vertices[0], "y")
        and hasattr(vertices[0], "z")
    ):
        vertices_floats = []
        for vert in vertices:
            vertices_floats.append((vert.x, vert.y, vert.z))
    else:
        vertices_floats = vertices

    face_ids_with_solid_ids = {}
    for volume_id, triangles_on_each_face in triangles_by_solid_by_face.items():
        for face_id, triangles_on_face in triangles_on_each_face.items():
            if face_id in face_ids_with_solid_ids:
                face_ids_with_solid_ids[face_id].append(volume_id)
            else:
                face_ids_with_solid_ids[face_id] = [volume_id]

    moab_core, tags = define_moab_core_and_tags()

    volume_sets_by_solid_id = {}
    for material_tag, (volume_id, triangles_on_each_face) in zip(
        material_tags, triangles_by_solid_by_face.items(), strict=False
    ):
        volume_set = moab_core.create_meshset()
        volume_sets_by_solid_id[volume_id] = volume_set

    added_surfaces_ids = {}
    for material_tag, (volume_id, triangles_on_each_face) in track(
        zip(material_tags, triangles_by_solid_by_face.items(), strict=False)
    ):
        volume_set = volume_sets_by_solid_id[volume_id]

        moab_core.tag_set_data(tags["global_id"], volume_set, volume_id)
        moab_core.tag_set_data(tags["geom_dimension"], volume_set, 3)
        moab_core.tag_set_data(tags["category"], volume_set, "Volume")

        group_set = moab_core.create_meshset()
        moab_core.tag_set_data(tags["category"], group_set, "Group")
        moab_core.tag_set_data(tags["name"], group_set, f"mat:{material_tag}")
        moab_core.tag_set_data(tags["global_id"], group_set, volume_id)
        # moab_core.tag_set_data(tags["geom_dimension"], group_set, 4)

        for face_id, triangles_on_face in triangles_on_each_face.items():
            if face_id not in added_surfaces_ids:
                surface_set = moab_core.create_meshset()
                moab_core.tag_set_data(tags["global_id"], surface_set, face_id)
                moab_core.tag_set_data(tags["geom_dimension"], surface_set, 2)
                moab_core.tag_set_data(tags["category"], surface_set, "Surface")

                if len(face_ids_with_solid_ids[face_id]) == 2:
                    other_solid_id = face_ids_with_solid_ids[face_id][1]
                    other_volume_set = volume_sets_by_solid_id[other_solid_id]
                    sense_data = np.array([other_volume_set, volume_set], dtype="uint64")
                else:
                    sense_data = np.array([volume_set, 0], dtype="uint64")

                moab_core.tag_set_data(tags["surf_sense"], surface_set, sense_data)

                moab_verts = moab_core.create_vertices(vertices)
                moab_core.add_entity(surface_set, moab_verts)

                for triangle in triangles_on_face:
                    tri = (
                        moab_verts[int(triangle[0])],
                        moab_verts[int(triangle[1])],
                        moab_verts[int(triangle[2])],
                    )

                    moab_triangle = moab_core.create_element(types.MBTRI, tri)
                    moab_core.add_entity(surface_set, moab_triangle)

                added_surfaces_ids[face_id] = surface_set
            else:
                surface_set = added_surfaces_ids[face_id]

                other_solid_id = face_ids_with_solid_ids[face_id][0]

                other_volume_set = volume_sets_by_solid_id[other_solid_id]

                sense_data = np.array([other_volume_set, volume_set], dtype="uint64")
                moab_core.tag_set_data(tags["surf_sense"], surface_set, sense_data)

            moab_core.add_parent_child(volume_set, surface_set)

        moab_core.add_entity(group_set, volume_set)

    if implicit_complement_material_tag:
        group_set = moab_core.create_meshset()
        moab_core.tag_set_data(tags["category"], group_set, "Group")
        moab_core.tag_set_data(
            tags["name"], group_set, f"mat:{implicit_complement_material_tag}_comp"
        )
        moab_core.tag_set_data(tags["geom_dimension"], group_set, 4)
        moab_core.add_entity(
            group_set, volume_set
        )  # volume is arbitrary but should exist in moab core

    all_sets = moab_core.get_entities_by_handle(0)

    file_set = moab_core.create_meshset(types.MBENTITYSET)
    tags_faceting_tol = moab_core.tag_get_handle(
        "FACETING_TOL",
        1,
        types.MB_TYPE_DOUBLE,
        types.MB_TAG_SPARSE | types.MB_TAG_CREAT,
        create_if_missing=True,
    )
    moab_core.tag_set_data(tags_faceting_tol, file_set, faceting_tolerance)

    moab_core.add_entities(file_set, all_sets)

    moab_core.write_file(f"{filename}.h5m")
    moab_core.write_file(f"{filename}.vtk")

    # # makes the folder if it does not exist
    # if Path(h5m_filename).parent:
    #     Path(h5m_filename).parent.mkdir(parents=True, exist_ok=True)

    # # moab_core.write_file only accepts strings
    # if isinstance(h5m_filename, Path):
    #     moab_core.write_file(str(h5m_filename))
    # else:
    #     moab_core.write_file(h5m_filename)

    # print(f"written DAGMC file {h5m_filename}")

    # return h5m_filename
