# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import gmsh
import pytest

from bluemira.base.components import Component, PhysicalComponent
from bluemira.geometry import tools
from bluemira.geometry.face import BluemiraFace
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf


@pytest.fixture(autouse=True)
def clean_gmsh():
    """Ensure each test starts and ends with Gmsh uninitialised."""
    if gmsh.is_initialized():
        gmsh.finalize()

    yield

    if gmsh.is_initialized():
        gmsh.finalize()


def make_square_surface(
    lcar: float,
    *,
    surface_lcar: float | None = None,
) -> BluemiraFace:
    poly = tools.make_polygon(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 0, 1],
            [0, 0, 1],
        ],
        closed=True,
        label="poly",
    )
    poly.mesh_options = {
        "lcar": lcar,
        "physical_group": "poly",
    }

    surface = BluemiraFace(poly, label="surf")
    surface.mesh_options = {
        "physical_group": "coil",
    }

    if surface_lcar is not None:
        surface.mesh_options = {
            "lcar": surface_lcar,
            "physical_group": "coil",
        }

    return surface


def make_box(x0: float, x1: float):
    rectangle = tools.make_polygon(
        [
            [x0, x1, x1, x0],
            [0, 0, 10, 10],
            [0, 0, 0, 0],
        ],
        closed=True,
    )

    return tools.extrude_shape(
        BluemiraFace(rectangle),
        [0, 0, 10],
    )


def make_cylinder():
    circle = tools.make_circle(
        radius=5,
        center=[0, 0, 0],
    )

    return tools.extrude_shape(
        BluemiraFace(circle),
        [0, 0, 10],
    )


def make_two_solid_component(shape_a, shape_b) -> Component:
    universe = Component("universe")

    PhysicalComponent(
        "shape_a",
        shape=shape_a,
        parent=universe,
    )
    PhysicalComponent(
        "shape_b",
        shape=shape_b,
        parent=universe,
    )

    return universe


def physical_groups_by_name(
    dim: int,
) -> dict[str, set[tuple[int, int]]]:
    return {
        gmsh.model.getPhysicalName(group_dim, tag): {
            (group_dim, int(entity_tag))
            for entity_tag in gmsh.model.getEntitiesForPhysicalGroup(
                group_dim,
                tag,
            )
        }
        for group_dim, tag in gmsh.model.getPhysicalGroups(dim)
    }


def assert_entities_have_elements(
    dim_tags: list[tuple[int, int]],
) -> None:
    for dim_tag in dim_tags:
        _, element_tags, _ = gmsh.model.mesh.getElements(*dim_tag)

        assert sum(len(tags) for tags in element_tags) > 0


# -----------------------------------------------------------------------------
# Gmsh session
# -----------------------------------------------------------------------------


def test_gmsh_session_context_manager_initialises_and_finalises():
    assert not gmsh.is_initialized()

    with meshing.GmshSession(logfile=None):
        assert gmsh.is_initialized()

    assert not gmsh.is_initialized()


def test_gmsh_session_explicit_lifecycle():
    session = meshing.GmshSession(logfile=None)

    assert not gmsh.is_initialized()

    session.initialize()

    assert gmsh.is_initialized()

    session.finalize()

    assert not gmsh.is_initialized()


def test_gmsh_session_lifecycle_is_idempotent():
    session = meshing.GmshSession(logfile=None)

    session.initialize()
    session.initialize()

    assert gmsh.is_initialized()

    session.finalize()
    session.finalize()

    assert not gmsh.is_initialized()


def test_gmsh_session_finalises_after_exception():
    with pytest.raises(RuntimeError, match="test error"):  # noqa: PT012, SIM117
        with meshing.GmshSession(logfile=None):
            assert gmsh.is_initialized()
            raise RuntimeError("test error")

    assert not gmsh.is_initialized()


def test_gmsh_session_context_manager_does_not_finalize_external_session():
    gmsh.initialize()

    with meshing.GmshSession(logfile=None):
        assert gmsh.is_initialized()

    # GmshSession did not initialise Gmsh, so it must not finalise it.
    assert gmsh.is_initialized()


def test_gmsh_session_explicit_lifecycle_does_not_finalize_external_session():
    gmsh.initialize()

    session = meshing.GmshSession(logfile=None)
    session.initialize()
    session.finalize()

    assert gmsh.is_initialized()


def test_gmsh_session_writes_log(tmp_path):
    logfile = tmp_path / "gmsh.log"

    with meshing.GmshSession(logfile=logfile):
        gmsh.logger.write("test message")

    assert logfile.exists()
    assert "test message" in logfile.read_text()


def test_mesh_requires_initialised_gmsh(tmp_path):
    surface = make_square_surface(lcar=0.5)
    meshfile = (tmp_path / "Mesh.msh").as_posix()

    with pytest.raises(
        RuntimeError,
        match="Gmsh is not initialised",
    ):
        meshing.Mesh(meshfile=meshfile)(surface)


# -----------------------------------------------------------------------------
# 2D meshing
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("lcar", "nodes_num"),
    [
        (0.1, 40),
        (0.25, 16),
        (0.5, 8),
    ],
)
def test_mesh_poly(lcar, nodes_num, tmp_path):
    surface = make_square_surface(lcar)
    meshfile = (tmp_path / "Mesh.msh").as_posix()

    with meshing.GmshSession(logfile=None):
        meshing.Mesh(meshfile=meshfile)(surface)

    msh_to_xdmf(
        "Mesh.msh",
        dimensions=(0, 1),
        directory=tmp_path,
    )

    _, boundaries, _, labels = import_mesh(
        "Mesh",
        directory=tmp_path,
        subdomains=True,
    )

    assert (boundaries.values == labels["poly"]).sum() == nodes_num


@pytest.mark.parametrize(
    ("lcar", "nodes_num"),
    [
        (0.1, 40),
        (0.25, 16),
        (0.5, 8),
    ],
)
def test_finer_surface_lcar_overrides_boundary_lcar(
    lcar,
    nodes_num,
    tmp_path,
):
    surface = make_square_surface(
        lcar,
        surface_lcar=lcar / 2,
    )
    meshfile = (tmp_path / "Mesh.msh").as_posix()

    with meshing.GmshSession(logfile=None):
        meshing.Mesh(meshfile=meshfile)(surface)

    msh_to_xdmf(
        "Mesh.msh",
        dimensions=(0, 1),
        directory=tmp_path,
    )

    _, boundaries, _, labels = import_mesh(
        "Mesh",
        directory=tmp_path,
        subdomains=True,
    )

    assert (boundaries.values == labels["poly"]).sum() == 2 * nodes_num


# -----------------------------------------------------------------------------
# 3D meshing
# -----------------------------------------------------------------------------


def test_mesh_3d_solid(tmp_path):
    cylinder = make_cylinder()
    cylinder.mesh_options = {
        "lcar": 1.0,
        "physical_group": "cylinder",
    }

    meshfile = (tmp_path / "cylinder.msh").as_posix()

    with meshing.GmshSession(logfile=None):
        entities = meshing.Mesh(meshfile=meshfile)(
            cylinder,
            dim=3,
        )

        volumes = gmsh.model.getEntities(3)

        assert len(volumes) == 1

        physical_groups = gmsh.model.getPhysicalGroups(3)

        assert len(physical_groups) == 1

        dim, physical_tag = physical_groups[0]

        assert dim == 3
        assert gmsh.model.getPhysicalName(dim, physical_tag) == "cylinder"

        grouped_volumes = gmsh.model.getEntitiesForPhysicalGroup(
            dim,
            physical_tag,
        )

        assert grouped_volumes.tolist() == [volumes[0][1]]

        assert_entities_have_elements(volumes)

        assert len(entities) == 1
        assert entities[0].physical_group == "cylinder"
        assert entities[0].dim_tags == volumes

    assert (tmp_path / "cylinder.msh").exists()


def test_mesh_3d_overlapping_solids_preserves_provenance(
    tmp_path,
):
    box_a = make_box(0, 10)
    box_b = make_box(5, 15)

    box_a.mesh_options = {
        "lcar": 2.0,
        "physical_group": "box_a",
    }
    box_b.mesh_options = {
        "lcar": 1.0,
        "physical_group": "box_b",
    }

    universe = make_two_solid_component(
        box_a,
        box_b,
    )

    meshfile = (tmp_path / "overlapping_boxes.msh").as_posix()

    with meshing.GmshSession(logfile=None):
        entities = meshing.Mesh(meshfile=meshfile)(
            universe,
            dim=3,
        )

        volumes = gmsh.model.getEntities(3)

        assert len(volumes) == 3

        by_name = {entity.physical_group: set(entity.dim_tags) for entity in entities}

        assert set(by_name) == {
            "box_a",
            "box_b",
        }

        assert len(by_name["box_a"]) == 2
        assert len(by_name["box_b"]) == 2

        overlap = by_name["box_a"] & by_name["box_b"]

        assert len(overlap) == 1
        assert (by_name["box_a"] | by_name["box_b"]) == set(volumes)

        physical_groups = physical_groups_by_name(3)

        assert physical_groups["box_a"] == by_name["box_a"]
        assert physical_groups["box_b"] == by_name["box_b"]

        assert_entities_have_elements(volumes)


def test_mesh_3d_touching_solids_are_conforming(
    tmp_path,
):
    box_a = make_box(0, 10)
    box_b = make_box(10, 20)

    box_a.mesh_options = {
        "lcar": 2.0,
        "physical_group": "box_a",
    }
    box_b.mesh_options = {
        "lcar": 1.0,
        "physical_group": "box_b",
    }

    universe = make_two_solid_component(
        box_a,
        box_b,
    )

    meshfile = (tmp_path / "touching_boxes.msh").as_posix()

    with meshing.GmshSession(logfile=None):
        entities = meshing.Mesh(meshfile=meshfile)(
            universe,
            dim=3,
        )

        volumes = gmsh.model.getEntities(3)

        assert len(volumes) == 2

        by_name = {entity.physical_group: set(entity.dim_tags) for entity in entities}

        assert len(by_name["box_a"]) == 1
        assert len(by_name["box_b"]) == 1
        assert by_name["box_a"].isdisjoint(by_name["box_b"])
        assert (by_name["box_a"] | by_name["box_b"]) == set(volumes)

        boundary_a = set(
            gmsh.model.getBoundary(
                list(by_name["box_a"]),
                combined=False,
                oriented=False,
                recursive=False,
            )
        )
        boundary_b = set(
            gmsh.model.getBoundary(
                list(by_name["box_b"]),
                combined=False,
                oriented=False,
                recursive=False,
            )
        )

        shared_surfaces = boundary_a & boundary_b

        # The two boxes meet along one complete face.
        assert len(shared_surfaces) == 1
        assert next(iter(shared_surfaces))[0] == 2

        assert_entities_have_elements(volumes)
