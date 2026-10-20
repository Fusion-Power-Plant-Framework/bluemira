# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from types import SimpleNamespace
from unittest.mock import Mock

import gmsh
import pytest

from bluemira.base.components import Component, PhysicalComponent
from bluemira.geometry import tools
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import PrincetonD
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


def number_of_elements(dim: int) -> int:
    """Return the number of generated mesh elements of a given dimension."""
    _, element_tags, _ = gmsh.model.mesh.getElements(dim)

    return sum(len(tags) for tags in element_tags)


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


# -----------------------------------------------------------------------------
# Mesh settings
# -----------------------------------------------------------------------------


def test_mesh_settings_defaults():
    settings = meshing.MeshSettings()

    assert settings.algorithm_2d == 6
    assert settings.algorithm_3d == 1
    assert settings.element_order == 1
    assert settings.mesh_size_min == pytest.approx(0.0)
    assert settings.mesh_size_max == pytest.approx(1e22)
    assert settings.optimise is True


@pytest.mark.parametrize(
    "kwargs",
    [
        {"element_order": 0},
        {"element_order": -1},
        {"mesh_size_min": -0.1},
        {"mesh_size_max": 0},
        {"mesh_size_max": -1},
        {
            "mesh_size_min": 2.0,
            "mesh_size_max": 1.0,
        },
    ],
)
def test_mesh_settings_reject_invalid_values(kwargs):
    with pytest.raises(ValueError):  # noqa: PT011
        meshing.MeshSettings(**kwargs)


def test_mesh_uses_default_settings_when_none_are_given():
    mesh = meshing.Mesh(meshfile="Mesh.msh")

    assert mesh.settings == meshing.MeshSettings()


def test_mesh_preserves_supplied_settings():
    settings = meshing.MeshSettings(
        algorithm_2d=5,
        element_order=2,
        mesh_size_max=0.5,
        optimise=False,
    )

    mesh = meshing.Mesh(
        meshfile="Mesh.msh",
        settings=settings,
    )

    assert mesh.settings is settings


def test_mesh_applies_gmsh_settings(tmp_path):
    surface = make_square_surface(lcar=1.0)

    settings = meshing.MeshSettings(
        algorithm_2d=5,
        algorithm_3d=4,
        element_order=2,
        mesh_size_min=0.1,
        mesh_size_max=0.5,
        optimise=False,
    )

    with meshing.GmshSession(logfile=None):
        meshing.Mesh(
            meshfile=(tmp_path / "mesh.msh").as_posix(),
            settings=settings,
        )(surface, dim=2)

        assert gmsh.option.get_number("Mesh.Algorithm") == 5
        assert gmsh.option.get_number("Mesh.Algorithm3D") == 4
        assert gmsh.option.get_number("Mesh.ElementOrder") == 2
        assert gmsh.option.get_number("Mesh.MeshSizeMin") == pytest.approx(0.1)
        assert gmsh.option.get_number("Mesh.MeshSizeMax") == pytest.approx(0.5)
        assert gmsh.option.get_number("Mesh.Optimize") == 0


def test_default_settings_replace_previous_custom_settings(
    tmp_path,
):
    surface = make_square_surface(lcar=1.0)

    custom = meshing.MeshSettings(
        algorithm_2d=5,
        algorithm_3d=4,
        element_order=2,
        mesh_size_min=0.1,
        mesh_size_max=0.5,
        optimise=False,
    )

    with meshing.GmshSession(logfile=None):
        meshing.Mesh(
            modelname="custom",
            meshfile=(tmp_path / "custom.msh").as_posix(),
            settings=custom,
        )(surface)

        assert gmsh.option.get_number("Mesh.Algorithm") == 5
        assert gmsh.option.get_number("Mesh.Algorithm3D") == 4
        assert gmsh.option.get_number("Mesh.ElementOrder") == 2
        assert gmsh.option.get_number("Mesh.MeshSizeMin") == pytest.approx(0.1)
        assert gmsh.option.get_number("Mesh.MeshSizeMax") == pytest.approx(0.5)
        assert gmsh.option.get_number("Mesh.Optimize") == 0

        meshing.Mesh(
            modelname="default",
            meshfile=(tmp_path / "default.msh").as_posix(),
        )(surface)

        assert gmsh.option.get_number("Mesh.Algorithm") == 6
        assert gmsh.option.get_number("Mesh.Algorithm3D") == 1
        assert gmsh.option.get_number("Mesh.ElementOrder") == 1
        assert gmsh.option.get_number("Mesh.MeshSizeMin") == 0
        assert gmsh.option.get_number("Mesh.MeshSizeMax") == pytest.approx(1e22)
        assert gmsh.option.get_number("Mesh.Optimize") == 1


@pytest.mark.parametrize("order", [1, 2])
def test_element_order_is_applied(
    order,
    tmp_path,
):
    surface = make_square_surface(lcar=0.25)

    settings = meshing.MeshSettings(
        element_order=order,
    )

    with meshing.GmshSession(logfile=None):
        meshing.Mesh(
            meshfile=(tmp_path / f"order_{order}.msh").as_posix(),
            settings=settings,
        )(surface, dim=2)

        element_types, _, _ = gmsh.model.mesh.getElements(
            dim=2,
        )

        generated_orders = {
            gmsh.model.mesh.getElementProperties(element_type)[2]
            for element_type in element_types
        }

        assert generated_orders == {order}


def test_smaller_mesh_size_max_creates_finer_mesh(
    tmp_path,
):
    coarse_surface = make_square_surface(lcar=10.0)

    with meshing.GmshSession(logfile=None):
        meshing.Mesh(
            modelname="coarse",
            meshfile=(tmp_path / "coarse.msh").as_posix(),
            settings=meshing.MeshSettings(
                mesh_size_max=0.5,
            ),
        )(coarse_surface, dim=2)

        coarse_elements = number_of_elements(2)

    fine_surface = make_square_surface(lcar=10.0)

    with meshing.GmshSession(logfile=None):
        meshing.Mesh(
            modelname="fine",
            meshfile=(tmp_path / "fine.msh").as_posix(),
            settings=meshing.MeshSettings(
                mesh_size_max=0.1,
            ),
        )(fine_surface, dim=2)

        fine_elements = number_of_elements(2)

    assert fine_elements > coarse_elements


def test_mesh_size_min_is_applied(tmp_path):
    surface = make_square_surface(lcar=0.01)

    with meshing.GmshSession(logfile=None):
        meshing.Mesh(
            meshfile=(tmp_path / "mesh.msh").as_posix(),
            settings=meshing.MeshSettings(
                mesh_size_min=0.2,
            ),
        )(surface, dim=2)

        assert gmsh.option.get_number("Mesh.MeshSizeMin") == pytest.approx(0.2)


@pytest.mark.parametrize("algorithm", [5, 6])
def test_2d_mesh_algorithms_generate_mesh(
    algorithm,
    tmp_path,
):
    surface = make_square_surface(lcar=0.25)

    settings = meshing.MeshSettings(
        algorithm_2d=algorithm,
    )

    with meshing.GmshSession(logfile=None):
        meshing.Mesh(
            meshfile=(tmp_path / f"algorithm_2d_{algorithm}.msh").as_posix(),
            settings=settings,
        )(surface, dim=2)

        assert gmsh.option.get_number("Mesh.Algorithm") == algorithm
        assert number_of_elements(2) > 0


@pytest.mark.parametrize("algorithm", [1, 4])
def test_3d_mesh_algorithms_generate_mesh(
    algorithm,
    tmp_path,
):
    cylinder = make_cylinder()
    cylinder.mesh_options = {
        "lcar": 1.0,
        "physical_group": "cylinder",
    }

    settings = meshing.MeshSettings(
        algorithm_3d=algorithm,
    )

    with meshing.GmshSession(logfile=None):
        meshing.Mesh(
            meshfile=(tmp_path / f"algorithm_3d_{algorithm}.msh").as_posix(),
            settings=settings,
        )(cylinder, dim=3)

        assert gmsh.option.get_number("Mesh.Algorithm3D") == algorithm
        assert number_of_elements(3) > 0


@pytest.mark.parametrize(
    ("optimise", "expected"),
    [
        (True, 1),
        (False, 0),
    ],
)
def test_optimise_setting_is_applied(
    optimise,
    expected,
    tmp_path,
):
    surface = make_square_surface(lcar=0.25)

    with meshing.GmshSession(logfile=None):
        meshing.Mesh(
            meshfile=(tmp_path / "mesh.msh").as_posix(),
            settings=meshing.MeshSettings(
                optimise=optimise,
            ),
        )(surface, dim=2)

        assert gmsh.option.get_number("Mesh.Optimize") == expected


# -----------------------------------------------------------------------------
# Mesh complex examples
# -----------------------------------------------------------------------------


def make_polygon_face(points, label, lcar):
    """Create a face with a named physical group and target mesh size."""
    wire = tools.make_polygon(points, closed=True, label=label)
    face = BluemiraFace(wire, label=label)
    face.mesh_options = {
        "lcar": lcar,
        "physical_group": label,
    }
    return face


def make_nested_rectangle_component(lcar):
    """Build a rectangular ring and its inner region as separate faces."""
    outer = tools.make_polygon(
        [
            [0, 0, 0],
            [4, 0, 0],
            [4, 0, 4],
            [0, 0, 4],
        ],
        closed=True,
        label="pf_outer",
    )
    inner = tools.make_polygon(
        [
            [0.15, 0, 0.15],
            [3.85, 0, 0.15],
            [3.85, 0, 3.85],
            [0.15, 0, 3.85],
        ],
        closed=True,
        label="pf_inner",
    )

    ring = BluemiraFace([outer, inner], label="pf_outer")
    ring.mesh_options = {
        "lcar": lcar / 2,
        "physical_group": "pf_outer",
    }

    core = BluemiraFace(inner, label="pf_inner")
    core.mesh_options = {
        "lcar": lcar,
        "physical_group": "pf_inner",
    }

    component = Component("pf_coil")
    PhysicalComponent("outer_region", ring, parent=component)
    PhysicalComponent("inner_region", core, parent=component)
    return component


@pytest.mark.parametrize("lcar", [0.15, 0.3, 0.6])
def test_mesh_nested_rectangles_with_thin_outer_region(lcar):
    """Thin outer rectangular regions and the inner region should mesh."""
    component = make_nested_rectangle_component(lcar)

    with meshing.GmshSession(logfile=None):
        meshing.Mesh()(component, dim=2)

        groups = physical_groups_by_name(2)
        assert "pf_outer" in groups
        assert "pf_inner" in groups

        assert_entities_have_elements([tag for tags in groups.values() for tag in tags])

        # The outer ring should have a finer target size than the core.
        # The exact number of elements is intentionally not prescribed.
        assert number_of_elements(2) > 0


def make_square_with_circular_inclusion(lcar):
    """Build a square ring around a circular region."""
    square = tools.make_polygon(
        [
            [-1, 0, -1],
            [1, 0, -1],
            [1, 0, 1],
            [-1, 0, 1],
        ],
        closed=True,
        label="square_boundary",
    )
    circle = tools.make_circle(
        radius=0.35,
        center=[0, 0, 0],
        label="circle_boundary",
        axis=(0, 1, 0),
    )

    ring = BluemiraFace([square, circle], label="square_region")
    ring.mesh_options = {
        "lcar": lcar,
        "physical_group": "square_region",
    }

    disk = BluemiraFace(circle, label="circular_region")
    disk.mesh_options = {
        "lcar": lcar / 8,
        "physical_group": "circular_region",
    }

    component = Component("square_with_circle")
    PhysicalComponent("square_region", ring, parent=component)
    PhysicalComponent("circular_region", disk, parent=component)
    return component


def test_mesh_circle_inside_square():
    """A circular inclusion should mesh alongside the surrounding square."""
    component = make_square_with_circular_inclusion(lcar=0.2)

    with meshing.GmshSession(logfile=None):
        meshing.Mesh()(component, dim=2)

        groups = physical_groups_by_name(2)
        assert "square_region" in groups
        assert "circular_region" in groups

        assert_entities_have_elements([tag for tags in groups.values() for tag in tags])
        assert number_of_elements(2) > 0


@pytest.fixture
def tf_centreline():
    return PrincetonD({
        "x1": {"value": 4, "fixed": True},
        "x2": {"value": 15, "fixed": True},
        "dz": {"value": 0, "fixed": True},
    }).create_shape()


@pytest.fixture
def tf_winding_pack_xz(tf_centreline):
    outer = tools.offset_wire(tf_centreline, 1.0)
    return BluemiraFace([outer, tf_centreline])


@pytest.fixture
def tf_winding_pack_xyz(tf_centreline):
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
    return tools.sweep_shape(cross_section, tf_centreline)


def test_tf_winding_pack_xz_meshes(tf_winding_pack_xz):
    tf_winding_pack_xz.mesh_options = {
        "lcar": 0.4,
        "physical_group": "tf_winding_pack_xz",
    }

    with meshing.GmshSession(logfile=None):
        meshing.Mesh()(tf_winding_pack_xz, dim=2)

        groups = physical_groups_by_name(2)
        assert "tf_winding_pack_xz" in groups
        assert_entities_have_elements([
            (2, tag) for _, tag in groups["tf_winding_pack_xz"]
        ])


def test_tf_winding_pack_xyz_meshes(tf_winding_pack_xyz):
    tf_winding_pack_xyz.mesh_options = {
        "lcar": 0.8,
        "physical_group": "tf_winding_pack_xyz",
    }

    with meshing.GmshSession(logfile=None):
        meshing.Mesh()(tf_winding_pack_xyz, dim=3)

        groups = physical_groups_by_name(3)
        assert "tf_winding_pack_xyz" in groups
        assert_entities_have_elements([
            (3, tag) for _, tag in groups["tf_winding_pack_xyz"]
        ])


# -----------------------------------------------------------------------------
# Mesh element type tests
# -----------------------------------------------------------------------------


@pytest.fixture
def gmsh_mock(monkeypatch):
    mesh = Mock()

    monkeypatch.setattr(
        meshing,
        "gmsh",
        SimpleNamespace(
            model=SimpleNamespace(
                mesh=mesh,
            )
        ),
    )

    return mesh


def make_entity(
    *,
    dim_tags,
    element_type=None,
    mesh_strategy="unstructured",
):
    source = SimpleNamespace(
        mesh_options=meshing.MeshOptions(
            element_type=element_type,
            mesh_strategy=mesh_strategy,
        )
    )

    return meshing.GmshEntity(
        source=source,
        dim_tags=dim_tags,
    )


def test_none_element_type_is_ignored(gmsh_mock):
    """Entities with no element_type configured are ignored."""
    entity = make_entity(
        dim_tags=[(2, 1)],
        element_type=None,
    )

    meshing._apply_element_type([entity])

    gmsh_mock.set_recombine.assert_not_called()
    gmsh_mock.set_transfinite_automatic.assert_not_called()


def test_invalid_mesh_strategy_raises():
    """An unsupported mesh strategy is rejected."""
    entity = make_entity(
        dim_tags=[(2, 1)],
        element_type="quad",
        mesh_strategy="banana",
    )

    with pytest.raises(
        meshing.MeshOptionsError,
        match="Unsupported mesh_strategy",
    ):
        meshing._apply_element_type([entity])


def test_no_dim_tags_raises():
    """Element type options cannot be applied to empty entities."""
    entity = make_entity(
        dim_tags=[],
        element_type="quad",
    )

    with pytest.raises(
        meshing.MeshOptionsError,
        match="no Gmsh entities were found",
    ):
        meshing._apply_element_type([entity])


def test_quad_not_allowed_on_volume():
    """Quad elements cannot be assigned to a volume."""
    entity = make_entity(
        dim_tags=[(3, 7)],
        element_type="quad",
    )

    with pytest.raises(
        meshing.MeshOptionsError,
        match="incompatible with dimension 3",
    ):
        meshing._apply_element_type([entity])


def test_hex_not_allowed_on_surface():
    """Hex elements cannot be assigned to surfaces."""
    entity = make_entity(
        dim_tags=[(2, 7)],
        element_type="hex",
    )

    with pytest.raises(
        meshing.MeshOptionsError,
        match="incompatible with dimension 2",
    ):
        meshing._apply_element_type([entity])


def test_tet_not_allowed_on_surface():
    """Tet elements cannot be assigned to surfaces."""
    entity = make_entity(
        dim_tags=[(2, 7)],
        element_type="tet",
    )

    with pytest.raises(meshing.MeshOptionsError):
        meshing._apply_element_type([entity])


def test_tri_not_allowed_on_volume():
    """Tri elements cannot be assigned to volumes."""
    entity = make_entity(
        dim_tags=[(3, 7)],
        element_type="tri",
    )

    with pytest.raises(meshing.MeshOptionsError):
        meshing._apply_element_type([entity])


def test_structured_tri_warns(monkeypatch):
    """Structured meshing requests for tri elements warn."""
    warning = Mock()
    monkeypatch.setattr(meshing, "bluemira_warn", warning)

    entity = make_entity(
        dim_tags=[(2, 1)],
        element_type="tri",
        mesh_strategy="structured",
    )

    meshing._apply_element_type([entity])

    warning.assert_called_once()


def test_structured_tet_warns(monkeypatch):
    """Structured meshing requests for tet elements warn."""
    warning = Mock()
    monkeypatch.setattr(meshing, "bluemira_warn", warning)

    entity = make_entity(
        dim_tags=[(3, 1)],
        element_type="tet",
        mesh_strategy="structured",
    )

    meshing._apply_element_type([entity])

    warning.assert_called_once()


def test_quad_unstructured_sets_recombine(gmsh_mock):
    """Unstructured quad surfaces request recombination"""
    entity = make_entity(
        dim_tags=[(2, 42)],
        element_type="quad",
        mesh_strategy="unstructured",
    )

    meshing._apply_element_type([entity])

    gmsh_mock.set_recombine.assert_called_once_with(2, 42)


def test_quad_structured_sets_transfinite(gmsh_mock):
    """Structured quad surfaces use transfinite meshing."""
    entity = make_entity(
        dim_tags=[(2, 42)],
        element_type="quad",
        mesh_strategy="structured",
    )

    meshing._apply_element_type([entity])

    gmsh_mock.set_transfinite_automatic.assert_called_once_with(
        [(2, 42)],
        recombine=True,
    )


def test_tri_generates_no_surface_constraints(gmsh_mock):
    """Tri surface meshing requires no extra constraints."""
    entity = make_entity(
        dim_tags=[(2, 12)],
        element_type="tri",
    )

    meshing._apply_element_type([entity])

    gmsh_mock.set_recombine.assert_not_called()
    gmsh_mock.set_transfinite_automatic.assert_not_called()


def test_hex_structured_sets_transfinite(gmsh_mock):
    """Structured hex volumes use transfinite meshing."""
    entity = make_entity(
        dim_tags=[(3, 8)],
        element_type="hex",
        mesh_strategy="structured",
    )

    meshing._apply_element_type([entity])

    gmsh_mock.set_transfinite_automatic.assert_called_once_with(
        [(3, 8)],
        recombine=True,
    )


def test_hex_unstructured_currently_no_api_calls(gmsh_mock):
    """Unstructured hex requests not implemented."""
    entity = make_entity(
        dim_tags=[(3, 8)],
        element_type="hex",
        mesh_strategy="unstructured",
    )

    meshing._apply_element_type([entity])

    gmsh_mock.set_transfinite_automatic.assert_not_called()
    gmsh_mock.set_recombine.assert_not_called()


def test_same_tag_same_options_allowed(gmsh_mock):
    """Identical mesh options may be applied to the same entity."""
    e1 = make_entity(
        dim_tags=[(2, 1)],
        element_type="quad",
    )

    e2 = make_entity(
        dim_tags=[(2, 1)],
        element_type="quad",
    )

    meshing._apply_element_type([e1, e2])

    gmsh_mock.set_recombine.assert_called_once_with(2, 1)


def test_same_surface_conflicting_element_type_raises():
    """Conflicting element types on the same surface are rejected."""
    e1 = make_entity(
        dim_tags=[(2, 1)],
        element_type="quad",
    )

    e2 = make_entity(
        dim_tags=[(2, 1)],
        element_type="tri",
    )

    with pytest.raises(
        meshing.MeshOptionsError,
        match="Conflicting mesh options",
    ):
        meshing._apply_element_type([e1, e2])


def test_same_surface_conflicting_strategy_raises():
    """Conflicting mesh strategies on the same surface are rejected."""
    e1 = make_entity(
        dim_tags=[(2, 1)],
        element_type="quad",
        mesh_strategy="structured",
    )

    e2 = make_entity(
        dim_tags=[(2, 1)],
        element_type="quad",
        mesh_strategy="unstructured",
    )

    with pytest.raises(meshing.MeshOptionsError):
        meshing._apply_element_type([e1, e2])


def test_dimension_one_entities_are_ignored(gmsh_mock):
    """Unsupported dimensions are silently ignored."""
    entity = make_entity(
        dim_tags=[(1, 99)],
        element_type="quad",
    )

    meshing._apply_element_type([entity])

    gmsh_mock.set_recombine.assert_not_called()
    gmsh_mock.set_transfinite_automatic.assert_not_called()


def test_same_numeric_tag_in_different_dimensions_not_conflict(gmsh_mock):
    """Surface and volume tags are tracked independently."""
    e1 = make_entity(
        dim_tags=[(2, 1)],
        element_type="quad",
    )

    e2 = make_entity(
        dim_tags=[(3, 1)],
        element_type="hex",
    )

    meshing._apply_element_type([e1, e2])

    gmsh_mock.set_recombine.assert_called_once_with(2, 1)
