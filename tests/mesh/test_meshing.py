# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path

import gmsh
import pytest

from bluemira.base.components import Component, PhysicalComponent
from bluemira.geometry import tools
from bluemira.geometry.face import BluemiraFace
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf


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


def make_two_solid_component(shape_a, shape_b):
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


class TestMeshing:
    @pytest.mark.parametrize(("lcar", "nodes_num"), [(0.1, 40), (0.25, 16), (0.5, 8)])
    def test_mesh_poly(self, lcar, nodes_num, tmp_path):
        poly = tools.make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], closed=True, label="poly"
        )

        poly.mesh_options = {"lcar": lcar, "physical_group": "poly"}

        surf = BluemiraFace(poly, label="surf")
        surf.mesh_options = {"physical_group": "coil"}

        meshfiles = [
            Path(tmp_path, p).as_posix() for p in ["Mesh.geo_unrolled", "Mesh.msh"]
        ]
        m = meshing.Mesh(meshfile=meshfiles)
        m(surf)

        msh_to_xdmf("Mesh.msh", dimensions=(0, 1), directory=tmp_path)

        _, boundaries, _, labels = import_mesh(
            "Mesh", directory=tmp_path, subdomains=True
        )

        arr = boundaries.values
        assert (arr == labels["poly"]).sum() == nodes_num

    @pytest.mark.parametrize(("lcar", "nodes_num"), [(0.1, 40), (0.25, 16), (0.5, 8)])
    def test_override_lcar_surf(self, lcar, nodes_num, tmp_path):
        poly = tools.make_polygon(
            [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]], closed=True, label="poly"
        )

        poly.mesh_options = {"lcar": lcar, "physical_group": "poly"}

        surf = BluemiraFace(poly, label="surf")
        surf.mesh_options = {"lcar": lcar / 2, "physical_group": "coil"}

        meshfiles = [
            Path(tmp_path, p).as_posix() for p in ["Mesh.geo_unrolled", "Mesh.msh"]
        ]
        m = meshing.Mesh(meshfile=meshfiles)
        m(surf)

        msh_to_xdmf("Mesh.msh", dimensions=(0, 1), directory=tmp_path)

        _, boundaries, _, labels = import_mesh(
            "Mesh", directory=tmp_path, subdomains=True
        )

        arr = boundaries.values
        assert (arr == labels["poly"]).sum() == nodes_num * 2

    def test_mesh_3d_solid(self, tmp_path):
        cylinder = make_cylinder()

        cylinder.mesh_options = {
            "lcar": 1.0,
            "physical_group": "cylinder",
        }

        meshfile = Path(tmp_path, "cylinder.msh").as_posix()

        entities = meshing.Mesh(meshfile=meshfile)(cylinder, dim=3)

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

        _, element_tags, _ = gmsh.model.mesh.getElements(
            *volumes[0],
        )

        assert sum(len(tags) for tags in element_tags) > 0

        assert len(entities) == 1
        assert entities[0].physical_group == "cylinder"
        assert entities[0].dim_tags == volumes

    def test_mesh_3d_overlapping_solids_preserves_provenance(self, tmp_path):
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

        meshfile = Path(tmp_path, "overlapping_boxes.msh").as_posix()

        entities = meshing.Mesh(meshfile=meshfile)(universe, dim=3)

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

        physical_groups = {
            gmsh.model.getPhysicalName(dim, tag): {
                (dim, int(entity_tag))
                for entity_tag in gmsh.model.getEntitiesForPhysicalGroup(
                    dim,
                    tag,
                )
            }
            for dim, tag in gmsh.model.getPhysicalGroups(3)
        }

        assert physical_groups["box_a"] == by_name["box_a"]
        assert physical_groups["box_b"] == by_name["box_b"]

        for volume in volumes:
            _, element_tags, _ = gmsh.model.mesh.getElements(
                *volume,
            )

            assert sum(len(tags) for tags in element_tags) > 0

    def test_mesh_3d_touching_solids_are_conforming(self, tmp_path):
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

        meshfile = Path(tmp_path, "touching_boxes.msh").as_posix()

        entities = meshing.Mesh(meshfile=meshfile)(universe, dim=3)

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

        shared_surface = next(iter(shared_surfaces))

        assert shared_surface[0] == 2

        # Both volumes should have tetrahedra.
        for volume in volumes:
            _, element_tags, _ = gmsh.model.mesh.getElements(
                *volume,
            )

            assert sum(len(tags) for tags in element_tags) > 0
