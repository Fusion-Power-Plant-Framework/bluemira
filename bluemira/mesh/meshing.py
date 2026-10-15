# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Core functionality for the bluemira mesh module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING

import gmsh

from bluemira.base.look_and_feel import bluemira_print
from bluemira.mesh.error import MeshOptionsError

if TYPE_CHECKING:
    from bluemira.base.components import Component
    from bluemira.geometry.base import BluemiraGeo


MESH_EXTENSIONS = {
    ".geo",
    ".geo_unrolled",
    ".msh",
    ".xdmf",
    ".h5",
    ".ini",
}


@dataclass
class GmshEntity:
    """
    Association between a Bluemira geometry object and the Gmsh
    entities representing it.
    """

    source: BluemiraGeo
    dim_tags: list[tuple[int, int]]

    @property
    def lcar(self) -> float | None:
        """Mesh size."""
        return self.source.mesh_options.lcar

    @property
    def physical_group(self):
        """Physical group name."""
        return self.source.mesh_options.physical_group


def _import_brep(obj: BluemiraGeo) -> list[tuple[int, int]]:
    """
    Import a Bluemira geometry object into the current Gmsh OCC model.

    Returns
    -------
    :
        Gmsh dimension-tag pairs created by the import.
    """
    with TemporaryDirectory() as directory:
        brep_file = (Path(directory) / "shape.brep").as_posix()

        obj.shape.exportBrep(brep_file)

        return gmsh.model.occ.importShapes(brep_file)


def _get_entities_recursive(
    dim_tags: list[tuple[int, int]],
    target_dim: int,
) -> list[tuple[int, int]]:
    """
    Get entities of ``target_dim`` belonging to the supplied Gmsh entities.

    Returns
    -------
    :
        Unique Gmsh dimension-tag pairs of the requested dimension.
    """
    entities: set[tuple[int, int]] = set()
    pending = list(dim_tags)

    while pending:
        dim_tag = pending.pop()
        dim, _ = dim_tag

        if dim == target_dim:
            entities.add(dim_tag)
            continue

        if dim < target_dim:
            continue

        boundary = gmsh.model.getBoundary(
            [dim_tag],
            combined=False,
            oriented=False,
            recursive=False,
        )

        pending.extend(boundary)

    return sorted(entities)


def _apply_mesh_sizes(
    entities: list[GmshEntity],
) -> None:
    """
    Apply mesh sizes to final Gmsh points.

    The smallest requested size takes precedence so that a finer
    parent or child mesh requirement is never overwritten by a
    coarser one.
    """
    point_sizes: dict[int, float] = {}

    for entity in entities:
        if entity.lcar is None:
            continue

        points = _get_entities_recursive(
            entity.dim_tags,
            target_dim=0,
        )

        for _, point_tag in points:
            point_sizes[point_tag] = min(
                point_sizes.get(point_tag, entity.lcar),
                entity.lcar,
            )

    points_by_size: dict[float, list[tuple[int, int]]] = {}

    for point_tag, size in point_sizes.items():
        points_by_size.setdefault(size, []).append((0, point_tag))

    for size, points in points_by_size.items():
        gmsh.model.mesh.setSize(
            points,
            size,
        )


def _apply_physical_group(entity: GmshEntity):
    if entity.physical_group is None:
        return

    entities_by_dim: dict[int, list[int]] = {}

    for dim, tag in entity.dim_tags:
        entities_by_dim.setdefault(dim, []).append(tag)

    for dim, tags in entities_by_dim.items():
        physical_tag = gmsh.model.addPhysicalGroup(dim, tags)

        gmsh.model.setPhysicalName(dim, physical_tag, entity.physical_group)


def _fragment_entities(entities: list[GmshEntity]) -> None:
    """
    Fragment imported OCC entities and update source provenance.

    Gmsh returns one output mapping entry for every input dim-tag.
    Preserve the original grouping of dim-tags by GmshEntity so
    each Bluemira object remains associated with all final entities
    produced from it.
    """
    if len(entities) <= 1:
        return

    input_dim_tags = [dim_tag for entity in entities for dim_tag in entity.dim_tags]

    if len(input_dim_tags) <= 1:
        return

    _, out_map = gmsh.model.occ.fragment(
        objectDimTags=input_dim_tags,
        toolDimTags=[],
    )

    gmsh.model.occ.synchronize()

    offset = 0

    for entity in entities:
        count = len(entity.dim_tags)

        mapped = out_map[offset : offset + count]

        entity.dim_tags = sorted({output for mappings in mapped for output in mappings})

        offset += count


def _mesh_brep_objects(objects: list[BluemiraGeo]) -> list[GmshEntity]:
    """
    Import, fragment, and configure Bluemira geometry in Gmsh.

    Returns
    -------
    :
        Source-to-Gmsh entity mappings after fragmentation.
    """
    entities = []

    for obj in objects:
        dim_tags = _import_brep(obj)

        entities.append(
            GmshEntity(
                source=obj,
                dim_tags=dim_tags,
            )
        )

    gmsh.model.occ.synchronize()

    _fragment_entities(entities)

    _apply_mesh_sizes(entities)

    for entity in entities:
        _apply_physical_group(entity)

    return entities


def _apply_mesh_settings(settings: MeshSettings) -> None:
    """Apply global Gmsh mesh settings."""
    options = {
        "Mesh.Algorithm": settings.algorithm_2d,
        "Mesh.Algorithm3D": settings.algorithm_3d,
        "Mesh.ElementOrder": settings.element_order,
        "Mesh.MeshSizeMin": settings.mesh_size_min,
        "Mesh.MeshSizeMax": settings.mesh_size_max,
        "Mesh.Optimize": int(settings.optimise),
    }

    for name, value in options.items():
        gmsh.option.set_number(name, value)


@dataclass
class MeshOptions:
    """Options controlling meshing of a Bluemira geometry."""

    lcar: float | None = None
    physical_group: str | None = None


class Meshable:
    """Mixin class to make a class meshable"""

    def __init__(self):
        super().__init__()
        self._mesh_options = MeshOptions()

    @property
    def mesh_options(self) -> MeshOptions:
        """
        The options that will be used to mesh the object.
        """
        return self._mesh_options

    @mesh_options.setter
    def mesh_options(self, value: MeshOptions | dict):
        if isinstance(value, MeshOptions):
            self._mesh_options = value
        elif isinstance(value, dict):
            for key, val in value.items():
                if hasattr(self._mesh_options, key):
                    setattr(self._mesh_options, key, val)
        else:
            raise MeshOptionsError(
                "Mesh options must be a MeshOptions instance or dictionary."
            )


@dataclass
class MeshSettings:
    """Global Gmesh mesh settings (defaults from Gmsh docs)."""

    algorithm_2d: int = 6
    algorithm_3d: int = 1
    element_order: int = 1
    mesh_size_min: float = 0.0
    mesh_size_max: float = 1e22
    optimise: bool = True

    def __post_init__(self):
        """
        Validate settings.

        Raises
        ------
        ValueError
            If settings are invalid.
        """
        if self.element_order < 1:
            raise ValueError("element_order must be at least 1.")
        if self.mesh_size_min < 0:
            raise ValueError("mesh_size_min must be positive.")
        if self.mesh_size_max <= 0:
            raise ValueError("mesh_size_max must be positive.")
        if self.mesh_size_min > self.mesh_size_max:
            raise ValueError("mesh_size_min cannot be greater than mesh_size_max.")


class Mesh:
    """
    A class for supporting the creation of meshes and writing out those meshes to files.
    """

    def __init__(
        self,
        modelname: str = "Mesh",
        meshfile: str | list[str] | None = None,
        settings: MeshSettings | None = None,
    ):
        self.modelname = modelname
        self.meshfile = (
            ["Mesh.geo_unrolled", "Mesh.msh"] if meshfile is None else meshfile
        )
        self.settings = MeshSettings() if settings is None else settings

    @staticmethod
    def _check_meshfile(meshfile: str | list) -> list[str]:
        """
        Check the mesh file input.

        Returns
        -------
        :
            The meshfile list

        Raises
        ------
        ValueError
            Meshfile list is empty
        TypeError
            Meshfile must be a string or list of strings
        """
        if isinstance(meshfile, str):
            meshfile = [meshfile]
        elif isinstance(meshfile, list):
            if len(meshfile) < 1:
                raise ValueError("meshfile is an empty list")
        else:
            raise TypeError("meshfile must be a string or a list of strings")

        for filename in meshfile:
            ext = Path(filename).suffix.lower()

            if ext not in MESH_EXTENSIONS:
                raise ValueError(f"Unsupported mesh file extension: {ext}")

        return meshfile

    @property
    def meshfile(self) -> list[str]:
        """
        The path(s) to the file(s) containing the meshes.
        """
        return self._meshfile

    @meshfile.setter
    def meshfile(self, meshfile: str | list[str]):
        self._meshfile = self._check_meshfile(meshfile)

    @staticmethod
    def _has_mesh_options(obj: Meshable) -> bool:
        return (
            obj.mesh_options.lcar is not None
            or obj.mesh_options.physical_group is not None
        )

    def _collect_geometry_objects(
        self,
        shape: BluemiraGeo,
    ) -> list[BluemiraGeo]:
        """
        Collect geometry requiring independent Gmsh provenance.

        The top-level geometry is always included. Nested geometry is included
        when it has explicit mesh options.

        Returns
        -------
        :
            Geometry objects to import independently into Gmsh.
        """
        objects = []
        seen = set()

        def visit(obj, *, include=False):
            if not isinstance(obj, Meshable):
                return

            obj_id = id(obj)

            if (include or self._has_mesh_options(obj)) and obj_id not in seen:
                seen.add(obj_id)
                objects.append(obj)

            for boundary in obj.boundary:
                if isinstance(boundary, Meshable):
                    visit(boundary)

        visit(shape, include=True)

        return objects

    def _collect_meshable_objects(
        self,
        comp: Component | Meshable,
    ) -> list[BluemiraGeo]:
        """
        Collect geometry objects that need independent Gmsh provenance.

        PhysicalComponent shapes are always included. Nested geometry is
        additionally included when it has explicit mesh options.

        Returns
        -------
        :
            Geometry objects to import independently into Gmsh.

        Raises
        ------
        TypeError
            If ``comp`` is neither a Component nor a Meshable object.
        """
        from bluemira.base.components import (  # noqa: PLC0415
            Component,
            PhysicalComponent,
        )

        objects = []
        seen = set()

        def add_geometry(shape):
            for obj in self._collect_geometry_objects(shape):
                obj_id = id(obj)

                if obj_id not in seen:
                    seen.add(obj_id)
                    objects.append(obj)

        def visit_component(component):
            if isinstance(component, PhysicalComponent):
                add_geometry(component.shape)

            for child in component.children:
                visit_component(child)

        if isinstance(comp, Component):
            visit_component(comp)
        elif isinstance(comp, Meshable):
            add_geometry(comp)
        else:
            raise TypeError(
                f"Only Component or Meshable objects can be meshed, got {type(comp)}"
            )

        return objects

    def __call__(self, comp: Component | Meshable, dim: int = 2) -> list[GmshEntity]:
        """
        Generate a Gmsh mesh from Bluemira geometry and save it to file.

        Returns
        -------
        comp:
            Geometry or component tree to mesh.
        dim:
            Dimension of mesh elements to generate.

        Returns
        -------
        :
            Mapping between the source Bluemira geometries and their
            final Gmsh entities.

        Raises
        ------
        RuntimeError
            If Gmsh has not been initialised.
        ValueError
            If ``dim`` is not 1, 2, or 3.
        """
        if not gmsh.is_initialized():
            raise RuntimeError(
                "Gmsh is not initialised. Initialise a GmshSession before meshing."
            )

        if dim not in {1, 2, 3}:
            raise ValueError(f"Mesh dimension must be 1, 2, or 3, got {dim}.")

        bluemira_print("Starting mesh process...")

        gmsh.model.add(self.modelname)

        objects = self._collect_meshable_objects(comp)
        entities = _mesh_brep_objects(objects)

        _apply_mesh_settings(self.settings)

        gmsh.model.mesh.generate(dim)

        for file in self.meshfile:
            gmsh.write(file)

        bluemira_print("Mesh process completed.")
        return entities


class GmshSession:
    """Manage the Gmsh Python API session."""

    def __init__(self, *, terminal: int = 0, logfile: str | Path | None = "gmsh.log"):
        self.terminal = terminal
        self.logfile = logfile
        self._owns_session = False
        self._active = False

    def __enter__(self):
        """
        Called upon entering ``with GmshSession():``

        Returns
        -------
        :
            Active Gmsh session.
        """
        return self.initialize()

    def __exit__(self, *_):
        """Called upon exiting ``with GmshSession():``"""
        self.finalize()

    def initialize(self):
        """
        Initialise the Gmsh session.

        Ideally this is called indirectly through ``with GmshSession()`` calling
        ``__enter``. However, an explicit method is implemented for notebook users
        that may want to carry a session over multiple cells.

        Returns
        -------
        :
            GmshSession.
        """
        if self._active:
            return self

        if not gmsh.is_initialized():
            gmsh.initialize()
            self._owns_session = True

        gmsh.option.setNumber("General.Terminal", self.terminal)
        gmsh.logger.start()

        self._active = True
        return self

    def finalize(self):
        """
        Finalise the Gmsh session.

        Again, this is ideally called automatically when a ``with GmshSession()`` block
        if exited, calling ``__exit__``. However, an explicit method is implemented for
        notebook users again.
        """
        if not self._active:
            return

        try:
            if self.logfile is not None:
                Path(self.logfile).write_text("\n".join(gmsh.logger.get()))
        finally:
            gmsh.logger.stop()

            if self._owns_session and gmsh.is_initialized():
                gmsh.finalize()

            self._active = False
            self._owns_session = False
