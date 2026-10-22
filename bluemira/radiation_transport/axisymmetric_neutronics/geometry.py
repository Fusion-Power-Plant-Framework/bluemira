# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Geometry for generalised axisymmetric neutronics
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print, bluemira_warn
from bluemira.base.reactor import ComponentManager
from bluemira.geometry.despliner import (
    check_face_has_splines,
    check_solid_has_revolution,
    despline_xz_face,
)
from bluemira.geometry.tools import (
    check_touching_geos,
    repair_gaps_between_faces,
    repair_overlapping_geos,
    revolve_shape,
)

if TYPE_CHECKING:
    from matproplib.material import Material

    from bluemira.geometry.face import BluemiraFace


@dataclass
class AxisymmetricNeutronicsCompData:
    """
    Store component data used during neutronics pre-processing.
    """

    name: str
    original_volume: float
    desplined_xz_face: BluemiraFace
    material: Material | None = None
    touching_components: list[str] | None = None

    def revolve_and_create_component(self) -> Component:
        """
        Revolve the desplined xz geometry and create the corresponding xyz
        component.

        Returns
        -------
        Component
        """
        xz_component = Component(
            "xz",
            children=[
                PhysicalComponent(
                    name=self.name,
                    shape=self.desplined_xz_face,
                    material=self.material,
                )
            ],
        )

        xyz_shape = revolve_shape(
            self.desplined_xz_face,
            base=(0, 0, 0),
            direction=(0, 0, 1),
            degree=360.0,
        )

        xyz_component = Component(
            "xyz",
            children=[
                PhysicalComponent(
                    name=self.name,
                    shape=xyz_shape,
                    material=self.material,
                )
            ],
        )

        return Component(self.name, children=[xz_component, xyz_component])


class AxisymmetricNeutronicsGeomManager(ComponentManager):
    """Manage components used to construct the neutronics geometry."""

    def __init__(
        self,
        component: Component,
        volume_differences: dict[str, float],
    ):
        super().__init__(component)
        self.volume_differences = volume_differences

    @staticmethod
    def check_touching(
        comp: Component,
        components: list[Component],
        overlap_tolerance: float,
    ) -> list[str]:
        """Return the names of components whose XZ faces touch ``comp``.

        ``comp`` is compared against the XZ face of each
        component in ``components``. Components with the same name as
        ``comp`` are excluded from the comparison.

        Parameters
        ----------
        comp
            Component whose XZ face is checked for touching.
        components
            Components whose XZ faces are checked against ``comp``.
        overlap_tolerance
            Tolerance used to determine whether two XZ face geometries
            are touching.

        Returns
        -------
        list[str]
            Names of components whose XZ faces touch the XZ face of ``comp``.
        """
        comp_xz_face = comp.get_component("xz").children[0].shape

        touching_components = []

        for other_comp in components:
            if other_comp.name == comp.name:
                continue

            other_xz_face = other_comp.get_component("xz").children[0].shape

            if check_touching_geos(
                comp_xz_face,
                other_xz_face,
                overlap_tolerance,
            ):
                touching_components.append(other_comp.name)

        return touching_components

    @classmethod
    def from_list_of_components(
        cls,
        components: list[Component],
        discretisations: list[int],
        overlap_tolerance: float = 1e-10,
        gap_tolerance: float = 1e-2,
        max_side_length: float = 0.15,
        min_side_length: float = 0.001,
    ) -> AxisymmetricNeutronicsGeomManager:
        """Create a neutronics geometry manager from components.

        Parameters
        ----------
        components
            Components to include in the neutronics geometry.
        discretisations
            Discretisation to use for each component.
        overlap_tolerance
            Tolerance used when checking whether XZ faces are touching.
        gap_tolerance
            Tolerance used when checking for gaps between components.
        max_side_length
            Maximum length of wire edges, in metres, used to warn
            when the requested discretisation may be insufficient.
        min_side_length
            Minimum length of wire edges, in metres. A warning is
            issued if any side is shorter than this length.

        Returns
        -------
        AxisymmetricNeutronicsGeomManager
            Manager containing the processed neutronics components.

        Raises
        ------
        ValueError
            If the number of discretisations does not match the number
            of components.
        ComponentError
            If the component does not have both xz and xyz geometry, or if the
            xz component does not contain exactly one face.
        """
        if len(discretisations) != len(components):
            raise ValueError(
                f"number of components {len(components)} "
                f"differs from provided number of discretisations "
                f"{len(discretisations)}"
            )

        # ---------------------------------------------------------
        # Desplining xz boundaries
        # * Assuming one child per component
        # If you have multiple children, unpack them before
        # running this
        # ---------------------------------------------------------
        all_neutronics_comps = []
        bluemira_print("Desplining xz components.")
        for comp, dscrt in zip(components, discretisations, strict=True):
            xz_component = comp.get_component("xz")
            xyz_component = comp.get_component("xyz")

            # only consider one sub-component
            if len(xz_component.children) > 1:
                bluemira_warn(
                    "Axissyeemtric neutronics module only supports "
                    "one child per xz component. Considering first child only."
                )

            face = xz_component.children[0].shape
            solid = xyz_component.children[0].shape
            material = xyz_component.children[0].material

            # Despline if the face has splines, or if the solid has any
            # face is of revolution type
            if not check_face_has_splines(face) and not check_solid_has_revolution(
                solid
            ):
                desplined_xz_face = face
            else:
                desplined_xz_face = despline_xz_face(
                    xz_face=face,
                    discretisation=dscrt,
                    name=xz_component.name,
                    parent_name=xz_component.parent.name,
                    max_side_length=max_side_length,
                    min_side_length=min_side_length,
                )

            all_neutronics_comps.append(
                AxisymmetricNeutronicsCompData(
                    name=comp.name,
                    original_volume=solid.volume,
                    touching_components=cls.check_touching(
                        comp,
                        components,
                        overlap_tolerance,
                    ),
                    desplined_xz_face=desplined_xz_face,
                    material=material,
                )
            )

        # ---------------------------------------------------------
        # Check and fix overlaps.
        # * Assuming one child per component
        # If you have multiple children, unpack them before
        # running this
        # ---------------------------------------------------------
        bluemira_print("Checking for possible overlaps and fixing them.")

        overlap_fixed_neutronics_comps = cls.fix_xz_overlaps(
            all_neutronics_comps,
            overlap_tolerance=overlap_tolerance,
        )

        gap_fixed_neutronics_comps = cls.fix_xz_gaps(
            overlap_fixed_neutronics_comps,
            gap_tolerance=gap_tolerance,
        )

        # ---------------------------------------------------------
        # Create XYZ components by revolution and assemble the
        # final component tree.
        # ---------------------------------------------------------
        component_tree = Component("Neutronics Geometry")
        volume_differences = {}

        for neutronics_comp in gap_fixed_neutronics_comps:
            component = neutronics_comp.revolve_and_create_component()

            component_tree.add_child(component)

            # Calculate and store the relative volume difference.
            relative_volume_difference = (
                abs(
                    component.get_component("xyz").children[0].shape.volume
                    - neutronics_comp.original_volume
                )
                / neutronics_comp.original_volume
            )

            volume_differences[neutronics_comp.name] = relative_volume_difference

            bluemira_debug(
                f"{neutronics_comp.name}: volume difference after pre-processing = "
                f"{relative_volume_difference:.4%}"
            )

        return cls(
            component_tree,
            volume_differences,
        )

    def get_all_components(self) -> list[Component]:
        """
        Return all component managers.

        Returns
        -------
        list[Component]
        """
        return self.component().children

    def inspect_xyz_overlaps(
        self,
        tolerance: float = 1e-10,
    ):
        """
        Inspect all xyz solids to ensure that there is no overlap
        Touching is allowed.

        Running on solids takes longer. Just use it for sanity
        checks.

        Parameters
        ----------
        tolerance
            Minimum intersection area considered to be an overlap.

        """
        all_comps = self.get_all_components()

        for i, comp_a in enumerate(all_comps):
            xyz_a = comp_a.get_component("xyz")
            name_a = comp_a.name
            xyz_a_solid = xyz_a.children[0].shape

            for _, comp_b in enumerate(
                all_comps[i + 1 :],
                start=i + 1,
            ):
                # assumes only one xz child
                name_b = comp_b.name
                xyz_b = comp_b.get_component("xyz")
                xyz_b_solid = xyz_b.children[0].shape
                intersection = xyz_a_solid.shape.intersect(xyz_b_solid.shape)
                intersection_vol = intersection.Volume()

                if intersection_vol > tolerance:
                    bluemira_warn(
                        f"Desplining created overlapping solids. "
                        f"{name_a} overlaps {name_b} by an volume of "
                        f"{intersection_vol} m^3"
                    )

    @staticmethod
    def fix_xz_overlaps(
        neutronics_comps: list[AxisymmetricNeutronicsCompData],
        overlap_tolerance: float = 1e-10,
    ) -> list[AxisymmetricNeutronicsCompData]:
        """Fix overlaps between desplined XZ faces.

        Components that should touch are identified from the
        ``touching_components`` stored on each ``AxisymmetricNeutronicsCompData``.

        For each component, all overlapping neighbours that should touch it
        are repaired in a single operation.

        Parameters
        ----------
        neutronics_comps
            Neutronics components containing the desplined XZ faces and
            names of components they should touch.
        overlap_tolerance
            Minimum intersection area considered to be an overlap.

        Returns
        -------
        list[AxisymmetricNeutronicsCompData]
            Neutronics component data with updated desplined XZ faces.
        """
        comps_by_name = {comp.name: comp for comp in neutronics_comps}

        for neutronics_comp in neutronics_comps:
            face_a = neutronics_comp.desplined_xz_face
            overlapping = []

            for touching_name in neutronics_comp.touching_components or []:
                face_b = comps_by_name[touching_name].desplined_xz_face

                intersection = face_a.shape.intersect(face_b.shape)
                intersection_area = intersection.Area()

                if intersection_area > overlap_tolerance:
                    overlapping.append((touching_name, face_b, intersection_area))

            if overlapping:
                bluemira_warn(
                    f"{neutronics_comp.name} overlaps "
                    f"{', '.join(name for name, _, _ in overlapping)} "
                    f"by areas "
                    f"{', '.join(f'{area} m^2' for _, _, area in overlapping)}. "
                    f"In the original geometry, they should touch instead. "
                    f"Running overlap fixing."
                )

                neutronics_comp.desplined_xz_face = repair_overlapping_geos(
                    face_a,
                    [face for _, face, _ in overlapping],
                )

        return neutronics_comps

    @staticmethod
    def fix_xz_gaps(
        neutronics_comps: list[AxisymmetricNeutronicsCompData],
        gap_tolerance: float = 1e-2,
    ) -> list[AxisymmetricNeutronicsCompData]:
        """Fix gaps between desplined XZ faces that should touch.

        Note that assumes Overlaps has already been removed.

        Components that should touch are identified from the
        ``touching_components`` stored on each ``AxisymmetricNeutronicsCompData``.

        Each touching pair is processed only once.

        Parameters
        ----------
        neutronics_comps
            Neutronics components containing the desplined XZ faces and
            names of components they should touch.
        gap_tolerance
            Maximum sewing tolerance for neighbouring XZ faces that should touch.

        Returns
        -------
        list[AxisymmetricNeutronicsCompData]
            Neutronics components with updated desplined XZ faces.
        """
        comps_by_name = {comp.name: comp for comp in neutronics_comps}
        repaired_pairs: set[frozenset[str]] = set()

        for neutronics_comp in neutronics_comps:
            for touching_name in neutronics_comp.touching_components or []:
                pair = frozenset((neutronics_comp.name, touching_name))

                if pair in repaired_pairs:
                    continue

                touching_comp = comps_by_name[touching_name]

                repaired_a, repaired_b = repair_gaps_between_faces(
                    [
                        neutronics_comp.desplined_xz_face,
                        touching_comp.desplined_xz_face,
                    ],
                    tolerance=gap_tolerance,
                )

                neutronics_comp.desplined_xz_face = repaired_a
                touching_comp.desplined_xz_face = repaired_b

                repaired_pairs.add(pair)

        return neutronics_comps
