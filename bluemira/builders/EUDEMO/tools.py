# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
A collection of tools used in the EU-DEMO design.
"""

import copy

import bluemira.base.components as bm_comp
import bluemira.geometry as bm_geo
from bluemira.builders.EUDEMO._varied_offset import varied_offset  # noqa :F401


def circular_pattern_component(
    component: bm_comp.Component,
    n_children: int,
    parent_prefix: str = "Sector",
    *,
    origin=(0.0, 0.0, 0.0),
    direction=(0.0, 0.0, 1.0),
    degree=360.0,
):
    """
    Pattern the provided Component equally spaced around a circle n_children times.

    The resulting components are assigned to a set of common parent Components having
    a name with the structure "{parent_prefix} {idx}", where idx runs from 1 to
    n_children. The Components produced under each parent are named according to the
    original Component with the corresponding idx value appended.

    Parameters
    ----------
    component: Component
        The original Component to use as the template for copying around the circle.
    n_children: int
        The number of children to produce around the circle.
    parent_prefix: str
        The prefix to provide to the new parent component, having a name of the form
        "{parent_prefix} {idx}", by default "Sector".
    origin: Tuple[float, float, float]
        The origin of the circle to pattern around, by default (0., 0., 0.).
    direction: Tuple[float, float, float]
        The surface normal of the circle to pattern around, by default (0., 0., 1.) i.e.
        the positive z axis, resulting in a counter clockwise circle in the x-y plane.
    degree: float
        The angular extent of the patterning in degrees, by default 360.
    """
    sectors = [
        bm_comp.Component(f"{parent_prefix} {idx+1}") for idx in range(n_children)
    ]

    def assign_component_to_sector(
        comp: bm_comp.Component,
        sector: bm_comp.Component,
        shape: bm_geo.base.BluemiraGeo = None,
    ):
        idx = int(sector.name.replace(f"{parent_prefix} ", ""))

        if shape is not None and not shape.label:
            shape.label = f"{comp.name} {idx}"

        comp = copy.deepcopy(comp)
        comp.name = f"{comp.name} {idx}"

        comp.children = []
        orig_parent: bm_comp.Component = comp.parent
        if orig_parent is not None:
            comp.parent = sector.get_component(f"{orig_parent.name} {idx}")
        if comp.parent is None:
            comp.parent = sector

        if isinstance(comp, bm_comp.PhysicalComponent):
            comp.shape = shape

    def assign_or_pattern(comp: bm_comp.Component):
        if isinstance(comp, bm_comp.PhysicalComponent):
            shapes = bm_geo.tools.circular_pattern(
                comp.shape,
                n_shapes=n_children,
                origin=origin,
                direction=direction,
                degree=degree,
            )
            for sector, shape in zip(sectors, shapes):
                assign_component_to_sector(comp, sector, shape)
        else:
            for sector in sectors:
                assign_component_to_sector(comp, sector)

    def process_children(comp: bm_comp.Component):
        for child in comp.children:
            assign_or_pattern(child)
            process_children(child)

    assign_or_pattern(component)
    process_children(component)

    return sectors
