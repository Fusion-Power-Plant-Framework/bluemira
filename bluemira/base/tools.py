# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Tool function and classes for the bluemira base module.
"""

from __future__ import annotations

import time
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, TypeVar

from bluemira.base.components import (
    Component,
    ComponentT,
    PhysicalComponent,
    get_properties_from_components,
)
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print
from bluemira.builders.tools import (
    circular_pattern_component,
    compound_from_components,
)
from bluemira.display.displayer import ComponentDisplayer
from bluemira.display.plotter import ComponentPlotter
from bluemira.geometry.compound import BluemiraCompound
from bluemira.geometry.tools import revolve_shape, save_cad, serialise_shape

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    import bluemira.codes._freecadapi as cadapi
    from bluemira.base.reactor import ComponentManager


_T = TypeVar("_T")


class CADConstructionType(Enum):
    """
    Enum for construction types for components
    """

    PATTERN_RADIAL = "PATTERN_RADIAL"
    REVOLVE_XZ = "REVOLVE_XZ"
    NO_OP = "NO_OP"


def _timing(
    func: Callable[..., _T],
    timing_prefix: str,
    info_str: str = "",
    *,
    debug_info_str: bool = False,
) -> Callable[..., _T]:
    """
    Time a function and push to logging.

    Returns
    -------
    :
        Wrapped function

    Parameters
    ----------
    func:
        Function to time
    timing_prefix:
        Prefix to print before time duration
    info_str:
        information to print before running function
    debug_info_str:
        send info_str to debug logger instead of info logger
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        """Time a function wrapper.

        Returns
        -------
        :
            Output of the function
        """
        if debug_info_str:
            bluemira_debug(info_str)
        else:
            bluemira_print(info_str)
        t1 = time.perf_counter()
        out = func(*args, **kwargs)
        t2 = time.perf_counter()
        bluemira_debug(f"{timing_prefix} {t2 - t1:.5g} s")
        return out

    return wrapper


def create_compound_from_component(comp: Component) -> BluemiraCompound:
    """
    Creates a BluemiraCompound from the children's shapes of a component.

    Parameters
    ----------
    comp:
        Component to create the compound from

    Returns
    -------
    :
        The BluemiraCompound component

    """
    if comp.is_leaf and hasattr(comp, "shape") and comp.shape:
        boundary = [comp.shape]
    else:
        boundary = [c.shape for c in comp.leaves if hasattr(c, "shape") and c.shape]

    return BluemiraCompound(label=comp.name, boundary=boundary)


def circular_pattern_xyz_components(
    comp: Component, n_sectors: int, degree: float
) -> Component:
    """
    Create a circular pattern of components in the XY plane.

    Parameters
    ----------
    comp:
        Component to pattern
    n_sectors:
        Number of sectors to pattern
    degree:
        Degree of the pattern

    Returns
    -------
    :
        The component with the circular pattern applied

    Raises
    ------
    ValueError
        If no xyz components are found in the component.
    """
    xyzs = comp.get_component(
        "xyz",
        first=False,
    )
    if xyzs is None:
        raise ValueError("No xyz components found in the component")
    xyzs = [xyzs] if isinstance(xyzs, Component) else xyzs
    for xyz in xyzs:
        xyz.children = circular_pattern_component(
            list(xyz.children),
            n_sectors,
            degree=degree,
        )
    return comp


def copy_and_filter_component(
    comp: Component,
    dim: str,
    component_filter: Callable[[ComponentT], bool] | None,
) -> Component:
    """
    Copies a component (deeply) then filters
    and returns the resultant component tree.

    Parameters
    ----------
    comp:
        Component to copy and filter
    dim:
        Dimension to filter (to keep)
    component_filter:
        Filter to apply to the components

    Returns
    -------
    :
        The copied and filtered component
    """
    c: Component = comp.copy()
    c.filter_components([dim], component_filter)
    return c


def save_components_cad(
    components: ComponentT | Iterable[ComponentT],
    filename: str,
    cad_format: str | cadapi.CADFileType = "stp",
    **kwargs,
):
    """
    Save the CAD build of the component.

    Parameters
    ----------
    components:
        Components to save
    filename:
        The filename of the
    cad_format:
        CAD file format
    """
    shapes, names = get_properties_from_components(components, ("shape", "name"))
    save_cad(shapes, filename, cad_format, names, **kwargs)


def show_components_cad(
    components: ComponentT | Iterable[ComponentT],
    **kwargs,
):
    """
    Show the CAD build of the component.
    """
    ComponentDisplayer().show_cad(
        components,
        **kwargs,
    )


def plot_component_dim(
    dim: str,
    component: ComponentT,
    **kwargs,
):
    """
    Plot the component in the specified dimension.
    """
    ComponentPlotter(view=dim, kwargs=kwargs).plot_2d(component)


def _construct_comp_manager_physical_comps(
    comp_manager: ComponentManager,
    component_filter: Callable[[ComponentT], bool] | None,
    n_sectors: int,
    sector_degrees: int,
) -> tuple[list[PhysicalComponent], str]:
    """
    Construct the compoent using the construction type
    and return the PhysicalComponents.

    Returns
    -------
    :
        A List of constructed PhysicalComponent's
        and the name to associate with them

    Raises
    ------
    ValueError
        If no components were constructed
    """
    # TODO: add construction params
    construction_type = comp_manager.cad_construction_type()
    # should cost nothing to get the component
    manager_comp: Component = comp_manager.component()
    manager_comp_name = manager_comp.name

    phy_comps = None
    match construction_type:
        case CADConstructionType.PATTERN_RADIAL:
            xyz_copy_and_filtered = copy_and_filter_component(
                manager_comp,
                "xyz",
                component_filter,
            )
            phy_comps = circular_pattern_xyz_components(
                xyz_copy_and_filtered,
                n_sectors,
                degree=sector_degrees,
            ).leaves
        case CADConstructionType.REVOLVE_XZ:
            xz_components = manager_comp.get_component("xz", first=False)
            xz_phy_comps: list[PhysicalComponent] = []
            for xz_c in xz_components:
                xz_phy_comps.extend(xz_c.leaves)
            phy_comps = [
                PhysicalComponent(
                    c.name,
                    revolve_shape(c.shape, degree=sector_degrees * n_sectors),
                    material=c.material,
                )
                for c in xz_phy_comps
            ]
        case CADConstructionType.NO_OP:
            phy_comps = copy_and_filter_component(
                manager_comp,
                "xyz",
                component_filter,
            ).leaves

    if not phy_comps:
        raise ValueError(f"No components were constructed for {manager_comp_name}")

    return phy_comps, manager_comp_name


def _group_physical_components_by_material(
    phy_comps: list[PhysicalComponent],
) -> dict[str, list[PhysicalComponent]]:
    """
    Group the physical components by material name.

    Returns
    -------
    :
        A dictionary of material name to list of physical components
    """
    mat_to_comps_map = {}
    for phy_comp in phy_comps:
        mat_name = "" if phy_comp.material is None else phy_comp.material.name
        if mat_name not in mat_to_comps_map:
            mat_to_comps_map[mat_name] = []
        mat_to_comps_map[mat_name].append(phy_comp)
    return mat_to_comps_map


def _build_compounds_from_map(
    mat_to_comps_map: dict[str, list[PhysicalComponent]],
    manager_name: str,
) -> list[PhysicalComponent]:
    """
    Build the compounds from the material to components map.

    Returns
    -------
    :
        A list of compounds
    """
    return [
        PhysicalComponent(
            name=f"{manager_name}_{comps[0].name}",
            shape=comps[0].shape,
            material=comps[0].material,
        )
        if len(comps) == 1
        # Component(
        #     f"{manager_name}_{mat_name}" if mat_name else manager_name,
        #     children=comps,
        # )
        else compound_from_components(
            comps,
            f"{manager_name}_{mat_name}" if mat_name else manager_name,
        )
        for mat_name, comps in mat_to_comps_map.items()
    ]


def build_comp_manager_save_xyz_cad_tree(
    comp_manager: ComponentManager,
    component_filter: Callable[[ComponentT], bool] | None,
    n_sectors: int,
    sector_degrees: int,
) -> Component:
    """
    Build the CAD of the component manager's components
    and save the CAD to a file.

    Parameters
    ----------
    comp_manager:
        Component manager
    component_filter:
        Filter to apply to the components

    Returns
    -------
    :
        The constructed component manager component for CAD saving
    """
    constructed_phy_comps, manager_name = _construct_comp_manager_physical_comps(
        comp_manager,
        component_filter,
        n_sectors,
        sector_degrees,
    )

    mat_to_comps_map = _group_physical_components_by_material(constructed_phy_comps)

    return_comp = Component(name=manager_name)
    return_comp.children = _build_compounds_from_map(
        mat_to_comps_map,
        manager_name,
    )

    return return_comp


def build_comp_manager_show_cad_tree(
    comp_manager: ComponentManager,
    dim: str,
    component_filter: Callable[[ComponentT], bool] | None,
    n_sectors: int,
    sector_degrees: int,
) -> Component:
    """
    Build the CAD of the component manager's components
    and save the CAD to a file.

    Parameters
    ----------
    comp_manager:
        Component manager
    dim:
        Dimension to build the CAD in
    component_filter:
        Filter to apply to the components

    Returns
    -------
    :
        The constructed component manager component for CAD showing
    """
    manager_comp = comp_manager.component()
    filtered_comp = copy_and_filter_component(
        manager_comp,
        dim,
        component_filter,
    )
    if dim == "xyz":
        filtered_comp = circular_pattern_xyz_components(
            filtered_comp,
            n_sectors,
            degree=sector_degrees,
        )
    return filtered_comp


# # =============================================================================
# # Serialise and Deserialise
# # =============================================================================
def serialise_component(comp: Component) -> dict:
    """
    Serialise a Component object.

    Parameters
    ----------
    comp:
        The Component object to serialise

    Returns
    -------
    :
        The serialised Component object as a dictionary
    """
    type_ = type(comp)

    if isinstance(comp, Component):
        output = [serialise_component(child) for child in comp.children]
        cdict = {"label": comp.name, "children": output}
        if isinstance(comp, PhysicalComponent):
            cdict["shape"] = serialise_shape(comp.shape)
        return {str(type(comp).__name__): cdict}
    raise NotImplementedError(f"Serialisation non implemented for {type_}")
