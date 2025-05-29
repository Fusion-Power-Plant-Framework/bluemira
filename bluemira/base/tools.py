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
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import TYPE_CHECKING, Any, TypeVar, TypedDict

from typing_extensions import NotRequired

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
from bluemira.geometry.tools import (
    revolve_shape,
    save_cad,
    serialise_shape,
)
from bluemira.materials.material import Material, Void
from bluemira.radiation_transport.neutronics.dagmc import save_cad_to_dagmc

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from pathlib import Path

    import bluemira.codes._freecadapi as cadapi
    from bluemira.base.reactor import ComponentManager


_T = TypeVar("_T")


class FilterMaterial:
    """
    Filter nodes by material

    Parameters
    ----------
    keep_material:
       materials to include
    reject_material:
       materials to exclude

    """

    __slots__ = ("keep_material", "reject_material")

    def __init__(
        self,
        keep_material: type[Material] | tuple[type[Material]] | None = None,
        reject_material: type[Material] | tuple[type[Material]] | None = Void,
    ):
        super().__setattr__("keep_material", keep_material)
        super().__setattr__("reject_material", reject_material)

    def __call__(self, node: ComponentT) -> bool:
        """Filter node based on material include and exclude rules.

        Parameters
        ----------
        node:
            The node to filter.

        Returns
        -------
        :
            True if the node should be kept, False otherwise.
        """
        if hasattr(node, "material"):
            return self._apply_filters(node.material)
        return True

    def __setattr__(self, name: str, value: Any):
        """
        Override setattr to force immutability

        This method makes the class nearly immutable as no new attributes
        can be modified or added by standard methods.

        See #2236 discussion_r1191246003 for further details

        Raises
        ------
        AttributeError
            FilterMaterial is immutable
        """
        raise AttributeError(f"{type(self).__name__} is immutable")

    def _apply_filters(self, material: Material | tuple[Material]) -> bool:
        bool_store = True

        if self.keep_material is not None:
            bool_store = isinstance(material, self.keep_material)

        if self.reject_material is not None:
            bool_store = not isinstance(material, self.reject_material)

        return bool_store


class CADConstructionType(Enum):
    """
    Enum for construction types for components
    """

    PATTERN_RADIAL = "PATTERN_RADIAL"
    REVOLVE_XZ = "REVOLVE_XZ"
    NO_OP = "NO_OP"


class ConstructionParams(TypedDict):
    """
    Parameters for the construction of CAD.
    """

    with_components: NotRequired[list[ComponentManager] | None]
    without_components: NotRequired[list[ComponentManager] | None]
    component_filter: NotRequired[Callable[[Component], bool] | None]
    n_sectors: NotRequired[int | None]
    total_sectors: NotRequired[int | None]
    group_by_materials: NotRequired[bool]
    disable_composite_grouping: NotRequired[bool]


@dataclass
class ConstructionParamValues:
    """
    Parameters for the construction of CAD.
    """

    with_components: list[ComponentManager] | None
    without_components: list[ComponentManager] | None
    component_filter: Callable[[Component], bool] | None
    n_sectors: int
    total_sectors: int
    group_by_materials: bool
    disable_composite_grouping: bool

    @classmethod
    def empty(cls) -> ConstructionParamValues:
        """
        Create an empty ConstructionParamValues object.

        Returns
        -------
        :
            The empty ConstructionParamValues object
        """
        return cls(
            with_components=None,
            without_components=None,
            component_filter=None,
            n_sectors=1,
            total_sectors=1,
            group_by_materials=False,
            disable_composite_grouping=False,
        )

    @classmethod
    def from_construction_params(cls, construction_params: ConstructionParams | None):
        """
        Create the ConstructionParamValues from the ConstructionParams.

        Parameters
        ----------
        construction_params:
            Construction parameters to extract values from.

        Returns
        -------
        :
            The ConstructionParamValues object
        """
        construction_params = construction_params or {}
        comp_filter = (
            construction_params["component_filter"]
            if "component_filter" in construction_params
            else FilterMaterial()
        )

        tot_secs = int(construction_params.get("total_sectors", 1))
        n_secs = int(construction_params.get("n_sectors", tot_secs))

        return cls(
            with_components=construction_params.get("with_components"),
            without_components=construction_params.get("without_components"),
            component_filter=comp_filter,
            n_sectors=n_secs,
            total_sectors=tot_secs,
            group_by_materials=construction_params.get("group_by_materials", False),
            disable_composite_grouping=construction_params.get(
                "disable_composite_grouping", False
            ),
        )


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
            bluemira_debug(info_str, stacklevel=4)
        else:
            bluemira_print(info_str, stacklevel=4)
        t1 = time.perf_counter()
        out = func(*args, **kwargs)
        t2 = time.perf_counter()
        bluemira_debug(f"{timing_prefix} {t2 - t1:.5g} s")
        return out

    return wrapper


def create_compound_from_component(comp: Component) -> BluemiraCompound:
    """
    Creates a BluemiraCompound from the shapes at the root of the component's
    component tree.

    Parameters
    ----------
    comp:
        Component to create the compound from

    Returns
    -------
    :
        The BluemiraCompound component

    """
    shapes = get_properties_from_components(comp, ("shape"))
    return BluemiraCompound(shapes, comp.name)


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
    filename: Path,
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
        The full filename path to save the CAD to
    cad_format:
        CAD file format
    """
    shapes, names, mats = get_properties_from_components(
        components, ("shape", "name", "material"), extract=False
    )

    if cad_format == "dagmc":
        save_cad_to_dagmc(
            shapes,
            names,
            filename,
            comp_mat_mapping={
                n: "undef_material" if m is None else m.name
                for n, m in zip(
                    names,
                    mats,
                    strict=False,
                )
            },
            converter_config=kwargs.get("converter_config"),
        )
    else:
        save_cad(shapes, filename, cad_format, names, **kwargs)


def show_components_cad(
    components: ComponentT | Iterable[ComponentT],
    **kwargs,
):
    """
    Show the CAD build of the component.
    """
    ComponentDisplayer().show_cad(components, **kwargs)


def plot_component_dim(
    dim: str,
    component: ComponentT,
    **kwargs,
):
    """
    Plot the component in the specified dimension.
    """
    ComponentPlotter(view=dim, **kwargs).plot_2d(component)


def _construct_comp_manager_physical_comps(
    comp_manager: ComponentManager,
    construction_params: ConstructionParamValues,
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
    construction_type = comp_manager.cad_construction_type()

    # should cost nothing to get the component
    manager_comp: Component = comp_manager.component()
    manager_comp_name = manager_comp.name

    component_filter = construction_params.component_filter
    tot_secs = construction_params.total_sectors
    n_secs = construction_params.n_sectors
    sec_degrees = int((360 / tot_secs) * n_secs)

    xyz_phy_comps = None
    if construction_type is CADConstructionType.REVOLVE_XZ:
        xz_phy_comps: list[PhysicalComponent] = copy_and_filter_component(
            manager_comp,
            "xz",
            component_filter,
        ).leaves

        xyz_phy_comps = [
            PhysicalComponent(
                c.name,
                revolve_shape(c.shape, degree=sec_degrees),
                material=c.material,
            )
            for c in xz_phy_comps
        ]
    else:
        xyz_copy_and_filtered = copy_and_filter_component(
            manager_comp,
            "xyz",
            component_filter,
        )
        match construction_type:
            case CADConstructionType.PATTERN_RADIAL:
                xyz_phy_comps = circular_pattern_xyz_components(
                    xyz_copy_and_filtered,
                    n_secs,
                    degree=sec_degrees,
                ).leaves
            case CADConstructionType.NO_OP:
                xyz_phy_comps = xyz_copy_and_filtered.leaves

    if not xyz_phy_comps:
        raise ValueError(f"No components were constructed for {manager_comp_name}")

    return xyz_phy_comps, manager_comp_name


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


def _build_compounds_from_mat_map(
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
        # recreate the PhysicalComponent to rename it
        PhysicalComponent(
            name=f"{manager_name}_mat_{mat_name}" if mat_name else manager_name,
            shape=comps[0].shape,
            material=comps[0].material,
        )
        if len(comps) == 1
        else compound_from_components(
            name=f"{manager_name}_mat_{mat_name}" if mat_name else manager_name,
            components=comps,
            # all comps in the list have the same material
            # (when not grouped by material correctly, in the map)
            material=comps[0].material,
        )
        for mat_name, comps in mat_to_comps_map.items()
    ]


def build_comp_manager_save_xyz_cad_tree(
    comp_manager: ComponentManager,
    construction_params: ConstructionParamValues,
) -> Component:
    """
    Build the CAD of the component manager's components.

    Parameters
    ----------
    comp_manager:
        Component manager
    construction_params:
        Construction parameters to use for CAD building.

    Returns
    -------
    :
        The constructed component manager component for CAD saving
    """
    constructed_phy_comps, manager_name = _construct_comp_manager_physical_comps(
        comp_manager, construction_params
    )

    return_comp = Component(name=manager_name)

    if construction_params.disable_composite_grouping:
        # if disabled, simply set the constructed components as the children
        # and return
        return_comp.children = constructed_phy_comps
        return return_comp

    if construction_params.group_by_materials:
        mat_to_comps_map = _group_physical_components_by_material(constructed_phy_comps)
    else:
        # note: by assigning the empty string as the key, we are
        # grouping all phy. components together. They could
        # have different materials, which will get lost.
        # Only the first material will be used in _build_compounds_from_mat_map.
        # This option makes the CAD output cleaner
        # (and material information is not saved in the CAD file)
        # so it is not a problem (usually), except for code that operates
        # on the shapes downstream (such as DAGMC exporting).
        mat_to_comps_map = {"": constructed_phy_comps}

    return_comp.children = _build_compounds_from_mat_map(
        mat_to_comps_map,
        manager_name,
    )

    return return_comp


def build_comp_manager_show_cad_tree(
    comp_manager: ComponentManager,
    dim: str,
    construction_params: ConstructionParamValues,
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
    component_filter = construction_params.component_filter

    manager_comp: Component = comp_manager.component()
    filtered_comp = copy_and_filter_component(
        manager_comp,
        dim,
        component_filter,
    )

    if dim == "xyz":
        tot_secs = construction_params.total_sectors
        n_secs = construction_params.n_sectors
        sec_degrees = int((360 / tot_secs) * n_secs)

        filtered_comp = circular_pattern_xyz_components(
            filtered_comp,
            n_secs,
            degree=sec_degrees,
        )

    return_comp = Component(name=manager_comp.name)
    return_comp.children = [filtered_comp]

    return return_comp


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
