# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Tool function and classes for the bluemira base module.
"""

import re
import time
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Iterable, TypeVar

import bluemira.codes._freecadapi as cadapi
from bluemira.base.components import (
    Component,
    ComponentT,
    PhysicalComponent,
    get_properties_from_components,
)
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print
from bluemira.builders.tools import circular_pattern_component
from bluemira.display.displayer import ComponentDisplayer
from bluemira.display.plotter import ComponentPlotter
from bluemira.geometry.compound import BluemiraCompound
from bluemira.geometry.tools import save_cad, serialise_shape

_T = TypeVar("_T")


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
    ComponentDisplayer().show_cad(
        components,
        **kwargs,
    )


def plot_component_dim(
    dim: str,
    component: ComponentT,
    **kwargs,
):
    ComponentPlotter(view=dim).plot_2d(component)


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
