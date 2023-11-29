# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Tool function and classes for the bluemira base module.
"""

import time
from functools import wraps
from typing import Callable, Dict, TypeVar

from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print
from bluemira.geometry.compound import BluemiraCompound
from bluemira.geometry.tools import serialize_shape

_T = TypeVar("_T")


def _timing(
    func: Callable[..., _T],
    timing_prefix: str,
    info_str: str = "",
    debug_info_str: bool = False,
) -> Callable[..., _T]:
    """
    Time a function and push to logging

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
        """Time a function wrapper"""
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
    """
    if comp.is_leaf and hasattr(comp, "shape") and comp.shape:
        boundary = [comp.shape]
    else:
        boundary = [c.shape for c in comp.leaves if hasattr(c, "shape") and c.shape]

    return BluemiraCompound(label=comp.name, boundary=boundary)


# # =============================================================================
# # Serialize and Deserialize
# # =============================================================================
def serialize_component(comp: Component) -> Dict:
    """
    Serialize a Component object.
    """
    type_ = type(comp)

    if isinstance(comp, Component):
        output = [serialize_component(child) for child in comp.children]
        cdict = {"label": comp.name, "children": output}
        if isinstance(comp, PhysicalComponent):
            cdict["shape"] = serialize_shape(comp.shape)
        return {str(type(comp).__name__): cdict}
    raise NotImplementedError(f"Serialization non implemented for {type_}")
