# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
Tool function and classes for the bluemira base module.
"""
import time
from typing import Callable

from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print
from bluemira.geometry.compound import BluemiraCompound
from bluemira.geometry.tools import serialize_shape


def timing(
    func: Callable, timing_str: str, print_str: str = "", print_name: bool = True
) -> Callable:
    """Time a function and push to logging"""

    def wrapper(*args, **kwargs):
        """Time a function wrapper"""
        if print_name:
            bluemira_print(print_str)
        else:
            bluemira_debug(print_str)
        t1 = time.perf_counter()
        out = func(*args, **kwargs)
        t2 = time.perf_counter()
        bluemira_debug(f"{timing_str} {t2 - t1:.5g} s")
        return out

    wrapper.__doc__ = func.__doc__
    return wrapper


def create_compound_from_component(comp):
    """
    Creates a BluemiraCompound from the children's shapes of a component.
    """
    boundary = []
    if comp.is_leaf and hasattr(comp, "shape") and comp.shape:
        boundary.append(comp.shape)
    else:
        for c in comp.leaves:
            if hasattr(c, "shape") and c.shape:
                boundary.append(c.shape)
    compound = BluemiraCompound(label=comp.name, boundary=boundary)
    return compound


# # =============================================================================
# # Serialize and Deserialize
# # =============================================================================
def serialize_component(comp: Component):
    """
    Serialize a Component object.
    """
    type_ = type(comp)

    output = []
    if isinstance(comp, Component):
        cdict = {"label": comp.name, "children": output}
        for child in comp.children:
            output.append(serialize_component(child))
        if isinstance(comp, PhysicalComponent):
            cdict["shape"] = serialize_shape(comp.shape)
        return {str(type(comp).__name__): cdict}
    else:
        raise NotImplementedError(f"Serialization non implemented for {type_}")
