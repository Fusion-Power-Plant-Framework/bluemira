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
Tool function and classes for the bluemira base module.
"""
from bluemira.base.components import Component, PhysicalComponent
from bluemira.geometry.compound import BluemiraCompound
from bluemira.geometry.tools import make_compound, serialize_shape


def create_compound_from_component(comp):
    """
    Creates a BluemiraCompound from the children's shapes of a component.
    """
    shape = []
    if comp.is_leaf and hasattr(comp, "shape") and comp.shape:
        shape.append(comp.shape)
    else:
        for c in comp.leaves:
            if hasattr(c, "shape") and c.shape:
                shape.append(c.shape)
    shape = make_compound(shape)
    compound = BluemiraCompound(label=comp.name, shape=shape)
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
        dict = {"label": comp.name, "children": output}
        for child in comp.children:
            output.append(serialize_component(child))
        if isinstance(comp, PhysicalComponent):
            dict["shape"] = serialize_shape(comp.shape)
        return {str(type(comp).__name__): dict}
    else:
        raise NotImplementedError(f"Serialization non implemented for {type_}")
