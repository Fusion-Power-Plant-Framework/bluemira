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
import bluemira.geometry as geo

from .components import Component, MagneticComponent, PhysicalComponent


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
            dict["shape"] = geo.tools.serialize_shape(comp.shape)
        return {str(type(comp).__name__): dict}
    else:
        raise NotImplementedError(f"Serialization non implemented for {type_}")
