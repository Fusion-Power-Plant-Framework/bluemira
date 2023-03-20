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
"""CAD builder for PF coils."""

from dataclasses import dataclass

from bluemira.base.components import Component
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame import Parameter, ParameterFrame, make_parameter_frame
from bluemira.builders.pf_coil import PFCoilBuilder, PFCoilPictureFrame


@dataclass
class PFCoilsBuilderParams(ParameterFrame):
    """
    Parameters for the `PFCoilsBuilder` class.
    """

    n_TF: Parameter[int]
    tk_pf_insulation: Parameter[float]
    tk_pf_casing: Parameter[float]
    tk_cs_insulation: Parameter[float]
    tk_cs_casing: Parameter[float]
    r_pf_corner: Parameter[float]
    r_cs_corner: Parameter[float]


def build_pf_coils_component(params, build_config, coilset):
    """
    Build the PF coils component
    """
    params = make_parameter_frame(params, PFCoilsBuilderParams)

    wires = []
    for name in coilset.name:
        coil = coilset[name]
        coil_type = coil.ctype
        r_corner = (
            params.r_pf_corner.value if coil_type == "PF" else params.r_cs_corner.value
        )
        if not (coil.dx == 0 or coil.dz == 0):
            wires.append(
                (
                    PFCoilPictureFrame(
                        {"r_corner": {"value": r_corner, "unit": "m"}}, coil
                    ),
                    coil_type,
                    name,
                )
            )
        else:
            bluemira_warn(f"Coil {name} has no size")

    pf_builders = []
    cs_builders = []
    for designer, coil_type, coil_name in wires:
        tk_ins = (
            params.tk_pf_insulation.value
            if coil_type.name == "PF"
            else params.tk_cs_insulation.value
        )
        tk_case = (
            params.tk_pf_casing.value
            if coil_type.name == "PF"
            else params.tk_cs_casing.value
        )
        bc = {
            **build_config,
            "name": coil_name,
        }
        builder = PFCoilBuilder(
            {
                "n_TF": {"value": params.n_TF.value, "unit": params.n_TF.unit},
                "tk_insulation": {"value": tk_ins, "unit": "m"},
                "tk_casing": {"value": tk_case, "unit": "m"},
                "ctype": {"value": coil_type.name, "unit": ""},
            },
            bc,
            designer.execute(),
        )
        if coil_type.name == "PF":
            pf_builders.append(builder)
        else:
            cs_builders.append(builder)

    pf_coils = Component(
        "PF coils", children=[builder.build() for builder in pf_builders]
    )
    cs_coils = Component(
        "CS coils", children=[builder.build() for builder in cs_builders]
    )

    return Component("Poloidal Coils", children=[pf_coils, cs_coils])
