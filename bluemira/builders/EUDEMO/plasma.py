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
A builder for Plasma properties and geometry
"""

from __future__ import annotations

import numpy as np
from typing import Callable, List, Optional

from bluemira.base.builder import Builder, BuildConfig
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.config import Configuration
from bluemira.base.constants import BLUEMIRA_PALETTE
from bluemira.base.error import BuilderError
from bluemira.base.look_and_feel import bluemira_print

from bluemira.equilibria.constants import (
    NBTI_J_MAX,
    NBTI_B_MAX,
    NB3SN_J_MAX,
    NB3SN_B_MAX,
)
from bluemira.equilibria import AbInitioEquilibriumProblem
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.shapes import JohnerLCFS
import bluemira.geometry as geo
from bluemira.geometry._deprecated_loop import Loop
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.utilities.tools import get_class_from_module


class PlasmaComponent(Component):
    """
    A component containing the equilibrium used to build a plasma.
    """

    def __init__(
        self,
        name: str,
        parent: Optional[Component] = None,
        children: Optional[List[Component]] = None,
        equilibrium: Optional[Equilibrium] = None,
    ):
        super().__init__(name, parent=parent, children=children)

        self._equilibrium = equilibrium

    @property
    def equilibrium(self) -> Equilibrium:
        """
        The equilibrium used to build this plasma.
        """
        return self._equilibrium

    @equilibrium.setter
    def equilibrium(self, val: Equilibrium):
        self._equilibrium = val


def run_equilibrium_callback(self: PlasmaBuilder, **kwargs):
    eq = create_equilibrium(self)
    analyse_equilibrium(self, eq)
    self._boundary = geo.tools.make_polygon(eq.get_LCFS().xyz.T, "LCFS")
    return eq


def read_equilibrium_callback(self: PlasmaBuilder, **kwargs):
    if "eqdsk_path" not in kwargs or kwargs["eqdsk_path"] is None:
        raise BuilderError(
            "Must supply eqdsk_path as a kwarg when using read_equilibrium_callback"
        )
    eq = read_equilibrium(self, kwargs["eqdsk_path"])
    self._boundary = geo.tools.make_polygon(eq.get_LCFS().xyz.T, "LCFS")
    return eq


def mock_equilibrium_callback(self: PlasmaBuilder, **kwargs):
    shape = JohnerLCFS()
    for var, param in {"r_0": "R_0", "a": "A"}.items():
        shape.adjust_variable(var, self._params[param])
    self._boundary = shape.create_shape()


class PlasmaBuilder(Builder):
    """
    A builder for Plasma properties and geometry.
    """

    _required_params: List[str] = [
        "R_0",
        "B_0",
        "A",
        "I_p",
        "beta_p",
        "P_sep",
        "l_i",
        "kappa_95",
        "kappa",
        "delta_95",
        "delta",
        "q_95",
        "shaf_shift",
        "div_L2D_ib",
        "div_L2D_ob",
        "r_cs_in",
        "r_tf_in_centre",
        "r_tf_out_centre",
        "tk_cs",
        "n_PF",
        "n_CS",
        "PF_material",
        "CS_material",
        "C_Ejima",
        "reactor_type",
        "plasma_type",
    ]

    _params: Configuration
    _boundary: geo.wire.BluemiraWire
    _plot_flag: bool
    _segment_angle: float

    def _extract_config(self, build_config: BuildConfig):
        super()._extract_config(build_config)

        self._plot_flag = build_config.get("plot_flag", False)
        self._segment_angle = build_config.get("segment_angle", 360.0)

    def __call__(
        self,
        params,
        eq_callback: Optional[Callable[[PlasmaBuilder], Optional[Equilibrium]]] = None,
        **kwargs,
    ) -> Component:
        if isinstance(eq_callback, str):
            eq_callback = get_class_from_module(
                eq_callback, default_module=__loader__.name
            )
        if eq_callback is None:
            eq_callback = run_equilibrium_callback

        self.reinitialise(params, **kwargs)
        eq = eq_callback(self, **kwargs)
        return self.build(equilibrium=eq, **kwargs)

    def reinitialise(self, params, **kwargs) -> None:
        super().reinitialise(params, **kwargs)

        self._boundary = None

    def build(self, equilibrium=None, **kwargs) -> Component:
        super().build(**kwargs)

        component = PlasmaComponent(self._name, equilibrium=equilibrium)

        component.add_child(self.build_xz(equilibrium=equilibrium, **kwargs))
        component.add_child(self.build_xy(**kwargs))
        component.add_child(self.build_xyz(**kwargs))

        return component

    def build_xz(self, equilibrium: Optional[Equilibrium] = None, **kwargs):
        component = Component("xz")

        if equilibrium is not None:
            sep_loop = equilibrium.get_separatrix()
            sep_wire = geo.tools.make_polygon(sep_loop.xyz.T, label="Separatrix")
            sep_component = PhysicalComponent("Separatrix", sep_wire)
            sep_component.plot_options.wire_options["color"] = BLUEMIRA_PALETTE[6]
            component.add_child(sep_component)

        lcfs_face = geo.face.BluemiraFace(self._boundary, label="LCFS")
        lcfs_component = PhysicalComponent("LCFS", lcfs_face)
        lcfs_component.plot_options.wire_options["color"] = BLUEMIRA_PALETTE[6]
        lcfs_component.plot_options.face_options["color"] = BLUEMIRA_PALETTE[7]
        component.add_child(lcfs_component)

        return component

    def build_xy(self, **kwargs):
        inner = geo.tools.make_circle(self._boundary.bounding_box.x_min, axis=[0, 1, 0])
        outer = geo.tools.make_circle(self._boundary.bounding_box.x_max, axis=[0, 1, 0])

        face = geo.face.BluemiraFace([outer, inner], label="LCFS")
        component = PhysicalComponent("LCFS", face)
        component.plot_options.wire_options["color"] = BLUEMIRA_PALETTE[6]
        component.plot_options.face_options["color"] = BLUEMIRA_PALETTE[7]

        return Component("xy").add_child(component)

    def build_xyz(self, segment_angle: Optional[float] = None, **kwargs):
        if segment_angle is None:
            segment_angle = self._segment_angle

        shell = geo.tools.revolve_shape(
            self._boundary, direction=(0, 0, 1), degree=segment_angle
        )
        component = PhysicalComponent("LCFS", shell)
        component.display_cad_options.color = BLUEMIRA_PALETTE[7]
        component.display_cad_options.transparency = 0.5

        return Component("xyz").add_child(component)


def read_equilibrium(self: PlasmaBuilder, eqdsk_path: str, **kwargs) -> Equilibrium:
    """
    Read a reference MHD equilibrium for the Reactor.
    """
    bluemira_print("Reading reference plasma MHD equilibrium.")

    return Equilibrium.from_eqdsk(eqdsk_path)


def create_equilibrium(self: PlasmaBuilder, **kwargs) -> Equilibrium:
    """
    Creates a reference MHD equilibrium for the Reactor.
    """
    bluemira_print("Generating reference plasma MHD equilibrium.")

    # First make an initial TF coil shape along which to auto-position
    # some starting PF coil locations. We will design the TF later
    rin, rout = self._params["r_tf_in_centre"], self._params["r_tf_out_centre"]

    # TODO: Handle other TF coil parameterisations?
    shape = PrincetonD()
    for key, val in {"x1": rin, "x2": rout, "dz": 0}.items():
        shape.adjust_variable(key, value=val)

    tf_boundary = shape.create_shape()
    if self._params.delta_95 < 0:  # Negative triangularity
        tf_boundary.rotate(tf_boundary.center_of_mass, direction=(0, 1, 0), degree=180)
    tf_boundary = geo.tools.offset_wire(tf_boundary, -0.5)

    # TODO: Avoid converting to (deprecated) Loop
    # TODO: Agree on numpy array dimensionality
    tf_boundary = Loop(*tf_boundary.discretize().T)

    profile = None

    # TODO: Can we make it so that the equilibrium problem being used can be
    # configured?
    a = AbInitioEquilibriumProblem(
        self._params.R_0.value,
        self._params.B_0.value,
        self._params.A.value,
        self._params.I_p.value * 1e6,  # MA to A
        self._params.beta_p.value / 1.3,  # TODO: beta_N vs beta_p here?
        self._params.l_i.value,
        # TODO: 100/95 problem
        # TODO: This is a parameter patch... switch to strategy pattern
        self._params.kappa_95.value,
        1.2 * self._params.kappa_95.value,
        self._params.delta_95.value,
        1.2 * self._params.delta_95.value,
        -20,
        5,
        60,
        30,
        self._params.div_L2D_ib.value,
        self._params.div_L2D_ob.value,
        self._params.r_cs_in.value + self._params.tk_cs.value / 2,
        self._params.tk_cs.value / 2,
        tf_boundary,
        self._params.n_PF.value,
        self._params.n_CS.value,
        c_ejima=self._params.C_Ejima.value,
        eqtype=self._params.plasma_type.value,
        rtype=self._params.reactor_type.value,
        profile=profile,
    )

    # TODO: Handle these through properties on actual materials.
    if self._params.PF_material.value == "NbTi":
        j_pf = NBTI_J_MAX
        b_pf = NBTI_B_MAX
    elif self._params.PF_material.value == "Nb3Sn":
        j_pf = NB3SN_J_MAX
        b_pf = NB3SN_B_MAX
    else:
        raise ValueError("Unrecognised material string")

    if self._params.CS_material.value == "NbTi":
        j_cs = NBTI_J_MAX
        b_pf = NBTI_B_MAX
    elif self._params.CS_material.value == "Nb3Sn":
        j_cs = NB3SN_J_MAX
        b_cs = NB3SN_B_MAX

    a.coilset.assign_coil_materials("PF", j_max=j_pf, b_max=b_pf)
    a.coilset.assign_coil_materials("CS", j_max=j_cs, b_max=b_cs)
    a.solve(plot=self._plot_flag)

    return a.eq


def analyse_equilibrium(self: PlasmaBuilder, eq: Equilibrium):
    """
    Analyse an equilibrium and store important values in the Reactor parameters.
    """
    plasma_dict = eq.analyse_plasma()

    dx_shaf = plasma_dict["dx_shaf"]
    dz_shaf = plasma_dict["dz_shaf"]
    shaf = np.hypot(dx_shaf, dz_shaf)

    params = {
        "I_p": plasma_dict["Ip"] / 1e6,
        "q_95": plasma_dict["q_95"],
        "beta_p": plasma_dict["beta_p"],
        "l_i": plasma_dict["li"],
        "delta_95": plasma_dict["delta_95"],
        "kappa_95": plasma_dict["kappa_95"],
        "delta": plasma_dict["delta"],
        "kappa": plasma_dict["kappa"],
        "shaf_shift": shaf,
    }
    self._params.update_kw_parameters(params, source="equilibria")
