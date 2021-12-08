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
EU-DEMO build classes for TF Coils.
"""
from typing import Type, Optional, List
from copy import deepcopy
import numpy as np

from bluemira.base.builder import Builder, BuildConfig
from bluemira.base.look_and_feel import bluemira_warn, bluemira_debug, bluemira_print
from bluemira.base.parameter import ParameterFrame
from bluemira.base.components import Component, PhysicalComponent
from bluemira.builders.shapes import ParameterisedShapeBuilder
from bluemira.builders.tf_coils import RippleConstrainedLengthOpt
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.optimisation import GeometryOptimisationProblem
from bluemira.geometry.tools import offset_wire, sweep_shape, make_polygon, boolean_cut
from bluemira.geometry.wire import BluemiraWire
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.solid import BluemiraSolid
from bluemira.utilities.optimiser import Optimiser
from bluemira.display.palettes import BLUE_PALETTE


class TFCoilsComponent(Component):
    def __init__(
        self,
        name: str,
        parent: Optional[Component] = None,
        children: Optional[List[Component]] = None,
        magnetics=None,
    ):
        super().__init__(name, parent=parent, children=children)
        self._magnetics = magnetics

    @property
    def magnetics(self):
        return self._magnetics

    @magnetics.setter
    def magnetics(self, magnetics):
        self._magnetics = magnetics


class TFWindingPackBuilder(Builder):
    """
    A class to build TF coil winding pack geometry
    """

    _required_params = []

    def __init__(self, params, build_config: BuildConfig):
        super().__init__(params, build_config)

    def set_tools(self, wp_centreline, wp_cross_section):
        self.wp_centreline = wp_centreline
        self.wp_cross_section = wp_cross_section

    def build_xy(self):
        # Should normally be gotten with wire_plane_intersect
        # (it's not OK to assume that the maximum x value occurs on the midplane)
        x_out = self.wp_centreline.bounding_box.x_max

        xs = deepcopy(self.wp_cross_section)
        xs2 = deepcopy(xs)
        xs2.translate((x_out - xs2.center_of_mass[0], 0, 0))

        return [
            PhysicalComponent(self.name, xs),
            PhysicalComponent(self.name, xs2),
        ]

    def build_xz(self):
        x_min = self.wp_cross_section.bounding_box.x_min
        x_centreline_in = self.wp_centreline.bounding_box.x_min
        dx = abs(x_min - x_centreline_in)
        outer = offset_wire(self.wp_centreline, dx)
        inner = offset_wire(self.wp_centreline, -dx)
        return PhysicalComponent(self.name, BluemiraFace([outer, inner], self.name))

    def build_xyz(self):
        solid = sweep_shape(
            self.wp_cross_section.boundary[0], self.wp_centreline, label=self.name
        )
        return PhysicalComponent(self.name, solid)


class TFInsulationBuilder(Builder):

    _required_params = ["tk_tf_ins"]

    def __init__(self, params, build_config: BuildConfig):
        super().__init__(params, build_config)

    def set_tools(self, wp_solid, wp_centreline, wp_cross_section):
        self.wp_solid = wp_solid
        self.wp_centreline = wp_centreline
        self.wp_cross_section = wp_cross_section

    def build_xy(self):
        outer_wire = offset_wire(
            self.wp_cross_section.boundary[0], self._params.tk_tf_ins.value
        )
        face = BluemiraFace([outer_wire, self.wp_cross_section.boundary[0]])

        # Should normally be gotten with wire_plane_intersect
        x_out = self.wp_centreline.bounding_box.x_max
        outer_face = deepcopy(face)
        outer_face.translate((x_out - outer_face.center_of_mass[0], 0, 0))
        return [
            PhysicalComponent(self.name, face),
            PhysicalComponent(self.name, outer_face),
        ]

    def build_xz(self):
        x_centreline_in = self.wp_centreline.bounding_box.x_min

        x_in_wp = self.wp_cross_section.bounding_box.x_min

        dx_wp = x_centreline_in - x_in_wp

        ins_xs = offset_wire(
            self.wp_cross_section.boundary[0], self._params.tk_tf_ins.value
        )
        x_in_ins = ins_xs.bounding_box.x_min

        dx_ins = x_centreline_in - x_in_ins
        outer = offset_wire(self.wp_centreline, dx_ins)
        inner = offset_wire(self.wp_centreline, dx_wp)

        outer_face = BluemiraFace([outer, inner])

        outer = offset_wire(self.wp_centreline, -dx_wp)
        inner = offset_wire(self.wp_centreline, -dx_ins)
        inner_face = BluemiraFace([outer, inner])
        return [
            PhysicalComponent(self.name, outer_face),
            PhysicalComponent(self.name, inner_face),
        ]

    def build_xyz(self):
        ins_xs = offset_wire(
            self.wp_cross_section.boundary[0], self._params.tk_tf_ins.value
        )

        solid = sweep_shape(ins_xs, self.wp_centreline)
        ins_solid = boolean_cut(solid, self.wp_solid)[0]
        return PhysicalComponent(self.name, ins_solid)


class TFCasingBuilder(Builder):

    _required_params: List[str] = [
        "n_TF",
        "tk_tf_nose",
        "tk_tf_front_ib",
        "tk_tf_side",
    ]

    def __init__(self, params, build_config: BuildConfig):
        super().__init__(params, build_config)

    def set_tools(self, ins_solid, wp_centreline, ins_cross_section):
        self.ins_solid = deepcopy(ins_solid)
        self.ins_cross_section = deepcopy(ins_cross_section)
        self.wp_centreline = deepcopy(wp_centreline)

    def build_xy(self):
        x_ins_in = self.ins_cross_section.bounding_box.x_min
        x_ins_out = self.ins_cross_section.bounding_box.x_max

        x_in = x_ins_in - self._params.tk_tf_nose.value
        x_out = x_ins_out + self._params.tk_tf_front_ib.value
        half_angle = np.pi / self._params.n_TF.value
        y_in = x_in * np.sin(half_angle)
        y_out = x_out * np.sin(half_angle)
        outer_wire = make_polygon(
            [[x_in, -y_in, 0], [x_out, -y_out, 0], [x_out, y_out, 0], [x_in, y_in, 0]],
            closed=True,
        )
        inner_face = BluemiraFace(
            [outer_wire, deepcopy(self.ins_cross_section.boundary[0])]
        )

        bb = self.ins_cross_section.bounding_box
        dx_ins = 0.5 * (bb.x_max - bb.x_min)
        dy_ins = 0.5 * (bb.y_max - bb.y_min)

        # Split the total radial thickness equally on the outboard
        # This could be done with input params too..
        tk_total = self._params.tk_tf_front_ib.value + self._params.tk_tf_nose.value
        tk = 0.5 * tk_total

        dx_out = dx_ins + tk
        dy_out = dy_ins + self._params.tk_tf_side.value
        outer_wire = make_polygon(
            [
                [-dx_out, -dy_out, 0],
                [dx_out, -dy_out, 0],
                [dx_out, dy_out, 0],
                [-dx_out, dy_out, 0],
            ],
            closed=True,
        )

        outer_ins = deepcopy(self.ins_cross_section.boundary[0])

        # Should normally be gotten with wire_plane_intersect
        x_out = self.wp_centreline.bounding_box.x_max
        outer_wire.translate((x_out - outer_wire.center_of_mass[0], 0, 0))
        outer_ins.translate((x_out - outer_ins.center_of_mass[0], 0, 0))
        outer_face = BluemiraFace([outer_wire, outer_ins])
        return [
            PhysicalComponent(self.name, inner_face),
            PhysicalComponent(self.name, outer_face),
        ]

    def build_xz(self):
        # Normally I'd do a variable thickness offset and actually build the xyz from
        # that. We can't do variable thickness offsets with primitives.. (yet, I suppose)

        # We could just section the xyz, but we can't do that yet either.
        pass

    def build_xyz(self):
        # Normally I'd do lots more here to get to a proper casing
        # This is just a proof-of-principle
        inner_xs, outer_xs = self.build_xy()
        inner_xs = inner_xs.shape.boundary[0]
        outer_xs = outer_xs.shape.boundary[0]

        solid = sweep_shape([inner_xs, outer_xs], self.wp_centreline)
        outer_ins_solid = BluemiraSolid(self.ins_solid.boundary[0])
        solid = boolean_cut(solid, outer_ins_solid)[0]

        return PhysicalComponent(self.name, solid)


class TFCoilsBuilder(ParameterisedShapeBuilder):
    _required_params: List[str] = [
        "R_0",
        "z_0",
        "B_0",
        "n_TF",
        "TF_ripple_limit",
        "r_tf_in",
        "tk_tf_nose",
        "tk_tf_front_ib",
        "tk_tf_side",
        "tk_tf_ins",
        # This isn't treated at the moment...
        "tk_tf_insgap",
        # Dubious WP depth from PROCESS (I used to tweak this when building the TF coils)
        "tf_wp_width",
        "tf_wp_depth",
    ]
    _required_config = ParameterisedShapeBuilder._required_config + []
    _params: ParameterFrame
    _param_class: Type[GeometryParameterisation]
    _default_run_mode: str = "run"
    _design_problem: Optional[GeometryOptimisationProblem] = None
    _centreline: BluemiraWire

    def __init__(self, params, build_config: BuildConfig, **kwargs):
        super().__init__(params, build_config, **kwargs)

        self.reinitialise(params)

    def _extract_config(self, build_config: BuildConfig):
        super()._extract_config(build_config)

    def reinitialise(self, params, **kwargs) -> None:
        """
        Initialise the state of this builder ready for a new run.

        Parameters
        ----------
        params: Dict[str, Any]
            The parameterisation containing at least the required params for this
            Builder.
        """
        bluemira_debug(f"Reinitialising {self.name}")
        self._reset_params(params)

        self._design_problem = None
        self._centreline = self._param_class().create_shape()
        self._wp_cross_section = self._make_wp_xs()
        self._wp_solid = None
        self._ins_cross_section = None
        self._ins_solid = None
        self._sub_builders = [
            TFWindingPackBuilder(self._params, {"name": "Winding pack"}),
            TFInsulationBuilder(self._params, {"name": "Insulation"}),
            TFCasingBuilder(self._params, {"name": "Casing"}),
        ]

    def _make_wp_xs(self):
        x_c = self.params.r_tf_in.value
        d_xc = 0.5 * self.params.tf_wp_width.value
        d_yc = 0.5 * self.params.tf_wp_depth.value
        wp_xs = make_polygon(
            [
                [x_c - d_xc, -d_yc, 0],
                [x_c + d_xc, -d_yc, 0],
                [x_c + d_xc, d_yc, 0],
                [x_c - d_xc, d_yc, 0],
            ],
            closed=True,
        )
        return BluemiraFace(wp_xs, "TF WP x-y cross-section")

    def run(self, separatrix, keep_out_zone=None, nx=1, ny=1):
        optimiser = Optimiser(
            "SLSQP",
            opt_conditions={
                "ftol_rel": 1e-3,
                "xtol_rel": 1e-12,
                "xtol_abs": 1e-12,
                "max_eval": 1000,
            },
        )
        self._design_problem = RippleConstrainedLengthOpt(
            self._param_class(),
            optimiser,
            self._params,
            self._wp_cross_section,
            separatrix=separatrix,
            keep_out_zone=keep_out_zone,
            rip_con_tol=1e-3,
            koz_con_tol=1e-3,
            nx=nx,
            ny=ny,
            n_koz_points=100,
        )
        self._design_problem.solve()
        self._centreline = self._design_problem.parameterisation.create_shape()

    def read(self, variables):
        parameterisation = self._param_class(variables)
        self._centreline = parameterisation.create_shape()

    def mock(self, variables):
        parameterisation = self._param_class(variables)
        self._centreline = parameterisation.create_shape()

    def reinitialise(self, params, **kwargs) -> None:
        """
        Initialise the state of this builder ready for a new run.

        Parameters
        ----------
        params: Dict[str, Any]
            The parameterisation containing at least the required params for this
            Builder.
        """
        bluemira_debug(f"Reinitialising {self.name}")
        self._reset_params(params)

    def build(self, label: str = "TF Coils", **kwargs) -> Component:
        """
        Build the TF Coils component.

        Returns
        -------
        component: TFCoilsComponent
            The Component built by this builder.
        """
        super().build(**kwargs)

        component = TFCoilsComponent(self.name)

        component.add_child(self.build_xz(label=label))
        component.add_child(self.build_xy(label=label))
        component.add_child(self.build_xyz(label=label))
        return component

    def build_xz(self, **kwargs):
        component = Component("xz")

        for sub_builder in self._sub_builders:
            sub_comp = sub_builder.build_xz()
            component.add_child(sub_comp)

        return component

    def build_xy(self, **kwargs):
        component = Component("xy")

        return component

    def build_xyz(self, **kwargs):
        component = Component("xyz")

        return component
