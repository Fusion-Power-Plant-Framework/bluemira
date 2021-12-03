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

from typing import Any, Dict
import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import PhysicalComponent
from bluemira.display.displayer import show_cad
from bluemira.geometry.face import BluemiraFace
from bluemira.utilities.optimiser import Optimiser


from bluemira.geometry.tools import (
    sweep_shape,
    make_polygon,
    offset_wire,
    circular_pattern,
    boolean_cut,
)
from bluemira.base.parameter import ParameterFrame
from bluemira.base.constants import MU_0

from bluemira.builders.tf_coils import RippleConstrainedLengthOpt


class BuildTFWindingPack:
    """
    A class to build TF coil winding pack geometry
    """

    name = "TFWindingPack"

    def __init__(self, wp_centreline, wp_cross_section):
        self.wp_centreline = wp_centreline
        self.wp_cross_section = wp_cross_section

    def build_xy(self):
        pass

    def build_xz(self):
        x_min = self.wp_cross_section.bounding_box.x_min
        x_centreline_in = self.wp_centreline.bounding_box.x_min
        dx = abs(x_min - x_centreline_in)
        outer = offset_wire(self.wp_centreline, dx)
        inner = offset_wire(self.wp_centreline, -dx)
        return PhysicalComponent(self.name, BluemiraFace([outer, inner], self.name))

    def build_xyz(self):
        solid = sweep_shape(self.wp_cross_section, self.wp_centreline, label=self.name)
        return PhysicalComponent(self.name, solid)


class BuildTFInsulation:
    name = "TFWPInsulation"

    def __init__(self, wp_solid, wp_centreline, wp_cross_section, insulation_thickness):
        self.wp_solid = wp_solid
        self.wp_centreline = wp_centreline
        self.wp_cross_section = wp_cross_section
        self.tk_insulation = insulation_thickness

    def build_xy(self):
        outer_wire = offset_wire(self.wp_cross_section, self.tk_insulation)
        face = BluemiraFace([outer_wire, self.wp_cross_section])
        return PhysicalComponent(self.name, face)

    def build_xz(self):
        x_centreline_in = self.wp_centreline.bounding_box.x_min

        x_in_wp = self.wp_cross_section.bounding_box.x_min

        dx_wp = x_centreline_in - x_in_wp

        ins_xs = offset_wire(self.wp_cross_section, self.tk_insulation)
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
        ins_xs = offset_wire(self.wp_cross_section, self.tk_insulation)

        solid = sweep_shape(ins_xs, self.wp_centreline)
        # This doesnt frigging work
        ins_solid = boolean_cut(solid, self.wp_solid)
        return PhysicalComponent(self.name, ins_solid)


class BuildTFCasing:
    name = "TFCasing"

    def __init__(
        self,
        ins_solid,
        wp_centreline,
        ins_cross_section,
        n_TF,
        tk_tf_nose,
        tk_tf_front_ib,
        tk_tf_side,
    ):
        self.ins_solid = ins_solid
        self.ins_cross_section = ins_cross_section
        self.wp_centreline = wp_centreline
        self.n_TF = n_TF
        self.tk_tf_nose = tk_tf_nose
        self.tk_tf_front_ib = tk_tf_front_ib
        self.tk_tf_side = tk_tf_side

    def build_xy(self):
        x_ins_in = self.ins_cross_section.bounding_box.x_min
        x_ins_out = self.ins_cross_section.bounding_box.x_max

        x_in = x_ins_in - self.tk_tf_nose
        x_out = x_ins_out + self.tk_tf_front_ib
        half_angle = np.pi / self.n_TF
        y_in = x_in * np.sin(half_angle)
        y_out = x_out * np.sin(half_angle)
        outer_wire = make_polygon(
            [[x_in, -y_in, 0], [x_out, -y_out, 0], [x_out, y_out, 0], [x_in, y_in, 0]],
            closed=True,
        )
        inner_face = BluemiraFace([outer_wire, self.ins_cross_section.boundary[0]])

        # Should normally be gotten with wire_plane_intersect
        x_out = self.wp_centreline.bounding_box.x_max
        # Split the total radial thickness equally on the outboard
        tk_total = self.tk_tf_front_ib + self.tk_tf_nose
        tk = 0.5 * tk_total
        outer_wire = make_polygon(
            [
                [x_out - tk, -self.tk_tf_side, 0],
                [x_out + tk, -self.tk_tf_side, 0],
                [x_out + tk, self.tk_tf_side, 0],
                [x_out - tk, self.tk_tf_side, 0],
            ],
            closed=True,
        )
        outer_ins = self.ins_cross_section.deepcopy()
        outer_ins.translate((x_out - outer_ins.center_of_mass[0], 0, 0))

        outer_face = BluemiraFace([outer_ins.boundary[0], outer_wire])
        return [
            PhysicalComponent(self.name, inner_face),
            PhysicalComponent(self.name, outer_face),
        ]

    def build_xz(self):
        pass

    def build_xyz(self):
        pass


class BuildTFCoils(Builder):
    """
    A class to build TF coils in the same way as BLUEPRINT.
    """

    _required_config = Builder._required_config + ["targets"]
    _required_params = [
        "R_0",
        "B_0",
        "n_TF",
        "r_tf_in",
        "tk_tf_wp",
        "tk_tf_nose",
        "tf_wp_depth",
    ]

    def __init__(self, params, build_config: Dict[str, Any], **kwargs):
        super().__init__(params, build_config, **kwargs)


class ToroidalFieldSystem:
    def __init__(self, params, wp_parameterisation):
        self.params = params
        self.wp_parameterisation = wp_parameterisation

    def build(self):
        # TODO: I see that nobody ever got to the bottom of the PROCESS insulation story
        r_wp_centroid = self.params.r_tf_in + self.params.tk_tf_wp
        self.wp_parameterisation.adjust_variable("x1", r_wp_centroid)
        wp_xs = self.make_wp_cross_section(r_wp_centroid)
        builder = BuildTFWindingPack(self.wp_parameterisation.create_shape(), wp_xs)
        builder.build()

        builder = BuildTFInsulation(
            self.wp_parameterisation.create_shape(), wp_xs, self.params.tk_tf_ins
        )

        builder = BuildTFCasing()
        builder.build()

    def optimise(self):
        pass

    def calculate_wp_current(self):
        # Back of the envelope
        bm = -self.params.B_0 * self.params.R_0
        current = abs(2 * np.pi * bm / (self.params.n_TF * MU_0))
        self.params.add_parameter(
            "I_tf", "TF coil current", current, "A", None, "bluemira"
        )

    def make_wp_cross_section(self, r_wp_centroid):
        r_wp_in = r_wp_centroid - 0.5 * self.params.tk_tf_wp
        r_wp_out = r_wp_centroid + 0.5 * self.params.tk_tf_wp
        y_down = -0.5 * self.params.tf_wp_depth
        y_up = -y_down
        return BluemiraFace(
            make_polygon(
                [
                    [r_wp_in, y_down, 0],
                    [r_wp_out, y_down, 0],
                    [r_wp_out, y_up, 0],
                    [r_wp_in, y_up, 0],
                ],
                closed=True,
            )
        )


if __name__ == "__main__":

    from bluemira.geometry.parameterisations import PrincetonD, TripleArc, PictureFrame
    from bluemira.geometry.face import BluemiraFace
    from bluemira.equilibria.shapes import JohnerLCFS
    from bluemira.base.parameter import ParameterFrame
    from bluemira.geometry.tools import sweep_shape, circular_pattern, revolve_shape
    from bluemira.display import show_cad
    from bluemira.display.displayer import DisplayCADOptions

    x_tf_wp_center = 3.2
    parameterisation = PrincetonD(
        {
            "x1": {"value": x_tf_wp_center, "fixed": True},
            "x2": {"lower_bound": 10, "value": 14, "upper_bound": 18},
            "dz": {"lower_bound": -0.5, "value": 0, "upper_bound": 0.5, "fixed": True},
        }
    )

    parameterisation = TripleArc(
        {
            "x1": {"value": x_tf_wp_center, "fixed": True},
            "z1": {"value": 0, "lower_bound": -2, "fixed": True},
        }
    )

    optimiser = Optimiser(
        "SLSQP",
        opt_conditions={
            "ftol_rel": 1e-3,
            "xtol_rel": 1e-12,
            "xtol_abs": 1e-12,
            "max_eval": 1000,
        },
    )

    # I just don't know where to get these any more
    params = ParameterFrame(
        [
            ["R_0", "Major radius", 9, "m", None, "Input", None],
            ["z_0", "Vertical height at major radius", 0, "m", None, "Input", None],
            ["B_0", "Toroidal field at R_0", 6, "T", None, "Input", None],
            ["n_TF", "Number of TF coils", 16, "N/A", None, "Input", None],
            ["TF_ripple_limit", "TF coil ripple limit", 0.6, "%", None, "Input", None],
        ]
    )

    separatrix = JohnerLCFS(
        {
            "r_0": {"value": 9},
            "z_0": {"value": 0},
            "a": {"value": 9 / 3.1},
            "kappa_u": {"value": 1.65},
            "kappa_l": {"value": 1.8},
        }
    ).create_shape()

    from bluemira.geometry.tools import offset_wire

    koz = offset_wire(separatrix, 2.0, join="arc")

    problem = RippleConstrainedLengthOpt(
        parameterisation, optimiser, params, separatrix, koz
    )
    problem.solve()

    centreline = parameterisation.create_shape()
    x_c = 4
    d_xc = 0.25
    d_yc = 0.4
    tk_ins = 0.05
    wp_xs = make_polygon(
        [
            [x_c - d_xc, -d_yc, 0],
            [x_c + d_xc, -d_yc, 0],
            [x_c + d_xc, d_yc, 0],
            [x_c - d_xc, d_yc, 0],
        ],
        closed=True,
    )

    tf_wp = sweep_shape(wp_xs, centreline)
    shapes = circular_pattern(tf_wp, n_shapes=16)
    options = 16 * [DisplayCADOptions(color=(0.2, 0.3, 0.4))]
    plasma = revolve_shape(BluemiraFace(separatrix), degree=360)
    shapes.append(plasma)
    options.append(DisplayCADOptions(color=(1.0, 0.2, 0.5), transparency=0.5))

    show_cad(shapes, options)

    wp_centreline = parameterisation.create_shape()

    # Move XS
    x_wp = wp_centreline.bounding_box.x_min
    x_xs = wp_xs.center_of_mass[0]
    dx = x_wp - x_xs
    wp_xs.translate((dx, 0, 0))

    builder = BuildTFWindingPack(wp_centreline, wp_xs)
    xz_comp = builder.build_xz()
    xyz_shape = builder.build_xyz().shape
    # xz.plot_2d()

    builder = BuildTFInsulation(xyz_shape, wp_centreline, wp_xs, tk_ins)
    xz_ins_comp = builder.build_xz()
    xy_ins_shape = builder.build_xy().shape
    # xyz_ins_shape = builder.build_xyz().shape
    builder = BuildTFCasing(None, wp_centreline, xy_ins_shape, 16, 0.4, 0.1, 0.1)
    xy_casing = builder.build_xy()

    xz_comps = [xz_comp]

    xz_comps.extend(xz_ins_comp)

    import matplotlib.pyplot as plt

    f, ax = plt.subplots()
    for shape in xz_comps:
        shape.plot_2d(ax=ax, show=False)

    # shapes = circular_pattern(xyz_shape.shape, n_shapes=16)
    # show_cad(shapes)
