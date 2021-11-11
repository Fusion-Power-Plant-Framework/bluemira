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
Coil CAD routines
"""
from BLUEPRINT.geometry.offset import offset_clipper
import numpy as np
from collections import OrderedDict

try:
    from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
    from OCC.Core.gp import gp_Pnt, gp_Ax1, gp_Dir, gp_Ax2, gp_Vec, gp_Circ
    from OCC.Core.GC import GC_MakeArcOfCircle
    from OCC.Core.BRepFill import BRepFill_PipeShell
    from OCC.Core.TopoDS import topods
except ImportError:
    from OCC.BRepBuilderAPI import BRepBuilderAPI_MakePolygon
    from OCC.gp import gp_Pnt, gp_Ax1, gp_Dir, gp_Ax2, gp_Vec, gp_Circ
    from OCC.GC import GC_MakeArcOfCircle
    from OCC.BRepFill import BRepFill_PipeShell
    from OCC.TopoDS import topods
from BLUEPRINT.base.palettes import BLUE
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.shell import Shell
from BLUEPRINT.cad.component import ComponentCAD
from BLUEPRINT.cad.cadtools import (
    boolean_cut,
    boolean_fuse,
    revolve,
    extrude,
    make_box,
    sweep,
    rotate_shape,
    make_axis,
    make_face,
    make_mixed_shell,
    make_mixed_face,
    translate_shape,
    make_compound,
    sew_shapes,
    make_circle,
    mirror_shape,
    make_wire,
    _make_OCCedge,
    _make_OCCwire,
    _make_OCCface,
    _make_OCCsolid,
)
from BLUEPRINT.geometry.boolean import (
    boolean_2d_difference_loop,
    clean_loop,
    simplify_loop,
)
from BLUEPRINT.geometry.parameterisations import picture_frame
from BLUEPRINT.geometry.geomtools import make_box_xz


class RingCAD:
    """
    CAD building object for circular components.
    """

    def _build_ring(self, ctype, angle=None):
        pf = self.args[0]

        if angle is None:
            angle = 360 / pf.params.n_TF

        for name, coil in pf.coils.items():
            if coil.ctype == ctype:
                pf_loop = Loop(x=coil.x_corner, z=coil.z_corner)
                pf_loop.close()
                pf_face = make_face(pf_loop)
                ax = make_axis((0, 0, 0), (0, 0, 1))
                shape = revolve(pf_face, ax, angle)
                shape = rotate_shape(shape, ax, -angle / 2)
                self.add_shape(shape, name=name)


class PFSystemCAD(ComponentCAD):
    """
    CAD building class for the entire PoloidalFieldSystem.
    """

    def __init__(self, pf_system, **kwargs):
        self.pf_system = pf_system
        self.name = "Poloidal field system"
        self.component = {
            "shapes": [],
            "names": [],
            "sub_names": [],
            "colors": [],
            "transparencies": [],
        }
        self.n_shape = 0  # number of shapes within component
        self.slice_flag = kwargs.get("slice_flag", False)
        if kwargs.get("neutronics", False):
            self.build_neutronics()
        else:
            self.build(from_compound=kwargs.get("from_compound", False))

    @staticmethod
    def _merge_components(pf_cad, cs_cad):
        component = pf_cad.component.copy()

        for key in component:
            component[key].extend(cs_cad.component[key])
        return component

    def build(self, **kwargs):
        """
        Build the CAD for all of the PF and CS coils.
        """
        pf_cad = PFCoilCAD(self.pf_system, **kwargs)
        cs_cad = CSCoilCAD(self.pf_system, **kwargs)
        self.component = self._merge_components(pf_cad, cs_cad)

    def build_neutronics(self, **kwargs):
        """
        Build the neutronics CAD for all of the PF and CS coils.
        """
        pf_cad = PFCoilCAD(self.pf_system, neutronics=True, **kwargs)
        cs_cad = CSCoilCAD(self.pf_system, neutronics=True, **kwargs)
        self.component = self._merge_components(pf_cad, cs_cad)


class PFCoilCAD(RingCAD, ComponentCAD):
    """
    CAD building class for the PF coils.
    """

    def __init__(self, pf, **kwargs):
        super().__init__(
            "Poloidal field coils", pf, palette=[BLUE["PF"][0]], n_colors=6, **kwargs
        )

    def build(self, **kwargs):
        """
        Build the CAD for the PF coils.
        """
        self._build_ring("PF")

    def build_neutronics(self, **kwargs):
        """
        Build the neutronics CAD for the PF coils.
        """
        self._build_ring("PF", angle=360)


class CSCoilCAD(RingCAD, ComponentCAD):
    """
    CAD building class for the central solenoid.
    """

    def __init__(self, pf, **kwargs):
        super().__init__(
            "Central solenoid", pf, palette=[BLUE["CS"][0]], n_colors=6, **kwargs
        )

    def build(self, **kwargs):
        """
        Build the CAD for the solenoid.
        """
        self._build_ring("CS")

    def build_neutronics(self, **kwargs):
        """
        Build the neutronics CAD for the solenoid.
        """
        self._build_ring("CS", angle=360)


class TFCoilCAD(ComponentCAD):
    """
    Toroidal field coil CAD constructor class
    """

    def __init__(self, tf, **kwargs):
        super().__init__(
            "Toroidal field coils", tf, palette=BLUE["TF"], n_colors=12, **kwargs
        )
        if tf.conductivity in ["SC"] and tf.shape_type in ["TP"]:
            raise NotImplementedError(
                "Superconducting Tapered Pictureframe coils not supported"
            )
        self.n_TF = None

    def build(self, **kwargs):
        """
        Build the CAD for the TF coils.
        """
        tf = self.args[0]
        self.n_TF = tf.params.n_TF

        tf_components = self._build(tf)

        for name in tf_components:

            tf_components[name] = rotate_shape(
                tf_components[name], axis=None, angle=180 / self.n_TF
            )
            self.add_shape(tf_components[name], name=name)

    @staticmethod
    def sanity_check(face):
        """
        Checks if the x-z geometries for CAD generation are in the
        right format, and converts them to OCC faces
        """
        if isinstance(face, Shell):
            geom = make_mixed_shell(face)

        elif isinstance(face, Loop):
            geom = make_mixed_face(face)

        else:
            raise TypeError("The face argument must be either a Shell or a Loop")

        return geom

    @staticmethod
    def wedge_from_xz(tf, face, coil_toroidal_angle, x_shift=0, case=False):
        """
        Build a coil cad with a wedge shaped inner leg. Done by sweeping
        2D x-z face through full toroidal angle, then cutting off excess bits

        Parameters
        ----------
        tf: ToroidalFieldCoils
            Toroidal field coil object used to make the CAD
        face: Shell or Loop
            2D x-z face of TF coil geometry
        coil_toroidal_angle: float
            2pi/n_TF radians
        x_shift: float
            Radial offset (from machine centre) for axis about
            which coil_toroidal_angle is used (useful if inboard
            legs aren't meant to touch, e.g WPs inside a casing)
        case: bool
            Checks if required geometry is a casing

        Outputs
        -------
        geom: OCC 3D geometry
            3D model of face with relevant tf parameters
        """
        # Shift face to lie along the 'zero toroidal angle' line
        geom = face.rotate(
            theta=-np.rad2deg(coil_toroidal_angle / 2),
            xo=[x_shift, 0, 0],
            dx=[0, 0, 1],
            update=False,
        )

        # Make the CAD face
        geom = TFCoilCAD.sanity_check(geom)

        # Sweep Winding Pack thorugh full toroidal angle
        geom_axis = make_axis([x_shift, 0, 0], [0, 0, 1])
        geom = revolve(geom, axis=geom_axis, angle=np.rad2deg(coil_toroidal_angle))

        # Now Cut away excess bits
        # from wp
        xmax = max(tf.p_in["x"]) + 15
        zmax = max(tf.p_in["z"]) + 15
        cutter_outer_2D = make_box_xz(x_min=0.0, x_max=xmax, z_min=-zmax, z_max=zmax)
        wp_half_depth = tf.section["winding_pack"]["depth"] / 2
        side = 0
        case_front = 0

        # Lateral extent of the cut defining the end of the wedge
        # ---
        # Tapered coils
        if tf.conductivity in ["R"]:
            cut_half_depth = tf.params.r_cp_top * np.sin(0.5 * coil_toroidal_angle)

        # Other coils (SC like)
        else:
            # Check if CAD being generated is TF casing or WP (for casing the
            # cutter will have to be shifted further of course)
            if case:
                side = tf.params.tk_tf_side
                case_front = tf.params.tk_tf_front_ib
            cut_half_depth = (
                wp_half_depth + side + case_front * np.tan(0.5 * coil_toroidal_angle)
            )

        # Translate the cut loop on the y direction
        # ordering of loop is important (boolean upper cut then lower cut)
        for cut_depth, ex_length in [[cut_half_depth, 15], [-cut_half_depth, -15]]:
            cutter = cutter_outer_2D.translate([0, cut_depth, 0], update=False)

            # Make the CAD object
            cutter = make_face(cutter)

            # Make it a massive square
            cutter = extrude(cutter, axis="y", length=ex_length)

            # Cut to obtain the final geometry
            geom = boolean_cut(geom, cutter)

        return geom

    @staticmethod
    def rect_from_xz(tf, face, coil_toroidal_angle, x_shift=0, case=False):
        """
        Build a tf coil cad with an inner leg with a rectangular cross
        section. Done by extruding 2D xz face by full toroidal depth
        then cutting away excess

        Parameters
        ----------
        tf: ToroidalFieldCoils
            Toroidal field coil object used to make the CAD
        face: Shell or Loop
            2D x-z face of TF coil geometry
        coil_toroidal_angle: float
            2pi/n_TF radians
        x_shift: float
            Radial offset (from machine centre) for axis about
            which coil_toroidal_angle is used (useful if inboard
            legs aren't meant to touch, e.g WPs inside a casing)
        case: bool
            Checks if required geometry is a casing

        Outputs
        -------
        geom: OCC 3D geometry
            3D model of face with relevant tf parameters
        """
        # Shift face to lie along the 'zero toroidal angle' line
        geom = face.translate([0, -0.5 * tf.params.tf_wp_depth, 0], update=False)

        # Make the CAD face
        geom = TFCoilCAD.sanity_check(geom)

        side = 0
        case_front = 0
        if case:
            side = tf.params.tk_tf_side
            case_front = tf.params.tk_tf_front_ib

        # Make the WP volume (extrude)
        geom = extrude(geom, axis="y", length=tf.params.tf_wp_depth + 2 * side)

        # Define the xz loop of the cutter objects
        # ---
        # Use r_cp_top for tapered coils
        if tf.conductivity in ["R"]:
            cutter_radius = tf.params.r_cp_top

        # SC coils
        else:
            geom_inner_leg_outboard_radius = np.min(tf.loops["wp_in"]["x"])
            cutter_radius = geom_inner_leg_outboard_radius + case_front + 0.001

        zmax = max(tf.p_in["z"]) + 5
        geom_inner_2D = Loop(
            x=cutter_radius * np.array([0.0, 1, 1, 0]),
            z=zmax * np.array([1.0, 1.0, -1.0, -1.0]),
        )
        geom_inner_2D.close()

        # Shift the xz loop of the cutter
        geom_inner_leg_inboard_radius = np.min(tf.loops["wp_out"]["x"]) - x_shift
        geom_inner_half_depth = geom_inner_leg_inboard_radius * np.sin(
            coil_toroidal_angle / 2
        )
        # ordering of loop is important (boolean upper cut then lower cut)
        extrude_length = np.max(tf.loops["wp_out"]["x"]) + 5
        for geom_depth, ex_length in [
            [geom_inner_half_depth, extrude_length],
            [-geom_inner_half_depth, -extrude_length],
        ]:
            geom_cutter = geom_inner_2D.translate([0, geom_depth, 0], update=False)
            # Make the cutter CAD face
            geom_cutter = make_face(geom_cutter)

            # Make the cutter volume
            geom_cutter = extrude(geom_cutter, axis="y", length=ex_length)

            # Cut the lateral objects
            geom = boolean_cut(geom, geom_cutter)

        return geom

    @staticmethod
    def _build(tf):
        """
        Strong and stable... and slow?. Suspect will help with neutronics
        Yes, slower by 0.4 seconds (without the second lbox boolean)
        But it works for everything... and a few 100 lines shorter
        """
        geom = tf.geom
        side = tf.section["case"]["side"]

        if tf.inputs["shape_type"] == "P":
            case_front = tf.params.tk_tf_front_ib
            case_nose = tf.params.tk_tf_nose
            coil_toroidal_angle = 2 * np.pi / tf.params.n_TF

            # Make 2D x-z cross section:
            # wp_in:
            x1_in = np.min(tf.loops["wp_in"]["x"])
            x2_in = np.max(tf.loops["wp_in"]["x"])
            z1_in = np.max(tf.loops["wp_in"]["z"])
            z2_in = -z1_in
            ro_in = np.max(
                [tf.params.r_tf_outboard_corner - tf.params.tk_tf_wp / np.sqrt(2), 0]
            )
            ri_in = np.max(
                [tf.params.r_tf_inboard_corner - tf.params.tk_tf_wp / np.sqrt(2), 0]
            )
            x, z = picture_frame(x1_in, x2_in, z1_in, z2_in, ri_in, ro_in, npoints=200)
            # x, z = tf.loops["wp_in"]["x"], tf.loops["wp_in"]["z"]
            wp_in_2D = Loop(x=x, z=z)
            wp_in_2D.close()
            wp_in_2D = clean_loop(wp_in_2D)
            wp_in_2D = simplify_loop(wp_in_2D)

            # wp_out:
            wp_initial_2D = Shell.from_offset(wp_in_2D, tf.params.tk_tf_wp)
            x_shift = side / np.tan(coil_toroidal_angle / 2)

            if tf.wp_shape == "R" or tf.wp_shape == "N":

                wp = TFCoilCAD.rect_from_xz(
                    tf, wp_initial_2D, coil_toroidal_angle, x_shift=x_shift
                )

            elif tf.wp_shape == "W":

                wp = TFCoilCAD.wedge_from_xz(
                    tf, wp_initial_2D, coil_toroidal_angle, x_shift=x_shift
                )
            # case:

            # case_in:
            x1_in = np.min(tf.loops["wp_in"]["x"]) + case_front
            x2_in = np.max(tf.loops["wp_in"]["x"]) - case_front
            z1_in = np.max(tf.loops["wp_in"]["z"]) - case_front
            z2_in = -z1_in
            ro_in = np.max(
                [
                    tf.params.r_tf_outboard_corner
                    - (tf.params.tk_tf_wp + case_front) / np.sqrt(2),
                    0,
                ]
            )
            ri_in = np.max(
                [
                    tf.params.r_tf_inboard_corner
                    - (tf.params.tk_tf_wp + case_front) / np.sqrt(2),
                    0,
                ]
            )
            x, z = picture_frame(x1_in, x2_in, z1_in, z2_in, ri_in, ro_in, npoints=200)
            # x, z = tf.loops["in"]["x"], tf.loops["in"]["z"]
            case_in_2D = Loop(x=x, z=z)
            case_in_2D.close()
            case_in_2D = clean_loop(case_in_2D)
            case_in_2D = simplify_loop(case_in_2D)

            # case_out:
            case_initial_2D = Shell.from_offset(
                case_in_2D, tf.params.tk_tf_wp + case_front + case_nose
            )
            # Sweep Case thorugh full toroidal angle

            case = TFCoilCAD.wedge_from_xz(
                tf, case_initial_2D, coil_toroidal_angle, x_shift=0, case=True
            )

            # from case
            case = boolean_cut(case, wp)

            rbox = cut_box(side="right", n_TF=tf.params.n_TF)
            lbox = cut_box(side="left", n_TF=tf.params.n_TF)
            case = boolean_cut(case, rbox)
            case = boolean_cut(case, lbox)

            # ---

        elif tf.inputs["shape_type"] in ["TP", "CP"]:
            # Coils with a tapered centrepost segemented from tf leg conductors
            # Central collumn dimensions
            coil_toroidal_angle = 2 * np.pi / tf.params.n_TF
            zmax_wp = np.max(tf.loops["wp_out"]["z"])  # Max z height of tfcoil

            if tf.conductivity in ["SC"]:
                # r_cp_top doesn't exist for SC coils, so need to define our
                # own r_cp (i.e outboard edge of Centrepost)
                r_cp = tf.params.r_tf_in + tf.params.tk_tf_inboard
                TF_depth_at_r_cp = 2 * (r_cp * np.tan(np.pi / tf.params.n_TF))
                x_shift = side / np.tan(coil_toroidal_angle / 2)

            else:
                TF_depth_at_r_cp = 2 * (
                    tf.params.r_cp_top * np.tan(np.pi / tf.params.n_TF)
                )
                x_shift = 0

            # Edit WP
            wp_loop = geom["TF WP"]
            winding_pack = TFCoilCAD.wedge_from_xz(
                tf, wp_loop, coil_toroidal_angle, x_shift=x_shift
            )

            # Central column
            tapered_cp = tf.geom["TF Tapered CP"]
            tapered_cp = TFCoilCAD.wedge_from_xz(
                tf, tapered_cp, coil_toroidal_angle, x_shift=x_shift
            )
            leg_conductor = boolean_cut(winding_pack, tapered_cp)

            if tf.conductivity in ["R"]:
                # Resistive tapered CP coils
                tk_case = tf.params.tk_tf_ob_casing
                if tf.shape_type in ["CP"]:
                    zmax_b_cyl = np.max(tf.loops["b_cyl"]["z"])
                else:
                    zmax_b_cyl = zmax_wp

                # Make B Cyl
                ri = np.min(tf.loops["b_cyl"]["x"])
                ro = np.max(tf.loops["b_cyl"]["x"])
                b_cyl_loop = make_box_xz(
                    x_min=ri, x_max=ro, z_min=-zmax_b_cyl, z_max=zmax_b_cyl
                )
                b_cyl = TFCoilCAD.wedge_from_xz(tf, b_cyl_loop, coil_toroidal_angle)

                # Define casing loop
                # Can't use Casing Loops since they aren't connected to each other
                # in the xz plane. Instead:
                # Take the leg conductor loop, offset it by the casing thickness
                # then cut to match the casing loops
                leg_conductor_loop = geom["TF Leg Conductor"]
                case = offset_clipper(leg_conductor_loop, tk_case)
                xmax_cut = 1.03 * (
                    (0.5 * TF_depth_at_r_cp + tk_case)
                    / np.tan(0.5 * coil_toroidal_angle)
                )
                inner_cut_loop = make_box_xz(
                    x_min=-5.0 * xmax_cut,
                    x_max=xmax_cut,
                    z_min=-(zmax_wp + 5.0),
                    z_max=zmax_wp + 5.0,
                )
                case = boolean_2d_difference_loop(case, inner_cut_loop)
                case.interpolate(800)

                # Shift the casing loop in the y direction prepare the extrusion
                half_depth_casing = 0.5 * TF_depth_at_r_cp + tf.params.tk_tf_ob_casing
                case = case.translate([0, -half_depth_casing, 0], update=False)

                # Make the case CAD object (extrusion/WP subtraction)
                case = TFCoilCAD.sanity_check(case)
                case = extrude(case, axis="y", length=2 * half_depth_casing)
                case = boolean_cut(case, leg_conductor)
                rbox = cut_box(side="right", n_TF=tf.params.n_TF)
                lbox = cut_box(side="left", n_TF=tf.params.n_TF)

            else:
                # For SC SEGMENTED coils
                case_front = tf.params.tk_tf_front_ib
                case_nose = tf.params.tk_tf_nose
                coil_toroidal_angle = 2 * np.pi / tf.params.n_TF

                # Make 2D x-z cross section:
                # wp:
                leg_initial_2D = geom["TF Leg Conductor"]

                # case:
                geom["TF case in"].inner.interpolate(800)
                geom["TF case out"].outer.interpolate(800)
                case_initial_2D = Shell(
                    geom["TF case in"].inner, geom["TF case out"].outer
                )

                leg_conductor = TFCoilCAD.wedge_from_xz(
                    tf, leg_initial_2D, coil_toroidal_angle, x_shift=x_shift
                )

                # Now make case
                case = TFCoilCAD.wedge_from_xz(
                    tf, case_initial_2D, coil_toroidal_angle, x_shift=0, case=True
                )

                # from case
                case = boolean_cut(case, leg_conductor)
                case = boolean_cut(case, tapered_cp)
                rbox = cut_box(side="right", n_TF=tf.params.n_TF)
                lbox = cut_box(side="left", n_TF=tf.params.n_TF)
                case = boolean_cut(case, rbox)
                case = boolean_cut(case, lbox)

                # ---

        else:
            case_front = tf.params.tk_tf_front_ib
            case_nose = tf.params.tk_tf_nose
            coil_toroidal_angle = 2 * np.pi / tf.params.n_TF

            # Make 2D x-z cross section:
            # wp:
            x_shift = side / np.tan(coil_toroidal_angle / 2)
            wp_initial_2D = geom["TF WP"]

            # case:
            case_initial_2D = Shell(geom["TF case in"].inner, geom["TF case out"].outer)

            if tf.wp_shape == "R" or tf.wp_shape == "N":

                wp = TFCoilCAD.rect_from_xz(
                    tf, wp_initial_2D, coil_toroidal_angle, x_shift=x_shift
                )

            elif tf.wp_shape == "W":

                wp = TFCoilCAD.wedge_from_xz(
                    tf, wp_initial_2D, coil_toroidal_angle, x_shift=x_shift
                )

            # Now make case

            case = TFCoilCAD.wedge_from_xz(
                tf, case_initial_2D, coil_toroidal_angle, x_shift=0, case=True
            )

            # from case
            case = boolean_cut(case, wp)

            rbox = cut_box(side="right", n_TF=tf.params.n_TF)
            lbox = cut_box(side="left", n_TF=tf.params.n_TF)
            case = boolean_cut(case, rbox)
            case = boolean_cut(case, lbox)

            # ---

        comp_dict = OrderedDict()

        if tf.conductivity in ["R"]:
            comp_dict["b_cyl"] = b_cyl
            comp_dict["leg_conductor"] = leg_conductor
            comp_dict["cp_conductor"] = tapered_cp
            comp_dict["case"] = case
        elif tf.shape_type in ["CP"]:
            comp_dict["leg_conductor"] = leg_conductor
            comp_dict["cp_conductor"] = tapered_cp
            comp_dict["case"] = case
        else:
            comp_dict["case"] = case
            comp_dict["wp"] = wp
        return comp_dict

    # Storage.. old stuff from the Architect days (needed for neutronics)
    def build_neutronics(self, **kwargs):
        """
        Build the neutronics CAD for the TF coils.
        """
        self.build(**kwargs)
        self.component_pattern(self.n_TF)
        component = {
            "shapes": [],
            "names": [],
            "sub_names": [],
            "colors": [],
            "transparencies": [],
        }  # Phoenix design pattern
        for i, name in enumerate(self.component["names"]):
            if name in ["Toroidal field coils_wp", "Toroidal field coils_case"]:
                for k, v in self.component.items():
                    component[k].append(v[i])
        # tfc = self.component['shapes'][0]
        # tfwp = self.component['shapes'][1]
        # tfc = boolean_cut(tfc, tfwp)
        # self.component['shapes'][0] = tfc
        self.component = component

    def for_neutronics(self):
        """
        An old decomposition for neutronics.
        """
        tfc = self.component["shapes"][0]
        tfwp = self.component["shapes"][1]
        self.component = {
            "shapes": [],
            "names": [],
            "sub_names": [],
            "colors": [],
        }  # Phoenix design pattern
        # Yes you made that name up
        tfc = boolean_cut(tfc, tfwp)
        self.add_shape(tfc, name="TF_case")
        self.add_shape(tfwp, name="TF_wp")
        return self.split(["TF_case", "TF_WP"], [["TF_case"], ["TF_wp"]])

    def _get_OIS_collision(self):  # noqa (N802)
        """
        Gets x value of OIS collision.
        """
        ois = self.args[0].loop["OIS"]
        m = []
        for k, v in ois.items():
            m.append(min(np.array(v["nodes"]).T[0]))
        return min(m)


class CoilStructureCAD(ComponentCAD):
    """
    Component CAD class for coil structures

    Parameters
    ----------
    atec: Nova::CoilArchitect
    """

    def __init__(self, atec, **kwargs):
        super().__init__(
            "Coil structures", atec, palette=BLUE["ATEC"], n_colors=0, **kwargs
        )
        self.geom = None
        self.n_TF = None

    def build(self, **kwargs):
        """
        Builds the CAD for the coil structures

        All items are built on the X-Z plane for simplicity, before being
        rotated onto the correct planes
        """
        atec = self.args[0]
        self.geom = atec.geom
        self.n_TF = atec.params.n_TF

        cad = {"CSseat": build_CS_seat(self.geom["feed 3D CAD"]["CS"])}

        for k, v in self.geom["feed 3D CAD"]["PF"].items():
            cad[k] = self._build_PF_seat(v)

        for k, v in self.geom["feed 3D CAD"]["OIS"].items():
            cad[k] = self._build_OIS(v)

        if self.geom["feed 3D CAD"]["Gsupport"]["gs_type"] == "JT60SA":
            cad["Gsupport"] = build_GS_JT60SA(self.geom["feed 3D CAD"]["Gsupport"])
        elif self.geom["feed 3D CAD"]["Gsupport"]["gs_type"] == "ITER":
            cad["Gsupport"] = build_GS_ITER(self.geom["feed 3D CAD"]["Gsupport"])

        for name in cad:
            cad[name] = rotate_shape(cad[name], axis=None, angle=180 / self.n_TF)
            self.add_shape(cad[name], name=name)

    def _build_OIS(self, support):  # noqa (N802)
        """
        Builds an individual outer inter-coil support

        Parameters
        ----------
        support: dict
            The OIS support dict

        Returns
        -------
        ois: OCC Compound
            The BRep of the OIS supports (on both sides of the TF)
        """
        theta = np.pi / self.n_TF
        yo = self.geom["feed 3D CAD"]["other"]["PF"]["space"]
        loop = support["loop"]
        loop.translate([0, yo, 0], update=True)  # Snap to TF coil case edge

        length = 10
        dx = -np.sin(theta) * length
        dy = np.cos(theta) * length

        x = [loop.centroid[0], loop.centroid[0] + dx]
        y = [yo, yo + dy]
        z = [loop.centroid[1], loop.centroid[1]]
        path = Loop(x=x, y=y, z=z)
        face = make_face(loop)
        path = make_wire(path)

        left_ois = sweep(face, path)
        cut = cut_box()
        left_ois = boolean_cut(left_ois, cut)

        axis = gp_Ax2(gp_Pnt(loop.x[0], 0, loop.z[0]), gp_Dir(0, 1, 0), gp_Dir(0, 0, 1))
        right_ois = mirror_shape(
            left_ois,
            axis,
        )

        return make_compound([left_ois, right_ois])

    def _build_PF_seat(self, seat):  # noqa (N802)
        """
        Builds an individual PF seat

        Parameters
        ----------
        seat: dict
            The PF support dict

        Returns
        -------
        pf_seat: OCC Compound
            The BRep object of the PF seat
        """
        tk = self.geom["feed 3D CAD"]["other"]["PF"]["tk"]
        space = self.geom["feed 3D CAD"]["other"]["PF"]["space"]
        n = self.geom["feed 3D CAD"]["other"]["PF"]["n"]

        # Rib base loop
        loop = seat["loop"]
        # Offset by half-thickness
        loop = loop.translate([0, -tk / 2, 0], update=False)

        loops = [loop]
        if n > 1:
            # Linear pattern support ribs
            l_loop = loop.translate([0, -space + tk / 2, 0], update=False)
            loops.append(l_loop)
            shift = (2 * space - tk) / (n - 1)

            for i in range(1, n):
                n_loop = l_loop.translate([0, i * shift, 0], update=False)
                loops.append(n_loop)

        pf_seat = []
        for rib_loop in loops:
            face = make_face(rib_loop)
            body = extrude(face, vec=[0, tk, 0])
            pf_seat.append(body)

        plate, dt = self._make_PF_plate(seat, space, tk)

        face = make_face(plate)

        plate = extrude(face, vec=(0, 0, dt))

        pf_seat.append(plate)
        pf_seat = make_compound(pf_seat)
        return pf_seat

    @staticmethod
    def _make_PF_plate(seat, width, thickness):  # noqa (N802)
        """
        Makes a flat plate for a PF coil to sit on.
        NOTE: Butt ugly generalisation

        Parameters
        ----------
        seat: dict
            The PF support dict
        width: float
            The width of the plate
        thickness: float
            The thickness of the plate

        Returns
        -------
        plate: Geometry::Loop
            The loop of the plate
        tk: float
            The thickness (directional) of the plate
        """
        loop = seat["loop"]
        if (seat["p"]["z"][0] - seat["p"]["z"][1]) < 0:
            dt = -thickness
            down = True
        else:
            dt = thickness
            down = False

        if down:
            z_ref = np.min(loop.z)
            mask = np.where(np.isclose(loop.z, z_ref, rtol=1e-3))
            x_vals = loop.x[mask]
            x_min = np.min(x_vals)
            x_max = np.max(x_vals)

        else:
            z_ref = np.max(loop.z)
            mask = np.where(np.isclose(loop.z, z_ref, rtol=1e-3))
            x_vals = loop.x[mask]
            x_min = np.min(x_vals)
            x_max = np.max(x_vals)

        plate = Loop(
            x=[x_min, x_max, x_max, x_min, x_min],
            y=[-width, -width, width, width, -width],
            z=z_ref,
        )
        return plate, dt


def cut_box(side="right", size=30, n_TF=16):
    """
    Builds an oriented box for cutting TF coil components

    Parameters
    ----------
    side: str from ['right', 'left']
        The side on which to make the cutbox
    size: float
        The size of the cube (in all directions) [m]
    n_TF: int
        The number of TF coils (sets the angle)

    Returns
    -------
    box: OCC BRep
        The oriented cut box
    """
    angle = np.pi / n_TF
    if side == "right":
        v1 = (size * np.cos(angle), size * np.sin(angle), 0)
        v2 = (-size * np.tan(angle), size * np.cos(angle), 0)
    elif side == "left":
        v1 = (size * np.cos(angle), -size * np.sin(angle), 0)
        v2 = (-size * np.sin(angle), -size * np.cos(angle), 0)
    else:
        raise ValueError(f"O que caralho e essa merda?! {side}")

    return make_box((0, 0, -size / 2), v1, v2, (0, 0, size))


def build_CS_seat(cs_support):  # noqa (N802)
    """
    Builds a single central solenoid support seat

    Parameters
    ----------
    cs_support: dict
        The CS seat support dict

    Returns
    -------
    body: OCC BRep
        The BRep of the CS support

    """
    ynose = cs_support["ynose"]
    yfactor = 0.8

    loop = cs_support["loop"].copy()
    loop = loop.translate([0, -yfactor * ynose / 2, 0], update=False)

    face = make_face(loop)
    body = extrude(face, vec=[0, yfactor * ynose, 0])
    return body


def build_GS_JT60SA(g_support):
    """
    Builds a single gravity support (a la JST60-SA)

    Parameters
    ----------
    g_support: dict
        The gravity support dictionary

    Returns
    -------
    g_support: OCC Compond
        The BRep object for the gravity support
    """
    # Extract dimensions
    gs_base_list = []
    node = g_support["base"]
    width = g_support["width"]
    depth = g_support["tf_wp_depth"]
    side = g_support["tk_tf_side"]
    pin2pin = g_support["pin2pin"]
    g_alpha = g_support["alpha"]
    spread = g_support["spread"]
    zbase = g_support["zground"]
    rtube, ttube = g_support["rtube"], g_support["ttube"]
    x_gs = np.mean([node[0][0], node[1][0]])
    z_gs = node[1][1]

    # Make GS base
    loop = g_support["loop"].copy()
    loop.translate([0, -depth / 2 - side, 0])

    face = make_face(loop)
    body = extrude(face, vec=[0, depth + 2 * side, 0])
    gs_base_list.append(body)

    gs_axis1 = make_axis((x_gs, 0, z_gs - width / 2), (1, 0, 0))
    pin = make_circle((x_gs, 0, z_gs - width / 2), (1, 0, 0), width / 4)

    pin = extrude(pin, vec=(width, 0, 0))
    pin = translate_shape(pin, [-width / 2, 0, 0])
    arc = GC_MakeArcOfCircle(
        gp_Pnt(x_gs, -width / 2, z_gs - width / 2),
        gp_Pnt(x_gs, 0, z_gs - width),
        gp_Pnt(x_gs, width / 2, z_gs - width / 2),
    )
    arc = _make_OCCedge(arc.Value())

    base = BRepBuilderAPI_MakePolygon()
    base.Add(gp_Pnt(x_gs, -width / 2, z_gs - width / 2))
    base.Add(gp_Pnt(x_gs, -width / 2, z_gs))
    base.Add(gp_Pnt(x_gs, width / 2, z_gs))
    base.Add(gp_Pnt(x_gs, width / 2, z_gs - width / 2))

    gs_loop = _make_OCCwire([arc, base.Wire()])
    gs_face = _make_OCCface(gs_loop)
    gs_body = extrude(gs_face, vec=(width / 3, 0, 0))
    gs_body = topods.Solid(gs_body)
    gs_tag = translate_shape(gs_body, [-width / 6, 0, 0])
    gs_base = translate_shape(gs_tag, [-width / 2 + width / 6, 0, 0])
    gs_base = boolean_fuse(
        gs_base, translate_shape(gs_tag, [width / 2 - width / 6, 0, 0])
    )
    gs_base = boolean_fuse(gs_base, pin)
    gs_base_list.append(gs_base)
    gs_base_final = make_compound(gs_base_list)
    gs_base = rotate_shape(gs_base, gs_axis1, 180)
    gs_base = translate_shape(gs_base, [0, 0, -pin2pin])
    gb_axis2 = make_axis((x_gs, 0, z_gs - width / 2 - pin2pin), (0, 0, 1))
    gs_base = rotate_shape(gs_base, gb_axis2, 90)

    x_1, x_2 = x_gs + width, x_gs - width
    z_1, z_2 = z_gs - width - pin2pin, z_gs - 3 * width - pin2pin
    loop = Loop(x=[x_1, x_1, x_2, x_2, x_1], z=[z_1, z_2, z_2, z_1, z_1])
    gs_face = make_face(loop)

    gs_block = extrude(gs_face, vec=gp_Vec(0, width, 0))
    gs_block = translate_shape(gs_block, [0, -width / 2, 0])
    gs_block = topods.Solid(gs_block)
    gs_base = make_compound([gs_block, gs_base])

    loop = Loop(
        x=x_gs,
        y=[spread, spread, -spread, -spread, spread],
        z=[zbase, zbase - 1.5 * width, zbase - 1.5 * width, zbase, zbase],
    )
    gs_face2 = make_face(loop)

    gs_trim = extrude(gs_face2, vec=(2 * width, 0, 0))
    gs_trim = translate_shape(gs_trim, [-width, 0, 0])
    gs_base_left = rotate_shape(gs_base, gs_axis1, -g_alpha * 180 / np.pi)
    gs_base_right = rotate_shape(gs_base_left, gs_axis1, 2 * g_alpha * 180 / np.pi)
    gs_base_left = boolean_cut(gs_base_left, gs_trim)
    gs_base_right = boolean_cut(gs_base_right, gs_trim)

    # GS legs
    gs_tag = rotate_shape(gs_tag, gs_axis1, 180)
    r2c = 1.5 * width

    # This is messy but it is the only thing that works
    rect = BRepBuilderAPI_MakePolygon()
    rect.Add(gp_Pnt(x_gs - width / 6, -width / 2, z_gs - width))
    rect.Add(gp_Pnt(x_gs - width / 6, width / 2, z_gs - width))
    rect.Add(gp_Pnt(x_gs + width / 6, width / 2, z_gs - width))
    rect.Add(gp_Pnt(x_gs + width / 6, -width / 2, z_gs - width))
    rect.Close()
    rect_wire = rect.Wire()
    rect_face = _make_OCCface(rect_wire)

    rpath = _make_OCCedge(gp_Pnt(x_gs, 0, z_gs - width), gp_Pnt(x_gs, 0, z_gs - r2c))
    rpath = _make_OCCwire(rpath)
    circ = gp_Circ()
    circ.SetRadius(rtube)
    circ.SetAxis(gp_Ax1(gp_Pnt(x_gs, 0, z_gs - r2c), gp_Dir(0, 0, 1)))
    circ = _make_OCCedge(circ)
    circ = _make_OCCwire(circ)
    circ_face = make_circle((x_gs, 0, z_gs - r2c), (0, 0, 1), rtube)
    circ_cut = gp_Circ()
    circ_cut.SetRadius(rtube - ttube)
    circ_cut.SetAxis(gp_Ax1(gp_Pnt(x_gs, 0, z_gs - r2c), gp_Dir(0, 0, 1)))
    circ_cut = _make_OCCedge(circ_cut)
    circ_cut = _make_OCCwire(circ_cut)
    circ_face_cut = _make_OCCface(circ_cut)
    pipe = BRepFill_PipeShell(rpath)
    pipe.Add(rect_wire)
    pipe.Add(circ)
    pipe.Build()
    quilt = sew_shapes(pipe.Shape(), rect_face, circ_face)

    gs_transition = _make_OCCsolid(quilt)
    gs_transition = boolean_fuse(gs_transition, gs_tag)
    gs_t_upper = boolean_cut(gs_transition, pin)  # make upper hole

    tube_body = extrude(circ_face, vec=(0, 0, -(pin2pin + width - 2 * r2c)))
    tube_body = topods.Solid(tube_body)
    tube_body_cut = extrude(circ_face_cut, vec=(0, 0, -(pin2pin + width - 2 * r2c)))
    tube_body_cut = topods.Solid(tube_body_cut)
    tube_body = boolean_cut(tube_body, tube_body_cut)
    gs_t_lower = rotate_shape(gs_t_upper, gs_axis1, 180)
    gs_t_lower = translate_shape(gs_t_lower, [0, 0, -pin2pin])
    gs_t_lower = rotate_shape(gs_t_lower, make_axis((x_gs, 0, z_gs), (0, 0, 1)), 90)
    leg = boolean_fuse(gs_t_upper, tube_body)
    leg = boolean_fuse(leg, gs_t_lower)
    leftleg = rotate_shape(leg, gs_axis1, -g_alpha * 180 / np.pi)
    rightleg = rotate_shape(leg, gs_axis1, g_alpha * 180 / np.pi)
    gs_strut = boolean_fuse(leftleg, rightleg)
    return make_compound([gs_base_final, gs_base_left, gs_base_right, gs_strut])


def build_GS_ITER(g_support):
    """
    Builds a single gravity support (a la ITER)

    Parameters
    ----------
    g_support: dict
        The gravity support dictionary

    Returns
    -------
    g_support: OCC Compond
        The BRep object for the gravity support
    """
    depth = g_support["tf_wp_depth"]
    side = g_support["tk_tf_side"]
    p_width = g_support["plate_width"]
    width = g_support["width"]

    compound_list = []
    # Make GS base
    loop = g_support["loop"].copy()
    loop.translate([0, -depth / 2 - side, 0])

    face = make_face(loop)
    body = extrude(face, vec=[0, depth + 2 * side, 0])
    compound_list.append(body)

    # Make GS plates
    n_plates = width / (2 * p_width)
    delta = (n_plates - int(n_plates)) * 2 * p_width
    x_left = g_support["Xo"] - width / 2 + delta
    loop = Loop(
        x=[-p_width / 2, p_width / 2, p_width / 2, -p_width / 2, -p_width / 2],
        y=[-width / 2, -width / 2, width / 2, width / 2, -width / 2],
        z=g_support["zfloor"],
    )
    loop.translate([x_left, 0, 0])
    for i in range(int(n_plates)):
        p_loop = loop.translate([i * 2 * p_width, 0, 0], update=False)
        face = make_face(p_loop)
        plate = extrude(face, vec=[0, 0, g_support["zbase"] - g_support["zfloor"]])
        compound_list.append(plate)

    # Make GS floor
    loop = Loop(
        x=[-width / 2, width / 2, width / 2, -width / 2, -width / 2],
        y=[-width / 2, -width / 2, width / 2, width / 2, -width / 2],
        z=g_support["zfloor"],
    )
    loop.translate([g_support["Xo"], 0, 0])
    face = make_face(loop)
    floor = extrude(face, vec=[0, 0, -3 * p_width])
    compound_list.append(floor)
    return make_compound(compound_list)


if __name__ == "__main__":
    pass
    # from BLUEPRINT import test
    #
    # test()
