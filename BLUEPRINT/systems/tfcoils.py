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
Toroidal field system
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Type
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline

from bluemira.base.constants import MU_0
from bluemira.base.parameter import ParameterFrame
from bluemira.base.look_and_feel import bluemira_warn

from BLUEPRINT.nova.coilcage import HelmholtzCage as CoilCage
from BLUEPRINT.systems.baseclass import ReactorSystem
from BLUEPRINT.base.error import SystemsError
from BLUEPRINT.geometry.offset import offset_smc, offset
from BLUEPRINT.geometry.boolean import (
    boolean_2d_difference_loop,
    boolean_2d_union,
    clean_loop,
    simplify_loop,
)
from BLUEPRINT.geometry.geomtools import length, lengthnorm, make_box_xz, rainbow_seg
from BLUEPRINT.geometry.loop import Loop, MultiLoop, make_ring
from BLUEPRINT.geometry.shell import Shell, MultiShell
from BLUEPRINT.geometry.shape import Shape
from BLUEPRINT.cad.coilCAD import TFCoilCAD
from BLUEPRINT.systems.mixins import Meshable
from BLUEPRINT.systems.plotting import ReactorSystemPlotter


class ToroidalFieldCoils(Meshable, ReactorSystem):
    """
    Reactor toroidal field (TF) coil system
    """

    config: Type[ParameterFrame]
    inputs: dict

    # fmt: off
    default_params = [
        ['R_0', 'Major radius', 9, 'm', None, 'Input'],
        ['B_0', 'Toroidal field at R_0', 6, 'T', None, 'Input'],
        ['n_TF', 'Number of TF coils', 16, 'N/A', None, 'Input'],
        ['rho_j', 'TF coil WP current density', 18.25, 'MA/m^2', None, 'Input'],
        ['twall', 'Wall thickness', 0.045, 'm', 'No idea yet', 'Nova'],
        ['tk_tf_nose', 'TF coil inboard nose thickness', 0.6, 'm', None, 'Input'],
        ['tk_tf_wp', 'TF coil winding pack thickness', 0.5, 'm', None, 'PROCESS'],
        ['tk_tf_front_ib', 'TF coil inboard steel front plasma-facing', 0.04, 'm', None, 'Input'],
        ['tk_tf_ins', 'TF coil ground insulation thickness', 0.08, 'm', None, 'Input'],
        ['tk_tf_insgap', 'TF coil WP insertion gap', 0.1, 'm', 'Backfilled with epoxy resin (impregnation)', 'Input'],
        ['tk_tf_side', 'TF coil inboard case minimum side wall thickness', 0.1, 'm', None, 'Input'],
        ['tk_tf_case_out_in', 'TF coil case thickness on the outboard inside', 0.35, 'm', None, 'Calc'],
        ['tk_tf_case_out_out', 'TF coil case thickness on the outboard outside', 0.4, 'm', None, 'Calc'],
        ['tf_wp_width', 'TF coil winding pack radial width', 0.76, 'm', 'Including insulation', 'PROCESS'],
        ['tf_wp_depth', 'TF coil winding pack depth (in y)', 1.05, 'm', 'Including insulation', 'PROCESS'],
        ['tk_tf_outboard', 'TF coil outboard thickness', 1, 'm', None, 'Input', 'PROCESS'],
        ['r_tf_in', 'Inboard radius of the TF coil inboard leg', 3.2, 'm', None, 'PROCESS'],
        ['TF_ripple_limit', 'TF coil ripple limit', 0.6, '%', None, 'Input'],
        ['h_cp_top', 'Height of the Tapered Section', 4.199, 'm', None, 'PROCESS'],
        ['r_cp_top', 'Radial Position of Top of taper', 1.31, 'm', None, 'PROCESS'],
        ['tf_taper_frac', "Height of straight portion as fraction of total tapered section height", 0.5, 'N/A', None, 'Input'],
        ['r_tf_outboard_corner', "Corner Radius of TF coil outboard legs", 0.8, 'm', None, 'Input'],
        ['r_tf_inboard_corner', "Corner Radius of TF coil inboard legs", 0.0, 'm', None, 'Input'],
        ["tk_tf_ob_casing", "TF outboard leg conductor casing thickness", 0.1, "m", None, "PROCESS"],
        ["r_tf_curve", "Radial position of the CP-leg conductor joint", 1.5, "m", None, "PROCESS"],
        ['h_tf_max_in', 'Plasma side TF coil maximum height', 11.5, 'm', None, 'PROCESS'],
        ['r_tf_in_centre', 'Inboard TF leg centre radius', 3.7, 'N/A', None, 'PROCESS'],
        ['tk_tf_inboard', 'TF coil inboard thickness', 1, 'm', None, 'Input'],
        ["r_tf_inboard_out", "Outboard Radius of the TF coil inboard leg tapered region", 0.8934, "m", None, "PROCESS"],
        ['h_tf_min_in', 'Plasma side TF coil min height', -6.5, 'm', None, 'PROCESS'],
    ]
    # fmt: on
    CADConstructor = TFCoilCAD

    def __init__(self, config, inputs):

        self.config = config
        self.inputs = inputs
        self._plotter = ToroidalFieldCoilsPlotter()

        self._init_params(self.config)

        # Constructors
        self.flag_new = False
        self.ripple = True
        self.loops = {}
        self.section = {}
        self.shp = None
        self.cage = None
        self.ro = None
        self.rc = None
        self.max_ripple = None
        self._maxL = None
        self._minL = None
        self._energy_norm = None

        self.sep = self.inputs["plasma"].copy()
        self.shape_type = self.inputs["shape_type"]
        self.wp_shape = self.inputs["wp_shape"]
        self.conductivity = self.inputs["conductivity"]
        self.ripple_limit = self.params.TF_ripple_limit

        # Number of points on the LCFS to evaluate ripple on
        self.nrippoints = int(self.inputs.get("nrip", 25))
        # Number of radial filaments
        self.nr = int(self.inputs.get("nr", 1))
        # Number of toroidal toroidal filaments
        self.ny = int(self.inputs.get("ny", 1))
        # Number of points to use for the vessel and plasma interpolation
        self.npoints = int(self.inputs.get("npoints", 80))

        self.initialise_loops()
        self.update_cage = False  # Wieso??
        self.initialise_profile()
        self.p_in = self.shp.parameterisation.draw()
        self.set_inner_loop()
        self.add_koz()
        self.set_cage()  # initalise cage

    def initialise_loops(self):
        """
        Initialise the dictionary of loop structures for the TF coil.
        """
        self.loops = {}
        for loop in [
            "in",
            "wp_in",
            "cl",
            "wp_out",
            "out",
            "nose",
            "loop",
            "trans_lower",
            "trans_upper",
            "cl_fe",
            "b_cyl",
        ]:
            self.loops[loop] = {"x": [], "z": []}

    def initialise_profile(self):
        """
        Carry out all shape parameterisation adjustments.

        Notes
        -----
        The TF geometry in BLUEPRINT is a trapezoidal wedge, so the mid-plane extent is
        less than the extent along the diagonal. While we parameterise based on the
        mid-plane, we also have to make sure to avoid collisions with the keep-out-zone
        along the diagonal.
        """
        self.shp = Shape(
            self.inputs["name"],
            family=self.inputs["shape_type"],
            wp_shape=self.inputs["wp_shape"],
            conductivity=self.inputs["conductivity"],
            objective=self.inputs["obj"],
            npoints=self.npoints,
            n_TF=self.params.n_TF,
            read_write=True,
            read_directory=self.inputs["read_folder"],
            write_directory=self.inputs["write_folder"],
        )

        # The outer point of the TF coil inboard in the mid-plane
        r_tf_inboard_out = (
            self.params.r_tf_in
            + self.params.tk_tf_nose
            + self.params.tk_tf_wp
            + self.params.tk_tf_front_ib
        )

        if self.conductivity in ["SC"] and self.wp_shape not in ["N"]:
            # Use the CORRECT process-provided values to define outboard edge of
            # centrepost/inboard leg in the midplane - replaces r_tf_inboard_out
            r_tf_inboard_out = self.params.r_tf_in + self.params.tk_tf_inboard
            # Adjust for the radial build discrepancy in the wp thickness
            # between a WP curved face (BLUEPRINT) and flat face (PROCESS)
            self.params.tk_tf_wp = (r_tf_inboard_out - self.params.tk_tf_front_ib) - (
                self.params.r_tf_in + self.params.tk_tf_nose
            )

        # The keep-out-zone at the mid-plane has to be scaled down from the keep-out-zone
        # at the maximum TF radius to avoid collisions on the inner leg.
        x_koz_min = np.min(self.inputs["koz_loop"].x) * np.cos(np.pi / self.params.n_TF)

        if self.wp_shape not in ["N"]:
            # Don't need the cosine term for curved inboard plasma facing faces
            x_koz_min = np.min(self.inputs["koz_loop"].x)

        x_koz_max = np.max(self.inputs["koz_loop"].x)

        if x_koz_min < r_tf_inboard_out:
            bluemira_warn(
                "TF coil radial build issue, resetting TF inboard outer edge in the "
                f"mid-plane from {r_tf_inboard_out:.6f} to {x_koz_min:.6f}."
            )
            r_tf_inboard_out = x_koz_min

        R_0 = self.params.R_0

        if self.inputs["shape_type"] == "S":
            # inner radius
            # NOTE: SLSQP doesn't like it when you remove x1 from the S
            # parameterisation...
            self.adjust_xo(
                "x1",
                lb=r_tf_inboard_out - 0.001,
                value=r_tf_inboard_out,
                ub=r_tf_inboard_out + 0.001,
            )

            # tailor limits on loop parameters (l -> loop tension)
            self.adjust_xo("l", lb=0.05)  # don't go too high (<1.2)
            self.adjust_xo("x2", value=16, lb=10, ub=25)  # outer radius
            self.adjust_xo("z2", value=0, lb=-0.9, ub=0.9)  # outer node vertical shift
            self.adjust_xo("height", value=18, lb=0.1, ub=30)  # full loop height
            self.adjust_xo("top", value=0.5, lb=0.05, ub=1)  # horizontal shift
            self.adjust_xo("upper", value=0.7, lb=0.2, ub=1)  # vertical shift

            self._minL, self._maxL = 0.35, 0.75  # Ripple search range
        elif self.inputs["shape_type"] == "D":
            # inner radius
            self.adjust_xo("x1", value=r_tf_inboard_out)
            self.shp.remove_oppvar("x1")

            # outer radius
            self.adjust_xo("x2", lb=x_koz_max, value=1.1 * x_koz_max, ub=2 * R_0)
            # vertical offset
            z_mid = np.average(self.sep["z"])
            self.adjust_xo("dz", value=z_mid)
            self.shp.remove_oppvar("dz")
            # self.adjust_xo("dz", lb=z_mid - 0.001, value=z_mid, ub=z_mid + 0.001)

            self._minL, self._maxL = 0.35, 0.75

        elif self.inputs["shape_type"] == "A":
            self.adjust_xo("xo", value=r_tf_inboard_out)
            self.shp.remove_oppvar("xo")
            self._minL, self._maxL = 0.2, 0.8

        elif self.inputs["shape_type"] == "P":
            # Drop the corner radius from the optimisation
            self.adjust_xo("ro", value=self.params.r_tf_outboard_corner)
            self.shp.remove_oppvar("ro")

            self.adjust_xo("ri", value=self.params.r_tf_inboard_corner)
            self.shp.remove_oppvar("ri")

            # Set the inner radius of the shape, and pseudo-remove from optimiser
            self.adjust_xo(
                "x1",
                value=r_tf_inboard_out,
            )
            self.shp.remove_oppvar("x1")

            # Adjust bounds to fit problem
            zmin = np.min(self.inputs["koz_loop"]["z"])
            zmax = np.max(self.inputs["koz_loop"]["z"])
            xmax = np.max(self.inputs["koz_loop"]["x"])
            self.adjust_xo("z1", value=zmax + 1e-4)
            self.adjust_xo("z2", value=zmin - 1e-4)
            self.shp.remove_oppvar("z1")
            self.shp.remove_oppvar("z2")
            self.adjust_xo("x2", lb=xmax, value=xmax + 5.0e-4, ub=xmax + 3)

            # Adjust the range of points on the separatrix to check for ripple
            self._minL, self._maxL = 0.2, 0.8

        elif self.inputs["shape_type"] == "TP":
            # Set the inner radius of the shape, and pseudo-remove from optimiser
            tk_case = self.params.tk_tf_ob_casing
            # We add tk_case in x1 since the casing will be deleted later as the
            # centrepost has no casing, however, for the initial Loop generation
            # the casing loop circulates the entire coil, with extra bits chopped
            # off later
            self.adjust_xo("x1", value=self.params.r_tf_inboard_out + tk_case)
            self.adjust_xo("z2", value=self.params.h_cp_top)
            self.adjust_xo("x2", value=self.params.r_cp_top + tk_case)
            self.adjust_xo("r", value=self.params.r_tf_outboard_corner - tk_case)
            self.adjust_xo("z1_frac", value=self.params.tf_taper_frac)
            self.shp.remove_oppvar("z2")
            self.shp.remove_oppvar("x1")
            self.shp.remove_oppvar("x2")
            self.shp.remove_oppvar("r")
            self.shp.remove_oppvar("z1_frac")

            # Adjust bounds to fit problem
            zmax = np.max(self.inputs["koz_loop"]["z"])
            xmax = np.max(self.inputs["koz_loop"]["x"])
            self.adjust_xo("z3", lb=zmax + 1e-4, value=zmax * 1.01, ub=zmax * 1.02)
            self.adjust_xo("x3", lb=xmax, value=xmax * 1.2, ub=xmax * 1.5)

            # Adjust the range of points on the separatrix to check for ripple
            self._minL, self._maxL = 0.2, 0.8

        elif self.inputs["shape_type"] == "CP":
            # Inboard mid-plane radius (plasma side)
            # Taper end z corrdinate (curve top end)

            if self.conductivity in ["R"]:
                self.adjust_xo(
                    "x_in",
                    value=self.params.r_tf_inboard_out + self.params.tk_tf_ob_casing,
                )
                self.adjust_xo("z_in", value=self.params.h_cp_top)
            else:
                self.adjust_xo("x_in", value=r_tf_inboard_out)
                self.adjust_xo("z_in", value=0)
            self.shp.remove_oppvar("x_in")
            self.shp.remove_oppvar("z_in")

            # Taper end x-coordinate (curve top end)
            if self.conductivity in ["R"]:
                self.adjust_xo(
                    "x_mid", value=self.params.r_cp_top + self.params.tk_tf_ob_casing
                )
            else:
                self.adjust_xo("x_mid", value=r_tf_inboard_out)
            self.shp.remove_oppvar("x_mid")

            # Top/bot doming start x-coordinate
            self.adjust_xo("x_curve_start", value=self.params.r_tf_curve)
            self.shp.remove_oppvar("x_curve_start")

            # Curvature end (hard coded)
            self.adjust_xo("r_c", value=0.5)
            self.shp.remove_oppvar("r_c")

            # Central column top and bottom z-coordinate
            z_mid_max = np.max(self.inputs["koz_loop"]["z"])
            z_mid_min = np.min(self.inputs["koz_loop"]["z"])

            z_top_val = (
                z_mid_max + 1e-3
                if self.params.h_tf_max_in == 0
                else self.params.h_tf_max_in
            )
            z_bottom_val = (
                z_mid_min - 1e-3
                if self.params.h_tf_min_in == 0
                else self.params.h_tf_min_in
            )
            adjustments = {
                "z_mid_up": z_mid_max + 1e-3,
                "z_mid_down": z_mid_min - 1e-3,
                "z_top": z_top_val,
                "z_bottom": z_bottom_val,
            }
            for key, value in adjustments.items():
                self.adjust_xo(key, value=value)
                self.shp.remove_oppvar(key)

            # Outboard leg position (ripple optimization variable)
            xmax = np.max(self.inputs["koz_loop"]["x"])
            self.adjust_xo("x_out", lb=xmax + 1e-4, value=xmax + 5.0e-1, ub=xmax * 1.5)

            # Adjust the range of points on the separatrix to check for ripple
            self._minL, self._maxL = 0.2, 0.8

        else:
            raise SystemsError(
                f"TF shape parameterisation "
                f'{self.inputs["shape_type"]} not recognised.'
            )

    def adjust_xo(self, name, **kwargs):
        """
        Adjust shape parameters of under-lying Shape object
        """
        self.shp.adjust_xo(name, **kwargs)
        self.p_in = self.shp.parameterisation.draw()
        self.set_inner_loop()

    def set_inner_loop(self):
        """
        Set the internal loop of the TF coil shape.
        """
        self.ro = np.min(self.p_in["x"])
        self.cross_section()
        self._get_loops(self.p_in)

    def cross_section(self):
        """
        Calculates the inboard cross-section and sets the outboard CS
        """
        # Keep this structure for future interfacing with structural solver
        self.section = {}
        # Rough casing radial build for external TF leg

        iocasthk = (self.params.tk_tf_front_ib + self.params.tk_tf_nose) / 2

        if self.wp_shape != "N":
            # For coils with constant casing thicknesses
            self.section["case"] = {
                "side": self.params.tk_tf_side,
                "nose": self.params.tk_tf_nose,
                "WP": self.params.tk_tf_wp,
                "inboard": self.params.tk_tf_front_ib,
                "outboard": self.params.tk_tf_front_ib,
                "external": self.params.tk_tf_nose,
            }
        else:
            # Casing thickness changing around the coil
            self.section["case"] = {
                "side": self.params.tk_tf_side,
                "nose": self.params.tk_tf_nose,
                "WP": self.params.tk_tf_wp,
                "inboard": self.params.tk_tf_front_ib,
                "outboard": iocasthk * 1.1,
                "external": iocasthk * 0.9,
            }

        if self.conductivity in ["R"]:
            # Resistive coils
            iocasthk = 0
            self.section["case"] = {
                "side": self.params.tk_tf_ob_casing,
                "nose": self.params.tk_tf_ob_casing,
                "WP": self.params.tk_tf_outboard,
                "inboard": self.params.tk_tf_ob_casing,
                "outboard": self.params.tk_tf_ob_casing,
                "external": self.params.tk_tf_ob_casing,
            }
        elif self.inputs["shape_type"] == "CP":
            # For CURVED SC coils
            iocasthk = 0
            self.section["case"] = {
                "side": self.params.tk_tf_side,
                "nose": self.params.tk_tf_nose,
                "WP": self.params.tk_tf_wp,
                "inboard": self.params.tk_tf_front_ib,
                "outboard": self.params.tk_tf_front_ib,
                "external": self.params.tk_tf_nose,
            }

        bm = -self.params.B_0 * self.params.R_0
        current = abs(2 * np.pi * bm / (self.params.n_TF * MU_0))
        self.add_parameter("I_tf", "TF coil current", current, "MA", None, "BLUEPRINT")
        r_wp_in = self.params.r_tf_in + self.params.tk_tf_nose
        # Rem : Not the same definition of WP depth is used between shapes !
        depth = 2 * (r_wp_in * np.tan(np.pi / self.params.n_TF) - self.params.tk_tf_side)

        if self.conductivity in ["SC"] and self.wp_shape != "N":
            # For 'nosed' wp shapes
            depth = self.params.tf_wp_depth

        elif self.conductivity in ["R"]:
            # For resistive coils
            r_wp_in = self.params.r_cp_top
            depth = 2 * (r_wp_in * np.tan(np.pi / self.params.n_TF))

        elif self.conductivity in ["SC"] and self.inputs["shape_type"] in ["CP"]:
            # For Curved SC coils (maybe not needed?)
            x_shift = self.params.tk_tf_side / np.tan(np.pi / self.params.n_TF)
            r_wp_in = (
                self.params.r_tf_in
                + self.params.tk_tf_inboard
                - self.params.tk_tf_front_ib
            )
            depth = 2 * ((r_wp_in - x_shift) * np.sin(np.pi / self.params.n_TF))

        if self.inputs["shape_type"] in ["TP", "CP"]:
            # For designs with a tapered Centrepost
            self.section["winding_pack"] = {
                "width": self.section["case"]["WP"],
                "depth": depth,
            }
            self.rc = (self.section["case"]["WP"] + depth) / 4 / self.nr
        else:
            self.section["winding_pack"] = {
                "width": self.params.tk_tf_wp,
                "depth": depth,
            }
            self.rc = (self.params.tk_tf_wp + depth) / 4 / self.nr

            # Update cross-sectional parameters
            source_name = "TF Cross Section"
            if self.wp_shape != "N":
                # For coils with constant casing thicknesses
                self.params.update_kw_parameters(
                    {
                        "tk_tf_case_out_in": self.params.tk_tf_front_ib,
                        "tk_tf_case_out_out": self.params.tk_tf_nose,
                        "tf_wp_width": self.params.tk_tf_wp,
                        "tf_wp_depth": depth,
                    },
                    source_name,
                )
            else:
                self.params.update_kw_parameters(
                    {
                        "tk_tf_case_out_in": iocasthk * 0.9,
                        "tk_tf_case_out_out": iocasthk * 1.1,
                        "tf_wp_width": self.params.tk_tf_wp,
                        "tf_wp_depth": depth,
                    },
                    source_name,
                )

    def _get_loops(self, p_in):
        """
        Warning: this is now a semi-deprecated function: if you wanted the
        optimsed TF loops after the fact, just call TF.l
        """
        x, z = p_in["x"], p_in["z"]
        wp = self.section["winding_pack"]
        case = self.section["case"]
        inboard_dt = [case["inboard"], wp["width"] / 2, wp["width"] / 2, case["nose"]]
        outboard_dt = [
            case["outboard"],
            wp["width"] / 2,
            wp["width"] / 2,
            case["external"],
        ]
        loops = ["wp_in", "cl", "wp_out", "out"]

        if self.conductivity in ["R"]:
            loops = ["wp_in", "cl", "wp_out", "out", "b_cyl"]
            inboard_dt = [
                self.params.tk_tf_ob_casing,
                wp["width"] / 2,
                wp["width"] / 2,
                self.params.tk_tf_ob_casing,
                0.00,
            ]
            outboard_dt = [
                self.params.tk_tf_ob_casing,
                wp["width"] / 2,
                wp["width"] / 2,
                self.params.tk_tf_ob_casing,
                0.00,
            ]
        self.loops["in"]["x"], self.loops["in"]["z"] = x, z
        index = self.transition_index(self.loops["in"]["x"], self.loops["in"]["z"])

        for loop, dt_in, dt_out in zip(loops, inboard_dt, outboard_dt):
            if self.conductivity in ["R"]:
                # Designs that might have a tapered CP or
                # if offset_clipper needs to be used
                dt = dt_in
            else:
                dt = self._loop_dt(x, z, dt_in, dt_out, index)

            if self.shape_type in ["P"]:
                # offset smc produces weird corners
                # offset_clipper breaks optimiser for some coils....sigh
                x, z = offset(x, z, np.mean(dt))

            else:
                x, z = offset_smc(x, z, dt, close_loop=True, min_steps=len(p_in["x"]))

            # TODO check that unwind does not effect stored energy calculation
            # self.p[loop] = unwind({'x': x, 'z': z})
            self.loops[loop]["x"], self.loops[loop]["z"] = x, z

            if self.conductivity in ["R"]:

                if loop == "b_cyl":

                    x, z = self.bucking_cylinder(self.loops["wp_out"])
                    if self.shape_type in ["CP"]:
                        # thanks to offset sharp corner weirdness
                        x, z = self.bucking_cylinder(self.loops["wp_in"])
                    self.loops[loop]["x"], self.loops[loop]["z"] = x, z

        return self.loops

    @staticmethod
    def transition_index(x_in, z_in, eps=1e-12):
        """
        Heilige Scheisse im Himmel was soll der Scheiss mensch!
        Entiendo que hace, pero no como funciona!
        """
        npoints = len(x_in)
        x_cl = x_in[0] + eps
        upper = npoints - next((i for i, x_in_ in enumerate(x_in) if x_in_ > x_cl))  # +1
        lower = next((i for i, x_in_ in enumerate(x_in) if x_in_ > x_cl))
        top, bottom = np.argmax(z_in), np.argmin(z_in)
        index = {"upper": upper, "lower": lower, "top": top, "bottom": bottom}
        return index

    @staticmethod
    def _loop_dt(x, z, dt_in, dt_out, index):
        le = lengthnorm(x, z)
        l_values = np.array(
            [
                0,
                le[index["lower"]],
                le[index["bottom"]],
                le[index["top"]],
                le[index["upper"]],
                1,
            ]
        )
        d_x = np.array([dt_in, dt_in, dt_out, dt_out, dt_in, dt_in])
        dt = interp1d(l_values, d_x)(le)
        return dt

    def set_cage(self):
        """
        Set up the CoilCage object for the ToroidalFieldCoils.
        """
        z_0 = np.average(self.sep["z"])
        self.cage = CoilCage(
            self.params.n_TF.value,
            self.params.R_0.value,
            z_0,
            self.params.B_0.value,
            self.sep,
            rc=None,
            ny=self.ny,
            nr=self.nr,
            npts=self.npoints,
            winding_pack=self.section["winding_pack"],
        )
        self.update_cage = True
        self.initalise_cloop()

    def initalise_cloop(self):
        """
        Initialise the Shape Parameterisation and update the CoilCage.
        """
        x = self.shp.parameterisation.xo.get_value()
        p_in = self.shp.parameterisation.draw(x=x)
        xloop = self._get_loops(p_in)  # update tf
        if self.update_cage:
            self.cage.set_coil(xloop["cl"])  # update coil cage
        return xloop

    def optimise(self, verbose=True, **kwargs):
        """
        Carry out the optimisation of the TF coil centreline shape.

        Parameters
        ----------
        verbose: bool (default = True)
            Verbosity of the scipy optimiser

        Other Parameters
        ----------------
        ripple: bool
            Whether or not to include a ripple constraint
        ripple_limit: float
            The maximum toroidal field ripple on the separatrix [%]
        ny: intz
            The number of current filaments in the y direction
        nr: int
            The number of current filaments in the radial direction
        nrippoints: int
            The number of points on the separatrix to check for ripple
        """
        # Handle kwargs and update corresponding attributes
        for attr in ["ripple", "ripple_limit", "nrippoints"]:
            setattr(self, attr, kwargs.get(attr, getattr(self, attr)))
            kwargs.pop(attr, None)
        for attr in ["nr", "ny"]:
            setattr(self, attr, kwargs.get(attr, getattr(self, attr)))
            setattr(self.cage, attr, kwargs.get(attr, getattr(self, attr)))
            kwargs.pop(attr, None)

        self.p_in = self.shp.parameterisation.draw()

        # Over-ride the Shape default constraints and objectives
        self.shp.f_ieq_constraints = self.constraints
        self.shp.f_objective = self.objective  # objective function
        self.shp.f_update = self.update_loop  # called on exit from minimizer

        # Perform optimisation with geometric and magnetic constraints
        self.ripple = True
        self.shp.args = (self.ripple, self.ripple_limit)
        self.shp.optimise(verbose=verbose, **kwargs)

        self.shp.write()
        self.cage.loop_ripple()
        self.sanity()

        # Update various loops and parameters with optimised result
        self.store_info()
        self._generate_xz_plot_loops()
        self._generate_xy_plot_loops()

    def store_info(self):
        """
        Store information in the ToroidalFieldCoils object, and update ParameterFrame.
        """
        self.geom["Centreline"] = simplify_loop(Loop(**self.loops["cl"]))
        self.geom["WP outer"] = simplify_loop(Loop(**self.loops["wp_out"]))
        self.geom["WP inner"] = simplify_loop(Loop(**self.loops["wp_in"]))
        if self.conductivity in ["R"]:
            self.geom["B Cyl"] = simplify_loop(Loop(**self.loops["b_cyl"]))
        self.add_parameter(
            "E_sto",
            "TF coil stored energy",
            self.cage.energy() / 1e9,
            "GJ",
            None,
            "Nova",
        )

    def sanity(self):
        """
        Perform sanity to check to see if the TF ripple is withing the specified
        limit, or whether the ripple is supra-optimal.
        """
        tol = 0.002
        self.max_ripple = max(self.cage.ripple)
        if self.max_ripple > (1 + tol) * self.ripple_limit:
            bluemira_warn(
                "TF coil ripple exceeds optimiser specification: "
                f"{self.max_ripple:.3f} %."
            )
        elif self.max_ripple < (1 - tol) * self.ripple_limit:
            bluemira_warn(
                "TF coil ripple is supra-optimised: " f"{self.max_ripple:.3f} %."
            )

    def load_shape(self):
        """
        Loads a stored TF shape parameterisation result from a file
        """
        self.shp.load()
        self.p_in = self.shp.parameterisation.draw()
        self.update_loop(self.shp.parameterisation.xo.get_xnorm())
        self.cage.loop_ripple()
        self.sanity()
        self.store_info()
        self._generate_xz_plot_loops()
        self._generate_xy_plot_loops()

    def get_TF_track(self, offset):
        """
        Builds the track upon which to pin the PF coils. Already clips the
        inboard leg and flattens the D

        Returns
        -------
        track: BLUEPRINT Loop
            The track along which to optimise the positions of the PF coils
        """
        x, z = self.loops["out"]["x"], self.loops["out"]["z"]
        amin, amax = np.argmin(z), np.argmax(z)
        zmin, zmax = np.min(z), np.max(z)
        xmin = np.min(x)
        if amin < amax:
            xx = np.array(xmin)
            xx = np.append(xx, x[amin:amax])
            xx = np.append(xx, xmin)
            zz = np.array(zmin)
            zz = np.append(zz, z[amin:amax])
            zz = np.append(zz, zmax)
        track = Loop(x=xx, z=zz)
        track.interpolate(200)
        return track.offset(offset)

    def add_koz(self):
        """
        Adds keep-out zones for the optimiser, and a radial build constraint
        """
        koz = self.inputs["koz_loop"].copy()
        koz.interpolate(self.npoints)
        rvv, zvv = koz.x, koz.z
        self.shp.add_bound({"x": rvv, "z": zvv}, "internal")  # vv+gap+vvts+gap

    def update_loop(self, xnorm, *args):
        """
        The TF coil update function, used after the optimisation is complete.

        Parameters
        ----------
        xnorm: np.array
            The normalised vector of shape variables
        args: tuple
            Additional arguments for the optimiser

        Returns
        -------
        xloop: dict
            The dictionary of TF loop dicts
        """
        x = self.shp.parameterisation.get_oppvar(xnorm)
        xloop = self._get_loops(self.shp.parameterisation.draw(x=x))  # update tf
        self.shp.parameterisation.set_input(x=x)  # inner loop
        if self.update_cage:
            self.cage.set_coil(xloop["cl"])  # update coil cage
        return xloop

    def constraints(self, xnorm, *args):
        """
        The TF coil constraint function used in the shape optimisation.

        Parameters
        ----------
        xnorm: np.array
            The normalised vector of shape variables
        args: tuple
            Additional arguments for the optimiser

        Returns
        -------
        constraint: np.array
            The array of constraint equation values
        """
        (
            ripple,
            ripple_limit,
        ) = args
        # de-normalize
        if ripple:  # constrain ripple contour
            xloop = self.update_loop(xnorm, *args)
            constraint = np.array([])
            if self.wp_shape in ["N"]:
                # For ST reactor coils, the KOZ is captured by the bounds set
                # in initialise_profile, since the coil shape is constrained in
                # all but the outboard leg x location, it greatly improves stability
                # and accuracy to switch off the geometric constraint. This
                # prevents a poorly defined/shaped KOZ from breaking the coil
                # optimisation
                for side, key in zip(
                    ["internal", "interior", "external"], ["in", "in", "out"]
                ):
                    constraint = np.append(
                        constraint, self.shp.dot_difference(xloop[key], side)
                    )
            max_ripple = self.cage.get_max_ripple()

            # NOTE: npoints is the resolution of the ripple calculation along
            # the separatrix. Turning it up will more or less guarantee that
            # the ripple constraint is met, but increases time significantly.
            edge_ripple = self.cage.edge_ripple(
                npoints=int(self.nrippoints), min_l=self._minL, max_l=self._maxL
            )
            constraint = np.append(constraint, ripple_limit - edge_ripple)
            constraint = np.append(constraint, ripple_limit - max_ripple)
        else:  # constraint from shape
            constraint = self.shp.geometric_constraints(xnorm, *args)
        return constraint

    def objective(self, xnorm, *args):
        """
        The TF coil objective function used in the shape optimisation.

        Parameters
        ----------
        xnorm: np.array
            The normalised vector of shape variables
        args: tuple
            Additional arguments for the optimiser

        Returns
        -------
        value: float
            The value of the objective function
        """
        # loop length or loop volume (torus)
        if self.shp.objective == "L" or self.shp.objective == "V":
            value = self.shp.geometric_objective(xnorm, *args)
        elif self.shp.objective == "E":
            value = self.energy(xnorm, *args)
        return value

    def energy(self, xnorm, *args):
        """
        The TF coil stored energy minimisation objective function used in the
        shape optimisation.

        Parameters
        ----------
        xnorm: np.array
            The normalised vector of shape variables
        args: tuple
            Additional arguments for the optimiser

        Returns
        -------
        energy_norm: float
            The normalised value of the TF coil cage stored energy, relative to
            an initial value calculated for a default shape.
        """
        self.update_cage = True
        self.update_loop(xnorm, *args)

        # We normalise the value of the energy to the first pass, so that the
        # values of the objective function are around 1.
        if self._energy_norm is None:
            self._energy_norm = self.cage.energy()
            return 1.0

        return self.cage.energy() / self._energy_norm

    def plot_ripple(self, ax=None, **kwargs):
        """
        Plots the toroidal field ripple.

        Parameters
        ----------
        ax: Axes, optional
            The optional Axes to plot onto, by default None.
            If None then the current Axes will be used.
        """
        loops = self._generate_xz_plot_loops()
        plasma = self.inputs["plasma"]
        self._plotter.plot_ripple(loops, plasma, self.cage, ax=ax, **kwargs)

    def plot_ripple_contours(self, ax=None, **kwargs):
        """
        Plots the toroidal field ripple contours.

        Parameters
        ----------
        ax: Axes, optional
            The optional Axes to plot onto, by default None.
            If None then the current Axes will be used.
        """
        loops = self._generate_xz_plot_loops()
        plasma = self.inputs["plasma"]
        self._plotter.plot_ripple_contours(loops, plasma, self.cage, ax=ax, **kwargs)

    def plot_field_xz(self, ax=None, theta=0, n=3e3, **kwargs):
        """
        Parameters
        ----------
        ax: Axes, optional
            The optional Axes to plot onto, by default None.
            If None then the current Axes will be used.
        theta: float
            Angle of the rotated x-z plane in radians
        n: int
            The reference resolution at which to plot the contours of the variable

        Other Parameters
        ----------------
        xmin: float
            The minimum x value for the grid
        xmax: float
            The maximum x value for the grid
        zmin: float
            The minimum z value for the grid
        zmax: float
            The maximum z value for the grid
        """
        self._generate_xz_plot_loops()
        loops = [self.geom["TF WP"].inner, self.geom["TF WP"].outer]
        self._plotter.plot_field_xz(loops, self.cage, ax=ax, theta=theta, n=n, **kwargs)

    def plot_field_xy(self, ax=None, z=0, n=3e3, **kwargs):
        """
        Parameters
        ----------
        ax: Axes, optional
            The optional Axes to plot onto, by default None.
            If None then the current Axes will be used.
        z: float
            Height of the x-z plane in meters
        n: int
            The reference resolution at which to plot the contours of the variable

        Other Parameters
        ----------------
        xmin: float
            The minimum x value for the grid
        xmax: float
            The maximum x value for the grid
        ymin: float
            The minimum y value for the grid
        ymax: float
            The maximum y value for the grid
        """
        self._generate_xy_plot_loops()
        loops = [self.geom["WP inboard X-Y"]]
        self._plotter.plot_field_xy(loops, self.cage, ax=ax, z=z, n=n, **kwargs)

    @property
    def xz_plot_loop_names(self):
        """
        The x-z loop names to plot.
        """
        if self.conductivity in ["R"]:
            return [
                "TF WP",
                "TF Tapered CP",
                "TF Leg Conductor",
                "TF case in",
                "TF case out",
                "B Cyl",
            ]
        elif self.inputs["shape_type"] in ["CP"]:
            return [
                "TF WP",
                "TF Tapered CP",
                "TF Leg Conductor",
                "TF case in",
                "TF case out",
            ]

        return ["TF case in", "TF WP", "TF case out"]

    @property
    def xy_plot_loop_names(self):
        """
        The x-y loop names to plot.
        """
        if self.conductivity in ["R"]:

            return [
                "WP inboard X-Y",
                "WP outboard X-Y",
                "B Cyl X-Y",
                "Leg Conductor Casing X-Y",
            ]

        else:
            # SC Coils
            return [
                "Case inboard X-Y",
                "Case outboard X-Y",
                "WP inboard X-Y",
                "WP outboard X-Y",
            ]

    def _generate_xz_plot_loops(self):
        """
        Transfer from .l to .geom with associated plotting uniformity
        """
        wp_in = Loop(**self.loops["wp_in"])
        wp_in = clean_loop(wp_in)
        wp_in = simplify_loop(wp_in)
        wp_in.reorder(0, 2)
        wp_out = Loop(**self.loops["wp_out"])
        wp_out = clean_loop(wp_out)
        wp_out = simplify_loop(wp_out)
        wp_out.reorder(0, 2)

        case_out = Loop(**self.loops["out"])
        case_out = clean_loop(case_out)
        case_out.reorder(0, 2)

        case_in = Loop(**self.loops["in"])
        case_in = clean_loop(case_in)
        case_in.reorder(0, 2)
        if self.shape_type in ["P"] and self.params.r_tf_inboard_corner == 0:
            wp_out = self.correct_inboard_corners(wp_out, 4)
            case_out = self.correct_inboard_corners(case_out, 4)
            wp_in = self.correct_inboard_corners(wp_in, self.params.tk_tf_wp)
            case_in = self.correct_inboard_corners(case_in, self.params.tk_tf_wp)

        self.geom["TF WP"] = Shell(wp_in, wp_out)

        if self.inputs["shape_type"] in ["TP", "CP"]:
            # For tapered cp based coils
            (
                wp_in,
                wp_out,
                case_in,
                case_out,
            ) = self.generate_tapered_centrepost(wp_in, wp_out, case_in, case_out)

            wp_out = clean_loop(wp_out)
            wp_out = simplify_loop(wp_out)
            wp_in = clean_loop(wp_in)
            wp_in = simplify_loop(wp_in)
            self.geom["TF WP"] = Shell(wp_in, wp_out)

            case_out = clean_loop(case_out)
            case_out = simplify_loop(case_out)
            case_in = clean_loop(case_in)
            case_in = simplify_loop(case_in)
            (
                centrepost,
                leg_conductor,
                case_in,
                case_out,
            ) = self.split_centrepost_from_coil(wp_in, wp_out, case_in, case_out)

            if self.conductivity in ["R"]:

                # Also define Bucking Cylinder Loop
                b_cyl = Loop(**self.loops["b_cyl"])
                b_cyl = clean_loop(b_cyl)
                b_cyl = simplify_loop(b_cyl)
                b_cyl.reorder(0, 2)
                self.geom["B Cyl"] = b_cyl

                # Now write into relevant geom dicts
                self.geom["TF WP"] = Shell(wp_in, wp_out)
                self.geom["TF Tapered CP"] = centrepost
                self.geom["TF Leg Conductor"] = leg_conductor
                self.geom["TF case out"] = case_out
                self.geom["TF case in"] = case_in

            else:
                # SC coils
                # Now write into relevant geom dicts
                self.geom["TF WP"] = Shell(wp_in, wp_out)
                self.geom["TF Tapered CP"] = centrepost
                self.geom["TF Leg Conductor"] = leg_conductor
                self.geom["TF case out"] = Shell(wp_out, case_out)
                self.geom["TF case in"] = Shell(case_in, wp_in)

        else:
            self.geom["TF case out"] = Shell(wp_out, case_out)
            self.geom["TF case in"] = Shell(case_in, wp_in)

        return super()._generate_xz_plot_loops()

    def _generate_xy_plot_loops(self):
        """
        Generates X-Y Loops and Shells for X-Y plotting
        """
        a = 2 * np.pi / self.params.n_TF  # [rad] port wall angles
        beta = a / 2  # [rad] port wall half angle

        if self.conductivity in ["SC"]:
            # Non tapered CPs
            # ---------------------------------
            # TF Coil Winding Pack (wp):
            # ---------------------------------

            # [m] radius and Depth of the winding pack at the coil nose:
            r_wp_in = self.params.r_tf_in + self.params.tk_tf_nose
            tf_wp_nose_depth = 2 * (
                r_wp_in * np.tan(np.pi / self.params.n_TF) - self.params.tk_tf_side
            )
            # [m] depth of the winding pack at the plasma-facing end
            tf_inboard_wp_out_depth = self.params.tf_wp_depth
            # [m] TF casing side wall thickness
            tf_case_sw = self.section["case"]["side"]
            # [m] TF casing outboard plasma facing thickness
            tf_case_fwo = self.section["case"]["inboard"]
            # [m] TF casing outboard outer thickness
            tf_case_owo = self.section["case"]["external"]

            tf = self.loops
            tf_radii = [min(tf["out"]["x"]), min(tf["in"]["x"])]
            tf_r_inner_inb = min(tf["wp_out"]["x"])  # riwi
            tf_r_inner_outb = min(tf["wp_in"]["x"])  # riwo
            tf_r_outer_outb = max(tf["wp_out"]["x"])  # rowo

            # [m] radial position of the inboard leg, inner and outer faces
            tf_wp_r = [tf_r_inner_inb, tf_r_inner_outb]

            # Inner Leg loop
            x_shift = tf_case_sw / np.tan(a / 2)
            x_wp_i, y_wp_i = rainbow_seg(
                tf_r_inner_inb - x_shift,
                tf_r_inner_outb - x_shift,
                h=(x_shift, 0),
                angle=np.rad2deg(a),
                npoints=50,
            )
            # Also useful to define a rectangular x-y C/S for the wp, whether the
            # inner leg wp C/S is rectangular or not
            x_wp_i_rect = [tf_wp_r[0], tf_wp_r[1], tf_wp_r[1], tf_wp_r[0], tf_wp_r[0]]
            y_wp_i_rect = [
                tf_inboard_wp_out_depth / 2,
                tf_inboard_wp_out_depth / 2,
                -tf_inboard_wp_out_depth / 2,
                -tf_inboard_wp_out_depth / 2,
                tf_inboard_wp_out_depth / 2,
            ]

            # For rectangular C/S wp shapes, or for EUDEMO-type, 'nosed' wp shapes,
            # we can use the above block directly
            if self.wp_shape == "R" or self.wp_shape == "N":

                x_wp_i = x_wp_i_rect
                y_wp_i = [
                    tf_wp_nose_depth / 2,
                    tf_wp_nose_depth / 2,
                    -tf_wp_nose_depth / 2,
                    -tf_wp_nose_depth / 2,
                    tf_wp_nose_depth / 2,
                ]

            # Most outboard legs have rectangular C/S', so we can use rectangular section
            # defined earlier and shift it outboard
            x_wp_o = [i + (tf_r_outer_outb - tf_r_inner_outb) for i in x_wp_i_rect]
            y_wp_o = y_wp_i_rect

            self.geom["WP inboard X-Y"] = Loop(x=x_wp_i, y=y_wp_i)
            self.geom["WP outboard X-Y"] = Loop(x=x_wp_o, y=y_wp_o)

            # WP side walls
            wp_side_x = [tf_r_inner_inb, tf_r_inner_inb, x_wp_o[1], x_wp_o[2], x_wp_i[0]]
            wp_side_y = [
                tf_wp_nose_depth / 2,
                tf_wp_nose_depth / 2,
                y_wp_o[1],
                y_wp_o[2],
                tf_wp_nose_depth / 2,
            ]
            self.geom["WP sidewalls X-Y"] = Loop(x=wp_side_x, y=wp_side_y)

            # --------------------------
            # TF Coil Casing:
            # --------------------------

            # [m] total width of coil:
            # NOTE: different for wp_shape == 'N'!
            tf_width_tot = (
                2 * (self.section["case"]["side"])
                + self.section["winding_pack"]["depth"]
            )

            # [m] radial position of innermost face of casing
            x_c_inner_inb = self.params.r_tf_in

            # Curved inner leg loop
            x_tf_ci, y_tf_ci = rainbow_seg(
                x_c_inner_inb,
                tf_r_inner_outb + tf_case_fwo,
                h=(0, 0),
                angle=np.rad2deg(a),
                npoints=50,
            )

            # For EUDEMO-like 'nosed' coils, the casign shape is
            # different and must be redrawn
            if self.wp_shape == "N":
                x_b = tf_width_tot / 2 / np.tan(beta)
                # inboard casing with nose
                x_tf_ci = [
                    tf_radii[0] * np.cos(beta),
                    x_b,
                    tf_radii[1],
                    tf_radii[1],
                    x_b,
                    tf_radii[0] * np.cos(beta),
                    tf_radii[0] * np.cos(beta),
                ]
                y_tf_ci = [
                    tf_radii[0] * np.sin(beta),
                    tf_width_tot / 2,
                    tf_width_tot / 2,
                    -tf_width_tot / 2,
                    -tf_width_tot / 2,
                    -tf_radii[0] * np.sin(beta),
                    tf_radii[0] * np.sin(beta),
                ]

            case_in = Loop(x=x_tf_ci, y=y_tf_ci)
            case_in.close()
            self.geom["Case inboard X-Y"] = Shell(self.geom["WP inboard X-Y"], case_in)

            # Outboard casing loop built around WP outer leg (midplane)
            x_tf_co = [
                x_wp_o[0] - tf_case_fwo,
                x_wp_o[1] + tf_case_owo,
                x_wp_o[1] + tf_case_owo,
                x_wp_o[0] - tf_case_fwo,
                x_wp_o[0] - tf_case_fwo,
            ]
            y_tf_co = [i + np.sign(i) * tf_case_sw for i in y_wp_i_rect]
            self.geom["Case outboard X-Y"] = Shell(
                self.geom["WP outboard X-Y"], Loop(x=x_tf_co, y=y_tf_co)
            )

            # Casing side walls
            tf_sidex = [tf_radii[1], x_tf_co[2], x_tf_co[2], tf_radii[1], tf_radii[1]]
            tf_sidey = [
                tf_wp_nose_depth / 2,
                tf_wp_nose_depth / 2,
                -tf_wp_nose_depth / 2,
                -tf_wp_nose_depth / 2,
                tf_wp_nose_depth / 2,
            ]
            self.geom["Case sidewalls X-Y"] = Loop(x=tf_sidex, y=tf_sidey)

        elif self.conductivity in ["R"]:
            # TF geometry numbers
            tf_width = self.section["winding_pack"]["depth"]
            tk_case = self.params.tk_tf_ob_casing

            tf = self.loops
            tf_radii = [min(tf["out"]["x"]), min(tf["in"]["x"])]
            r_wp_inner_inb = min(tf["wp_out"]["x"])
            r_wp_inner_outb = min(tf["wp_in"]["x"])
            # radial thickness of outboard leg wp
            tk_wp_outb = self.params.tk_tf_outboard
            r_wp_outer_outb = max(tf["wp_out"]["x"])
            tf_wp_r = [r_wp_inner_inb, r_wp_inner_outb]

            # TF coil winding pack (includes insulation and insertion gap)
            # Rainbow seg for wp_i
            x_wp_i, y_wp_i = rainbow_seg(
                r_wp_inner_inb,
                r_wp_inner_outb,
                h=(0, 0),
                angle=np.rad2deg(a),
                npoints=50,
            )

            # make arcs for wp inboard side and for b_cyl using tf radii from above
            # stuff below makes a rectangle for inboard leg, but leave it as it's used
            # for outer leg
            x_wp_i_rect = [
                tf_wp_r[0],
                tf_wp_r[0] + tk_wp_outb,
                tf_wp_r[0] + tk_wp_outb,
                tf_wp_r[0],
                tf_wp_r[0],
            ]
            y_wp_i_rect = [
                tf_width / 2,
                tf_width / 2,
                -tf_width / 2,
                -tf_width / 2,
                tf_width / 2,
            ]
            x_wp_o = [i + (r_wp_outer_outb - r_wp_inner_outb) for i in x_wp_i_rect]

            # Bucking Cylinder loop
            b_cyl_ri = self.params.r_tf_in
            b_cyl_ro = b_cyl_ri + self.params.tk_tf_nose
            b_cyl = make_ring(b_cyl_ri, b_cyl_ro, angle=360, centre=(0, 0), npoints=200)
            self.geom["B Cyl X-Y"] = b_cyl

            # Leg Conductor Casing
            leg_casing = offset(x_wp_o, y_wp_i_rect, -tk_case)
            leg_casing = Loop(x=leg_casing[0], y=leg_casing[1])
            self.geom["WP inboard X-Y"] = Loop(x=x_wp_i, y=y_wp_i)
            self.geom["WP outboard X-Y"] = Loop(x=x_wp_o, y=y_wp_i_rect)
            self.geom["Leg Conductor Casing X-Y"] = Shell(
                self.geom["WP outboard X-Y"], leg_casing
            )
        betas = np.arange(0 + beta, 2 * np.pi + beta, 2 * beta)
        betas = np.rad2deg(betas)

        def pattern(p_loop, angles):
            """
            Patterns the GeomBase objects based on the number of sectors
            """
            loops = [
                p_loop.rotate(a, update=False, p1=[0, 0, 0], p2=[0, 0, 1])
                for a in angles
            ]
            if isinstance(p_loop, Loop):
                return MultiLoop(loops, stitch=False)
            elif isinstance(p_loop, Shell):
                return MultiShell(loops)
            else:
                raise TypeError("wtf")

        if self.conductivity in ["R"]:
            for key in [
                "WP inboard X-Y",
                "WP outboard X-Y",
                "Leg Conductor Casing X-Y",
            ]:

                loop = self.geom[key]
                self.geom[key + " single"] = loop
                self.geom[key] = pattern(loop, betas)

        else:
            for key in [
                "WP inboard X-Y",
                "WP outboard X-Y",
                "Case inboard X-Y",
                "Case outboard X-Y",
            ]:
                loop = self.geom[key]
                self.geom[key + " single"] = loop
                self.geom[key] = pattern(loop, betas)

        return super()._generate_xy_plot_loops()

    def loop_interpolators(self, trim=[0, 1], offset=0.75, full=False):
        """
        Outer loop coordinate interpolators
        """
        funcs = {"in": {}, "out": {}}
        # inner/outer loop offset
        for side, sign in zip(["in", "out", "cl"], [-1, 1, 0]):
            x, z = self.loops[side]["x"], self.loops[side]["z"]
            index = self.transition_index(x, z)
            x = x[index["lower"] + 1 : index["upper"]]
            z = z[index["lower"] + 1 : index["upper"]]
            x, z = offset_smc(x, z, sign * offset)
            if full:  # full loop (including nose)
                rmid, zmid = np.mean([x[0], x[-1]]), np.mean([z[0], z[-1]])
                x = np.append(rmid, x)
                x = np.append(x, rmid)
                z = np.append(zmid, z)
                z = np.append(z, zmid)
            le = lengthnorm(x, z)
            lt = np.linspace(trim[0], trim[1], int(np.diff(trim) * len(le)))
            x, z = interp1d(le, x)(lt), interp1d(le, z)(lt)
            le = np.linspace(0, 1, len(x))
            funcs[side] = {
                "x": InterpolatedUnivariateSpline(le, x),
                "z": InterpolatedUnivariateSpline(le, z),
            }
            funcs[side]["L"] = length(x, z)[-1]
            funcs[side]["dx"] = funcs[side]["x"].derivative()
            funcs[side]["dz"] = funcs[side]["z"].derivative()
        return funcs

    def correct_inboard_corners(
        self, loop, x_thick, tapered=False, xmin=None, zmax=None, zmin=None
    ):
        """
        Fix inboard corner to be 90 degrees

        Parameters
        ----------
        loop: Loop
            Loop to be corrected
        x_thick: float
            Radial thickness of correction

        Returns
        -------
        corrected_loop: Loop
            Loop with corrected corners
        """
        if xmin is None:
            xmin = np.min(loop.x)
        if zmax is None:
            zmax = np.max(loop.z)
        if zmin is None:
            zmin = np.min(loop.z)

        xmax = xmin + x_thick
        if tapered:
            zmin = self.params.h_cp_top
            corrector = make_box_xz(xmin, xmax, zmin, zmax)
        else:
            corrector = make_box_xz(xmin, xmax, zmin, zmax)

        corrected_loop = boolean_2d_union(loop, corrector)[0]
        if tapered:
            zmin2 = -zmax
            zmax = -zmin
            corrector_2 = make_box_xz(xmin, xmax, zmin2, zmax)
            corrected_loop = boolean_2d_union(corrected_loop, corrector_2)[0]

        return corrected_loop

    def generate_tapered_centrepost(self, wp_in, wp_out, case_in, case_out):
        """
        Generates tapered Centrepost. This involves correcting the wp and outer
        casing loops generating through offsetting to correctly
        reflect the tapered centrepost shape.

        Also corrects lack of sharp inboad edge corners produced by offsetting
        methods

        Returns
        -------
        wp_in: Loop
            Winding Pack inner (closer to plasma) x-z Loop
        wp_out: Loop
            Winding Pack outer (away from plasma) x-z Loop
        case_in: Loop
            Casing inner (closer to plasma) x-z Loop
        case_out: Loop
            Casing outer (away from plasma) x-z Loop

        """
        #  First define some useful quantities
        z_max = np.max(self.loops["out"]["z"])
        # TEMPORARY -  values for correcting corner chamfering in offset
        r_taper_out = np.min(self.loops["wp_in"]["x"])
        xmin = np.min(self.loops["wp_out"]["x"])
        tk_tapered_wp = r_taper_out - xmin

        # Must Correct the wp_out and out loops to have straight edges
        # Do this by either redrawing loops (TP) or cutting inboard edges
        # of loops (CP)

        # Remember - need to define outer radius of centrepost differently
        # for Resistive and SC coils. r_cp_top only exists for Resistive

        if self.conductivity in ["R"]:
            # Define useful quantities
            r_cp = self.params.r_cp_top
            tk_case_ob = self.params.tk_tf_ob_casing
            tk_case_ib = self.params.tk_tf_ob_casing
            r_wp_inb_in = self.params.r_tf_in + self.params.tk_tf_nose
            case_cutter_xmax = r_wp_inb_in - tk_case_ob
            xmin_case_in_corrector = r_cp + tk_case_ib
            xmin_wp_in_corrector = r_cp
            tapered = True
        else:
            r_cp = self.params.r_tf_in + self.params.tk_tf_inboard
            tk_case_ib = self.params.tk_tf_front_ib
            tk_case_ob = self.params.tk_tf_nose
            r_wp_inb_in = self.params.r_tf_in + self.params.tk_tf_nose
            case_cutter_xmax = self.params.r_tf_in
            xmin_case_in_corrector = r_cp
            xmin_wp_in_corrector = r_cp - tk_case_ib
            tapered = False

        # Define wp cutters
        wp_out_cutter = make_box_xz(-r_wp_inb_in * 5, r_wp_inb_in, -z_max, z_max)
        # Define case cutters
        case_out_cutter = make_box_xz(-r_wp_inb_in * 5, case_cutter_xmax, -z_max, z_max)

        # Use centrepost and casing definitions to correctly address
        # corner issues for casing
        # Cut wp_out and case_out loops
        wp_out = boolean_2d_difference_loop(wp_out, wp_out_cutter)
        case_out = boolean_2d_difference_loop(case_out, case_out_cutter)

        if self.shape_type in ["TP"]:

            wp_out = self.correct_inboard_corners(wp_out, 3)
            wp_in = self.correct_inboard_corners(wp_in, tk_tapered_wp, xmin=r_cp)
            case_out = self.correct_inboard_corners(case_out, 3)
            case_in = self.correct_inboard_corners(
                case_in, tk_tapered_wp, xmin=r_cp + tk_case_ib
            )

        elif self.shape_type in ["CP"]:
            # Need some special variables due to doming and x_curve
            # correct_l avoids notching issue, and gives a nice straight line till
            # x_curve start
            # TODO: Replace correct_l with joint location when variable available
            correct_l = self.shp.parameterisation.xo["x_curve_start"]["value"]
            zmax_abs = np.max(self.loops["out"]["z"])
            zmin_abs = np.min(self.loops["out"]["z"])
            xmax_abs = np.max(self.loops["out"]["x"])
            # Specifiy Zmax here is z_mid, not the max height of dome
            tapered_cp_in_temp = boolean_2d_difference_loop(
                wp_in, make_box_xz(correct_l - 0.25, xmax_abs, zmin_abs, zmax_abs)
            )
            zmax_in = np.max(tapered_cp_in_temp.z)
            zmin_in = np.min(tapered_cp_in_temp.z)
            zmax_out = zmax_in + self.section["case"]["WP"]
            zmin_out = zmin_in - self.section["case"]["WP"]
            wp_out = self.correct_inboard_corners(
                wp_out, correct_l, zmax=zmax_out, zmin=zmin_out
            )

            wp_in = self.correct_inboard_corners(
                wp_in,
                correct_l,
                tapered=tapered,
                xmin=xmin_wp_in_corrector,
                zmax=zmax_in,
                zmin=zmin_in,
            )
            # TODO: Find a more general variable for zmax_in when available
            zmax_in = np.max(tapered_cp_in_temp.z) - tk_case_ib
            zmax_out = zmax_in + self.section["case"]["WP"] + tk_case_ob + tk_case_ib
            zmin_in = zmin_in + tk_case_ib
            zmin_out = zmin_out - tk_case_ob

            case_out = self.correct_inboard_corners(
                case_out, correct_l, zmax=zmax_out, zmin=zmin_out
            )
            case_in = self.correct_inboard_corners(
                case_in,
                correct_l,
                tapered=tapered,
                xmin=xmin_case_in_corrector,
                zmax=zmax_in,
                zmin=zmin_in,
            )

        wp_out = clean_loop(wp_out)
        wp_out = simplify_loop(wp_out)
        wp_in = clean_loop(wp_in)
        wp_in = simplify_loop(wp_in)

        case_out = clean_loop(case_out)
        case_out = simplify_loop(case_out)
        case_in = clean_loop(case_in)
        case_in = simplify_loop(case_in)

        # Add corrected loops to their relevant entries
        self.loops["wp_out"]["x"] = wp_out.x
        self.loops["wp_out"]["z"] = wp_out.z
        self.loops["wp_in"]["x"] = wp_in.x
        self.loops["wp_in"]["z"] = wp_in.z
        self.loops["out"]["x"] = case_out.x
        self.loops["out"]["z"] = case_out.z
        self.loops["in"]["x"] = case_in.x
        self.loops["in"]["z"] = case_in.z

        return wp_in, wp_out, case_in, case_out

    def split_centrepost_from_coil(self, wp_in, wp_out, case_in, case_out):
        """
        Splits centrepost from the rest of coil

        Parameters
        ----------
        wp_in: Loop
            Winding Pack inner (closer to plasma) x-z Loop
        wp_out: Loop
            Winding Pack outer (away from plasma) x-z Loop
        case_in: Loop
            Casing inner (closer to plasma) x-z Loop
        case_out: Loop
            Casing outer (away from plasma) x-z Loop

        Returns
        -------
        centrepost: Loop
            Centrepost x-z Loop
        leg_conductor: Loop
            x-z Loop of remaining wp legs
        case_in: Loop
            Casing inner (closer to plasma) x-z Loop (just
            over leg_conductor for resistive coils)
        case_out: Loop
            Casing outer (away from plasma) x-z Loop (just
            over leg_conductor for resistive coils)
        """
        # Define useful quantities
        if self.conductivity in ["R"]:
            # Resistive Coils
            r_cp = self.params.r_cp_top
        else:
            # SC coils
            r_cp = self.params.r_tf_in + self.params.tk_tf_inboard

        z_max = np.max(np.abs(self.loops["out"]["z"]))
        x_max = np.max(np.abs(self.loops["out"]["x"]))
        TF_depth_at_r_cp = self.section["winding_pack"]["depth"]

        # Make a Loop to cut the outer section of the TF coil to
        # select the centrepost
        outer_cut_loop = make_box_xz(r_cp, x_max, -(z_max + 1), z_max + 1)

        # Cut the TF shell to get the CP loop
        tfcoil_loop = self.geom["TF WP"]
        centrepost = boolean_2d_difference_loop(tfcoil_loop, outer_cut_loop)
        centrepost = clean_loop(centrepost)
        centrepost = simplify_loop(centrepost)

        # Make a Loop the cut the inner section of the TF coil to
        # select the leg conductor
        inner_cut_loop = make_box_xz(0, r_cp, -(z_max + 1), z_max + 1)
        inner_cut_loop.close()
        leg_conductor = boolean_2d_difference_loop(tfcoil_loop, inner_cut_loop)
        leg_conductor = clean_loop(leg_conductor)
        leg_conductor = simplify_loop(leg_conductor)
        leg_conductor.close()

        if self.conductivity in ["R"]:

            # Resistive Coils
            tk_case_ob = self.params.tk_tf_ob_casing
            case_out = Shell(wp_out, case_out)
            case_in = Shell(case_in, wp_in)
            # Make Resistive coil casing:
            # The casing only covers the 'legs', i.e not the centrepost

            # Make CP cutter loop - remember that the casing cannot extend all the
            # way upto r_cp_top, as they will intersect with each other
            x_val = (
                1.03
                * (0.5 * TF_depth_at_r_cp + tk_case_ob)
                / np.tan(np.pi / self.params.n_TF)
            )
            inner_cut_loop = make_box_xz(-5 * x_val, x_val, -(z_max + 1), z_max + 1)

            # Cut the  casing Shells using inner cut loop
            case_out = boolean_2d_difference_loop(case_out, inner_cut_loop)
            case_out.close()
            case_out = clean_loop(case_out)
            case_out = simplify_loop(case_out)
            case_in = boolean_2d_difference_loop(case_in, inner_cut_loop)
            case_in.close()
            case_in = clean_loop(case_in)
            case_in = simplify_loop(case_in)

        return centrepost, leg_conductor, case_in, case_out

    def bucking_cylinder(self, p_in):
        """
        Draw Bucking Cylinder Loop
        """
        xmin = self.params.r_tf_in
        xmax = xmin + self.params.tk_tf_nose
        zmax = np.max(p_in["z"])
        if self.shape_type in ["CP"]:
            # CP is weird because max height of coil is the dome, not the b_cyl height
            correct_l = self.shp.parameterisation.xo["x_curve_start"]["value"]
            tapered_cp_temp = boolean_2d_difference_loop(
                Loop(**p_in), make_box_xz(correct_l - 0.25, 20, -25, 25)
            )
            zmax = np.max(tapered_cp_temp.z) + self.section["case"]["WP"]

        x = np.array([xmin, xmax, xmax, xmin, xmin])
        z = zmax * np.array([1, 1, -1, -1, 1])

        return x, z


class ToroidalFieldCoilsPlotter(ReactorSystemPlotter):
    """
    The plotter for Toroidal Field Coils.
    """

    def __init__(self):
        super().__init__()
        self._palette_key = "TF"

    def plot_xz(self, plot_objects, ax=None, **kwargs):
        """
        Plot the ToroidalFieldCoils in the x-z plane.
        """
        self._apply_default_styling(kwargs)
        colors = kwargs["facecolor"]
        colors.append(colors[0])
        kwargs["facecolor"] = colors
        super().plot_xz(plot_objects, ax=ax, **kwargs)

    def plot_xy(self, plot_objects, ax=None, **kwargs):
        """
        Plot the ToroidalFieldCoils in the x-y plane.
        """
        self._apply_default_styling(kwargs)
        colors = kwargs["facecolor"]
        colors = [colors[0], colors[0], colors[1], colors[1]]
        kwargs["facecolor"] = colors
        super().plot_xy(plot_objects, ax=ax, **kwargs)

    def plot_ripple(self, loops, plasma, cage, ax=None, **kwargs):
        """
        Plots the toroidal field ripple.

        Parameters
        ----------
        loops: List[Loop]
            A list of Loops to be plotted.
        plasma: systems.plasma.Plasma
            The Plasma to be plotted.
        cage: CoilCage
            The CoilCage to be plotted.
        ax: Axes, optional
            The optional Axes to plot onto, by default None.
            If None then the current Axes will be used.
        """
        self.plot_xz(loops, ax=ax, **kwargs)
        plasma.plot(ax=ax, fill=False, edgecolor="r")
        scalar_mappable = cage.plot_loops(ax=ax)

        color_bar = plt.gcf().colorbar(scalar_mappable)
        color_bar.ax.set_ylabel("Toroidal field ripple [%]")

    def plot_ripple_contours(self, loops, plasma, cage, ax=None, **kwargs):
        """
        Plots the toroidal field ripple contours.

        Parameters
        ----------
        loops: List[Loop]
            A list of Loops to be plotted.
        plasma: systems.plasma.Plasma
            The Plasma to be plotted.
        cage: CoilCage
            The CoilCage to be plotted.
        ax: Axes, optional
            The optional Axes to plot onto, by default None.
            If None then the current Axes will be used.
        """
        self.plot_xz(loops, ax=ax, **kwargs)
        plasma.plot(ax=ax, edgecolor="pink", facecolor="pink", alpha=0.3, **kwargs)
        cage.plot_contours_xz(variable="ripple", ax=ax)

    def plot_field_xy(self, loops, cage, ax=None, z=0, n=3e3, **kwargs):
        """
        Plot the total field contours in x-y due to the TF coils.

        Parameters
        ----------
        loops: List[Loop]
            A list of Loops to be plotted.
        cage: CoilCage
            The CoilCage to be plotted.
        ax: Axes, optional
            The optional Axes to plot onto, by default None.
            If None then the current Axes will be used.
        z: float
            Height of the x-z plane in meters
        n: int
            The reference resolution at which to plot the contours of the variable

        Other Parameters
        ----------------
        xmin: float
            The minimum x value for the grid
        xmax: float
            The maximum x value for the grid
        ymin: float
            The minimum y value for the grid
        ymax: float
            The maximum y value for the grid
        """
        self.plot_xy(loops, ax=ax, fill=False, linewidth=3)
        scalar_mappable = cage.plot_contours_xy(
            ax=None,
            z=z,
            n=n,
            **kwargs,
        )
        ax = plt.gca()
        color_bar = plt.gcf().colorbar(scalar_mappable)
        color_bar.ax.set_ylabel("$|B|$ [T]")
        ax.set_title(f"z = {z:.2f}")

    def plot_field_xz(self, loops, cage, ax=None, theta=0, n=3e3, **kwargs):
        """
        Plot the total field contours in x-z due to the TF coils.

        Parameters
        ----------
        loops: List[Loop]
            A list of Loops to be plotted.
        cage: CoilCage
            The CoilCage to be plotted.
        ax: Axes, optional
            The optional Axes to plot onto, by default None.
            If None then the current Axes will be used.
        theta: float
            Angle of the rotated x-z plane in radians
        n: int
            The reference resolution at which to plot the contours of the variable

        Other Parameters
        ----------------
        xmin: float
            The minimum x value for the grid
        xmax: float
            The maximum x value for the grid
        zmin: float
            The minimum z value for the grid
        zmax: float
            The maximum z value for the grid
        """
        self.plot_xz(loops, ax=ax, fill=False, linewidth=3)
        scalar_mappable = cage.plot_contours_xz(
            ax=None, variable="field", theta=theta, n=n
        )
        ax = plt.gca()
        color_bar = plt.gcf().colorbar(scalar_mappable)
        color_bar.ax.set_ylabel("$|B|$ [T]")
        ax.set_title(f"$\\theta$ = {np.rad2deg(theta):.1f}")


if __name__ == "__main__":
    from BLUEPRINT import test

    test(plotting=True)
