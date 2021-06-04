# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
from BLUEPRINT.nova.coilcage import HelmholtzCage as CoilCage
from BLUEPRINT.base.constants import MU_0
from BLUEPRINT.base import ReactorSystem, ParameterFrame
from BLUEPRINT.base.lookandfeel import bpwarn
from BLUEPRINT.base.error import SystemsError
from BLUEPRINT.geometry.offset import offset_smc
from BLUEPRINT.geometry.boolean import clean_loop, simplify_loop
from BLUEPRINT.geometry.geomtools import length, lengthnorm, rainbow_seg
from BLUEPRINT.geometry.loop import Loop, MultiLoop, make_ring
from BLUEPRINT.geometry.shell import Shell, MultiShell
from BLUEPRINT.geometry.shape import Shape
from BLUEPRINT.cad.coilCAD import TFCoilCAD
from BLUEPRINT.systems.mixins import Meshable
from BLUEPRINT.systems.plotting import ReactorSystemPlotter
from BLUEPRINT.geometry.parameterisations import tapered_picture_frame


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
        ["r_tf_inboard_out", "Outboard Radius of the TF coil inboard leg tapered region", 0.6265, "m", None, "PROCESS"],
        ['tf_taper_frac', "Height of straight portion as fraction of total tapered section height", 0.5, 'N/A', None, 'Input'],
        ['r_tf_outboard_corner', "Corner Radius of TF coil outboard legs", 0.8, 'm', None, 'Input'],
    ]
    # fmt: on
    CADConstructor = TFCoilCAD

    def __init__(self, config, inputs):

        self.config = config
        self.inputs = inputs
        self._plotter = ToroidalFieldCoilsPlotter()

        self.params = ParameterFrame(self.default_params.to_records())
        self.params.update_kw_parameters(self.config)

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
            objective=self.inputs["obj"],
            npoints=self.npoints,
            n_TF=self.params.n_TF,
            read_write=True,
            read_directory=self.inputs["read_folder"],
            write_directory=self.inputs["write_folder"],
        )

        # The outer point of the TF coil inboard in the mid-plane
        r_tf_in_out_mid = (
            self.params.r_tf_in
            + self.params.tk_tf_nose
            + self.params.tk_tf_wp
            + self.params.tk_tf_front_ib
        )

        # The keep-out-zone at the mid-plane has to be scaled down from the keep-out-zone
        # at the maximum TF radius to avoid collisions on the inner leg.
        x_koz_min = np.min(self.inputs["koz_loop"].x) * np.cos(np.pi / self.params.n_TF)
        x_koz_max = np.max(self.inputs["koz_loop"].x)

        if x_koz_min < r_tf_in_out_mid:
            bpwarn(
                "TF coil radial build issue, resetting TF inboard outer edge in the "
                f"mid-plane from {r_tf_in_out_mid:.6f} to {x_koz_min:.6f}."
            )
            r_tf_in_out_mid = x_koz_min

        R_0 = self.params.R_0

        if self.inputs["shape_type"] == "S":
            # inner radius
            # NOTE: SLSQP doesn't like it when you remove x1 from the S
            # parameterisation...
            self.adjust_xo(
                "x1",
                lb=r_tf_in_out_mid - 0.001,
                value=r_tf_in_out_mid,
                ub=r_tf_in_out_mid + 0.001,
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
            self.adjust_xo("x1", value=r_tf_in_out_mid)
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
            self.adjust_xo("xo", value=r_tf_in_out_mid)
            self.shp.remove_oppvar("xo")
            self._minL, self._maxL = 0.2, 0.8

        elif self.inputs["shape_type"] == "P":
            # Drop the corner radius from the optimisation
            self.shp.remove_oppvar("r")

            # Set the inner radius of the shape, and pseudo-remove from optimiser
            self.adjust_xo(
                "x1",
                lb=r_tf_in_out_mid - 0.001,
                value=r_tf_in_out_mid,
                ub=r_tf_in_out_mid + 0.001,
            )

            # Adjust bounds to fit problem
            zmin = np.min(self.inputs["koz_loop"]["z"])
            zmax = np.max(self.inputs["koz_loop"]["z"])
            xmax = np.max(self.inputs["koz_loop"]["x"])
            self.adjust_xo("z1", lb=zmax, value=zmax + 0.1, ub=zmax + 2)
            self.adjust_xo("z2", lb=zmin - 2, value=zmin - 0.1, ub=zmin)
            self.adjust_xo("x2", lb=xmax, value=xmax + 0.5, ub=xmax + 3)

            # Adjust the range of points on the separatrix to check for ripple
            self._minL, self._maxL = 0.2, 0.8

        elif self.inputs["shape_type"] == "TP":

            # Set the inner radius of the shape, and pseudo-remove from optimiser
            self.adjust_xo(
                "x1", value=self.params.r_tf_inboard_out - self.params.tk_tf_wp / 2
            )
            self.adjust_xo("z2", value=self.params.h_cp_top)
            self.adjust_xo(
                "x2",
                value=0.5
                * (
                    self.params.r_cp_top
                    + self.params.r_tf_inboard_out
                    - self.params.tk_tf_wp
                ),
            )
            self.adjust_xo("r", value=self.params.r_tf_outboard_corner)
            self.adjust_xo("z1_frac", value=self.params.tf_taper_frac)
            self.shp.remove_oppvar("z2")
            self.shp.remove_oppvar("x1")
            self.shp.remove_oppvar("x2")
            self.shp.remove_oppvar("r")
            self.shp.remove_oppvar("z1_frac")

            # Adjust bounds to fit problem
            zmax = np.max(self.inputs["koz_loop"]["z"]) + self.params.tk_tf_outboard / 2
            xmax = np.max(self.inputs["koz_loop"]["x"]) + self.params.tk_tf_outboard / 2
            self.adjust_xo("z3", lb=zmax, value=zmax * 1.2, ub=zmax * 2)
            self.adjust_xo("x3", lb=xmax, value=xmax * 1.2, ub=xmax * 1.5)

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

        self.section["case"] = {
            "side": self.params.tk_tf_side,
            "nose": self.params.tk_tf_nose,
            "WP": self.params.tk_tf_wp,
            "inboard": self.params.tk_tf_front_ib,
            "outboard": iocasthk * 1.1,
            "external": iocasthk * 0.9,
        }
        if self.inputs["shape_type"] == "TP":
            iocasthk = 0
            self.section["case"] = {
                "side": 0,
                "nose": 0,
                "WP": self.params.tk_tf_outboard,
                "inboard": 0,
                "outboard": 0,
                "external": 0,
            }

        bm = -self.params.B_0 * self.params.R_0
        current = abs(2 * np.pi * bm / (self.params.n_TF * MU_0))
        self.add_parameter("I_tf", "TF coil current", current, "MA", None, "BLUEPRINT")
        rwpin = self.params.r_tf_in + self.params.tk_tf_nose

        # Rem : Not the same definition of WP depth is used between shapes !
        depth = 2 * (rwpin * np.tan(np.pi / self.params.n_TF) - self.params.tk_tf_side)
        if self.inputs["shape_type"] == "TP":
            depth = 2 * self.params.r_cp_top * np.sin(np.pi / self.params.n_TF)
            self.section["winding_pack"] = {
                "width": self.params.tk_tf_outboard,
                "depth": depth,
            }
            self.rc = self.params.tk_tf_outboard / 2
            # Update cross-sectional parameters
            self.params.update_kw_parameters(
                {
                    "tk_tf_case_out_in": iocasthk * 0.9,
                    "tk_tf_case_out_out": iocasthk * 1.1,
                    "tf_wp_width": self.params.tk_tf_outboard,
                    "tf_wp_depth": depth,
                }
            )

        else:
            self.section["winding_pack"] = {
                "width": self.params.tk_tf_wp,
                "depth": depth,
            }
            self.rc = (self.params.tk_tf_wp + depth) / 4 / self.nr

            # Update cross-sectional parameters
            self.params.update_kw_parameters(
                {
                    "tk_tf_case_out_in": iocasthk * 0.9,
                    "tk_tf_case_out_out": iocasthk * 1.1,
                    "tf_wp_width": self.params.tk_tf_wp,
                    "tf_wp_depth": depth,
                }
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

        if self.inputs["shape_type"] == "TP":
            loops = ["wp_in", "cl", "wp_out", "out", "b_cyl"]
            inboard_dt = [
                case["inboard"],
                wp["width"] / 2,
                wp["width"] / 2,
                case["nose"],
                0.00,
            ]
            outboard_dt = [
                case["outboard"],
                wp["width"] / 2,
                wp["width"] / 2,
                case["external"],
                0.00,
            ]

        self.loops["in"]["x"], self.loops["in"]["z"] = x, z
        index = self.transition_index(self.loops["in"]["x"], self.loops["in"]["z"])

        for loop, dt_in, dt_out in zip(loops, inboard_dt, outboard_dt):
            dt = self._loop_dt(x, z, dt_in, dt_out, index)
            x, z = offset_smc(x, z, dt, close_loop=True)
            # TODO check that unwind does not effect stored energy calculation
            # self.p[loop] = unwind({'x': x, 'z': z})
            self.loops[loop]["x"], self.loops[loop]["z"] = x, z

            if self.inputs["shape_type"] == "TP":

                if loop == "wp_in":
                    # Make the loop for the varying thickness WP for tapered Pictureframe
                    x1 = self.params.r_tf_inboard_out
                    x2 = self.params.r_cp_top
                    x3 = (
                        self.shp.parameterisation.xo["x3"]["value"]
                        - self.params.tk_tf_outboard / 2
                    )
                    z1_frac = self.shp.parameterisation.xo["z1_frac"]["value"]
                    z3 = (
                        self.shp.parameterisation.xo["z3"]["value"]
                        - self.params.tk_tf_outboard / 2
                    )
                    z2 = self.params.h_cp_top
                    r = (
                        self.shp.parameterisation.xo["r"]["value"]
                        - self.params.tk_tf_outboard / 2
                    )
                    xx, zz = tapered_picture_frame(
                        x1, x2, x3, z1_frac, z2, z3, r, npoints=len(p_in["x"])
                    )
                    self.loops[loop]["x"], self.loops[loop]["z"] = xx, zz

                if loop == "wp_out":
                    # Make the loop for the varying thickness WP for tapered Pictureframe
                    xmin = self.params.r_tf_in + self.params.tk_tf_nose
                    xmax = np.max(p_in["x"]) + (self.params.tk_tf_outboard / 2)
                    zmax = np.max(p_in["z"]) + (self.params.tk_tf_outboard / 2)

                    corner_rad = self.shp.parameterisation.xo["r"]["value"] + (
                        self.params.tk_tf_outboard / 2
                    )

                    xx, zz = tapered_picture_frame(
                        xmin,
                        xmin,
                        xmax,
                        0,
                        1.0,
                        zmax,
                        corner_rad,
                        npoints=len(p_in["x"]),
                    )

                    self.loops[loop]["x"], self.loops[loop]["z"] = xx, zz

                if loop == "b_cyl":

                    x, z = self.bucking_cylinder(p_in)
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
            self.params.n_TF,
            self.params.R_0,
            z_0,
            self.params.B_0,
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
        if self.inputs["shape_type"] == "TP":
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
            bpwarn(
                "TF coil ripple exceeds optimiser specification: "
                f"{self.max_ripple:.3f} %."
            )
        elif self.max_ripple < (1 - tol) * self.ripple_limit:
            bpwarn("TF coil ripple is supra-optimised: " f"{self.max_ripple:.3f} %.")

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
        ripple, ripple_limit = args
        # de-normalize
        if ripple:  # constrain ripple contour
            xloop = self.update_loop(xnorm, *args)
            constraint = np.array([])
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
        if self.inputs["shape_type"] == "TP":
            return ["TF WP", "B Cyl"]

        return ["TF case in", "TF WP", "TF case out"]

    @property
    def xy_plot_loop_names(self):
        """
        The x-y loop names to plot.
        """
        if self.inputs["shape_type"] == "TP":

            return [
                "WP inboard X-Y",
                "WP outboard X-Y",
                "B Cyl X-Y",
            ]

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
        wp_in.reorder(0, 2)
        wp_out = Loop(**self.loops["wp_out"])
        wp_out = clean_loop(wp_out)
        wp_out.reorder(0, 2)
        self.geom["TF WP"] = Shell(wp_in, wp_out)

        if self.inputs["shape_type"] == "TP":
            b_cyl = Loop(**self.loops["b_cyl"])
            b_cyl = clean_loop(b_cyl)
            b_cyl.reorder(0, 2)
            self.geom["B Cyl"] = b_cyl
            return super()._generate_xz_plot_loops()

        case_out = Loop(**self.loops["out"])
        case_out = clean_loop(case_out)
        case_out.reorder(0, 2)
        self.geom["TF case out"] = Shell(wp_out, case_out)

        case_in = Loop(**self.loops["in"])
        case_in = clean_loop(case_in)
        case_in.reorder(0, 2)
        self.geom["TF case in"] = Shell(case_in, wp_in)
        return super()._generate_xz_plot_loops()

    def _generate_xy_plot_loops(self):
        """
        Generates X-Y Loops and Shells for X-Y plotting
        """
        a = 2 * np.pi / self.params.n_TF  # [rad] port wall angles
        beta = a / 2  # [rad] port wall half angle
        if self.inputs["shape_type"] != "TP":
            # TF geometry numbers
            tf_width = (
                2 * (self.section["case"]["side"])
                + self.section["winding_pack"]["depth"]
            )
            # [m] TF casing side wall thickness
            tf_c_sw = self.section["case"]["side"]
            # [m] TF casing outboard plasma facing thickness
            tf_c_fwo = self.section["case"]["inboard"]
            # [m] TF casing outboard outer thickness
            tf_c_owo = self.section["case"]["external"]
            tf = self.loops
            tf_radii = [min(tf["out"]["x"]), min(tf["in"]["x"])]
            tf_riwi = min(tf["wp_out"]["x"])
            tf_riwo = min(tf["wp_in"]["x"])
            tf_rowo = max(tf["wp_out"]["x"])
            tf_wp_r = [tf_riwi, tf_riwo]

            # TF coil winding pack (includes insulation and insertion gap)
            x_wp_i = [tf_wp_r[0], tf_wp_r[1], tf_wp_r[1], tf_wp_r[0], tf_wp_r[0]]
            y_wp_i = [
                tf_width / 2 - tf_c_sw,
                tf_width / 2 - tf_c_sw,
                -tf_width / 2 + tf_c_sw,
                -tf_width / 2 + tf_c_sw,
                tf_width / 2 - tf_c_sw,
            ]
            x_wp_o = [i + (tf_rowo - tf_riwo) for i in x_wp_i]
            self.geom["WP inboard X-Y"] = Loop(x=x_wp_i, y=y_wp_i)
            self.geom["WP outboard X-Y"] = Loop(x=x_wp_o, y=y_wp_i)
            # WP side walls
            wp_side_x = [x_wp_i[0], x_wp_i[0], x_wp_o[1], x_wp_o[2], x_wp_i[0]]
            wp_side_y = [y_wp_i[0], y_wp_i[3], y_wp_i[2], y_wp_i[1], y_wp_i[0]]
            self.geom["WP sidewalls X-Y"] = Loop(x=wp_side_x, y=wp_side_y)
            # TF casing
            x_b = tf_width / 2 / np.tan(beta)
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
                tf_width / 2,
                tf_width / 2,
                -tf_width / 2,
                -tf_width / 2,
                -tf_radii[0] * np.sin(beta),
                tf_radii[0] * np.sin(beta),
            ]
            self.geom["Case inboard X-Y"] = Shell(
                self.geom["WP inboard X-Y"], Loop(x=x_tf_ci, y=y_tf_ci)
            )
            # outboard casing built around WP outer leg (midplane)
            x_tf_co = [
                x_wp_o[0] - tf_c_fwo,
                x_wp_o[1] + tf_c_owo,
                x_wp_o[1] + tf_c_owo,
                x_wp_o[0] - tf_c_fwo,
                x_wp_o[0] - tf_c_fwo,
            ]
            y_tf_co = [i + np.sign(i) * tf_c_sw for i in y_wp_i]
            self.geom["Case outboard X-Y"] = Shell(
                self.geom["WP outboard X-Y"], Loop(x=x_tf_co, y=y_tf_co)
            )
            # Casing side walls
            tf_sidex = [tf_radii[1], x_tf_co[2], x_tf_co[2], tf_radii[1], tf_radii[1]]
            tf_sidey = [y_tf_ci[2], y_tf_ci[2], -y_tf_ci[2], -y_tf_ci[2], y_tf_ci[2]]
            self.geom["Case sidewalls X-Y"] = Loop(x=tf_sidex, y=tf_sidey)

        else:
            # TF geometry numbers
            tf_width = self.section["winding_pack"]["depth"]

            tf = self.loops
            tf_radii = [min(tf["out"]["x"]), min(tf["in"]["x"])]
            tf_riwi = min(tf["wp_out"]["x"])
            tf_riwo = min(tf["wp_in"]["x"])
            tf_rowo = max(tf["wp_out"]["x"])
            tf_wp_r = [tf_riwi, tf_riwo]

            # TF coil winding pack (includes insulation and insertion gap)
            # Rainbow seg for wp_i
            wp_i_sec_x, wp_i_sec_y = rainbow_seg(
                tf_riwi, tf_riwo, h=(0, 0), angle=np.rad2deg(a), npoints=50
            )

            # make arcs for wp inboard side and for b_cyl using tf radii from above
            # stuff below makes a rectangle for inboard leg, but leave it as it's used
            # for outer leg too
            x_wp_i = [tf_wp_r[0], tf_wp_r[1], tf_wp_r[1], tf_wp_r[0], tf_wp_r[0]]
            y_wp_i = [
                tf_width / 2,
                tf_width / 2,
                -tf_width / 2,
                -tf_width / 2,
                tf_width / 2,
            ]
            x_wp_o = [i + (tf_rowo - tf_riwo) for i in x_wp_i]

            # Bucking Cylinder loop
            b_cyl_ri = self.params.r_tf_in
            b_cyl_ro = b_cyl_ri + self.params.tk_tf_nose
            b_cyl = make_ring(b_cyl_ri, b_cyl_ro, angle=360, centre=(0, 0), npoints=200)

            self.geom["WP inboard X-Y"] = Loop(x=wp_i_sec_x, y=wp_i_sec_y)
            self.geom["WP outboard X-Y"] = Loop(x=x_wp_o, y=y_wp_i)
            self.geom["B Cyl X-Y"] = b_cyl

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

        if self.inputs["shape_type"] == "TP":
            for key in [
                "WP inboard X-Y",
                "WP outboard X-Y",
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

    def bucking_cylinder(self, p_in):
        """
        Draw Bucking Cylinder Loop
        """
        xmin = self.params.r_tf_in
        xmax = xmin + self.params.tk_tf_nose
        zmax = np.max(p_in["z"]) + (self.params.tk_tf_outboard / 2)
        zmin = -zmax
        npoints = len(p_in["x"])
        x = np.zeros(npoints)
        z = np.zeros(npoints)
        x[0 : int(npoints / 8)] = xmin
        x[int(npoints / 8) : int(3 * npoints / 8)] = np.linspace(
            xmin, xmax, int(npoints / 4)
        )
        x[int(3 * npoints / 8) : int(5 * npoints / 8)] = xmax
        x[int(5 * npoints / 8) : int(7 * npoints / 8)] = np.linspace(
            xmax, xmin, int(npoints / 4)
        )
        x[int(7 * npoints / 8) : int(npoints)] = xmin

        z[0 : int(npoints / 8)] = np.linspace(0, zmax, int(npoints / 8))
        z[int(npoints / 8) : int(3 * npoints / 8)] = zmax
        z[int(3 * npoints / 8) : int(5 * npoints / 8)] = np.linspace(
            zmax, zmin, int(npoints / 4)
        )
        z[int(5 * npoints / 8) : int(7 * npoints / 8)] = zmin
        z[int(7 * npoints / 8) : int(npoints)] = np.linspace(
            zmin, 0, int(npoints / 8), endpoint=True
        )

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
