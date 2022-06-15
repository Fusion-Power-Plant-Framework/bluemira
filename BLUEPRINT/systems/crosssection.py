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
Reactor 2-D cross-section
"""
from collections import OrderedDict
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.optimize import minimize

from bluemira.base.parameter import ParameterFrame
from bluemira.geometry._deprecated_tools import innocent_smoothie
from BLUEPRINT.base.error import SystemsError
from BLUEPRINT.geometry.boolean import boolean_2d_difference, boolean_2d_union
from BLUEPRINT.geometry.geomtools import inloop, lengthnorm, normal, order, unique
from BLUEPRINT.geometry.loop import Loop, MultiLoop
from BLUEPRINT.geometry.offset import max_steps, offset_smc
from BLUEPRINT.geometry.parameterisations import PictureFrame, PolySpline
from BLUEPRINT.geometry.shape import Shape
from BLUEPRINT.geometry.shell import Shell
from BLUEPRINT.nova.firstwall import DivertorProfile, FirstWallProfile
from BLUEPRINT.systems.baseclass import ReactorSystem
from BLUEPRINT.systems.plotting import ReactorSystemPlotter


class ReactorCrossSection(ReactorSystem):
    """
    Reactor cross-section object used to calculate 2-D cross-sections from the
    radial build

    Parameters
    ----------
    sf: BLUEPRINT.nova StreamFlow object
        The reference equilibrium StreamFlow object to be used in the
        creation of the preliminary FW and DIV Profiles
    """

    config: Type[ParameterFrame]
    inputs: dict

    div_profile: Type[DivertorProfile]

    # fmt: off
    default_params = [
        ['Name', 'Reactor name', 'Cambridge', 'dimensionless', None, 'Input'],
        ['n_TF', 'Number of TF coils', 16, 'dimensionless', None, 'Input'],
        ['A', 'Plasma aspect ratio', 3.1, 'dimensionless', None, 'Input'],
        ['R_0', 'Major radius', 9, 'm', None, 'Input'],
        ["kappa_95", "95th percentile plasma elongation", 1.6, "dimensionless", None, "Input"],
        ['npoints', 'Number of points', 500, 'dimensionless', None, 'Default'],
        ['plasma_type', 'Type of plasma', 'SN', 'dimensionless', None, 'Input'],
        ['fw_psi_n', 'Normalised psi boundary to fit FW to', 1.07, 'dimensionless', None, 'Input'],
        ['fw_dx', 'Minimum distance of FW to separatrix', 0.225, 'm', None, 'Input'],
        ['div_L2D_ib', 'Inboard divertor leg length', 1, 'm', None, 'Input'],
        ['div_L2D_ob', 'Outboard divertor leg length', 1.36, 'm', None, 'Input'],
        ['div_graze_angle', 'Divertor SOL grazing angle', 1.5, '°', None, 'Input'],
        ['div_psi_o', 'Divertor flux offset', 0.75, 'm', None, 'Input'],
        ['div_Ltarg', 'Divertor target length', 0.5, 'm', None, 'Input'],
        ['div_open', 'Divertor open/closed configuration', False, 'dimensionless', None, 'Input'],
        ['tk_div', 'Divertor thickness', 0.5, 'm', None, 'Input'],
        ['c_rm', 'Remote maintenance clearance', 0.02, 'm', 'Distance between IVCs', None],
        ['r_vv_ib_in', 'Inboard vessel inner radius', 5.1, 'm', None, 'PROCESS'],
        ['r_vv_ob_in', 'Outboard vessel inner radius', 14.5, 'm', None, 'PROCESS'],
        ['tk_bb_ib', 'Inboard blanket thickness', 0.775312675901681, 'm', None, 'Input'],
        ['tk_bb_ob', 'Outboard blanket thickness', 1.300269571583071, 'm', None, 'Input'],
        ['tk_sh_in', 'Inboard shield thickness', 0.3, 'm', None, 'Input'],
        ['tk_sh_out', 'Outboard shield thickness', 0.3, 'm', None, 'Input'],
        ['tk_vv_in', 'Inboard vacuum vessel thickness', 0.3, 'm', None, 'Input'],
        ['tk_vv_out', 'Outboard vacuum vessel thickness', 0.8, 'm', None, 'Input'],
        ['tk_sol_ib', 'Inboard SOL thickness', 0.225, 'm', None, 'Input'],
        ['tk_sol_ob', 'Outboard SOL thickness', 0.225, 'm', None, 'Input'],
        ['tk_ts', 'TS thickness', 0.05, 'm', None, 'Input'],
        ['g_vv_ts', 'Gap between VV and TS', 0.05, 'm', None, 'Input'],
        ['g_cs_tf', 'Gap between CS and TF', 0.05, 'm', None, 'Input'],
        ['g_ts_tf', 'Gap between TS and TF', 0.05, 'm', None, 'Input'],
        ['g_vv_bb', 'Gap between VV and BB', 0.05, 'm', None, 'Input'],
        ['g_ts_pf', 'Clearances to PFs', 0.075, 'm', None, 'Input'],
        ['g_vv_div_add', 'Additional divertor/VV gap', 0, 'm', None, 'Input']
    ]
    # fmt: on

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs
        self._plotter = ReactorCrossSectionPlotter()

        self._init_params(self.config)

        self.configuration = self.params.Name + "_cross_section"
        self.sf = self.inputs["sf"]

        # Constructors
        self.targets = OrderedDict()  # targets structure (ordered)

        # FW coordinates (easy access)
        self.x_b = None
        self.z_b = None

        self.initialise_targets()

        self._generate_subsystem_classes(self.inputs)

    def initialise_targets(self):
        """
        Target datastructure for flux intercepting surfaces
        """
        self.targets["default"] = {}
        self.targets["inner"] = {}
        self.targets["outer"] = {}
        self.targets["default"]["L2D"] = self.params.div_L2D_ib
        self.targets["default"]["open"] = self.params.div_open
        self.targets["default"]["graze"] = self.params.div_graze_angle * np.pi / 180
        self.targets["default"]["dPlate"] = self.params.div_Ltarg
        self.targets["default"]["dR"] = 0
        self.targets["inner"]["L2D"] = self.params.div_L2D_ib
        self.targets["outer"]["L2D"] = self.params.div_L2D_ob

    def build(self, first_wall: Type[FirstWallProfile]):
        """
        Calculates the reactor crosssection of the in-vessel components

        Parameters
        ----------
        first_wall: FirstWallProfile
            The reactor FirstWallProfile object
        """
        # Make an initial Loop for the first wall (some treatments to help opt)
        x, z = first_wall.draw(npoints=500)
        istart = np.argmin((x - self.sf.x_point[0]) ** 2 + (z - self.sf.x_point[1]) ** 2)
        x = np.append(x[istart:], x[: istart + 1])
        z = np.append(z[istart:], z[: istart + 1])
        x, z = unique(x, z)[:2]
        self.geom["inner_loop"] = Loop(x=x, z=z)
        self.geom["initial_bb"] = varied_offset(
            self.geom["inner_loop"],
            [self.params.tk_bb_ib, self.params.tk_bb_ob],
            ref_o=3 / 10 * np.pi,
            dref=np.pi / 3,
        )

        # Ensure the initial blanket is a closed shell.
        self.geom["initial_bb"].inner.close()
        self.geom["initial_bb"].outer.close()

        to_dp = {
            "sf": self.sf,
            "targets": self.targets,
            "debug": False,
            "flux_conformal": False,
        }
        DivertorProfileClass = self.get_subsystem_class("div_profile")
        self.div_profile = DivertorProfileClass(self.params, to_dp)
        self.targets = self.div_profile.targets  # Update target struct

        if self.params.plasma_type == "DN":
            self._build_DN()
        elif self.params.plasma_type == "SN":
            self._build_SN()
        else:
            raise SystemsError(
                f"Plasma type {self.params.plasma_type} has not yet"
                "been parameterised and/or tested."
            )

        self.vessel_fill()
        self.x_b = self.geom["first_wall"]["x"]
        self.z_b = self.geom["first_wall"]["z"]

    def _build_SN(self):  # noqa :N802
        div_geom = self.div_profile.make_divertor(
            self.geom["inner_loop"], location="lower"
        )

        # Join the inner divertor Loop to the base FW inner loop
        self.geom["first_wall"] = boolean_2d_union(
            self.geom["inner_loop"], div_geom["divertor_inner"]
        )[0]

        # Cut the divertor and gap away from the initial blanket
        self.geom["blanket"] = boolean_2d_difference(
            self.geom["initial_bb"], div_geom["divertor_gap"]
        )[0]

        # Assign the divertor geometries dict to a sub-dict in geom
        self.geom["divertor"] = div_geom

        # Now we have to make an open Loop...
        inner = self.geom["inner_loop"].copy()
        div_koz = div_geom["divertor_gap"]
        # Let's find a point on the inner loop that is inside the divertor KOZ
        count = 0
        for i, point in enumerate(inner):
            if div_koz.point_inside(point):
                # Now we re-order the loop and open it, such that it is open
                # inside the KOZ
                if count > 0:
                    inner.reorder(i, 0)
                    inner.open_()
                    break
                count += 1  # (Second point inside the loop)

        # Now perform a boolean cut on an open loop
        self.geom["blanket_inner"] = boolean_2d_difference(inner, div_koz)[0]

        blanket_out = self.geom["initial_bb"].outer
        blanket_out.close()
        self.geom["blanket_outer"] = blanket_out
        self.geom["blanket_fw"] = boolean_2d_union(
            blanket_out, div_geom["divertor_gap"]
        )[0]

    def _build_DN(self):  # noqa :N802
        div_lower = self.div_profile.make_divertor(
            self.geom["inner_loop"], location="lower"
        )
        div_upper = self.div_profile.make_divertor(
            self.geom["inner_loop"], location="upper"
        )

        # Join the inner divertor Loop to the base FW inner loop
        first_wall = boolean_2d_union(
            self.geom["inner_loop"], div_lower["divertor_inner"]
        )[0]
        first_wall = boolean_2d_union(first_wall, div_upper["divertor_inner"])[0]
        self.geom["first_wall"] = first_wall

        # Cut the divertor and gap away from the initial blanket
        blanket = boolean_2d_difference(
            self.geom["initial_bb"], div_lower["divertor_gap"]
        )[0]
        blanket = boolean_2d_difference(blanket, div_upper["divertor_gap"])
        self.geom["blanket"] = MultiLoop(blanket)

        # Assign the divertor geometries dict to a sub-dict in geom
        self.geom["divertor"] = {"lower": div_lower, "upper": div_upper}

        # Now we have to make an open Loop...
        inner = self.geom["inner_loop"].copy()
        div_koz = div_lower["divertor_gap"]
        # Let's find a point on the inner loop that is inside the divertor KOZ
        count = 0
        for i, point in enumerate(inner):
            if div_koz.point_inside(point):
                # Now we re-order the loop and open it, such that it is open
                # inside the KOZ
                if count > 1:
                    inner.reorder(0, i)
                    inner.open_()
                    break
                count += 1  # (Second point inside the loop)

        # Now perform a boolean cut on an open loop
        blanket_inner = boolean_2d_difference(inner, div_koz)[0]

        # Now again for the upper divertor, and this time we will have 2 loops
        blanket_inner = boolean_2d_difference(blanket_inner, div_upper["divertor_gap"])[
            :2
        ]
        self.geom["blanket_inner"] = MultiLoop(blanket_inner)

        blanket_out = self.geom["initial_bb"].outer
        blanket_out.close()
        self.geom["blanket_outer"] = blanket_out
        blanket_fw = boolean_2d_union(blanket_out, div_lower["divertor_gap"])[0]
        blanket_fw = boolean_2d_union(blanket_fw, div_upper["divertor_gap"])[0]
        self.geom["blanket_fw"] = blanket_fw

    @property
    def xz_plot_loop_names(self):
        """
        The x-z loop names to plot.
        """
        loop_names = ["first_wall", "blanket", "vessel_shell", "lower_divertor"]
        if self.params.plasma_type == "DN":
            loop_names += ["upper_divertor"]
        return loop_names

    def _generate_xz_plot_loops(self):
        if self.params.plasma_type == "DN":
            self.geom["lower_divertor"] = self.geom["divertor"]["lower"]["divertor"]
            self.geom["upper_divertor"] = self.geom["divertor"]["upper"]["divertor"]
        else:
            self.geom["lower_divertor"] = self.geom["divertor"]["divertor"]
        return super()._generate_xz_plot_loops()

    def _set_vessel_bounds(
        self, vessel_shape, r_inboard, r_outboard, height, x_koz, z_koz
    ):
        if isinstance(vessel_shape.parameterisation, PolySpline):
            vessel_shape.adjust_xo(
                "x1", value=r_inboard, lb=r_inboard, ub=r_inboard + 0.001
            )  # inboard inner radius
            vessel_shape.adjust_xo(
                "x2", value=r_outboard, lb=r_outboard, ub=r_outboard * 1.1
            )  # outboard inner radius
            vessel_shape.adjust_xo(
                "z2", value=0, lb=-0.9, ub=0.9
            )  # outer node vertical shift
            vessel_shape.adjust_xo(
                "height", value=height, lb=height - 0.001, ub=50
            )  # full loop height
            vessel_shape.adjust_xo("top", value=0.5, lb=0.05, ub=1)  # horizontal shift
            vessel_shape.adjust_xo("upper", value=0.7, lb=0.2, ub=1)  # vertical shift
            vessel_shape.parameterisation.set_lower()
            vessel_shape.adjust_xo("dz", value=0, lb=-5, ub=5)  # vertical offset
            vessel_shape.adjust_xo(
                "flat", value=0, lb=0, ub=0.8
            )  # fraction outboard straight
            vessel_shape.adjust_xo(
                "tilt", value=0, lb=-45, ub=45
            )  # outboard angle [deg]
            vessel_shape.adjust_xo("upper", lb=0.6)
            vessel_shape.adjust_xo("lower", lb=0.6)
            vessel_shape.adjust_xo("l", lb=0.6)
        elif isinstance(vessel_shape.parameterisation, PictureFrame):
            # Drop the corner radius from the optimisation
            vessel_shape.remove_oppvar("r")

            # Set the inner radius of the shape, and pseudo-remove from optimiser
            vessel_shape.adjust_xo(
                "x1", value=r_inboard, lb=r_inboard - 0.001, ub=r_inboard + 0.001
            )

            vessel_shape.adjust_xo(
                "x2", value=r_outboard, lb=r_outboard - 0.001, ub=r_outboard + 0.001
            )

            half_height = height * 0.5

            # Adjust bounds to fit problem
            vessel_shape.adjust_xo(
                "z1", lb=half_height, value=half_height + 0.1, ub=half_height * 2.0
            )
            vessel_shape.adjust_xo(
                "z2", lb=-half_height * 2.0, value=-half_height - 0.1, ub=-half_height
            )

        vessel_shape.add_bound({"x": x_koz, "z": z_koz}, "internal")

    def vessel_fill(self, gap=True):
        """
        Optimise the vessel inner and outer shell geometry.
        """
        x_koz, z_koz = self.geom["blanket_fw"].offset(self.params.g_vv_bb).d2

        shp = Shape(
            self.configuration + "_vv",
            family=self.inputs["VV_parameterisation"],
            objective="L",
            npoints=400,
            read_write=False,
        )

        r_minor = self.params.R_0 / self.params.A
        height = (self.params.kappa_95 * r_minor) * 2

        ib_in_radius = self.params.r_vv_ib_in
        ob_in_radius = self.params.r_vv_ob_in

        self._set_vessel_bounds(shp, ib_in_radius, ob_in_radius, height, x_koz, z_koz)

        shp.optimise()

        x = shp.parameterisation.draw()
        xin, zin = x["x"], x["z"]

        tk_in = self.params.tk_vv_in + self.params.tk_sh_in
        tk_out = self.params.tk_vv_out + self.params.tk_sh_out

        ib_out_radius = ib_in_radius - tk_in
        ob_out_radius = ob_in_radius + tk_out

        loop = NLoop(xin, zin)
        x, z = loop.fill(
            dt=[tk_in, tk_out],
            ref_o=2 / 8 * np.pi,
            dref=np.pi / 6,
        )
        shp.clear_bound()
        shp.add_bound({"x": x, "z": z}, "internal")  # vessel outer bounds
        self._set_vessel_bounds(shp, ib_out_radius, ob_out_radius, height, x, z)
        shp.optimise()
        x = shp.parameterisation.draw()
        x, z = x["x"], x["z"]
        if self.params.plasma_type == "SX" or gap is True:
            vv = Wrap({"x": xin, "z": zin}, {"x": x, "z": z})
        else:
            vv = Wrap({"x": x_koz, "z": z_koz}, {"x": x, "z": z})
        vv.sort_z("inner", select=self.sf.xp_location)
        vv.sort_z("outer", select=self.sf.xp_location)

        self.geom["vessel_shell"] = Shell(Loop(x=xin, z=zin), Loop(x=x, z=z))
        self.geom["vessel"] = Loop(**vv.fill()[1])

    def get_sol(self, plot=False):
        """
        Get the scrape-off layer.
        """
        self.trim_sol(plot=plot)
        for leg in list(self.sf.legs)[2:]:
            l2d, l3d, x_sol, z_sol = self.sf.connection(leg, 0)
            x_o, z_o = x_sol[-1], z_sol[-1]
            l2d_edge, l3d_edge = self.sf.connection(leg, -1)[:2]
            if leg not in self.targets:
                self.targets[leg] = {}
            expansion = self.sf.expansion([x_o], [z_o])
            graze, theta = np.zeros(self.sf.n_sol), np.zeros(self.sf.n_sol)
            pannel = self.sf.legs[leg]["pannel"]
            for i in range(self.sf.n_sol):
                xo = self.sf.legs[leg]["X"][i][-1]
                zo = self.sf.legs[leg]["Z"][i][-1]
                graze[i] = self.sf.get_graze((xo, zo), pannel[i])
                theta[i] = self.sf.strike_point(expansion, graze[i])
            self.targets[leg]["graze_deg"] = graze * 180 / np.pi
            self.targets[leg]["theta_deg"] = theta * 180 / np.pi
            self.targets[leg]["L2Do"] = l2d
            self.targets[leg]["L3Do"] = l3d
            self.targets[leg]["L2Dedge"] = l2d_edge
            self.targets[leg]["L3Dedge"] = l3d_edge
            self.targets[leg]["Xo"] = x_o
            self.targets[leg]["Zo"] = z_o
            self.targets[leg]["Xsol"] = x_sol
            self.targets[leg]["Zsol"] = z_sol

    def trim_sol(self, color="k", plot=False):
        """
        Trim the scrape-off layer.
        """
        self.sf.sol()
        for c, leg in enumerate(self.sf.legs.keys()):
            if "core" not in leg:
                x_sol = self.sf.legs[leg]["X"]
                z_sol = self.sf.legs[leg]["Z"]
                self.sf.legs[leg]["pannel"] = [[] for _ in range(self.sf.n_sol)]
                for i in range(self.sf.n_sol):
                    if len(x_sol[i]) > 0:
                        x, z = x_sol[i], z_sol[i]
                        for j in range(2):  # predict - correct
                            x, z, pannel = self._trim(self.x_b, self.z_b, x, z)
                        self.sf.legs[leg]["X"][i] = x  # update sf
                        self.sf.legs[leg]["Z"][i] = z
                        self.sf.legs[leg]["pannel"][i] = pannel
                        if plot:
                            if color != "k" and i > 0:
                                plt.plot(x, z, "-", color=0.7 * np.ones(3))
                            elif color == "k":
                                plt.plot(x, z, "-", color="k", alpha=0.15)
                            else:
                                plt.plot(x, z, "--", color=[0.5, 0.5, 0.5])

    @staticmethod
    def _trim(x_loop, z_loop, x, z):
        def get_bstep(ss, dss, bb, x_loop_b, z_loop_b, j_b):
            d_b = np.array(
                [
                    x_loop_b[j_b[1]] - x_loop_b[j_b[0]],
                    z_loop_b[j_b[1]] - z_loop_b[j_b[0]],
                ]
            )
            b_step = np.cross(bb - ss, ds) / np.cross(dss, d_b)
            return b_step, d_b

        x_loop, z_loop = order(x_loop, z_loop)
        length_norm = lengthnorm(x, z)
        index = np.append(np.diff(length_norm) != 0, True)
        x, z = x[index], z[index]  # remove duplicates

        nx_loop, nz_loop = normal(x_loop, z_loop)
        x_in, z_in = np.array([]), np.array([])
        for x_i, z_i in zip(x, z):
            i = np.argmin((x_i - x_loop) ** 2 + (z_i - z_loop) ** 2)
            dx = [x_loop[i] - x_i, z_loop[i] - z_i]
            dn = [nx_loop[i], nz_loop[i]]
            if np.dot(dx, dn) > 0:
                x_in, z_in = np.append(x_in, x_i), np.append(z_in, z_i)
        i = np.argmin((x_in[-1] - x) ** 2 + (z_in[-1] - z) ** 2)
        # extend past target
        x_in, z_in = x[: i + 2], z[: i + 2]
        # sol crossing bndry
        i = np.argmin((x_in[-1] - x) ** 2 + (z_in[-1] - z) ** 2)
        jo = np.argmin((x[i] - x_loop) ** 2 + (z[i] - z_loop) ** 2)
        j = np.array([jo, jo + 1])
        s = np.array([x[i], z[i]])
        ds = np.array([x[i] - x[i - 1], z[i] - z[i - 1]])
        b = np.array([x_loop[j[0]], z_loop[j[0]]])
        bstep, db = get_bstep(s, ds, b, x_loop, z_loop, j)
        if bstep < 0:
            j = np.array([jo - 1, jo])  # switch target pannel
            bstep, db = get_bstep(s, ds, b, x_loop, z_loop, j)
        step = np.cross(b - s, db) / np.cross(ds, db)
        intersect = s + step * ds  # predict - correct
        if step < 0:  # step back
            x_in, z_in = x_in[:-1], z_in[:-1]
        x_in, z_in = np.append(x_in, intersect[0]), np.append(z_in, intersect[1])
        return x_in, z_in, db


class ReactorCrossSectionPlotter(ReactorSystemPlotter):
    """
    The plotter for Reactor Cross Sections.
    """

    def plot_xy(self, plot_objects, ax=None, **kwargs):
        """
        The x-y plot is not available for a ReactorCrossSection
        """
        raise NotImplementedError("ReactorCrossSection is only define in the x-z plane.")


def varied_offset(
    inner_loop,
    thicknesses: list,
    ref_o=4 / 8 * np.pi,
    dref=np.pi / 4,
):
    """
    Builds a varied offset shell from an inner loop

    Parameters
    ----------
    inner_loop: Loop
        The inner base loop from which to build the offset
    thicknesses: List[float]
        The list of [inner, outer] thickness offsets
    ref_o: float
        The reference blend range (check - can't remember)
    dref: float
        The reference blend point (check - can't remember)

    Returns
    -------
    shell: Shell
        The varied offset shell
    """
    gloop = NLoop(inner_loop.x, inner_loop.z)
    x, z = gloop.fill(dt=thicknesses, ref_o=ref_o, dref=dref)
    outer_loop = Loop(x=x, z=z)

    return Shell(inner_loop, outer_loop)


class Wrap:
    """
    Geometrical wrapping object.
    """

    def __init__(self, inner_points, outer_points):
        self.loops = OrderedDict()
        self.loops["inner"] = {"points": inner_points}
        self.loops["outer"] = {"points": outer_points}
        self.segment = None
        self.patch = None
        self.indx = None

    def _get_segment(self, loop):
        segment = self.loops[loop]["points"]
        return segment["x"], segment["z"]

    def _set_segment(self, loop, x, z):
        self.loops[loop]["points"] = {"x": x, "z": z}

    def sort_z(self, loop, select="lower"):
        """
        Sort the loop by z based on upper/lower.
        """
        x, z = self._get_segment(loop)
        x, z = order(x, z, anti=True)  # order points
        if select == "lower":
            imin = np.argmin(z)  # locate minimum
            x = np.append(x[imin:], x[:imin])  # sort
            z = np.append(z[imin:], z[:imin])
        else:
            imax = np.argmax(z)  # locate minimum
            x = np.append(x[imax:], x[:imax])  # sort
            z = np.append(z[imax:], z[:imax])
        self._set_segment(loop, x, z)

    def offset(self, loop, dt, **kwargs):
        """
        Offset the base loop.
        """
        x, z = self._get_segment(loop)
        gloop = NLoop(x, z)
        x, z = gloop.fill(dt=dt, **kwargs)
        self._set_segment(loop, x, z)

    def interpolate(self):
        """
        Interpolate the loops.
        """
        for loop in self.loops:
            x, z = self._get_segment(loop)
            x, z, le = unique(x, z)
            interpolant = {
                "x": InterpolatedUnivariateSpline(le, x),
                "z": InterpolatedUnivariateSpline(le, z),
            }
            self.loops[loop]["fun"] = interpolant

    def _interp(self, loop, l_points):

        interpolant = self.loops[loop]["fun"]
        return interpolant["x"](l_points), interpolant["z"](l_points)

    def _seed(self, dl, n_points=500):  # coarse search
        le = np.linspace(dl[0], dl[1], n_points)
        x, z = np.zeros((n_points, 2)), np.zeros((n_points, 2))
        for i, loop in enumerate(self.loops):
            x[:, i], z[:, i] = self._interp(loop, le)
        dx_min, i_in, i_out = np.max(x[:, 1]), 0, 0
        for i, (xin, zin) in enumerate(zip(x[:, 0], z[:, 0])):
            dx_all = np.sqrt((x[:, 1] - xin) ** 2 + (z[:, 1] - zin) ** 2)
            dx = np.min(dx_all)
            if dx < dx_min:
                dx_min = dx
                i_in = i
                i_out = np.argmin(dx_all)
        return le[i_in], le[i_out]

    def _cross(self, l_vector):
        x, z = np.zeros(2), np.zeros(2)
        for i, (loop, l) in enumerate(zip(self.loops, l_vector)):
            x[i], z[i] = self._interp(loop, l)
        err = (x[0] - x[1]) ** 2 + (z[0] - z[1]) ** 2
        return err

    def _index(self, loop, l_points):
        xp, zp = self._interp(loop, l_points)
        x, z = self._get_segment(loop)
        i = np.argmin((x - xp) ** 2 + (z - zp) ** 2)
        return i

    def close_loop(self):
        """
        Close the loops in the Wrap.
        """
        for loop in self.loops:
            x, z = self._get_segment(loop)
            if (x[0] - x[-1]) ** 2 + (z[0] - z[-1]) ** 2 != 0:
                x = np.append(x, x[0])
                z = np.append(z, z[0])
                self._set_segment(loop, x, z)

    @staticmethod
    def _concentric(xin, zin, xout, zout):
        points = inloop(xout, zout, xin, zin, side="out")
        if np.shape(points)[1] == 0:
            return True
        else:
            return False

    def fill(self):
        """
        Perform the optimisation of the Wrap.
        """
        xin, zin = self._get_segment("inner")
        xout, zout = self._get_segment("outer")
        concentric = self._concentric(xin, zin, xout, zout)
        if concentric:
            self.close_loop()
            xin, zin = self._get_segment("inner")
            xout, zout = self._get_segment("outer")
        self.interpolate()  # construct interpolators
        self.indx = {
            "inner": np.array([0, len(xin)], dtype=int),
            "outer": np.array([0, len(xout)], dtype=int),
        }
        if not concentric:
            self.indx = {
                "inner": np.zeros(2, dtype=int),
                "outer": np.zeros(2, dtype=int),
            }
            # low feild / high feild
            for i, dl in enumerate([[0, 0.5], [0.5, 1]]):
                lo = self._seed(dl)
                lengths = minimize(
                    self._cross, lo, method="L-BFGS-B", bounds=([0, 1], [0, 1])
                ).x
                for loop, l in zip(self.loops, lengths):
                    self.indx[loop][i] = self._index(loop, l)

        x = np.append(
            xout[self.indx["outer"][0] : self.indx["outer"][1]],
            xin[self.indx["inner"][0] : self.indx["inner"][1]][::-1],
        )
        z = np.append(
            zout[self.indx["outer"][0] : self.indx["outer"][1]],
            zin[self.indx["inner"][0] : self.indx["inner"][1]][::-1],
        )
        self.patch = {"x": x, "z": z}
        x = np.append(
            xin[: self.indx["inner"][0]],
            xout[self.indx["outer"][0] : self.indx["outer"][1]],
        )
        x = np.append(x, xin[self.indx["inner"][1] :])
        z = np.append(
            zin[: self.indx["inner"][0]],
            zout[self.indx["outer"][0] : self.indx["outer"][1]],
        )
        z = np.append(z, zin[self.indx["inner"][1] :])
        # Hanging chads are born here
        self.segment = {"x": x, "z": z}

        return self.segment, self.patch


class NLoop:
    """
    Utility NLoop for varied offsetting.
    """

    def __init__(self, x, z, **kwargs):
        self.x = x
        self.z = z

        self.xo = kwargs.get("xo", (np.mean(x), np.mean(z)))

    def fill(
        self,
        trim=None,
        dx=0,
        dt=0,
        ref_o=4 / 8 * np.pi,
        dref=np.pi / 4,
        color="k",
        label=None,
        alpha=0.8,
        referance="theta",
        part_fill=True,
        loop=False,
        s=0,
        gap=0,
    ):
        """
        Perform the varied offsetting operation.
        """
        dt_max = 0.1  # 2.5
        if not part_fill:
            dt_max = dt
        if isinstance(dt, list):
            dt = self._blend(dt, ref_o=ref_o, dref=dref, referance=referance, gap=gap)
        dt, nt = max_steps(dt, dt_max)
        x_in, z_in = offset_smc(self.x, self.z, dx)  # gap offset
        for i in range(nt):
            self._part_fill(
                trim=trim,
                dt=dt,
                color=color,
                label=label,
                alpha=alpha,
                loop=loop,
                s=s,
                plot=False,
            )
        return self.x, self.z

    def _part_fill(
        self,
        trim=None,
        dt=0,
        color="k",
        label=None,
        alpha=0.8,
        loop=False,
        s=0,
        plot=False,
    ):
        x_in, z_in = self.x, self.z
        if loop:
            n_append = 5
            x = np.append(self.x, self.x[:n_append])
            x = np.append(self.x[-n_append:], x)
            z = np.append(self.z, self.z[:n_append])
            z = np.append(self.z[-n_append:], z)
            x, z = innocent_smoothie(x, z, n=len(x), s=s)
            if isinstance(dt, (np.ndarray, list)):
                dt = np.append(dt, dt[:n_append])
                dt = np.append(dt[-n_append:], dt)
            x_out, z_out = offset_smc(x, z, dt)
            x_out, z_out = x_out[n_append:-n_append], z_out[n_append:-n_append]
            x_out[-1], z_out[-1] = x_out[0], z_out[0]
        else:
            x, z = innocent_smoothie(self.x, self.z, n=len(self.x), s=s)
            x_out, z_out = offset_smc(x, z, dt)
        self.x, self.z = x_out, z_out  # update
        if trim is None:
            l_index = [0, len(x_in)]
        else:
            l_index = self._trim(trim)
        if plot:
            flag = 0
            for i in np.arange(l_index[0], l_index[1] - 1):
                x_fill = np.array([x_in[i], x_out[i], x_out[i + 1], x_in[i + 1]])
                z_fill = np.array([z_in[i], z_out[i], z_out[i + 1], z_in[i + 1]])
                if flag == 0 and label is not None:
                    flag = 1
                    plt.fill(
                        x_fill,
                        z_fill,
                        facecolor=color,
                        alpha=alpha,
                        edgecolor="none",
                        label=label,
                    )
                else:
                    plt.fill(
                        x_fill, z_fill, facecolor=color, alpha=alpha, edgecolor="none"
                    )

    def _blend(self, dt, ref_o=4 / 8 * np.pi, dref=np.pi / 4, gap=0, referance="theta"):
        if referance == "theta":
            theta = np.arctan2(self.z - self.xo[1], self.x - self.xo[0]) - gap
            theta[theta > np.pi] = theta[theta > np.pi] - 2 * np.pi
            tblend = dt[0] * np.ones(len(theta))  # inner
            tblend[(theta > -ref_o) & (theta < ref_o)] = dt[1]  # outer
            if dref > 0:
                for updown in [-1, 1]:
                    blend_index = (updown * theta >= ref_o) & (
                        updown * theta < ref_o + dref
                    )
                    tblend[blend_index] = dt[1] + (dt[0] - dt[1]) / dref * (
                        updown * theta[blend_index] - ref_o
                    )
        else:
            length_norm = lengthnorm(self.x, self.z)
            tblend = dt[0] * np.ones(len(length_norm))  # start
            tblend[length_norm > ref_o] = dt[1]  # end
            if dref > 0:
                blend_index = (length_norm >= ref_o) & (length_norm < ref_o + dref)
                tblend[blend_index] = dt[0] + (dt[1] - dt[0]) / dref * (
                    length_norm[blend_index] - ref_o
                )
        return tblend

    @staticmethod
    def _trim(trim, x, z):
        length_norm = lengthnorm(x, z)
        index = []
        for t in trim:
            index.append(np.argmin(np.abs(length_norm - t)))
        return index
