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
Flux surface attributes and first wall profile based on heat flux calculation
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Type

from bluemira.base.parameter import ParameterFrame

from BLUEPRINT.base.baseclass import ReactorSystem
from BLUEPRINT.base.error import SystemsError, GeometryError
from BLUEPRINT.cad.firstwallCAD import FirstWallCAD
from bluemira.equilibria.find import find_flux_surfs
from BLUEPRINT.geometry.loop import Loop, MultiLoop
from BLUEPRINT.geometry.shell import Shell
from BLUEPRINT.geometry.boolean import (
    convex_hull,
    boolean_2d_union,
    boolean_2d_difference,
    boolean_2d_difference_loop,
    boolean_2d_difference_split,
    boolean_2d_common_loop,
    simplify_loop,
)
from BLUEPRINT.geometry.geomtools import (
    get_intersect,
    check_linesegment,
    loop_plane_intersect,
    index_of_point_on_loop,
    make_box_xz,
)
from BLUEPRINT.geometry.geombase import make_plane
from BLUEPRINT.geometry.geomtools import rotate_vector_2d
from functools import partial
from BLUEPRINT.systems.plotting import ReactorSystemPlotter
from BLUEPRINT.utilities.csv_writer import write_csv


class EqInputs:
    """
    Class to extract equilibrium parameters relevant to the first wall

    Parameters
    ----------
    lcfs_shift: float
        Sometimes the separatrix is not well defined.
        This parameter take an open flux surface, close to the lcfs
        to replace the separatrix
    x_point_shift:
        A shift to slightly move away from the x-point and avoid singularities

    Attributes
    ----------
    self.lcfs_shift: float
        Save input parameter used to set separatrix
    self.x_point_shift: float
        Save input parameter used to set separatrix
    self.lcfs: Loop
        Contour corresponding to last closed flux surface
    self.points: dict
        Store the x,z coordinates of the O and X points in the flux field
    self.mid_plane: Plane
        A plane having z-normal and containing the O point
    self.x_omp_lcfs: float
        Outer x-coordinate of the intersection between the last closed flux
        surface and the mid-plane
    self.x_imp_lcfs: float
        Inner x-coordinate of the intersection between the last closed flux
        surface and the mid-plane
    self.sep: separatrix: Union[Loop, MultiLoop]
        The separatrix loop(s) (Loop for SN, MultiLoop for DN)

    Notes
    -----
    According to the refinement of the eqdsk file, the separatrix
    is extrapolated differently.
    For the SN, there are apparently no issues and the separatrix always
    comes as a closed loop that contains the main plasma and an
    open loop for the two legs. The shared point is the x-point.
    For the DN, if the separatrix is extrapolated correctly, it is a
    Multiloop with two open loops that share two points: upper and lower
    x-point. The two loops are the half separatrix at the lfs and
    half separatrix at the hfs.
    If the separatrix is not extrapolated correctly, this turns out to be
    an upside down "SN like separatrix". Thus, only the upper legs exist.
    """

    def __init__(self, lcfs_shift=0.001, x_point_shift=0.1):

        # Save inputs
        self.lcfs_shift = lcfs_shift
        self.x_point_shift = x_point_shift

        # First find the last closed flux surface
        self.lcfs = self.equilibrium.get_LCFS()

        # Find the local maxima (O) and inflection (X) points in the flux field
        o_point, x_point = self.equilibrium.get_OX_points()
        self.points = {
            "x_point": {
                "x": x_point[0][0],
                "z_low": x_point[0][1],
                "z_up": x_point[1][1],
            },
            "o_point": {"x": o_point[0][0], "z": round(o_point[0][1], 5)},
        }
        if self.points["x_point"]["z_low"] > self.points["x_point"]["z_up"]:
            self.points["x_point"]["z_low"] = x_point[1][1]
            self.points["x_point"]["z_up"] = x_point[0][1]

        # Define the mid-plane as having z-normal and containing O point.
        self.mid_plane = make_plane(self.points["o_point"]["z"], 2)

        # Find the intersection between the mid-plane and the last closed
        # flux surface
        mp_ints = loop_plane_intersect(self.lcfs, self.mid_plane)

        # Find the outer and inner mid-plane intersections
        self.x_omp_lcfs = find_outer_point(mp_ints, self.points["o_point"]["x"])[0]
        self.x_imp_lcfs = find_inner_point(mp_ints, self.points["o_point"]["x"])[0]

        # Here we check if it is a DN. If it is a DN, we have two mirrored x-points.
        if round(self.points["x_point"]["z_low"], 3) == -round(
            self.points["x_point"]["z_up"], 3
        ):
            # Here we pick the separatrix (right or wrong).
            self.sep = self.equilibrium.get_separatrix()

            # We check the separatrix against the lower x-point and we move
            # away from it to avoid a singularity.
            x_point_limit = self.points["x_point"]["z_low"] - self.x_point_shift

            # Here we take the lowest point contained in the separatrix loop.
            sep_limit = min(self.sep[0].z)

            # If we find a point, below the fixed limit, it means that the lower
            # legs are present, and the extrapolated sepatrix is correct.
            if sep_limit < x_point_limit:
                self.separatrix = self.sep

            # If we do not find a point below the fixed limit, it means that the
            # lower legs are not present. The extrapolated separatrix is not correct.
            else:

                # We need to "make" a separatrix.
                # As separatrix we take the first open flux surface (Ideally,
                # the first flux surfaces outside the LCFS).
                loops = self.equilibrium.get_flux_surface_through_point(
                    self.x_omp_lcfs + self.lcfs_shift, 0
                )

                # We re-order the loops as conventionally is done for the
                # separatrix and we make it a MultiLoop, as conventionally is
                # done for the separatrix.
                loops.reverse()
                self.separatrix = MultiLoop(loops[:2])

        # Here we check if it is a SN. If it is a SN, we do have more than one
        # x-points, but they are not mirrored.
        elif round(self.points["x_point"]["z_low"], 3) != -round(
            self.points["x_point"]["z_up"], 3
        ):
            # As the SN does not give concerns, we just have to pick
            # the separatrix.
            self.separatrix = [self.equilibrium.get_separatrix()]


class FirstWall(EqInputs, ReactorSystem):
    """
    Reactor First Wall (FW) system abstract base class
    """

    config: Type[ParameterFrame]
    inputs: dict
    CADConstructor = FirstWallCAD

    # fmt: off
    base_default_params = [
        ["n_TF", "Number of TF coils", 16, "N/A", None, "Input"],
        ["A", "Plasma aspect ratio", 3.1, "N/A", None, "Input"],
        ["psi_norm", "Normalised flux value of strike-point contours",
         1, "N/A", None, "Input"],
        ['tk_fw_in', 'Inboard first wall thickness', 0.052, 'm', None, 'Input'],
        ['tk_fw_out', 'Outboard first wall thickness', 0.052, 'm', None, 'Input'],
        ['tk_fw_div', 'First wall thickness around divertor', 0.052, 'm', None, 'Input'],
        ['tk_div_cass', 'Minimum thickness between inner divertor profile and cassette', 0.3, 'm', None, 'Input'],
        ['tk_div_cass_in', 'Additional radial thickness on inboard side relative to to inner strike point', 0.1, 'm', None, 'Input'],
    ]
    # fmt: on

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs
        self.init_params()
        self.init_equilibrium()
        self.build()
        self._plotter = FirstWallPlotter()

    def init_params(self):
        """
        Initialise First Wall parameters from config.
        """
        self._init_params(self.config)

    def init_equilibrium(self):
        """
        Initialise equilibrium inputs.
        """

        EqInputs.__init__(self)

    def optimise_fw_profile(self, hf_limit=0.2, n_iteration_max=5):
        """
        Optimises the initial preliminary profile in terms of heat flux.
        The divertor will be attached to this profile.

        Parameters
        ----------
        n_iteration_max: integer
            Max number of iterations after which the optimiser is stopped.
        hf_limit: float
            Heat flux limit for the optimisation.

        Returns
        -------
        profile: Loop
            Optimised profile
        """
        initial_profile = self.make_preliminary_profile()
        self.preliminary_profile = initial_profile
        self.make_flux_surfaces(profile=initial_profile)

        profile = initial_profile
        for i_it in range(n_iteration_max):

            x_wall, z_wall, hf_wall = self.hf_firstwall_params(profile)

            for x_hf, z_hf, hf in zip(x_wall, z_wall, hf_wall):
                if hf > hf_limit:
                    profile = self.modify_fw_profile(profile, x_hf, z_hf)

            heat_flux_max = max(hf_wall)
            print(heat_flux_max)
            self.optimised_profile = profile
            if heat_flux_max < hf_limit:
                break

        return profile

    def make_divertor_demo_like(self, fw_loop):
        """
        Create a DEMO like divertor loop for the single null configuration.

        Parameters
        ----------
        fw_loop: Loop
            first wall preliminary profile

        Returns
        -------
        divertor_loop: list
            List of Loops for the divertor geometry (single entry for SN)
        """
        # Some shorthands
        z_low = self.points["x_point"]["z_low"]
        x_x_point = self.points["x_point"]["x"]

        # Pick some flux loops to use to locate strike points
        flux_loops = self.pick_flux_loops()

        # Find the strike points
        inner, outer = self.find_strike_points(flux_loops)

        # Find the intersection of the first wall loop and
        # the x-y plane containing the lower X point
        z_norm = 2
        fw_int_point = get_intersection_point(
            z_low, z_norm, fw_loop, x_x_point, inner=False
        )

        # Define the left and right limits of the divertor entrance
        # relative to the separatrix x point given gap parameters
        div_left = x_x_point - self.params.xpt_inner_gap
        div_right = max(x_x_point + self.params.xpt_outer_gap, fw_int_point[0])

        # Define the x coordinates for the divertor
        x_div = [
            div_left,
            inner[0] - self.params.tk_inner_target_sol,
            inner[0] + self.params.tk_inner_target_pfr,
            outer[0] - self.params.tk_outer_target_pfr,
            outer[0] + self.params.tk_outer_target_sol,
            div_right,
        ]

        # Define the z coordinates for the divertor
        z_div = [z_low, inner[1], inner[1], outer[1], outer[1], z_low]

        # Create the loop and return as a list
        divertor_loop = Loop(x=x_div, z=z_div)
        divertor_loop.close()

        return [divertor_loop]

    def find_koz_flux_loop_ints(self, koz, flux_loops):
        """
        Find intersections between the keep-out-zone loop
        and the given flux loops.  Only upper and lower most
        intersections for each flux line are returned.

        Parameters
        ----------
        koz : Loop
            Loop representing the keep-out-zone
        flux_loops: list of Loop
            List of flux loops used to find intersections

        Returns
        -------
        all_points : list
             List of the [x,z] coordinates of the intersections
        """
        # For each flux loop find the intersections with the koz
        all_points = []
        for loop in flux_loops:
            # Expectation is that flux loop is open
            if loop.closed:
                raise SystemsError(
                    "Selected flux contour is closed, please check psi_norm"
                )

            # Get the intersections
            int_x, int_z = get_intersect(koz, loop)

            # Combine into [x,z] points
            points = list(map(list, zip(int_x, int_z)))
            all_points.extend(points)

        return all_points


class FirstWallDN(FirstWall):
    def init_params(self):
        """
        Initialise First Wall DN parameters from config.
        """
        super().init_params()
        self.fw_p_sol_near_omp = self.params.fw_p_sol_near * self.params.p_rate_omp
        self.fw_p_sol_far_omp = self.params.fw_p_sol_far * self.params.p_rate_omp
        self.fw_p_sol_near_imp = self.params.fw_p_sol_near * self.params.p_rate_imp
        self.fw_p_sol_far_imp = self.params.fw_p_sol_far * self.params.p_rate_imp


class FirstWallPlotter(ReactorSystemPlotter):
    """
    The plotter for a First Wall and relevant Heat Flux distribution
    """

    def __init__(self):
        super().__init__()
        self._palette_key = "FW"

    def plot_xz(self, plot_objects, ax=None, **kwargs):
        """
        Plot the first wall in x-z.
        """
        super().plot_xz(plot_objects, ax=ax, **kwargs)

    def plot_hf(
        self,
        separatrix,
        loops,
        x_int,
        z_int,
        hf_int,
        fw_profile,
        koz=None,
        ax=None,
        **kwargs,
    ):
        """
        Plots the 2D heat flux distribution.

        Parameters
        ----------
        separatrix: Union[Loop, MultiLoop]
            The separatrix loop(s) (Loop for SN, MultiLoop for DN)
        loops: [MultiLoop]
            The flux surface loops
        x_int: [float]
            List of all the x coordinates at the intersections of concern
        z_int: [float]
            List of all the z coordinates at the intersections of concern
        hf_int: [float]
            List of all hf values at the intersections of concern
        fw_profile: Loop
            Inner profile of a First wall
        koz: Loop
            Loop representing the keep-out-zone
        ax: Axes, optional
            The optional Axes to plot onto, by default None.
        """
        fw_profile.plot(ax=ax, fill=False, edgecolor="k", linewidth=1)
        loops.plot(ax=ax, fill=False, edgecolor="r", linewidth=0.2)
        separatrix.plot(ax=ax, fill=False, edgecolor="r", linewidth=1)
        if koz is not None:
            koz.plot(ax=ax, fill=False, edgecolor="g", linewidth=1)
        ax = plt.gca()
        cs = ax.scatter(x_int, z_int, s=25, c=hf_int, cmap="viridis", zorder=100)
        bar = plt.gcf().colorbar(cs, ax=ax)
        bar.set_label("Heat Flux [MW/m^2]")
