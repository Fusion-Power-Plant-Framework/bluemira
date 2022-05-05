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
Classes that provide divertor shapes built from tracking psi and grazing angle at a given
leg length.
"""

import operator
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from bluemira.base.look_and_feel import bluemira_print_flush, bluemira_warn
from bluemira.base.parameter import ParameterFrame
from bluemira.equilibria.find import find_OX_points
from BLUEPRINT.base.error import SystemsError
from BLUEPRINT.geometry.boolean import boolean_2d_difference, boolean_2d_union
from BLUEPRINT.geometry.geomtools import rotate_vector_2d, unique, xz_interp
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.offset import offset_clipper
from BLUEPRINT.nova.firstwall import DivertorProfile


class Location:
    """
    Type-checking struct for location values (lower/upper)
    """

    __name__ = "Location"
    lower = "lower"
    upper = "upper"


class Leg:
    """
    Type-checking struct for leg values (inner/outer)
    """

    __name__ = "Leg"
    inner = "inner"
    outer = "outer"


class DivertorSilhouette(DivertorProfile):
    """
    A divertor shape that tracks along psi and grazing angle at the provided leg length

    Builds the divertor using the following algorithm:

    1. Define the leg length and grazing angle.
    2. Draw the target, with specified length and grazing angle at the given leg length.
    3. Draw the dome by following psi from the end of the target.
    4. Connect the target and dome.
    5. Draw the baffle by maintaining the grazing angle through psi values.
    6. Connect the divertor in 2D by following the vacuum vessel shape.
    """

    config: Type[ParameterFrame]
    inputs: dict

    # fmt: off
    default_params = [
        ["plasma_type", "Type of plasma", "SN", "dimensionless", None, "Input"],
        ["div_L2D_ib", "Inboard divertor leg length", 1.1, "m", None, "Input"],
        ["div_L2D_ob", "Outboard divertor leg length", 1.3, "m", None, "Input"],
        ["div_graze_angle", "Divertor SOL grazing angle", 1.5, "°", None, "Input"],
        ["div_Ltarg", "Divertor target length", 0.5, "m", None, "Input"],
        ['div_open', 'Divertor open/closed configuration', False, 'dimensionless', None, 'Input'],
        ["div_r_min", "Divertor minimum radial extent", 0.25, "m", None, "Input"],
        ["tk_div", "Divertor thickness", 0.1, "m", None, "Input"],
        ["bb_gap", "Gap between divertor and breeding blanket", 0.05, "m", None, "Input"],
    ]
    # fmt: on

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs

        self._init_params(self.config)

        self._validate_inputs()
        self.sf = self.inputs["sf"]
        self.targets = self.inputs["targets"]
        self.debug = self.inputs.get("debug", False)
        self.baffle_max_iter = self.inputs.get("baffle_max_iter", 500)

        o_points, x_points = find_OX_points(self.sf.x2d, self.sf.z2d, self.sf.psi)
        self.o_points = np.array([[point[0], point[1]] for point in o_points])
        self.x_points = np.array([[point[0], point[1]] for point in x_points])

        self.leg_lengths = {
            Leg.inner: self.params.div_L2D_ib,
            Leg.outer: self.params.div_L2D_ob,
        }

        self._initialise_geometry()

    def _validate_inputs(self):
        """
        Check that the inputs dictionary is valid.

        Raises
        ------
        SystemsError
            If the inputs dictionary does not contain all of the required keys.
        """
        required_keys = {"sf", "targets"}
        missing_keys = []
        for key in required_keys:
            if key not in self.inputs:
                missing_keys.append(key)
        if len(missing_keys) > 0:
            raise SystemsError(
                f"{self.__class__.__name__} must be provided with "
                f"{', '.join(missing_keys)} keys in the input dictionary"
            )

    def _initialise_geometry(self):
        """
        Initialise the geometry dictionary.
        """
        self.geom["targets"] = {}
        self.geom["domes"] = {}
        self.geom["baffles"] = {}
        self.geom["divertors"] = {}
        self.geom["inner"] = {}
        self.geom["outer"] = {}
        self.geom["first_wall"] = {}
        self.geom["gap"] = {}
        self.geom["blanket_inner"] = {}
        self.geom["vessel_gap"] = {}

    def _select_location(self, location: Location):
        """
        Select the location to query in the `StreamFlow` object.

        Parameters
        ----------
        location: Location
            The location to select in the `StreamFlow` object.
        """
        self.sf.get_x_psi(select=location)
        self.sf.sol()
        self.sf.get_legs()

    def get_sol_leg(self, location: Location, leg: Leg):
        """
        Get the requested scrape off layer leg at the given location.

        Parameters
        ----------
        location: Location
            The location of the scrape off layer leg to retrieve.
        leg: Leg
            The scrape off layer leg to retrieve at the given location.

        Returns
        -------
        sol_leg: np.array(N, 2)
            The line representing the requested scrape off layer leg.
        """
        self._select_location(location)

        # Snip the SOL leg at the specified length and repack as numpy array.
        x_sol, z_sol = self.select_layer(leg)
        return np.array([x_sol, z_sol])

    def get_x_point(self, location: Location):
        """
        Get the x-point for the location that the divertor will be built around.

        Parameters
        ----------
        location: Location
            The location at which the x-point will be obtained.

        Returns
        -------
        x_point: List[float]
            The x-point.
        """
        for x_point in self.x_points:
            if location == Location.upper and x_point[1] > 0:
                return x_point
            elif location == Location.lower and x_point[1] < 0:
                return x_point
        else:
            raise SystemsError(f"Unable to find x-point for {location} location")

    def _get_target_side(self, location: Location, leg: Leg):
        """
        Get the side (1 or -1) of the target for the given location and leg.

        Parameters
        ----------
        location: Location
            The location of the target.
        leg: Leg
            The leg of the target.

        Returns
        -------
        side: int
            The side of the target.
        """
        side = 1
        if location == Location.lower and leg == Leg.outer:
            side = -1
        elif location == Location.upper and leg == Leg.inner:
            side = -1

        if self.targets[leg]["open"]:
            side *= -1

        return side

    def _match_grazing_angle(self, strike_point, initial_vector, side):
        """
        Rotate the `initial_vector` in the direction defined by `side` to match the
        grazing angle at the `strike_point`.

        Parameters
        ----------
        strike_point: np.array(2)
            The point at which to measure the grazing angle.
        initial_vector: np.array(2)
            The vector to rotate to minimise the grazing angle.
        side: int (-1 or 1)
            The side on which to rotate the vector (-1 for anti-clockwise, 1 for
            clockwise).
        """
        if side not in [-1, 1]:
            message = (
                "Rotation side must be either -1 for anti-clockwise or 1 for clockwise."
            )
            bluemira_warn(message)
            raise SystemsError(message)

        def opt(theta):
            target_unit_vector = rotate_vector_2d(
                initial_vector, np.radians(theta) * side
            )

            grazing_angle = self.sf.get_graze(
                [strike_point[0], strike_point[1]], target_unit_vector
            )

            if self.debug:
                bluemira_print_flush(
                    f"opt: {theta} ... {np.degrees(grazing_angle)} ... "
                    f"{np.abs(grazing_angle - np.radians(self.params.div_graze_angle))}"
                )

            return np.abs(grazing_angle - np.radians(self.params.div_graze_angle))

        opts = {"disp": True} if self.debug else None

        # Minimise the deviation from the specified grazing angle, but restrict the
        # resulting rotation to be within 90 degrees in the specified direction (side).
        res = minimize_scalar(opt, bounds=(0, 90), method="bounded", options=opts)

        if res.success:
            theta = res.x

            target_unit_vector = rotate_vector_2d(
                initial_vector, np.radians(theta) * side
            )

            return target_unit_vector
        else:
            raise SystemsError("Unable to match grazing angle for divertor target.")

    def make_target(self, location: Location, leg: Leg):
        """
        Make a divertor target leg at the specified location.

        Rotates the target until the `div_graze_angle` is found and then extends to the
        `div_Ltarg` length.

        Parameters
        ----------
        location: Location
            The location for the target.
        leg: Leg
            The leg for the target.

        Returns
        -------
        target: np.array(2, 2)
            The target line, made up of two points.
        """
        sol = self.get_sol_leg(location, leg)

        side = self._get_target_side(location, leg)

        # Rotate by the grazing angle
        sol_end_vector = sol.T[-2] - sol.T[-1]

        target_unit_vector = self._match_grazing_angle(sol.T[-1], sol_end_vector, side)

        # Extend unit vector to target length
        target = target_unit_vector * self.params.div_Ltarg / 2
        target = (
            np.array([-target, target]) if side == 1 else np.array([target, -target])
        )
        target = target + sol.T[-1]

        if location == Location.lower:
            target = target[::-1]

        return target

    def make_dome(self, location, start, end):
        """
        Make the dome by connecting the start and end points by following psi from the
        start point.

        Parameters
        ----------
        location: Location
            The location of the divertor to which this dome belongs (upper/lower).
        start: np.array(2)
            The starting point for the dome.
        end: np.array(2)
            The end point for the dome.

        Returns
        -------
        dome: np.array(N, 2)
            The points representing the resulting dome.
        """
        # Find the psi contour corresponding to the private region for the location.
        psi = self.sf.point_psi(start)
        contours = self.sf.get_contour([psi])[0]
        comp = operator.lt if location == Location.lower else operator.gt
        contour = None
        for contour in contours:
            if all(comp(contour.T[1], 0)):
                break
        else:
            raise SystemsError("Unable to find psi contour to position dome.")

        # Get the indexes of the closes points on the psi contour to the target points
        idx = np.zeros(2, dtype=int)
        idx[0] = np.argmin(np.hypot(*(contour - start).T))
        idx[1] = np.argmin(np.hypot(*(contour - end).T))

        # Make sure start and end are in the right order for the contour
        dome_contour = None
        if idx[0] > idx[1]:
            idx = idx[::-1]
            dome_contour = contour[idx[0] + 1 : idx[1]]
            dome_contour = dome_contour[::-1]
        else:
            dome_contour = contour[idx[0] + 1 : idx[1]]

        dome = np.array([])
        dome = np.append(dome, [start])
        dome = np.append(dome, dome_contour)
        dome = np.append(dome, [end])
        dome.shape = (len(dome) // 2, 2)
        if self.debug:
            plt.plot(*dome.T, lw=3)

        return dome

    def make_baffle(self, location, leg, target, step_size=0.01, end_point_scaling=0.8):
        """
        Make a baffle at the specified location and leg, for a provided target.

        Draws a line from the end of the target, minimising the deviation of the grazing
        angle from the `div_graze_angle` parameter at each point.

        Parameters
        ----------
        location: Location
            The location at which to make the baffle.
        leg: Leg
            The leg to make the baffle for.
        target: np.array(2, 2)
            The target line to build the baffle from.
        step_size: float, optional
            The step size with which to iterate along the baffle, by default 0.01.
        end_point_scaling: float, optional
            Scaling value to apply to the x-point z location to terminate baffle build,
            by default 0.8.
        """
        _, end_point_z = self.get_x_point(location)
        end_point_z *= end_point_scaling

        side = self._get_target_side(location, leg)
        if leg == Leg.inner:
            target = target[::-1]

        target_unit_vector = (np.diff(target.T) / np.hypot(*np.diff(target.T))).T
        baffle_section_start = target[-1]
        baffle_section_end = target_unit_vector * step_size
        baffle_section_end = (baffle_section_end + baffle_section_start)[0]

        if self.debug:
            _, ax = plt.subplots()
            ax.set_title(f"Baffle : {location} - {leg}")

        comp = operator.lt if location == Location.lower else operator.gt
        i = 0
        baffle = [baffle_section_start]
        baffle.append(baffle_section_end)
        while (
            comp(baffle_section_end.T[1], end_point_z) and baffle_section_end.T[0] > 0.5
        ):
            if i > self.baffle_max_iter:
                raise SystemsError(
                    f"Maximum baffle iterations {self.baffle_max_iter} exceeded."
                )
            initial_vector = np.diff(
                np.array([baffle_section_start, baffle_section_end]).T
            ).T[0]
            section_vector = self._match_grazing_angle(
                baffle_section_start, initial_vector, side
            )
            baffle_section_start = baffle_section_end
            baffle_section_end = section_vector * step_size
            baffle_section_end = baffle_section_end + baffle_section_start
            baffle.append(baffle_section_end)
            i += 1
            if self.debug:
                debug_section = np.array([baffle_section_start, baffle_section_end])
                section_diff = np.diff(debug_section.T)
                graze = np.degrees(self.sf.get_graze(baffle_section_start, section_diff))
                bluemira_print_flush(f"baffle: {graze}")
                ax.plot(*debug_section.T)  # type: ignore (reportUnboundVariable)
        baffle = np.array(baffle)
        return baffle if leg == Leg.outer else baffle[::-1]

    def _build_plasma_facing_shape(self, location: Location):
        """
        Draws the plasma-facing shape at the specified location.

        Populates the targets, baffles, and domes in the `geom` dictionary.

        Parameters
        ----------
        location: Location
            The location at which to build the plasma-facing shape.

        Returns
        -------
        plasma_facing_shape: Loop
            The open loop that defines the plasma-facing shape at the requested location.
        """
        self.geom["targets"][location] = {}
        self.geom["baffles"][location] = {}

        for leg in [Leg.inner, Leg.outer]:
            self.set_target(leg)
            self.geom["targets"][location][leg] = self.make_target(location, leg)

            self.geom["baffles"][location][leg] = self.make_baffle(
                location,
                leg,
                self.geom["targets"][location][leg],
            )

        # Find the outer (in the z-direction) target point and use the psi value
        # to construct the dome.
        point1 = self.geom["targets"][location][Leg.inner][-1]
        point2 = self.geom["targets"][location][Leg.outer][0]
        if self.debug:
            _, ax = plt.subplots()
        self.geom["domes"][location] = self.make_dome(location, point1, point2)

        # Build the plasma-facing divertor loop
        div = np.array([])
        div = np.append(div, self.geom["baffles"][location][Leg.inner])
        div = np.append(div, self.geom["targets"][location][Leg.inner][0])
        div = np.append(div, self.geom["domes"][location])
        div = np.append(div, self.geom["targets"][location][Leg.outer][-1])
        div = np.append(div, self.geom["baffles"][location][Leg.outer])
        div.shape = (len(div) // 2, 2)

        div = Loop(x=div.T[0], z=div.T[1])

        xd, zd = unique(*div.d2)[:2]

        if location == Location.lower:
            zindex = zd <= self.sf.x_point[1] + 0.5 * (
                self.sf.o_point[1] - self.sf.x_point[1]
            )
        else:
            zindex = zd >= self.sf.x_point[1] + 0.5 * (
                self.sf.o_point[1] - self.sf.x_point[1]
            )
        xd, zd = xd[zindex], zd[zindex]  # remove upper points
        xd, zd = xz_interp(xd, zd)  # resample

        return div

    def make_divertor(self, fw_loop: Type[Loop], location: str):
        """
        Make a divertor geometry dictionary for the provided `first_wall` at the given
        location.

        Parameters
        ----------
        fw_loop: Loop
            The loop representing the first wall to build the divertor onto.
        location: Location
            The location at which to make the divertor.

        Returns
        -------
        div_geom: dict
            The dictionary representing the divertor geometry, which can be used by a
            `ReactorCrossSection` object to reserve the divertor shape.
        """
        div = self._build_plasma_facing_shape(location)
        div.close()

        self.geom[Leg.inner][location] = div
        self.geom["first_wall"][location] = boolean_2d_union(
            fw_loop, self.geom[Leg.inner][location]
        )[0]
        self.geom[Leg.outer][location] = offset_clipper(
            self.geom[Leg.inner][location], self.params.tk_div, method="square"
        )
        self.geom[Leg.outer][location] = boolean_2d_difference(
            self.geom[Leg.outer][location], fw_loop
        )[0]
        self.geom["gap"][location] = offset_clipper(
            self.geom[Leg.outer][location], self.params.bb_gap
        )
        self.geom["vessel_gap"][location] = boolean_2d_union(
            fw_loop, self.geom["gap"][location]
        )[0]
        self.geom["divertors"][location] = boolean_2d_difference(
            self.geom[Leg.outer][location], self.geom[Leg.inner][location]
        )[0]

        inner = fw_loop.copy()
        div_koz = self.geom["gap"][location]

        count = 0
        for i, point in enumerate(inner):
            if div_koz.point_inside(point):
                # Now we re-order the loop and open it, such that it is open
                # inside the KOZ
                if count > 1:
                    inner.reorder(i, 0)
                    inner.open_()
                    break
                count += 1  # (Second point inside the loop)

        self.geom["blanket_inner"][location] = boolean_2d_difference(inner, div_koz)[0]

        div_geom = {
            "divertor_inner": self.geom["inner"][location],
            "divertor_gap": self.geom["gap"][location],
            "divertor": self.geom["divertors"][location],
        }

        return div_geom

    def build(self, first_wall):
        """
        Build the divertors onto the provided first wall.
        """
        # Handle single null and double null plasmas
        locations = [Location.lower]
        if self.params.plasma_type == "DN":
            locations.append(Location.upper)

        # Do the build, populating the geometry dictionary
        for location in locations:
            self.make_divertor(first_wall, location)


class DivertorSilhouetteFlatDome(DivertorSilhouette):
    """
    A divertor shape that tracks along psi at the provided leg length with a flat dome

    Builds the divertor using the following algorithm:

    1. Define the leg length and grazing angle.
    2. Draw the target, with specified length and grazing angle at the given leg length.
    3. Draw the dome by connecting the targets with a straight line.
    4. Connect the target and dome.
    5. Draw the baffle by maintaining the grazing angle through psi values.
    6. Connect the divertor in 2D by following the vacuum vessel shape.
    """

    default_params = DivertorSilhouette.default_params.to_records()

    def make_dome(self, location, start, end):
        """
        Make the dome by connecting the start and end points with a straight line.

        Parameters
        ----------
        location: Location
            The location of the divertor to which this dome belongs (upper/lower).
        start: np.array(2)
            The starting point for the dome.
        end: np.array(2)
            The end point for the dome.

        Returns
        -------
        dome: np.array(2, 2)
            The points representing the resulting dome.
        """
        return np.array([start, end])


class DivertorSilhouettePsiBaffle(DivertorSilhouette):
    """
    A divertor shape that tracks along psi at the provided leg length.

    Builds the divertor using the following algorithm:

    1. Define the leg length and grazing angle.
    2. Draw the target, with specified length and grazing angle at the given leg length.
    3. Draw the dome by following psi from the end of the target.
    4. Connect the target and dome.
    5. Draw the baffle by following the psi contour from the end of the target.
    6. Connect the divertor in 2D by following the vacuum vessel shape.
    """

    default_params = DivertorSilhouette.default_params.to_records()

    def make_baffle(self, location, leg, target, end_point_scaling=0.8):
        """
        Make a baffle at the specified location and leg, for a provided target.

        Draws a line from the end of the target, following the psi contour.

        Parameters
        ----------
        location: Location
            The location at which to make the baffle.
        leg: Leg
            The leg to make the baffle for.
        target: np.array(2, 2)
            The target line to build the baffle from.
        """
        end_point_x, end_point_z = self.get_x_point(location)
        end_point_z *= end_point_scaling

        if leg == Leg.inner:
            target = target[::-1]

        psi = self.sf.point_psi(target[-1])
        contours = self.sf.get_contour([psi])[0]
        comp = operator.lt if leg == Leg.inner else operator.gt
        contour = np.array([])
        for cont in contours:
            selector = np.where(comp(cont.T[0], end_point_x))
            contour = np.append(contour, cont[selector])
        contour.shape = (len(contour) // 2, 2)

        if len(contour) == 0:
            raise SystemsError("Unable to find psi contour to position baffle.")

        diff = target[-1] - contour
        comp = operator.lt if location == Location.lower else operator.gt
        if (leg == Leg.inner and location == Location.lower) or (
            leg == Leg.outer and location == Location.upper
        ):
            contour_end = np.where(comp(contour.T[1], end_point_z))[0][-1]
            contour_start = np.argmin(np.hypot(*diff.T))
        else:
            contour_end = np.argmin(np.hypot(*diff.T))
            contour_start = np.where(comp(contour.T[1], end_point_z))[0][0]
        baffle = contour[contour_start : contour_end + 1]

        # Find which end is closest to the target to connect
        d_first_point = np.hypot(*(target[-1] - baffle[0]))
        d_last_point = np.hypot(*(target[-1] - baffle[-1]))
        if leg == Leg.inner:
            # Last point should connect
            if d_last_point > d_first_point:
                baffle = baffle[::-1]
        else:
            # First point should connect
            if d_first_point > d_last_point:
                baffle = baffle[::-1]
        return baffle


class DivertorSilhouetteFlatDomePsiBaffle(DivertorSilhouettePsiBaffle):
    """
    A divertor shape that tracks along psi at the provided leg length with a flat dome

    Builds the divertor using the following algorithm:

    1. Define the leg length and grazing angle.
    2. Draw the target, with specified length and grazing angle at the given leg length.
    3. Draw the dome by connecting the targets with a straight line.
    4. Connect the target and dome.
    5. Draw the baffle by following the psi contour from the end of the target.
    6. Connect the divertor in 2D by following the vacuum vessel shape.
    """

    def make_dome(self, location, start, end):
        """
        Make the dome by connecting the start and end points with a straight line.

        Parameters
        ----------
        location: Location
            The location of the divertor to which this dome belongs (upper/lower).
        start: np.array(2)
            The starting point for the dome.
        end: np.array(2)
            The end point for the dome.

        Returns
        -------
        dome: np.array(2, 2)
            The points representing the resulting dome.
        """
        return np.array([start, end])
