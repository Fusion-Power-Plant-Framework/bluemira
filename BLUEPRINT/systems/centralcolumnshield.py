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
Central column neutron shield system
"""
import numpy as np

import bluemira.geometry._deprecated_loop as new_loop
from bluemira.geometry.error import GeometryError
from BLUEPRINT.cad.centralcolumnshieldCAD import CentralColumnShieldCAD
from BLUEPRINT.geometry.boolean import boolean_2d_difference_loop, simplify_loop
from BLUEPRINT.geometry.geombase import Plane
from BLUEPRINT.geometry.geomtools import loop_plane_intersect
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.shell import Shell
from BLUEPRINT.systems.baseclass import ReactorSystem
from BLUEPRINT.systems.mixins import Meshable
from BLUEPRINT.systems.plotting import ReactorSystemPlotter


class CentralColumnShield(Meshable, ReactorSystem):
    """
    Central column neutron shield system

    Parameters
    ----------
    config: ParameterFrame
    inputs: dict
       Expects:

         - inputs["FW_outer"] : Loop

            Outer loop of the first wall shell

         - inputs["VV_inner"] : Loop

            Inner loop loop of the vacuum vessel shell

         - inputs["Div_cassettes"] : list

            List of Loop representing the divertor cassette

    Attributes
    ----------
    geom : dict
        Dictionary to specify 2D geometry

         - geom["2D profile"] : Loop

            Central column shield 2D profile
    """

    # fmt: off
    default_params = [
        ["n_TF", "Number of TF coils", 16, "dimensionless", None, "Input"],
        ["g_ccs_vv_add", "Additional gap between the central column shield and the vacuum vessel", 0.0, "m", None, "Input"],
        ["g_ccs_fw", "Gap between the central column shield and the first wall", 0.05, "m", None, "Input"],
        ["g_ccs_div", "Gap between the central column shield and the divertor cassette", 0.05, "m", None, "Input"],
        ["tk_ccs_min", "Minimum thickness of the central column shield", 0.1, "m", None, "Input"],
        ["r_ccs", "Outer radius of the central column shield", 2.5, "m", None, "Input"]
    ]
    # fmt: on

    CADConstructor = CentralColumnShieldCAD

    # Some error messages - defined here so we can check them in the test
    THICKNESS_ERR = "Mid-plane thickness of central shield too small"
    FIRSTWALL_ERR = "(Offset) First wall not inside the (offset) vacuum vessel"
    INPUT_TYPE_ERR = "First wall and vacuum vessel profiles must be Loop objects"
    OFFSET_VAL_ERR = "Loop offsets should be non-negative."
    LARGE_RADIUS_ERR = "Central column radius is too big"
    SMALL_RADIUS_ERR = "Central column radius is too small"

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs

        self._init_params(self.config)

        # Construct the 2D profile
        self.build_profile()

        # Set the object responsible for 2D plotting
        self._plotter = CCSPlotter()

    def build_profile(self):
        """
        Construct the 2D profile of the central column shield
        """
        # First retrieve the first wall and vacuum vessel loops from inputs
        fw_outer = self.inputs["FW_outer"]
        vv_inner = self.inputs["VV_inner"]

        # Retrieve and check offset params
        fw_offset = self.params.g_ccs_fw
        vv_offset = self.params.g_ccs_vv_add
        div_offset = self.params.g_ccs_div

        # Create new offset loops
        fw_offset_loop = self._offset_loop(fw_outer, fw_offset, offset_inwards=False)
        vv_offset_loop = self._offset_loop(vv_inner, vv_offset, offset_inwards=True)

        # Create a shell from the vacuum vessel and first wall
        try:
            ccs_shell = Shell(fw_offset_loop, vv_offset_loop)
        except GeometryError:
            # Catch failure and change error message
            raise GeometryError(self.FIRSTWALL_ERR)

        # Cut the shell to everything contained within a given radius
        ccs_loop = self._cut_shell(ccs_shell, self.params.r_ccs)

        # Subtract the divertor cassettes if provided
        if "Div_cassettes" in self.inputs:
            div_cassettes = self.inputs["Div_cassettes"]
            for cassette in div_cassettes:
                # Apply an offset
                cassette_offset = cassette.offset_clipper(div_offset, method="miter")
                ccs_loop = boolean_2d_difference_loop(ccs_loop, cassette_offset)

        # Check the thickness at the midplane (z=0)
        mp_thickness = self._get_midplane_thickness(ccs_loop)
        if mp_thickness < self.params.tk_ccs_min:
            raise GeometryError(self.THICKNESS_ERR)

        # Save
        self.geom["2D profile"] = ccs_loop

    def _offset_loop(self, loop, offset, offset_inwards=True):
        """
        Return offset (and simplified) loop
        """
        if not isinstance(loop, (Loop, new_loop.Loop)):
            raise TypeError(self.INPUT_TYPE_ERR)

        if offset < 0.0:
            raise ValueError(self.OFFSET_VAL_ERR)

        if offset == 0.0:
            # Nothing to do
            offset_loop = loop
            return offset_loop

        if offset_inwards:
            offset *= -1.0

        offset_loop = loop.offset(offset)
        offset_loop = simplify_loop(offset_loop)
        return offset_loop

    def _cut_shell(self, ccs_shell, radius):
        """
        Cut the given shell to everything contained within a given radius
        """
        inner = ccs_shell.inner
        outer = ccs_shell.outer

        # Get the bounding coords of the loops in xz plane
        in_min, in_max = self._get_loop_extrema_xz(inner)
        out_min, out_max = self._get_loop_extrema_xz(outer)

        # Check that the inner loop is entirely inside the outer
        for icoord in range(0, 2):
            if (in_min[icoord] < out_min[icoord]) or (in_max[icoord] > out_max[icoord]):
                raise GeometryError(self.FIRSTWALL_ERR)

        # Check the cutting radius is inside the correct bounds
        if radius < out_min[0]:
            raise ValueError(self.SMALL_RADIUS_ERR)
        elif radius > in_max[0]:
            raise ValueError(self.LARGE_RADIUS_ERR)

        # Define a rectangle around the bounds of the vv loop and
        # the specified radius
        xmin = radius
        zmin = out_min[1]
        xmax, zmax = out_max
        rect_x = [xmin, xmin, xmax, xmax]
        rect_z = [zmin, zmax, zmax, zmin]
        ccs_rect = Loop(x=rect_x, y=None, z=rect_z)
        ccs_rect.close()

        # Crop the shell to everything within the given radius using our rectangle
        ccs_loop = boolean_2d_difference_loop(ccs_shell, ccs_rect)
        return ccs_loop

    def _get_loop_extrema_xz(self, loop: Loop):
        """
        Return the minimal and maximum points of a loop in the x,z plane.
        """
        # Get the min and max points of the loop
        x, y, z = loop.xyz
        xmax, xmin = np.max(x), np.min(x)
        ymax, ymin = np.max(y), np.min(y)
        zmax, zmin = np.max(z), np.min(z)

        # Check that the loop is in the xz plane
        if ymax != ymin:
            raise GeometryError("Profile should be in the x-z plane")

        min = [xmin, zmin]
        max = [xmax, zmax]
        return min, max

    def _get_midplane_thickness(self, loop):
        """
        Retrieve the thickness of given loop when intersected with a plane
        at z=0.
        """
        # Get the plane at z=0 (need three points in plane to specify)
        midplane = Plane([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])

        # Retrieve the intersections with z=0 x-y plane
        intersect = loop_plane_intersect(loop, midplane)
        if intersect is None:
            raise GeometryError("Central shield loop does not intersect midplane")
        # Expect precisely two intersections
        if intersect.shape != (2, 3):
            raise GeometryError(
                "Expect precisely two intersections of loop with midplane."
            )

        # Compute and return thickness from intersect
        thickness = abs(intersect[1, 0] - intersect[0, 0])
        return thickness

    @property
    def xz_plot_loop_names(self):
        """
        The x-z loop names to plot.
        """
        names = ["2D profile"]
        return names


class CCSPlotter(ReactorSystemPlotter):
    """
    The plotter for the central column shield
    """

    def __init__(self):
        super().__init__()
        self._palette_key = "CCS"

    def plot_xz(self, plot_objects, ax=None, **kwargs):
        """
        Plot the central column shield in x-z.
        """
        super().plot_xz(plot_objects, ax=ax, **kwargs)
