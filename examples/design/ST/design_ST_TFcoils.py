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
Build and show a standalone TF coil
"""

from bluemira.builders.ST.tf_coils import TFCoilsBuilder
from bluemira.equilibria.shapes import flux_surface_manickam
from bluemira.geometry.tools import make_polygon

# %%[markdown]
# # Configuring and building a simple TF coilset
#
# This example shows how to build a standalone TF coil.
#
# ## Configuring the design
#
# First we have to specify a `build_config` for our design. This will have details
# of the optimisation problem we want to, or don't want to run, as well as the shape
# paramterisation details

# %%
build_config = {
    "name": "test_ST_TF",
    "runmode": "run",  # ["run", "read", "mock"]
    "param_class": "TopDomeCurvedPictureFrame",
    "problem_class": "bluemira.builders.tf_coils::RippleConstrainedLengthOpt",
    "variables_map": {
        "x_mid": {"value": "r_tf_in_centre", "fixed": True},
        "x_curve_start": {"value": 6.8, "fixed": True},
        "x_out": {"value": 15, "lower_bound": 12, "upper_bound": 16},
        "z_mid_up": {"value": 11.5, "fixed": True},
        "z_mid_down": {"value": -11.5, "fixed": True},
        "z_max_up": {"value": 13, "fixed": True},
        "r_c": {"value": 0.6, "fixed": True},
    },
    "algorithm_name": "COBYLA",
    "problem_settings": {
        "n_rip_points": 50,
        "nx": 2,
        "ny": 2,
    },
}

# %%[markdown]
# Next, we input some parameters containing all the necessary information to build a coilset.
# The list below contains the minimum information required.

# %%
params = {
    "R_0": 9,
    "z_0": 0,
    "B_0": 6,
    "n_TF": 16,
    "TF_ripple_limit": 0.6,
    "r_tf_in": 3.2,
    "r_tf_in_centre": 3.7,
    "tk_tf_nose": 0.6,
    "tk_tf_front_ib": 0.04,
    "tk_tf_side": 0.1,
    "tk_tf_ins": 0.08,
    # This isn't treated at the moment...
    "tk_tf_insgap": 0.1,
    # Dubious WP depth from PROCESS (I used to tweak this when building the TF coils)
    "tf_wp_width": 0.76,
    "tf_wp_depth": 1.05,
}

# %%[markdown]
# We now must build a dummy LCFS and run the builder, using said LCFS and specifying no Keep-out-Zone.
# We can make and use a KOZ if needed. We can also specify that the code save the optimised shape
# if we run the optimiser, so for future runs we can simply select "read" and not have to rerun the
# optimiser

# %%
lcfs = flux_surface_manickam(9, 0, 3, 1.5)
lcfs = make_polygon(lcfs.xyz)
builder = TFCoilsBuilder(params, build_config, lcfs, None)
builder.run()
cad = builder.build()
xz = builder.build_xz()
xz.plot_2d()
cad.show_cad()

if builder.runmode == "run":
    builder.save_shape()
print("done")
