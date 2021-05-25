# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.
"""
An example file to make the Tapered Picture Frame version of the ToroidalFieldCoils
object, optimized for the minimum length

"""

import os
import matplotlib.pyplot as plt
from BLUEPRINT.base.file import make_BP_path
from BLUEPRINT.base import ParameterFrame
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.systems.tfcoils import ToroidalFieldCoils
from BLUEPRINT.equilibria.shapes import flux_surface_manickam
from BLUEPRINT.cad.model import CADModel

# BASED ON GV_SCR_03 from the PROCESS-STEP repository
# fmt: off
params = [
    ["R_0", "Major radius", 3.42, "m", None, "Input"],
    ["B_0", "Toroidal field at R_0", 2.2, "T", None, "Input"],
    ["n_TF", "Number of TF coils", 12, "N/A", None, "Input"],
    ["tk_tf_nose", "Bucking Cylinder Thickness", 0.17, "m", None, "Input"],
    ["tk_tf_wp", "TF coil winding pack thickness", 0.4505, "m", None, "PROCESS"],
    ["tk_tf_front_ib", "TF coil inboard steel front plasma-facing", 0.02252, "m", None, "Input"],  # casthi
    ["tk_tf_ins", "TF coil ground insulation thickness", 0.0, "m", None, "Input"],
    ["tk_tf_insgap", "TF coil WP insertion gap", 0.0, "m", "Backfilled with epoxy resin (impregnation)", "Input"],
    ["r_tf_in", "Inner Radius of the TF coil inboard leg", 0.176, "m", None, "PROCESS"],
    ["r_tf_inboard_out", "Outboard Radius of the TF coil inboard leg tapered region", 0.6265, "m", None, "PROCESS"],
    ['TF_ripple_limit', 'TF coil ripple limit', 1.0, '%', None, 'Input'],
    ["npts", "Number of points", 200, "N/A", "Used for vessel and plasma", "Input"],
    ["h_cp_top", "Height of the Tapered Section", 6.199, "m", None, "PROCESS"],
    ["r_cp_top", "Radial Position of Top of taper", 0.8934, "m", None, "PROCESS"],
    ["tf_wp_depth", "TF coil winding pack depth (in y)", 0.4625, "m", "Including insulation", "PROCESS"],
    ['tk_tf_outboard', 'TF coil outboard thickness', 1, 'm', None, 'Input', 'PROCESS'],
    ['tf_taper_frac', "Height of straight portion as fraction of total tapered section height", 0.5, 'N/A', None, 'Input'],
    ['r_tf_outboard_corner', "Corner Radius of TF coil outboard legs", 0.8, 'm', None, 'Input'],

]
# fmt: on

parameters = ParameterFrame(params)

read_path = make_BP_path("Geometry", subfolder="data")
write_path = make_BP_path("Geometry", subfolder="generated_data")

lcfs = flux_surface_manickam(3.42, 0, 2.137, 2.9, 0.55, n=40)
lcfs.close()

# lcfs.translate([0.15, 0, 0])

name = os.sep.join([read_path, "KOZ_PF_test1.json"])
ko_zone = Loop.from_file(name)

to_tf = {
    "name": "Example_PolySpline_TF",
    "plasma": lcfs,
    "koz_loop": ko_zone,
    "shape_type": "TP",  # This is the shape parameterisation to use
    "obj": "L",  # This is the optimisation objective: minimise length
    "ny": 3,  # This is the number of current filaments to use in y
    "nr": 2,  # This is the number of current filaments to use in x
    "nrip": 4,  # This is the number of points on the separatrix to calculate ripple for
    "read_folder": read_path,  # This is the path that the shape will be read from
    "write_folder": write_path,  # This is the path that the shape will be written to
}

tf1 = ToroidalFieldCoils(parameters, to_tf)

tf1.optimise()
# print(tf1.geom["Centreline"].length)

f1, ax1 = plt.subplots()
ko_zone.plot(ax1, edgecolor="b", fill=False)
tf1.plot_ripple(ax=ax1)
plt.gca().set_aspect("equal")
plt.show()

f, ax = plt.subplots()
tf1.plot_xy(ax=ax)
plt.show()

n_tf = tf1.params.n_TF
model = CADModel(n_tf)
model.add_part(tf1.build_CAD())
model.display("full")
# model.save_as_STEP_assembly(write_path, scale=1e3)
