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
An example file to make the Tapered Picture Frame version of the ToroidalFieldCoils
object, optimized for the minimum length

"""

import os

import matplotlib.pyplot as plt

from bluemira.base.file import make_bluemira_path
from bluemira.base.parameter import ParameterFrame
from bluemira.equilibria.shapes import flux_surface_manickam
from BLUEPRINT.cad.model import CADModel
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.systems.tfcoils import ToroidalFieldCoils

# BASED ON GV_SPR_08 from the PROCESS-STEP repository
# fmt: off
params = [
    ["R_0", "Major radius", 3.639, "m", None, "Input"],
    ["B_0", "Toroidal field at R_0", 2.0, "T", None, "Input"],
    ["n_TF", "Number of TF coils", 12, "dimensionless", None, "Input"],
    ["tk_tf_nose", "TF coil inboard nose thickness", 0.0377, "m", None, "Input"],
    ['tk_tf_side', 'TF coil inboard case minimum side wall thickness', 0.02, 'm', None, 'Input'],
    ["tk_tf_wp", "TF coil winding pack thickness", 0.569, "m", None, "PROCESS"],
    ["tk_tf_front_ib", "TF coil inboard steel front plasma-facing", 0.02, "m", None, "Input"],
    ["tk_tf_ins", "TF coil ground insulation thickness", 0.008, "m", None, "Input"],
    ["tk_tf_insgap", "TF coil WP insertion gap", 1.0E-7, "m", "Backfilled with epoxy resin (impregnation)", "Input"],
    ["r_tf_in", "Inboard radius of the TF coil inboard leg", 0.148, "m", None, "PROCESS"],
    ["tf_wp_depth", 'TF coil winding pack depth (in y)', 0.3644, 'm', 'Including insulation', 'PROCESS'],
    ["ripple_limit", "Ripple limit constraint", 0.6, "%", None, "Input"],
    ['r_tf_outboard_corner', "Corner Radius of TF coil outboard legs", 0.8, 'm', None, 'Input'],
    ['r_tf_inboard_corner', "Corner Radius of TF coil inboard legs", 0.0, 'm', None, 'Input'],
    ['tk_tf_inboard', 'TF coil inboard thickness', 0.6267, 'm', None, 'PROCESS'],

]
# fmt: on

parameters = ParameterFrame(params)

read_path = make_bluemira_path("Geometry", subfolder="data/BLUEPRINT")
write_path = make_bluemira_path("SC_P_coil", subfolder="generated_data/BLUEPRINT")

lcfs = flux_surface_manickam(3.639, 0, 2.183, 2.8, 0.543, n=40)
lcfs.close()

name = os.sep.join([read_path, "KOZ_PF_test1.json"])
ko_zone = Loop.from_file(name)

to_tf = {
    "name": "Example_PolySpline_TF",
    "plasma": lcfs,
    "koz_loop": ko_zone,
    "shape_type": "P",  # This is the shape parameterisation to use
    "wp_shape": "W",  # This is the winding pack shape choice for the inboard leg
    "conductivity": "SC",  # Resistive (R) or Superconducting (SC)
    "npoints": 200,
    "obj": "L",  # This is the optimisation objective: minimise length
    "ny": 3,  # This is the number of current filaments to use in y
    "nr": 2,  # This is the number of current filaments to use in x
    "nrip": 4,  # This is the number of points on the separatrix to calculate ripple for
    "read_folder": read_path,  # This is the path that the shape will be read from
    "write_folder": write_path,  # This is the path that the shape will be written to
}

tf1 = ToroidalFieldCoils(parameters, to_tf)

tf1.optimise()
plt.plot(tf1.p_in["x"], tf1.p_in["z"])
plt.gca().set_aspect("equal")
# plt.show()
# print(tf1.geom["Centreline"].length)
f1, ax1 = plt.subplots()

ko_zone.plot(ax1, edgecolor="b", fill=False)
tf1.plot_ripple(ax=ax1)
plt.gca().set_aspect("equal")
# plt.show()

f, ax = plt.subplots()
tf1.plot_xy(ax=ax)
plt.show()

n_tf = tf1.params.n_TF
model = CADModel(n_tf)
model.add_part(tf1.build_CAD())
model.display(pattern="q")
# model.save_as_STEP_assembly(write_path, scale=1e3)
