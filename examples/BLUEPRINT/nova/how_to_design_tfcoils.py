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
An example file to make ToroidalFieldCoils object, using different shapes, and
optimisation objectives.

It is meant to be worked through, line by line, so best to just run these in
the console one by one.
"""

# %%[markdown]
# # How To Design TF Coils

# %%
import os
import matplotlib.pyplot as plt

from bluemira.base.parameter import ParameterFrame
from bluemira.base.look_and_feel import bluemira_print

from BLUEPRINT.base.file import make_BP_path
from BLUEPRINT.systems.baseclass import ReactorSystem
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.systems.tfcoils import ToroidalFieldCoils

# %%[markdown]
# ## INSTANTIATE A TF COIL OBJECT
#
# First, let's initialise a TF coil object. In order to do this, we need two
# things:
# *  ParameterFrame: (contains all of the reactor parameters we need)
# *  dictionary:     (which contains some specific info for the TF coil, and
#                     some more complicated objects)
#
#
# Here is the input for the ParameterFrame for a TF coil object
# It is populated with defaults

# %%
# fmt: off
params = [
    ["R_0", "Major radius", 9, "m", None, "Input"],
    ["B_0", "Toroidal field at R_0", 6, "T", None, "Input"],
    ["n_TF", "Number of TF coils", 16, "N/A", None, "Input"],
    ["rho_j", "TF coil WP current density", 18.25, "MA/m^2", None, "Input"],
    ["tk_tf_nose", "TF coil inboard nose thickness", 0.6, "m", None, "Input"],
    ["tk_tf_wp", "TF coil winding pack thickness", 0.5, "m", None, "PROCESS"],
    ["tk_tf_front_ib", "TF coil inboard steel front plasma-facing", 0.04, "m", None, "Input"],
    ["tk_tf_ins", "TF coil ground insulation thickness", 0.08, "m", None, "Input"],
    ["tk_tf_insgap", "TF coil WP insertion gap", 0.1, "m", "Backfilled with epoxy resin (impregnation)", "Input"],
    ["r_tf_in", "Inboard radius of the TF coil inboard leg", 3.2, "m", None, "PROCESS"],
    ["ripple_limit", "Ripple limit constraint", 0.6, "%", None, "Input"],
]
# fmt: on

parameters = ParameterFrame(params)

# %%[markdown]
# Now we need to build the dictionary, which contains some optimiser information
# and some shapes we need to define the optimisation problem.
#
# Geometry is stored in Loop objects (collections of 2/3-D coordinates, and
# associated methods).
# Load a separatrix shape, along which we will calculate the toroidal field
# ripple.

# %%
read_path = make_BP_path("Geometry", subfolder="data/BLUEPRINT")
write_path = make_BP_path("Geometry", subfolder="generated_data/BLUEPRINT")
name = os.sep.join([read_path, "LCFS.json"])
lcfs = Loop.from_file(name)

# %%[markdown]
# Load a keep-out zone for the TF coil shape

# %%
name = os.sep.join([read_path, "KOZ.json"])
ko_zone = Loop.from_file(name)

# %%[markdown]
# Specify some inputs to the TF shape (that are not physical parameters)
# We also put the Loops in this dictionary.

# %%
to_tf = {
    "name": "Example_PolySpline_TF",
    "plasma": lcfs,
    "koz_loop": ko_zone,
    "shape_type": "S",  # This is the shape parameterisation to use
    "obj": "L",  # This is the optimisation objective: minimise length
    "ny": 1,  # This is the number of current filaments to use in y
    "nr": 1,  # This is the number of current filaments to use in x
    "nrip": 10,  # This is the number of points on the separatrix to calculate ripple for
    "read_folder": read_path,  # This is the path that the shape will be read from
    "write_folder": write_path,  # This is the path that the shape will be written to
}

# %%[markdown]
# So now we have everything we need to instantiate the TF coil object:

# %%
tf1 = ToroidalFieldCoils(parameters, to_tf)

# %%[markdown]
# In BLUEPRINT, Reactor objects and# sub-system objects all inherit from a base
# class: ReactorSystem
# Let's take a moment to introduce ourselves to the ReactorSystem.. It gives a
# lot of the flavour to the different sub-systems, and makes them all behave in
# similar ways.
# *  Methods
#    1. plotting (plot_xy and plot_xz)
#    2. CAD (build_CAD and show_CAD)
# *  Attributes
#    *  'params': the ParameterFrame of the ReactorSystem
#    *  'geom': the dictionary of geometry objects for the ReactorSystem
#    *  'requirements': the dictionary of requirements (I don't really use this)

# %%
query_subclass = issubclass(ToroidalFieldCoils, ReactorSystem)
query_instance = isinstance(tf1, ReactorSystem)
print(f"are ToroidalFieldCoils a ReactorSystem?: {query_subclass}")
print(f"are my TF coils an instance of ReactorSystem?: {query_instance}")

# %%[markdown]
# Let's take a look at the ParameterFrame for the TF coils we instantiated.

# %%
print(tf1.params)

# %%[markdown]
# Notice that there are a few more parameters in there that we didn't specify..
#
# Some of them were calculated upon instantiation, others.. well.
# There are lots of parameters required for designing ReactorSystems. Often,
# defaults are used - and these defaults are often "good guesses" or simply
# engineering judgement for things we don't quite know about yet.
#
# Let's plot our TF coils and we what we have so far and look at the TF coil
# itself (in the x-z plane)

# %%
f1, ax1 = plt.subplots()

lcfs.plot(ax1, edgecolor="r", fill=False)
ko_zone.plot(ax1, edgecolor="b", fill=False)

tf1.plot_xz(ax=ax1)

# %%[markdown]
# And in the x-y plane (at the midplane)

# %%
f, ax = plt.subplots()
tf1.plot_xy(ax=ax)

# %%[markdown]
# Looks weird, but that's because we haven't really designed the TF coil yet...
# The "default" shape is actually just the underlying shape parameterisation
# populated with some dummy variables. We need to find the optimum shape.
#
# We can see that the TF coil is also encroaching upon its keep-out zone..
#
# What about the ripple?

# %%
f, ax2 = plt.subplots()
tf1.plot_ripple(ax=ax2)

# %%[markdown]
# The maximum TF ripple is lower than we specified (0.6 %)
#
# We can also look at the CAD..

# %%
tf1.show_CAD()

# %%[markdown]
# Close the pop-up window to continue!

# %%[markdown]
# ## OPTIMISE A TF COIL OBJECT USING A SPLINE PARAMETERISATION
#
# Second, let's design our TF coil object. In order to do this, we need a few
# things (we have already done most of these when we instantiated our coil)
# *  we need to specify a shape parameterisation (here we are using a
#    PolySpline parameterisation - shape_type = 'S')
# *  we need to specify an optimisation objective (default = winding pack
#    length)
# *  we need to specify what our ripple constraint is (default = 0.6%)
# *  we need to specify a keep-out-zone for the TF coil
# *  we need to specify a shape upon which we want to contrain the ripple
# *  we need to optimise the TF coil shape parameterisation
#
#
# Run the optimisation problem (minimise length, constrain ripple)

# %%
bluemira_print("Optimising 'S' TF coil... could take about 20 s..")
tf1.optimise()

# %%[markdown]
# Hmm.. a warning. Our maximum ripple is a little higher than we specified.
# This is because we didn't specify many points at which to check the ripple
#
# Let's look at it

# %%
f, ax = plt.subplots()
tf1.plot_xz(ax=ax)
tf1.plot_ripple(ax=ax, fig=f)

# %%[markdown]
# NOTE: The bigger ny, ny, and nrip are, the longer the optimisation will take
# (and the better the result)
#
# In practice, we know a bit about this problem...
# The TF ripple is usually the worst on the low field side (outboard portion)
# So we speed up the problem by only checking points on the LFS
#
# We're going to ignore this warning for now, as will we address this problem
# later.

# %%[markdown]
# ## OPTIMISE A TF COIL OBJECT USING A PRINCETON-D PARAMETERISATION
#
# Now let's do the same thing for a different shape parameterisation
# We can use the same ParameterFrame as before, and the same dictionary
# We're just going to change the type of shape we are using

# %%
to_tf["shape_type"] = "D"  # This is the key for a Princeton-D shape

tf2 = ToroidalFieldCoils(parameters, to_tf)
bluemira_print("Optimising 'D' TF coil... could take about 4 s..")
tf2.optimise()

# %%[markdown]
# That looks bad; the optimiser had a hard time and failed. Let's look
# at what we got:

# %%
f, ax = plt.subplots()

tf2.plot_ripple(ax=ax, fig=f)

# %%[markdown]
# So now we have a few
# options:
# *  tweak the bounds and initial values
# *  tweak the optimiser parameters
#
#
# Let's try just tweaking the optimiser parameters for now.

# %%
tf2 = ToroidalFieldCoils(parameters, to_tf)
# eps is a step size parameter for the gradient-based algorithm.
tf2.optimise(eps=0.21)
f, ax = plt.subplots()
tf2.plot_ripple(ax=ax, fig=f)

# %%[markdown]
# That's a bit better.. but this isn't really a robust way of doing this.
# It's because this parameterisation is very simple, and we're using an
# optimisation algorithm that does a better job on more complex problems!

# It's still a bit over-optimised, too. We'll deal with this kind of stuff
# later on.

# %%

# %%[markdown]
# ## OPTIMISE A TF COIL OBJECT USING A PICTUREFRAME PARAMETERISATION
#
# Now let's do the same thing for a different shape parameterisation

# %%
to_tf["shape_type"] = "P"  # This is the key for a PictureFrame shape

tf3 = ToroidalFieldCoils(parameters, to_tf)
bluemira_print("Optimising 'P' TF coil... could take about 3 s..")
tf3.optimise()

# %%[markdown]
# Let's have a look at what we got:

# %%
f, ax = plt.subplots()
tf3.plot_ripple(ax=ax, fig=f)
lcfs.plot(ax, edgecolor="r", fill=False)
ko_zone.plot(ax, edgecolor="b", fill=False)

# %%[markdown]
# Alright, so this is better! But our ripple is still a little higher than we
# wanted. This is because we are only actually checking ripple at certain
# places, and probably we are not capturing the worst place.
#
# Let's start over, and increase the number of current filaments we are using
# and the number of points on the separatrix at which we are checking the
# TF ripple

# %%
to_tf = {
    "name": "Example_PictureFrame_TF",
    "plasma": lcfs,
    "koz_loop": ko_zone,
    "shape_type": "P",  # This is the shape parameterisation to use
    "obj": "L",  # This is the optimisation objective: minimise length
    "ny": 3,  # This is the number of current filaments to use in y
    "nr": 1,  # This is the number of current filaments to use in x
    "nrip": 30,  # This is the number of points on the separatrix to calculate ripple for
    "read_folder": read_path,
    "write_folder": write_path,
}

tf3 = ToroidalFieldCoils(parameters, to_tf)

bluemira_print(
    "Optimising a rectangular TF coil with multiple current filaments..\n"
    "This is going to take about 44 seconds."
)
tf3.optimise()

# %%[markdown]
# Let's take a look at our result now

# %%
f, ax = plt.subplots()
tf3.plot_ripple(ax=ax, fig=f)
lcfs.plot(ax, edgecolor="r", fill=False)
ko_zone.plot(ax, edgecolor="b", fill=False)

# %%[markdown]
# Hmm... OK so it's all good (the TF coil is outside the keep-out zone, and the
# ripple is below 0.6% at all locations (but not lower than 0.6%).

# The coil also doesn't look like it could be any smaller, so this is
# what we wanted to achieve.
