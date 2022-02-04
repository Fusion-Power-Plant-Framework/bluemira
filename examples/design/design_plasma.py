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
A basic tutorial for configuring and running a design with a parameterised plasma.
"""

# %%
from bluemira.base.design import Design
from bluemira.builders.plasma import MakeParameterisedPlasma

# %%[markdown]
# # Configuring and Running a Simple Parameterised Design
#
# This example shows how to set up a parameterised design with a single build stage.
# The build stage takes the provided major radius (`R_0`) aspect ratio (`A`) parameters,
# maps these onto a parameterised shape, which in this case is a Johner parameterisation
# of the last closed flux surface (LCFS).
#
# ## Configuring the design
#
# First we have to specify a `build_config` for our design. This gives the build stages
# that will be run within the design, what `Builder` class will be used for the build
# at that stage, and how that `Builder` should be configured. Here we see that we are
# specifying a build stage called "Plasma", that will use the `MakeParameterisedPlasma`
# class of `Builder`. We specify that the parameterisation class to use should be
# `JohnerLCFS`, the implementation of which is in the `bluemira.equilibria.shapes`
# module. We map the `R_0` and `A` external parameters onto the `r_0` and `a` parameters
# for that shape, and give the `PhysicalComponent` that results from the build a label of
# LCFS.

# %%

build_config = {
    "Plasma": {
        "class": "MakeParameterisedPlasma",
        "param_class": "bluemira.equilibria.shapes::JohnerLCFS",
        "variables_map": {
            "r_0": "R_0",
            "a": "A",
        },
        "label": "LCFS",
    },
}

# %%[markdown]
# ## Parameterising the Design
#
# Now that we have set up the design to be performed, we need to provide the values to
# which to set each of the parameters that are needed for the design. We also need to
# provide a `Name` for our design so that it can be identified.

# %%

params = {
    "Name": "A Plasma Design",
    "R_0": (9.0, "Input"),
    "A": (3.5, "Input"),
}

# %%[markdown]
# ## Running the Design
#
# After configuring and parameterising our design, we can now run it as below.

# %%
design = Design(params, build_config)
result = design.run()

# %%[markdown]
# ## Inspecting the Design Results
#
# Our design has now completed and we have the resulting `Component` available in our
# `result` variable. This component gives us a tree of values that we can navigate to
# see the shapes produced by our design in different dimensions (or views). In this case
# our top level `Component` is what we named our design, then the next level is the
# system that has been built as part of our build stages. We then see the representation
# of that system in various dimensions, followed by the physical components the define
# the shapes that make up that system in the given view.

# %%
print(result.tree())

# %%[markdown]
# We can also plot our components in each of the provided dimensions, as well as viewing
# the 3D cad for the xyz dimension. Finally we can also access the `Builder` that was
# used by the build stage in order to re-run part of the design. In this case we are
# able to generate a new 3D component using the same `Builder` but this time we sweep it
# through 270 degrees, rather than the full 360.

# %%
color = (0.80078431, 0.54, 0.80078431)
for dims in ["xz", "xy"]:
    lcfs = result.get_component("Plasma").get_component(dims).get_component("LCFS")
    lcfs.plot_options.face_options["color"] = color
    lcfs.plot_2d()

# %%
lcfs = result.get_component("Plasma").get_component("xyz").get_component("LCFS")
lcfs.display_cad_options.color = color
lcfs.display_cad_options.transparency = 0.2
lcfs.show_cad()

# %%
plasma_builder: MakeParameterisedPlasma = design.get_builder("Plasma")
lcfs = plasma_builder.build_xyz(degree=270.0)
lcfs.display_cad_options.color = color
lcfs.show_cad()
