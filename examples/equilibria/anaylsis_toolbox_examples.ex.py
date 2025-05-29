# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Example of equilibria analysis utilities.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.constants import CoilType
from bluemira.base.file import get_bluemira_root
from bluemira.equilibria.analysis import (
    EqAnalysis,
    MultiEqAnalysis,
    select_eq,
    select_multi_eqs,
)
from bluemira.equilibria.diagnostics import (
    CSData,
    EqDiagnosticOptions,
    EqPlotMask,
    EqSubplots,
    FluxSurfaceType,
    PsiPlotType,
)
from bluemira.geometry.coordinates import Coordinates

# %% [markdown]
# # Analysis Examples
#
# Below we highlight some example uses of the analysis classes in equilibria/analysis.py.
#
# **EqAnalysis:**
#
# Used to compare selected equilibrium to a reference equilibrium,
# e.g., for comparing optimisation results to a starting equilibrium.
#
# **MultiEqAnalysis:**
#
# Used to compare the properties of two or more equilibria,
# e.g, for examining the effects of different optimisation runs
# on key physics parameters.
#
# **Note:**
#
# Both EqAnalysis and MultiEqAnalysis can take equilibria of different types
# (i.e, fixed or free), and/or with different grid sizes and grid resolutions.
# However, it should be noted that not all of the available tools can be applied
# to the fixed boundary equilibria.

# %% [markdown]
# ## Inputs
#
# We are going to use MAST-U-like and DEMO-like equilibria
# as the inputs for our examples.
#
# The function 'select_eq' can be used to load a free-boundary
# or fixed plasma equilibrium object from a given file path.
#
# We will also set the label of each equilibria to a value that is useful for plotting.

# %%
# MAST-U-like
root_path = get_bluemira_root()

masty_path = Path(root_path, "tests/equilibria/test_data/SH_test_file.json")
masty_eq = select_eq(masty_path)
masty_eq.label = "MAST"
# DEMO-like-SN
single_demoish_path = Path(root_path, "tests/equilibria/test_data/eqref_OOB.json")
single_demoish_eq = select_eq(single_demoish_path, from_cocos=7)
single_demoish_eq.label = "DEMO-SN"
# DEMO-like-DN
double_demoish_path = Path(root_path, "tests/equilibria/test_data/DN-DEMO_eqref.json")
double_demoish_eq = select_eq(double_demoish_path)
double_demoish_eq.label = "DEMO-DN"

# %% [markdown]
# ## EqAnalysis Part 1
#
# First we will look at the utilities available in the EqAnalysis class
# for our input equilibria.

# %%
# We can create an EqAnalysis object without any inputs.
eq_analysis_double = EqAnalysis()

# Or we can add the input equilibrium we are interested in investigating.
eq_analysis_single = EqAnalysis(input_eq=single_demoish_eq)

# Note: we have not input any diagnostic plotting settings,
# these will be default values.

# %%
# Plot equilibrium.
ax = eq_analysis_single.plot()

# %%
# Plot equilibrium - we can input an equilibrium using set_input
# if we had not already specified its value when
# creating an EqAnalysis object.
eq_analysis_double.set_input(double_demoish_eq)
ax = eq_analysis_double.plot()

# %%
# Plot equilibrium normalised profiles.
ax = eq_analysis_single.plot_profiles()

# %%
# Both together - note that the eq we used as an input for plot has been saved.
ax = eq_analysis_double.plot_equilibria_with_profiles()

# %%
# Plot field components.
ax = eq_analysis_double.plot_field()

# %%
# Plot 1-D profiles at magnetic axis.
ax = eq_analysis_double.plot_eq_core_mag_axis()

# %%
# Plot an assortment of physics parameters for the plasma core.
# Note that the dataclass with the results is also output.
core_results, ax = eq_analysis_single.plot_eq_core_analysis()

# %%
# Key parameters as a table.
# Note that an a dataclass is also output.
physics_datclass = eq_analysis_double.physics_info_table()

# %%
# Control coil information in a table.
# Note that control can be a coil type, list of control coil names,
# or None if all coils are control coils.
table = eq_analysis_single.control_coil_table(control=CoilType.PF)

# %% [markdown]
# ## EqAnalysis Part 2
#
# Now we will look at the utilities available in the EqAnalysis class
# for comparison to a reference equilibria.
#
# The EqDiagnosticOptions class contains the information needed to
# set up the comparison plots used during an optimisation or
# (as is the case here) for analysis purposes.
#
# We will use the single null DEMO-like equilibrium as our reference equilibrium.

# %%
# Diagnostic settings for looking at the psi difference
diag_ops_1 = EqDiagnosticOptions(
    psi_diff=PsiPlotType.PSI_DIFF,
    split_psi_plots=EqSubplots.XZ,
)
# Diagnostic settings for looking at the relative psi difference,
# with the plasma and coilset psi contributions plotted separately.
# We have also added a mask so that we only plot values from inside
# the reference LCFS.
diag_ops_2 = EqDiagnosticOptions(
    psi_diff=PsiPlotType.PSI_REL_DIFF,
    split_psi_plots=EqSubplots.XZ_COMPONENT_PSI,
    plot_mask=EqPlotMask.IN_REF_LCFS,
)
# Here we create our two analysis classes
eq_analysis_1 = EqAnalysis(
    input_eq=double_demoish_eq, diag_ops=diag_ops_1, reference_eq=single_demoish_eq
)
eq_analysis_2 = EqAnalysis(
    input_eq=double_demoish_eq, diag_ops=diag_ops_2, reference_eq=single_demoish_eq
)

# %%
# PLot comparison of equilibrium profiles to reference equilibrium profiles
ax = eq_analysis_1.plot_compare_profiles()

# %%
# PLot the equilibrium and reference equilibrium seperatrices
ax = eq_analysis_1.plot_compare_separatrix()

# %%
# Plot a comparison of the input and reference equilibria.
# For eq_analysis_1 we chose to use diag_ops_1,
# which is a total psi diff plot.
eq_analysis_1.plot_compare_psi()

# %%
# What about with a completely different equilibria?
# Let's set the reference equilibria to be our MAST-U plasma.
# Notice that in the case of MAST vs DEMO SN,
# the two equilibria grids do not overlap at all.
eq_analysis_1.set_reference(masty_eq)
eq_analysis_1.plot_compare_psi()
print(
    "Psi range for reference equilibrium = ",
    np.round(np.min(masty_eq.psi()), 2),
    " to ",
    np.round(np.max(masty_eq.psi()), 2),
)
print(
    "Psi range for input equilibrium = ",
    np.round(np.min(double_demoish_eq.psi()), 2),
    " to ",
    np.round(np.max(double_demoish_eq.psi()), 2),
)

# %%
# For eq_analysis_2 we chose to use diag_ops_2, which is absolute psi diff
# for separate plasma and coilset contributions. A mask has also been applied,
# so that only the values inside the reference LCFS are visible.
eq_analysis_2.plot_compare_psi()

# %%
# Plot the leg flux for a given divertor target and compare to reference.
ax, _, _ = eq_analysis_2.plot_target_flux(
    target="lower_outer",
    target_coords=Coordinates({"x": [10, 11], "z": [-7.5, -7.5]}),
    vertical=False,
)

# %%
ax, _, _ = eq_analysis_2.plot_target_flux(
    target="upper_outer",
    target_coords=Coordinates({"x": [10, 11], "z": [7.5, 7.5]}),
)

# %%
ax, _, _ = eq_analysis_2.plot_target_flux(
    target="lower_inner",
    target_coords=Coordinates({"x": [5.5, 6.5], "z": [-7.5, -7.5]}),
)

# %%
ax, _, _ = eq_analysis_2.plot_target_flux(
    target="upper_inner",
    target_coords=Coordinates({"x": [5.5, 6.5], "z": [7.5, 7.5]}),
)

# %% [markdown]
# ## MultiEqAnalysis
#
# Now we will look at the utilities available in the
# MultiEqAnalysis class for multiple equilibria.
#
# We will use the same set of Equilibria as before,
# but the way that they are input is different.

# %%
# First we create a list of paths to equilibria of interest.
paths = [masty_path, double_demoish_path, single_demoish_path]
# Corresponding list of names for plt legends etc.
# If these are not chosen them the equilibria names are set to
# Eq_1, Eq_2, etc.
equilibrium_names = ["MAST", "DEMO-DN", "DEMO-SN"]
# Don't forget to make sure the correct cocos value is used
# if they are not all the same
from_cocos = [3, 3, 7]
# Load all the equilibria info into a dictionary.
# This will out input for MultiEqAnalysis.
equilibria_dictionary = select_multi_eqs(
    equilibrium_input=paths,
    equilibrium_names=equilibrium_names,
    from_cocos=from_cocos,
    control_coils=CoilType.PF,
)
# Note: equilibrium_input cam also be a list of equilibrium
# objects, in which case, fixed_or_free, dummy_coils, from_cocos,
# to_cocos, and qpsi_positive parameters are not necessary.
equilibria_dictionary = select_multi_eqs(
    equilibrium_input=[masty_eq, double_demoish_eq, single_demoish_eq],
    equilibrium_names=equilibrium_names,
    control_coils=CoilType.PF,
)
# Now create the analysis class for multiple equilibria
multi_analysis = MultiEqAnalysis(equilibria_dictionary)

# %%
# This outputs same physics info as for the EqAnalysis but for all listed equilibria.
table = multi_analysis.physics_info_table()

# %%
# Plot physics parameters for the plasma core
# Note that a list with the results is also output
core_results, ax = multi_analysis.plot_core_physics()

# %%
# Plot the normalised profiles
ax = multi_analysis.plot_compare_profiles()

# %%
# Plot a selected flux surface from each equilibria
# Default if LCFS but can also have seperatricies...
ax = multi_analysis.plot_compare_flux_surfaces()

# %%
# ... or a flux surface with a user selected normalised psi value.
ax = multi_analysis.plot_compare_flux_surfaces(
    flux_surface=FluxSurfaceType.PSI_NORM, psi_norm=1.05
)


# %%
# Now we will plot some divertor leg values of interest,
# but first we set up some dummy first wall coordinates to use in our example.
def make_dummy_pfb(radius, offset):
    """
    Make circular poloidal component of first wall coords.

    Parameters
    ----------
    radius:
        radius of circle
    offset:
        x-offset for circle centre

    Returns
    -------
    :
        Coordinates
    """
    theta = (np.arange(0, 101) * 0.02 * np.pi) - np.pi
    x = radius * np.sin(theta) + offset
    z = radius * np.cos(theta)
    return Coordinates({"x": x, "z": z})


pfb_masty = make_dummy_pfb(1.5, 0.5)
pfb_sin_demoish = make_dummy_pfb(7.0, 8.0)
pfb_dou_demoish = make_dummy_pfb(7.0, 10.0)

# PLot dummy first walls with seperatricies
f, ax = plt.subplots()
ax.plot(pfb_masty.x, pfb_masty.z, linestyle="--")
ax.plot(pfb_sin_demoish.x, pfb_sin_demoish.z, linestyle="--")
ax.plot(pfb_dou_demoish.x, pfb_dou_demoish.z, linestyle="--")
_ax = multi_analysis.plot_compare_flux_surfaces(
    flux_surface=FluxSurfaceType.SEPARATRIX, ax=ax
)

# %%
# Plot grazing angle and connection length for equilibria divertor legs,
# for a given number of flux surfaces and a given spacing between flux surfaces,
# defaults are n_layers = 10 and dx_off = 0.10 [m] respectfully.

# First wall coordinates are an input to the plotting function,
# will default to the grid edges if no first wall is set.
# We have set 'radians' to False to plot in degrees.
ax = multi_analysis.plot_divertor_length_angle(
    plasma_facing_boundary_list=[pfb_masty, pfb_sin_demoish, pfb_dou_demoish],
    dx_off=0.5,
    n_layers=10,
    radian=False,
)

# %%
# Print a table comparing coilset information,
# the equilibria can have different coilsets.
# Note: when we defined multi_analysis, we set the control_coils to be PF type coils,
# so only coils with that type are printed for each equilibria.
coilset_table = multi_analysis.coilset_info_table()

# %%
# Default value_type in coilset comparison table is coil current,
# but we can also choose from: x-position, z-position, coil field, and coil force.
coilset_table = multi_analysis.coilset_info_table(value_type=CSData.B)
