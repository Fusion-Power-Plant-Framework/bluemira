#!/usr/bin/env python
# coding: utf-8

# # SHApproximation Fuction
#
# This example illustrates the input and output of the Bluemira spherical harmonics approximation fuction (SHApproximation) which can be used in coilset current and position optimisation for spherical tokamaks. For an example of how SHApproximation is used, please see the notebook called 'Use of Spherical Harmonic Approximation in Optimisation.ipynb'.

# In[1]:


# from bluemira.equilibria.equilibrium import Equilibrium
import matplotlib.pyplot as plt
import numpy as np

from bluemira.display.auto_config import plot_defaults
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.harmonics import SHApproximation
from bluemira.equilibria.plotting import PLOT_DEFAULTS, EquilibriumPlotter

plot_defaults()

get_ipython().run_line_magic("pdb", "")


# In[5]:


# Data from EQDSK file
file_path = "SH_test_file.json"

# Plot
eq = Equilibrium.from_eqdsk(file_path)
f, ax = plt.subplots()
eq.plot(ax=ax)
eq.coilset.plot(ax=ax)
plt.show()


# ### Inputs
#
# #### Required
#
# - eq = Our chosen Bluemira Equilibrium
#
# #### Optional
#
# - n_points: Number of desired collocation points
# - point_type: How the collocation points are distributed
# - acceptable_fit_metric: how 'good' we require the approximation to be
# - r_t: typical lengthscale for spherical harmonic approximation
# - extra_info: set this to true if you wish to return additional information and plot the results.

# In[3]:


# Information needed for SH Approximation
shapprox = SHApproximation(
    eq,
    n_points=50,
    point_type="random_plus_extrema",
    acceptable_fit_metric=0.03,
    extra_info=True,
)


# ### Outputs
#
# SHApproximation outputs a dictionary of results that can be used in optimisation.
#
# #### Always output
#
# - "coilset", coilset to use with SH approximation
# - "r_t", typical lengthscale for spherical harmonic approximation
# - "harmonic_amplitudes", SH coefficients/amplitudes for required number of degrees
# - "max_degree", number of degrees required for a SH approx with the desired fit metric

# In[11]:


print(shapprox["coilset"])


# In[10]:


print(shapprox["r_t"])


# In[9]:


print(shapprox["harmonic_amplitudes"])


# In[8]:


print(shapprox["max_degree"])


# #### Ouput on request
#
# - "fit_metric_value", fit metric acheived
# - "approx_total_psi", the total psi obtained using the SH approximation

# In[14]:


print(shapprox["fit_metric_value"])


# In[19]:


psi = shapprox["approx_total_psi"]
levels = np.linspace(np.amin(psi), np.amax(psi), 50)
plot = plt.subplot2grid((2, 2), (0, 0), rowspan=2, colspan=1)
plot.set_title("approx_total_psi")
plot.contour(eq.grid.x, eq.grid.z, psi, levels=levels, cmap="viridis", zorder=8)
plt.show()
