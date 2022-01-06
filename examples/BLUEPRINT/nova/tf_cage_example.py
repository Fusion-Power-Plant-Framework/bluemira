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
An example file to perform a simple beam FE analysis on a coil cage
"""
import os

import matplotlib.pyplot as plt

# %%
from IPython import get_ipython

from bluemira.base.config import SingleNull
from bluemira.base.file import BM_ROOT, get_bluemira_path
from bluemira.display.auto_config import plot_defaults
from bluemira.materials import MaterialCache
from BLUEPRINT.nova.structuralsolver import StructuralSolver
from BLUEPRINT.reactor import Reactor

try:
    get_ipython().run_line_magic("matplotlib", "qt")
except AttributeError:
    pass

plot_defaults()

# %%[markdown]

# This is a worked example for how to use analyse the coil cage with a simple
# beam FE model.

# The first thing we'll do for this is build a full reactor object. Like that we
# have access to all the information and geometry.

# We start by sub-classing the Reactor object, and setting some configuration
# parameters.

# %%
config = {
    "Name": "Coil_Structures_Example",
    "P_el_net": 500,
    "tau_flattop": 3600,  # TODO
    "plasma_type": "SN",
    "reactor_type": "Normal",
    "CS_material": "Nb3Sn",
    "PF_material": "NbTi",
    "A": 3.1,
    "n_CS": 5,
    "n_PF": 6,
    "n_TF": 16,
}

build_config = {
    "generated_data_root": f"{BM_ROOT}/generated_data/BLUEPRINT",
    "plot_flag": False,
    "process_mode": "mock",
    "plasma_mode": "run",
    "tf_mode": "run",
    # TF coil config
    "TF_type": "S",
    "TF_objective": "L",
    "GS_type": "ITER",
    # FW and VV config
    "VV_parameterisation": "S",
    "FW_parameterisation": "S",
    "BB_segmentation": "radial",
    "lifecycle_mode": "life",
    "HCD_method": "power",
}

build_tweaks = {
    # TF coil optimisation tweakers (n ripple filaments)
    "nr": 1,
    "ny": 1,
    "nrippoints": 20,  # Number of points to check edge ripple on
}


class SingleNullReactor(Reactor):
    """
    A single-null fusion power reactor class.
    """

    config: dict
    build_config: dict
    build_tweaks: dict
    default_params = SingleNull().to_records()


# %%[markdown]
# We'll need to know some information about materials, so let's build a cache of known
# material properties

# %%
material_data_path = get_bluemira_path("materials", subfolder="data")
material_cache = MaterialCache()
material_cache.load_from_file(os.sep.join([material_data_path, "materials.json"]))
material_cache.load_from_file(os.sep.join([material_data_path, "mixtures.json"]))

# %%[markdown]

# Now we build the entire Reactor object:

# %%
R = SingleNullReactor(config, build_config, build_tweaks)
R.build()

# %%[markdown]
# When BLUEPRINT carries out the full Reactor design procedure, it designs the
# coil cage in a preliminary fashion with the CoilArchitect object.

# This object is responsible for the coil structures: connections between PF and TF
# coils, and the inter-TF-coil structures and cold mass gravity supports.

# This object can be viewed as shown below (we combine with a plot of the TF coils,
# for clarity)

# %%
f, ax = plt.subplots()
R.TF.plot_xz(ax)
R.ATEC.plot_xz(ax)

# %%[markdown]

# The inter-coil structures are optimised by:
#  * Placing the maximum number of OIS structures in between the ports, if the
#    size is greater than some value (e.g. > 1 m)
#  * Maximising the length of straight OIS (fixed thickness)
#  * Within the geometric constraints of the TF coil case

# %%[markdown]

# Next we're going to use an object which sets up and runs the FE problem
#  * The material properties are set
#  * The geometry is set
#  * The element cross-sectional properties are set
#  * Model boundary conditions are set
#  * Cyclic symmetry is used

# We need to give it the load cases for the reference equilibria: Breakdown,
# start-of-flattop (SOF), and end-of-flattop (EOF).

# The loads added are:
#  * Bursting forces
#  * Toppling forces
#  * PF and CS vertical loads
#  * Gravity

# %%
# First, we need to gather all of the Equilibrium objects for which we want to calculate
# the TF cage response.

all_equilibria = [snapshot.eq for snapshot in R.EQ.snapshots.values()]
SS = StructuralSolver(R.ATEC, R.TF.cage, all_equilibria, material_cache)

# You can take a look at the FE model (without loads)
SS.model.plot()

# %%[markdown]
# Now we solve the FE problem for 3 different load cases
# You can look at them individually like this:

# %%
results = SS.solve()

# Let's look at the result for the end of flat-top:
results[2].plot(deflection=True)

# %%
