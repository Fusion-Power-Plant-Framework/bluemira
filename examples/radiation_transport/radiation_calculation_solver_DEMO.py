# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
Example core and scraper-off layer radiation source calculation
"""

# %%
import os

import matplotlib.pyplot as plt

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria import Equilibrium
from bluemira.geometry.coordinates import Coordinates
from bluemira.radiation_transport.advective_transport import ChargedParticleSolver
from bluemira.radiation_transport.flux_surfaces_maker import FluxSuraceSolver

# %% [markdown]
# # Double Null radiation
#
# First we load an up equilibrium

# %%
read_path = get_bluemira_path("equilibria", subfolder="data")
eq_name = "DN-DEMO_eqref.json"
eq_name = os.path.join(read_path, eq_name)
eq = Equilibrium.from_eqdsk(eq_name)


# %% [markdown]
#
# Now we load a first wall geometry, so that the solver can determine where the flux
# surfaces intersect the first wall.

# %%
read_path = get_bluemira_path("radiation_transport/test_data", subfolder="tests")
fw_name = "DN_fw_shape.json"
fw_name = os.path.join(read_path, fw_name)
fw_shape = Coordinates.from_json(fw_name)

# %% [markdown]
#
# Then we define some input `Parameter`s for the solver.

# %%
params = {
    "rho_ped_n": 0.94,
    "n_el_0": 21.93e19,
    "n_el_ped": 8.117e19,
    "n_e_sep": 1.623e19,
    "alpha_n": 1.15,
    "rho_ped_t": 0.976,
    "T_el_0": 21.442,
    "T_e_ped": 5.059,
    "T_el_sep": 0.16,
    "alpha_t": 1.905,
    "t_beta": 2.0,
    "P_sep": 150,
    "k_0": 2000.0,
    "gamma_sheath": 7.0,
    "eps_cool": 25.0,
    "f_ion_t": 0.01,
    "det_t": 0.0015,
    "lfs_p_fraction": 0.9,
    "div_p_sharing": 0.5,
    "theta_outer_target": 5.0,
    "theta_inner_target": 5.0,
    "f_p_sol_near": 0.65,
    "fw_lambda_q_near_omp": 0.003,
    "fw_lambda_q_far_omp": 0.1,
    "fw_lambda_q_near_imp": 0.003,
    "fw_lambda_q_far_imp": 0.1,
}

# %%
config = {
            "core plasma" : {
                "f_imp_core" : {
                    "H": 0.7,
                    "He": 0.05,
                    "Xe": 0.25e-3,
                    "W": 0.1e-3
                },
                "f_imp_sol" : {
                    "H": 0,
                    "He": 0,
                    "Ar": 0.1e-2,
                    "Xe": 0,
                    "W": 0
                }
            }
        }

# %% [markdown]
#
# Initialising the `TempFsSolverChargedParticleSolver` and run it.

# %%
flux_surface_solver = FluxSuraceSolver(equilibrium=eq, dx_mp=0.001)
flux_surface_solver.analyse(first_wall=fw_shape)