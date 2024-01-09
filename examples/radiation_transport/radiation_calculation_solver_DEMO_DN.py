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
import numpy as np

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria import Equilibrium
from bluemira.geometry.coordinates import Coordinates
from bluemira.radiation_transport.midplane_temperature_density import MidplaneProfiles
from bluemira.radiation_transport.radiation_profile import (
    RadiationSource, 
    linear_interpolator, 
    interpolated_field_values, 
)
from bluemira.radiation_transport.radiation_tools import (
    filtering_in_or_out, 
    pfr_filter,
    grid_interpolator,
    FirstWallRadiationSolver,
)

# %% [markdown]
# # Double Null radiation
#
# First we load an equilibrium.

# %%
read_path = get_bluemira_path("equilibria", subfolder="data")
eq_name = "DN-DEMO_eqref.json"
eq_name = os.path.join(read_path, eq_name)
eq = Equilibrium.from_eqdsk(eq_name)


# %% [markdown]
#
# Now we load a first wall geometry.

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
    "rho_ped_n": {
        "value": 0.94,
        "unit": "dimensionless"
    },
    "n_e_0": {
        "value": 21.93e19,
        "unit": "1/m^3"
    },
    "n_e_ped": {
        "value": 8.117e19,
        "unit": "1/m^3"
    },
    "n_e_sep": {
        "value": 1.623e19,
        "unit": "1/m^3"
    },
    "alpha_n": {
        "value":1.15,
        "unit": "dimensionless"
    },
    "rho_ped_t": {
        "value": 0.976,
        "unit": "dimensionless"
    },
    "T_e_0": {
        "value": 21.442,
        "unit": "keV"
    },
    "T_e_ped": {
        "value": 5.059,
        "unit": "keV"
    },
    "T_e_sep": {
        "value": 0.16,
        "unit": "keV"
    },
    "alpha_t": {
        "value": 1.905,
        "unit": "dimensionless"
    },
    "t_beta": {
        "value": 2.0,
        "unit": "dimensionless"
    },
    "P_sep": {
        "value": 150,
        "unit": "MW"
    },
    "k_0": {
        "value": 2000.0,
        "unit": "dimensionless"
    },
    "gamma_sheath": {
        "value": 7.0,
        "unit": "dimensionless"
    },
    "eps_cool": {
        "value": 25.0,
        "unit": "eV"
    },
    "f_ion_t": {
        "value": 0.01,
        "unit": "keV"
    },
    "det_t": {
        "value": 0.0015,
        "unit": "keV"
    },
    "lfs_p_fraction": {
        "value": 0.9,
        "unit": "dimensionless"
    },
    "theta_outer_target": {
        "value": 5.0,
        "unit": "deg"
    },
    "theta_inner_target": {
        "value": 5.0,
        "unit": "deg"
    },
    "fw_lambda_q_near_omp": {
        "value": 0.002,
        "unit": "m"
    },
    "fw_lambda_q_far_omp": {
        "value": 0.1,
        "unit": "m"
    },
    "fw_lambda_q_near_imp": {
        "value": 0.002,
        "unit": "m"
    },
    "fw_lambda_q_far_imp": {
        "value": 0.1,
        "unit": "m"
    },
}

# %%
config = {
            "f_imp_core" : {
                "H": 1e-1,
                "He": 1e-2,
                "Xe": 1e-4,
                "W": 1e-5
            },
            "f_imp_sol" : {
                "H": 0,
                "He": 0,
                "Ar": 0.003,
                "Xe": 0,
                "W": 0
            }
        }


# %% [markdown]
#
# Get the core impurity fractions
f_impurities_core = config["f_imp_core"]
f_impurities_sol = config["f_imp_sol"]

# Get core midplane profiles
Profiles = MidplaneProfiles(params=params)
psi_n = Profiles.psi_n
ne_mp = Profiles.ne_mp
te_mp = Profiles.te_mp

# %% [markdown]
#
# Initialising the `RadiationSolver` and run it.

# %%
source = RadiationSource(
        eq=eq,
        firstwall_shape=fw_shape,
        params=params,
        psi_n = psi_n,
        ne_mp = ne_mp,
        te_mp = te_mp,
        core_impurities=f_impurities_core,
        sol_impurities=f_impurities_sol,
    )
source.analyse(firstwall_geom=fw_shape)
source.rad_map(fw_shape)

# %% [markdown]
#
# Defining whether to run the radiation source only [MW/m^3]
# or to calculate radiation loads on the first wall [MW/m^2].

# %%
def main(only_source=False):

    if only_source:
        source.plot()
        plt.show()
    
    else:
        # Core and SOL source: coordinates and radiation values
        x_core = source.core_rad.x_tot
        z_core = source.core_rad.z_tot
        rad_core = source.core_rad.rad_tot
        x_sol = source.sol_rad.x_tot
        z_sol = source.sol_rad.z_tot
        rad_sol = source.sol_rad.rad_tot

        # Coversion required for CHERAB
        rad_core = rad_core * 1.0e6
        rad_sol = rad_sol * 1.0e6

        # Core and SOL interpolating function
        f_core = linear_interpolator(x_core, z_core, rad_core)
        f_sol = linear_interpolator(x_sol, z_sol, rad_sol)

        # SOL radiation grid
        x_sol = np.linspace(min(fw_shape.x), max(fw_shape.x), 1000)
        z_sol = np.linspace(min(fw_shape.z), max(fw_shape.z), 1500)
        rad_sol_grid = interpolated_field_values(x_sol, z_sol, f_sol)

        # Filter in/out zones
        wall_filter = filtering_in_or_out(fw_shape.x, fw_shape.z)
        pfr_x_down, pfr_z_down = pfr_filter(
            source.sol_rad.separatrix, source.sol_rad.points["x_point"]["z_low"]
        )
        pfr_x_up, pfr_z_up = pfr_filter(
            source.sol_rad.separatrix, source.sol_rad.points["x_point"]["z_up"]
        )
        pfr_down_filter = filtering_in_or_out(pfr_x_down, pfr_z_down, False)
        pfr_up_filter = filtering_in_or_out(pfr_x_up, pfr_z_up, False)

        # Fetch lcfs
        lcfs = source.lcfs
        core_filter_in = filtering_in_or_out(lcfs.x, lcfs.z)
        core_filter_out = filtering_in_or_out(lcfs.x, lcfs.z, False)
        for i in range(len(x_sol)):
            for j in range(len(z_sol)):
                point = x_sol[i], z_sol[j]
                if core_filter_in(point):
                    rad_sol_grid[j, i] = interpolated_field_values(
                        x_sol[i], z_sol[j], f_core
                    )
                else:
                    rad_sol_grid[j, i] = (
                        rad_sol_grid[j, i]
                        * (wall_filter(point) * 1.0)
                        * (pfr_down_filter(point) * 1.0)
                        * (pfr_up_filter(point) * 1.0)
                        * (core_filter_out(point) * 1.0)
                    )

        func = grid_interpolator(x_sol, z_sol, rad_sol_grid)
        # Calculate radiation of FW points
        solver = FirstWallRadiationSolver(source_func=func, firstwall_shape=fw_shape)
        solver.solve()

if __name__ == "__main__":
    main()