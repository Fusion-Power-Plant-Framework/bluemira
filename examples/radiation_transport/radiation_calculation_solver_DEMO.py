# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Example core and scraper-off layer radiation source calculation
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.constants import raw_uc
from bluemira.base.file import get_bluemira_path
from bluemira.equilibria import Equilibrium
from bluemira.geometry.coordinates import Coordinates
from bluemira.radiation_transport.midplane_temperature_density import MidplaneProfiles
from bluemira.radiation_transport.radiation_profile import (
    RadiationSource,
    interpolated_field_values,
    linear_interpolator,
)
from bluemira.radiation_transport.radiation_tools import (
    FirstWallRadiationSolver,
    filtering_in_or_out,
    grid_interpolator,
    pfr_filter,
)

# %% [markdown]
# # Double Null radiation
#
# First we load an equilibrium.

# %%

SINGLE_NULL = False

if SINGLE_NULL:
    eq_name = "EU-DEMO_EOF.json"
    fw_name = "first_wall.json"
    sep_corrector = 5e-2
    lfs_p_fraction = 1
    tungsten_fraction = 1e-4
else:
    eq_name = "DN-DEMO_eqref.json"
    fw_name = "DN_fw_shape.json"
    sep_corrector = 5e-3
    lfs_p_fraction = 0.9
    tungsten_fraction = 1e-5


eq = Equilibrium.from_eqdsk(
    Path(get_bluemira_path("equilibria", subfolder="data"), eq_name)
)


# %% [markdown]
#
# Now we load a first wall geometry.

# %%
fw_shape = Coordinates.from_json(
    Path(get_bluemira_path("radiation_transport/test_data", subfolder="tests"), fw_name)
)

# %% [markdown]
#
# Then we define some input `Parameter`s for the solver.

# %%
params = {
    "sep_corrector": {"value": sep_corrector, "unit": "dimensionless"},
    "alpha_n": {"value": 1.15, "unit": "dimensionless"},
    "alpha_t": {"value": 1.905, "unit": "dimensionless"},
    "det_t": {"value": 0.0015, "unit": "keV"},
    "eps_cool": {"value": 25.0, "unit": "eV"},
    "f_ion_t": {"value": 0.01, "unit": "keV"},
    "fw_lambda_q_near_omp": {"value": 0.002, "unit": "m"},
    "fw_lambda_q_far_omp": {"value": 0.1, "unit": "m"},
    "fw_lambda_q_near_imp": {"value": 0.002, "unit": "m"},
    "fw_lambda_q_far_imp": {"value": 0.1, "unit": "m"},
    "gamma_sheath": {"value": 7.0, "unit": "dimensionless"},
    "k_0": {"value": 2000.0, "unit": "dimensionless"},
    "lfs_p_fraction": {"value": lfs_p_fraction, "unit": "dimensionless"},
    "n_e_0": {"value": 21.93e19, "unit": "1/m^3"},
    "n_e_ped": {"value": 8.117e19, "unit": "1/m^3"},
    "n_e_sep": {"value": 1.623e19, "unit": "1/m^3"},
    "P_sep": {"value": 150, "unit": "MW"},
    "rho_ped_n": {"value": 0.94, "unit": "dimensionless"},
    "rho_ped_t": {"value": 0.976, "unit": "dimensionless"},
    "n_points_core_95": {"value": 30, "unit": "dimensionless"},
    "n_points_core_99": {"value": 15, "unit": "dimensionless"},
    "n_points_mantle": {"value": 10, "unit": "dimensionless"},
    "t_beta": {"value": 2.0, "unit": "dimensionless"},
    "T_e_0": {"value": 21.442, "unit": "keV"},
    "T_e_ped": {"value": 5.059, "unit": "keV"},
    "T_e_sep": {"value": 0.16, "unit": "keV"},
    "theta_inner_target": {"value": 5.0, "unit": "deg"},
    "theta_outer_target": {"value": 5.0, "unit": "deg"},
}

# if SINGLE_NULL:
#     params["f_p_sol_near"] = {"value": 0.65, "unit": "dimensionless"}

# %%
config = {
    "f_imp_core": {"H": 1e-1, "He": 1e-2, "Xe": 1e-4, "W": tungsten_fraction},
    "f_imp_sol": {"H": 0, "He": 0, "Ar": 0.003, "Xe": 0, "W": 0},
}


# %% [markdown]
#

# Get core midplane profiles
# %%

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
    psi_n=psi_n,
    ne_mp=ne_mp,
    te_mp=te_mp,
    core_impurities=config["f_imp_core"],
    sol_impurities=config["f_imp_sol"],
)
source.analyse(firstwall_geom=fw_shape)
source.rad_map(fw_shape)

# %% [markdown]
#
# Defining whether to run the radiation source only [MW/m^3]
# or to calculate radiation loads on the first wall [MW/m^2].


# %%
def main(only_source=False):  # noqa: D103
    if only_source:
        source.plot()
        plt.show()

    else:
        # Core and SOL source: coordinates and radiation values
        x_core = source.core_rad.x_tot
        z_core = source.core_rad.z_tot
        x_sol = source.sol_rad.x_tot
        z_sol = source.sol_rad.z_tot

        # Coversion required for CHERAB
        # Core and SOL interpolating function
        f_core = linear_interpolator(
            x_core, z_core, raw_uc(source.core_rad.rad_tot, "MW", "W")
        )
        f_sol = linear_interpolator(
            x_sol, z_sol, raw_uc(source.sol_rad.rad_tot, "MW", "W")
        )

        # SOL radiation grid
        x_sol = np.linspace(min(fw_shape.x), max(fw_shape.x), 1000)
        z_sol = np.linspace(min(fw_shape.z), max(fw_shape.z), 1500)
        rad_sol_grid = interpolated_field_values(x_sol, z_sol, f_sol)

        # Filter in/out zones
        wall_filter = filtering_in_or_out(fw_shape.x, fw_shape.z)
        pfr_x_down, pfr_z_down = pfr_filter(
            source.sol_rad.separatrix, source.sol_rad.points["x_point"]["z_low"]
        )

        pfr_down_filter = filtering_in_or_out(
            pfr_x_down, pfr_z_down, include_points=False
        )

        if not SINGLE_NULL:
            pfr_x_up, pfr_z_up = pfr_filter(
                source.sol_rad.separatrix, source.sol_rad.points["x_point"]["z_up"]
            )
            pfr_up_filter = filtering_in_or_out(pfr_x_up, pfr_z_up, include_points=False)

        # Fetch lcfs
        lcfs = source.lcfs
        core_filter_in = filtering_in_or_out(lcfs.x, lcfs.z)
        core_filter_out = filtering_in_or_out(lcfs.x, lcfs.z, include_points=False)
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
