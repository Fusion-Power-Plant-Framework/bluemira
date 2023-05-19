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

import bluemira.codes.process as process
from bluemira.base.file import get_bluemira_path
from bluemira.equilibria import Equilibrium
from bluemira.base.parameter_frame import ParameterFrame
from bluemira.geometry.coordinates import Coordinates
from bluemira.radiation_transport.radiation_profile import (
    RadiationSolver, 
    linear_interpolator, 
    interpolated_field_values, 
    filtering_in_or_out, 
    pfr_filter,
    grid_interpolator,
    build_wall_detectors,
    detect_radiation,
    plot_radiation_loads,
)
from bluemira.radiation_transport.flux_surfaces_maker import FluxSurfaceMaker

# CHERAB imports
from cherab.core.math import AxisymmetricMapper
from cherab.tools.emitters import RadiationFunction
from raysect.core import translate
from raysect.optical import World
from raysect.optical.material import VolumeTransform
from raysect.primitive import Cylinder

# %% [markdown]
# # Double Null radiation
#
# First we load an up equilibrium

# %%
read_path = get_bluemira_path("equilibria", subfolder="data")
eq_name = "EU-DEMO_EOF.json"
eq_name = os.path.join(read_path, eq_name)
eq = Equilibrium.from_eqdsk(eq_name)


# %% [markdown]
#
# Now we load a first wall geometry, so that the solver can determine where the flux
# surfaces intersect the first wall.

# %%
read_path = get_bluemira_path("radiation_transport/test_data", subfolder="tests")
fw_name = "first_wall.json"
fw_name = os.path.join(read_path, fw_name)
fw_shape = Coordinates.from_json(fw_name)

# %% [markdown]
#
# Then we define some input `Parameter`s for the solver.

# %%
params = {
    "rho_ped_n": 0.94,
    "n_e_0": 21.93e19,
    "n_e_ped": 8.117e19,
    "n_e_sep": 1.623e19,
    "alpha_n": 1.15,
    "rho_ped_t": 0.976,
    "T_e_0": 21.442,
    "T_e_ped": 5.059,
    "T_e_sep": 0.16,
    "alpha_t": 1.905,
    "t_beta": 2.0,
    "P_sep": 150,
    "k_0": 2000.0,
    "gamma_sheath": 7.0,
    "eps_cool": 25.0,
    "f_ion_t": 0.01,
    "det_t": 0.0015,
    "lfs_p_fraction": 1,
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

# %% [markdown]
#
# Initialising the `TempFsSolverChargedParticleSolver` and run it.

# %%
flux_surface_solver = FluxSurfaceMaker(equilibrium=eq, dx_mp=0.001)
flux_surface_solver.analyse(first_wall=fw_shape)

# %% [markdown]
#
# Getting impurity data.

# %%

def get_impurity_data(impurities_list: list = ["H", "He"]):
    """
    Function getting the PROCESS impurity data
    """
    # This is a function
    imp_data_getter = process.Solver.get_species_data

    impurity_data = {}
    for imp in impurities_list:
        impurity_data[imp] = {
            "T_ref": imp_data_getter(imp)[0],
            "L_ref": imp_data_getter(imp)[1],
        }

    return impurity_data

def create_radiation_source(
    eq: Equilibrium,
    params: ParameterFrame,
    impurity_content_core: dict,
    impurity_data_core: dict,
    impurity_content_sol: dict,
    impurity_data_sol: dict,
    only_source=True,
):

    # Make the radiation source
    rad_solver = RadiationSolver(
        eq=eq,
        flux_surf_solver=flux_surface_solver,
        params=params,
        impurity_content_core=impurity_content_core,
        impurity_data_core=impurity_data_core,
        impurity_content_sol=impurity_content_sol,
        impurity_data_sol=impurity_data_sol,
    )

    if only_source:

        rad_solver.analyse(fw_shape)
        rad_solver.rad_map(fw_shape)
        # Call plot from the Solver
        rad_solver.plot()
        # Core Radiation distribution
        rad_solver.core_rad.plot_radiation_distribution()
        # Core Radiation profile at the midplane
        rad_solver.core_rad.build_mp_radiation_profile()
        rad_solver.core_rad.plot_mp_radiation_profile()
        # Plot Radiative loss function
        rad_solver.core_rad.plot_lz_vs_tref()

        # SOL Radiation Distribution
        rad_solver.sol_rad.plot_poloidal_radiation_distribution(fw_shape)
        # Individual plots for temperature and density from the mid-plane to the target
        # LFS
        rad_solver.sol_rad.poloidal_distribution_plot(
            rad_solver.sol_rad.flux_tubes_lfs,
            rad_solver.sol_rad.t_and_n_pol["lfs"][0],
        )
        rad_solver.sol_rad.poloidal_distribution_plot(
            rad_solver.sol_rad.flux_tubes_lfs,
            rad_solver.sol_rad.t_and_n_pol["lfs"][1],
            temperature=False,
        )
        # HFS
        rad_solver.sol_rad.poloidal_distribution_plot(
            rad_solver.sol_rad.flux_tubes_hfs,
            rad_solver.sol_rad.t_and_n_pol["hfs"][0],
        )
        rad_solver.sol_rad.poloidal_distribution_plot(
            rad_solver.sol_rad.flux_tubes_hfs,
            rad_solver.sol_rad.t_and_n_pol["hfs"][1],
            temperature=False,
        )

        # Temperature VS Density for the first open flux surface
        first_sol_fs_lfs = rad_solver.sol_rad.flux_tubes_lfs[0]
        t_distribution = rad_solver.sol_rad.t_and_n_pol["lfs"][0][0]
        n_distribution = rad_solver.sol_rad.t_and_n_pol["lfs"][1][0]
        rad_solver.sol_rad.plot_t_vs_n(first_sol_fs_lfs, t_distribution, n_distribution)

        first_sol_fs_hfs = rad_solver.sol_rad.flux_tubes_hfs[0]
        t_distribution = rad_solver.sol_rad.t_and_n_pol["hfs"][0][0]
        n_distribution = rad_solver.sol_rad.t_and_n_pol["hfs"][1][0]
        rad_solver.sol_rad.plot_t_vs_n(first_sol_fs_hfs, t_distribution, n_distribution)

        plt.show()

    else:
        rad_solver.analyse(firstwall_geom=fw_shape)
        rad_solver.rad_map(fw_shape)
        # Core and SOL source: coordinates and radiation values
        x_core = rad_solver.core_rad.x_tot
        z_core = rad_solver.core_rad.z_tot
        rad_core = rad_solver.core_rad.rad_tot
        x_sol = rad_solver.sol_rad.x_tot
        z_sol = rad_solver.sol_rad.z_tot
        rad_sol = rad_solver.sol_rad.rad_tot

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
            rad_solver.sol_rad.separatrix, rad_solver.sol_rad.points["x_point"]["z_low"]
        )

        pfr_down_filter = filtering_in_or_out(pfr_x_down, pfr_z_down, False)

        # Fetch lcfs
        lcfs = rad_solver.lcfs
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
                        * (core_filter_out(point) * 1.0)
                    )

        func = grid_interpolator(x_sol, z_sol, rad_sol_grid)

        return func
    
def fw_radiation(rad_source, plot=True):

    rad_3d = AxisymmetricMapper(rad_source)
    ray_stepsize = 1.0  # 2.0e-4
    emitter = VolumeTransform(
        RadiationFunction(rad_3d, step=ray_stepsize * 0.1),
        translate(0, 0, np.max(fw_shape.z)),
    )
    world = World()
    Cylinder(
        np.max(fw_shape.x),
        2.0 * np.max(fw_shape.z),
        transform=translate(0, 0, np.max(fw_shape.z)),
        parent=world,
        material=emitter,
    )
    max_wall_len = 10.0e-2
    X_WIDTH = 0.01
    wall_detectors = build_wall_detectors(fw_shape.x, fw_shape.z, max_wall_len, X_WIDTH)
    wall_loads = detect_radiation(wall_detectors, 500, world)

    if plot:
        plot_radiation_loads(
            rad_3d, wall_detectors, wall_loads, "SOL & divertor radiation loads", fw_shape
        )

    return wall_loads

def main(only_source=True):

    # Get the core impurity fractions
    f_impurities_core = config["f_imp_core"]
    f_impurities_sol = config["f_imp_sol"]
    impurities_list_core = [imp for imp in f_impurities_core]
    impurities_list_sol = [imp for imp in f_impurities_sol]

    # Get the impurities data
    impurity_data_core = get_impurity_data(impurities_list=impurities_list_core)
    impurity_data_sol = get_impurity_data(impurities_list=impurities_list_sol)

    # Make the radiation sources
    rad_source_func = create_radiation_source(
        eq=eq,
        params=params,
        impurity_content_core=f_impurities_core,
        impurity_data_core=impurity_data_core,
        impurity_content_sol=f_impurities_sol,
        impurity_data_sol=impurity_data_sol,
        only_source=only_source,
    )

    if only_source is False:
        # Calculate radiation of FW points
        fw_radiation(rad_source=rad_source_func)

if __name__ == "__main__":
    main()