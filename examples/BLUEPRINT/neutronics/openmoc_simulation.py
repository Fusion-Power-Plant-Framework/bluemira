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
Example of an OpenMOC simulation using an imported BLUEPRINT geometry
"""

import os
import warnings

# Common imports
from datetime import datetime

import numpy as np

# Import OpenMOC and it's plotter
import openmoc
from matplotlib.cbook import mplDeprecation
from openmoc import plotter as plotter

# Some BLUEPRINT imports
from bluemira.base.look_and_feel import bluemira_print
from BLUEPRINT.neutronics.openmoc_geometry_tools import (
    get_source_fsr_map,
    populate_source_cells,
)
from BLUEPRINT.neutronics.openmoc_plotting_tools import plot_spatial_custom_fluxes
from BLUEPRINT.neutronics.plasma_source import PlasmaSource

# Get the geometry from the example
from examples.neutronics.openmoc_geometry import (
    geometry,
    plasma,
    plasma_cells,
    plasma_sections,
)

####################################
# Set the main simulation parameters
####################################

start = datetime.now()

options = openmoc.options.Options()

# Avoid OpenMOC grabbing all available threads
num_threads = max(options.num_omp_threads - 1, 1)

# Set other options
azim_spacing = 0.25
num_azim = 128
num_polar = options.num_polar
tolerance = options.tolerance
max_iters = options.max_iters


###################################################
# Create the TrackGenerator and generate the tracks
###################################################

bluemira_print("Initializing the track generator")

track_generator = openmoc.TrackGenerator(geometry, num_azim, azim_spacing)
track_generator.setNumThreads(num_threads)
track_generator.setZCoord(0.1)
track_generator.generateTracks()

######################
# Setup the simulation
######################

bluemira_print("Setting up the simulation")

solver = openmoc.CPULSSolver(track_generator)
solver.setNumThreads(num_threads)
solver.setConvergenceThreshold(tolerance)


##########################
# Create the plasma source
##########################

bluemira_print("Creating the plasma source")

neutron_source = plasma.export_neutron_source()
plasma_source = PlasmaSource(**neutron_source)

num_fsr = geometry.getNumFSRs()

# Build a cache of FSRs corresponding to plasma cells
plasma_fsr_map = get_source_fsr_map(geometry, plasma_cells)

populate_source_cells(geometry, solver, plasma_source, plasma_cells, plasma_sections)

bluemira_print("Created the plasma source")

####################
# Run the simulation
####################

solver.computeSource(max_iters)
solver.printTimerReport()

end = datetime.now()

bluemira_print(f"Simulation completed in {end - start}")

#######################
# Generate some plots
#######################

bluemira_print("Plotting data")

start = datetime.now()

with warnings.catch_warnings():
    # OpenMOC raises some matplotlib deprecation warnings
    warnings.filterwarnings("ignore", category=mplDeprecation)

    # Use OpenMOC plotting routines
    plotter.plot_tracks(track_generator)
    plotter.plot_materials(geometry, gridsize=500, plane="xy", offset=0.0)
    plotter.plot_cells(geometry, gridsize=500, plane="xy", offset=0.0)
    plotter.plot_flat_source_regions(geometry, gridsize=500, plane="xy", offset=0.0)

    log10_fluxes = np.log10(openmoc.process.get_scalar_fluxes(solver))
    plot_spatial_custom_fluxes(
        solver,
        log10_fluxes,
        energy_groups=[1, 2, 3, 4, 5, 6, 7],
        plane="xy",
        offset=0.0,
        suptitle="Log10 Fluxes (Group {0})",
        filename="fsr-log10-flux-group-{0}-z-{1}",
    )

    noplasma_log10_fluxes = np.array(
        [
            log10_fluxes[idx]
            if idx not in plasma_fsr_map.values()
            else [np.nan] * len(log10_fluxes[idx])
            for idx in range(num_fsr)
        ],
        np.float64,
    )
    plot_spatial_custom_fluxes(
        solver,
        noplasma_log10_fluxes,
        energy_groups=[1],
        plane="xy",
        offset=0.0,
        suptitle="Log10 Fluxes No Plasma (Group {0})",
        filename="fsr-log10-flux-noplasma-group-{0}-z-{1}",
    )

    plasma_log10_fluxes = np.array(
        [
            log10_fluxes[idx]
            if idx in plasma_fsr_map.values()
            else [np.nan] * len(log10_fluxes[idx])
            for idx in range(num_fsr)
        ],
        np.float64,
    )
    plot_spatial_custom_fluxes(
        solver,
        plasma_log10_fluxes,
        energy_groups=[1],
        plane="xy",
        offset=0.0,
        suptitle="Log10 Fluxes Only Plasma (Group {0})",
        filename="fsr-log10-flux-onlyplasma-group-{0}-z-{1}",
    )

# Plots are generated in a plots sub-directory of the current working directory
bluemira_print(f"Plots available in {os.getcwd()}/plots")

end = datetime.now()

bluemira_print(f"Plotting completed in {end - start}")

bluemira_print("Finished")
