# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Different ways of running openmc, written as context managers so that we clean up the
files created by openmc.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, Sequence, Union

import openmc

from bluemira.base.look_and_feel import bluemira_debug
from bluemira.base.tools import _timing
from bluemira.neutronics.neutronics_axisymmetric import (
    create_parametric_plasma_source,
    create_ring_source,
)
from bluemira.neutronics.tallying import _create_tallies_from_filters, filter_new_cells

if TYPE_CHECKING:
    from bluemira.neutronics.make_materials import MaterialsLibrary
    from bluemira.neutronics.make_pre_cell import BlanketCellArray
    from bluemira.neutronics.params import (
        OpenMCSimulationRuntimeParameters,
        PlasmaSourceParametersPPS,
    )


# TODO rewrite this whole module as coarntext managers
class RunMode:
    """
    Generic run method.
    Usage:
    with ChildClassOfRunMode(cross_section_xml, cells, material_lib) as run_mode:
        run_mode.setup(...)
        run_mode.run()
    """

    run_mode_str = ""

    def __init__(
        self,
        cross_section_xml: Union[Path, str],
        cells: Iterable[openmc.Cell],  # TODO: turn this into two arguments:
        # blanket_cell_array and divertor_cell_array
        material_lib: MaterialsLibrary,
    ):
        """Basic set-up to openmc applicable to all run modes."""
        self.files_created = set()
        if type(self) == RunMode:
            raise TypeError(
                "RunMode is a baseclass that is not meant to be initialized!"
            )
        try:
            from openmc.config import config  # noqa: PLC0415

            config["cross_sections"] = cross_section_xml

        except ModuleNotFoundError:
            # Not new enought openmc
            import os  # noqa: PLC0415

            os.environ["OPENMC_CROSS_SECTIONS"] = str(cross_section_xml)

        self.settings = openmc.Settings(
            run_mode=self.run_mode_str, output={"summary": False}
        )
        self.universe = openmc.Universe(cells=cells)
        self.cells = cells
        self.geometry = openmc.Geometry(self.universe)
        self.material_lib = material_lib

    def _set_tallies(
        self, blanket_cell_array: BlanketCellArray, bodge_material_dict: Dict
    ):
        filter_list = filter_new_cells(bodge_material_dict, blanket_cell_array)
        _create_tallies_from_filters(*filter_list)
        self.files_created.add("tallies.xml")

    def __enter__(self):  # noqa: D105
        return self

    def _run_setup(self) -> None:
        """
        Set up basic xml files used in every simulation, i.e. settings, geometry and
        materials.
        """
        self.settings.export_to_xml()
        self.files_created.add("settings.xml")
        self.geometry.export_to_xml()
        self.files_created.add("geometry.xml")
        self.material_lib.export()
        self.files_created.add("materials.xml")

    def __exit__(self, exception_type, exception_value, traceback):
        """Remove files generated during the run (mainly .xml files.)"""
        base_path = Path.cwd()
        for file_name in self.files_created:
            if (f := Path(base_path, file_name)).exists():
                f.unlink()
                bluemira_debug(f"Removed file {f}.")

    def run(self, *setup_args, **setup_kwargs):
        """A generic run method that does both setup and call the openmc executable."""
        self._run_setup(*setup_args, **setup_kwargs)
        self._run_openmc_executable(output=True)

    @staticmethod
    def _run_openmc_executable(*args, output=False, **kwargs) -> None:
        """Complete the run"""
        _timing(openmc.run, "Executed in", "Running OpenMC", debug_info_str=False)(
            *args, output=output, **kwargs
        )


class Plotting(RunMode):
    """Plotting only"""

    run_mode_str = "plot"

    def _run_setup(self, plot_widths: Sequence[float], pixel_per_meter: float) -> None:
        """Set up the plot parameters"""
        super()._run_setup()
        self.plot = openmc.Plot()
        self.plot.basis = "xz"
        self.plot.pixels = [
            int(plot_widths[0] * pixel_per_meter),
            int(plot_widths[1] * pixel_per_meter),
        ]
        self.plot.width = plot_widths
        self.plot_list = openmc.Plots([self.plot])

        self.plot_list.export_to_xml()
        self.files_created.add("plots.xml")


class SourceSimulation(RunMode):
    """Generic base class for running a fixed source."""

    run_mode_str = "fixed source"

    @staticmethod
    def make_source(source_parameters: PlasmaSourceParametersPPS) -> openmc.Source:  # noqa: ARG004
        """Abstract base class method that returns a dummy openmc source."""
        return openmc.Source(...)

    def _run_setup(
        self,
        source_parameters: PlasmaSourceParametersPPS,
        runtime_variables: OpenMCSimulationRuntimeParameters,
        blanket_cell_array: BlanketCellArray,
        bodge_material_dict: Dict,
    ) -> None:
        """
        Break open the :class:`~bluemira.neutronics.params.PlasmaSourceParametersPPS`
        and the :class:`~bluemira.neutronics.params.OpenMCSimulationRuntimeParameters`
        to enter the corresponding values into openmc.Settings
        """
        self.settings.particles = runtime_variables.particles
        self.settings.source = self.make_source(source_parameters)
        self.settings.batches = runtime_variables.batches
        self.settings.photon_transport = runtime_variables.photon_transport
        self.settings.electron_treatment = runtime_variables.electron_treatment
        self._set_tallies(blanket_cell_array, bodge_material_dict)
        super()._run_setup()
        self.files_created.add("summary.h5")
        self.files_created.add("statepoint.1.h5")


class PlasmaSourceSimulation(SourceSimulation):
    """Run with our standard pps_isotropic source"""

    @staticmethod
    def make_source(source_parameters: PlasmaSourceParametersPPS) -> openmc.Source:
        """Make a plasma source"""
        return create_parametric_plasma_source(
            # tokamak geometry
            major_r=source_parameters.major_radius,
            minor_r=source_parameters.minor_radius,
            elongation=source_parameters.elongation,
            triangularity=source_parameters.triangularity,
            # plasma geometry
            peaking_factor=source_parameters.peaking_factor,
            temperature=source_parameters.temperature,
            radial_shift=source_parameters.shaf_shift,
            vertical_shift=source_parameters.vertical_shift,
            # plasma type
            mode="DT",
        )


class RingSourceSimulation(SourceSimulation):
    """Run with a simple annular (ring) source"""

    @staticmethod
    def make_source(source_parameters: PlasmaSourceParametersPPS) -> openmc.Source:
        """Create the ring source"""
        return create_ring_source(
            source_parameters.major_radius, source_parameters.shaf_shift
        )


class VolumeCalculation(RunMode):
    """Run Monte Carlo to get the volume"""

    run_mode_str = "volume"

    def _run_setup(
        self,
        num_particles: int,
        min_xyz: Sequence[float],
        max_xyz: Sequence[float],
        blanket_cell_array: BlanketCellArray,
        bodge_material_dict: Dict,
    ) -> None:
        """Set up openmc for volume calculation"""
        self.settings.volume_calculations = openmc.VolumeCalculation(
            self.cells,
            num_particles,
            min_xyz,
            max_xyz,
        )
        self._set_tallies(blanket_cell_array, bodge_material_dict)
        super()._run_setup()
        self.files_created.add("summary.h5")
        self.files_created.add("statepoint.1.h5")
