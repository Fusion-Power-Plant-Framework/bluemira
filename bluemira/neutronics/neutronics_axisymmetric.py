# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
[ ]Tests?
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import openmc
from numpy import pi
from pps_isotropic.source import create_parametric_plasma_source

import bluemira.neutronics.constants as neutronics_const
import bluemira.neutronics.make_csg as mcsg
import bluemira.neutronics.make_geometry as mg
import bluemira.neutronics.result_presentation as present
from bluemira.base.constants import raw_uc
from bluemira.base.tools import _timing
from bluemira.neutronics.constants import dt_neutron_energy
from bluemira.neutronics.make_materials import MaterialsLibrary
from bluemira.neutronics.params import (
    BreederTypeParameters,
    OpenMCSimulationRuntimeParameters,
    PlasmaGeometry,
    PlasmaGeometryBase,
    TokamakGeometry,
    TokamakGeometryBase,
    TokamakOperationParameters,
    TokamakOperationParametersBase,
)
from bluemira.neutronics.volume_functions import stochastic_volume_calculation
from bluemira.plasma_physics.reactions import n_DT_reactions

if TYPE_CHECKING:
    from bluemira.geometry.wire import BluemiraWire


def create_ring_source(major_r: float, shaf_shift: float) -> openmc.Source:
    """
    Creating simple line ring source lying on the Z=0 plane,
        at r = major radius + shafranov shift,
        producing 14.1 MeV neutrons with no variation in energy.
    A more accurate source will slightly affect the wall loadings and dpa profiles.

    Parameters
    ----------
    major_r: major radius [m]
    shaf_shift: shafranov shift [m]
    """
    ring_source = openmc.Source()
    source_radii_cm = openmc.stats.Discrete(
        [raw_uc(major_r + shaf_shift, "m", "cm")], [1]
    )
    source_z_values = openmc.stats.Discrete([0], [1])
    source_angles = openmc.stats.Uniform(a=0.0, b=2 * pi)
    ring_source.space = openmc.stats.CylindricalIndependent(
        r=source_radii_cm, phi=source_angles, z=source_z_values, origin=(0.0, 0.0, 0.0)
    )
    ring_source.angle = openmc.stats.Isotropic()
    ring_source.energy = openmc.stats.Discrete(
        [raw_uc(dt_neutron_energy, "J", "eV")], [1]
    )

    return ring_source


def setup_openmc(
    plasma_source: openmc.Source,
    openmc_params: OpenMCSimulationRuntimeParameters,
) -> None:
    """Configure openmc.Settings, so that it's ready for the run() step.

    Parameters
    ----------
    plasma_source: openmc.Source
        Openmc.Source used to emulate the neutron emission of the plasma.
    openmc_params:

    Notes
    -----
    Exports the settings to an xml file.

    We run the simulation with the assumption that temperature = 293K,
    as the nuclear cross-section values are evaluated at this temperature
    """
    try:
        from openmc.config import config  # noqa: PLC0415

        config["cross_sections"] = openmc_params.cross_section_xml

    except ModuleNotFoundError:
        # Not new enought openmc
        import os  # noqa: PLC0415

        os.environ["OPENMC_CROSS_SECTIONS"] = str(openmc_params.cross_section_xml)
    settings = openmc.Settings()

    settings.source = plasma_source
    settings.particles = openmc_params.particles
    settings.batches = openmc_params.batches
    settings.photon_transport = openmc_params.photon_transport
    settings.electron_treatment = openmc_params.electron_treatment
    settings.run_mode = (
        openmc_params.run_mode
        if isinstance(openmc_params.run_mode, str)
        else openmc_params.run_mode.value
    )
    settings.output = {"summary": openmc_params.openmc_write_summary}
    settings.export_to_xml()


def create_and_export_materials(
    breeder_materials: BreederTypeParameters,
) -> MaterialsLibrary:
    """
    Parameters
    ----------
    breeder_materials:
        dataclass containing attributes: 'blanket_type', 'enrichment_fraction_Li6'
    """
    material_lib = MaterialsLibrary.create_from_blanket_type(
        breeder_materials.blanket_type,
        raw_uc(breeder_materials.enrichment_fraction_Li6, "", "%"),
    )
    material_lib.export()
    return material_lib


class TBRHeatingSimulation:
    """
    Contains all the data necessary to run the openmc simulation of the tbr,
    and the relevant pre-and post-processing.
    """

    def __init__(
        self,
        runtime_params: OpenMCSimulationRuntimeParameters,
        operation_variable: TokamakOperationParametersBase,
        breeder_materials: BreederTypeParameters,
        plasma_geometry: PlasmaGeometryBase,
        tokamak_geometry: TokamakGeometryBase,
    ):
        self.runtime_params = runtime_params
        self.operation_variable = TokamakOperationParameters.from_si(operation_variable)
        self.breeder_materials = breeder_materials
        self.plasma_geometry = PlasmaGeometry.from_si(plasma_geometry)
        self.tokamak_geometry = TokamakGeometry.from_si(tokamak_geometry)

        self.cells = None
        self.material_lib = None
        self.universe = None

    def setup(
        self,
        fw_deconstruction,
        divertor_wire: BluemiraWire,
        new_major_radius: float,
        new_aspect_ratio: float,
        new_elong: float,
        plot_geometry: bool = True,
    ) -> None:
        """Plot the geometry and saving them as .png files with hard-coded names.
        Input parameters' units are still in SI;
        but after this step everything should be in cgs as data get parsed to openmc.

        Parameters
        ----------
        blanket_wire: BluemiraWire
            units [m]

        divertor_wire: BluemiraWire
            units [m]

        new_major_radius: [m]
            (new) major radius in SI units,
            separate to the one provided in self.plasma_geometry

        new_aspect_ratio: [dimensionless]
            scalar denoting the aspect ratio of the device (major/minor radius)

        new_elong: [dimensionless]
            (new) elongation variable,
            separate to the one provided in self.plasma_geometry

        plot_geometry: bool
            Should openmc plot the .png files or not.
        """
        material_lib = create_and_export_materials(self.breeder_materials)
        self.material_lib = material_lib
        mg.check_geometry(self.plasma_geometry, self.tokamak_geometry)
        if self.runtime_params.parametric_source:
            source = create_parametric_plasma_source(
                # plasma geometry
                major_r=self.plasma_geometry.cgs.major_r,
                minor_r=self.plasma_geometry.cgs.minor_r,
                elongation=self.plasma_geometry.cgs.elong,
                triangularity=self.plasma_geometry.cgs.triang,
                # plasma operating variable
                peaking_factor=self.operation_variable.plasma_physics_units.peaking_factor,
                temperature=self.operation_variable.plasma_physics_units.temperature,
                radial_shift=self.operation_variable.plasma_physics_units.shaf_shift,
                vertical_shift=self.operation_variable.plasma_physics_units.vertical_shift,
                # plasma type
                mode="DT",
            )
        else:
            source = create_ring_source(
                self.plasma_geometry.major_r, self.operation_variable.shaf_shift
            )

        setup_openmc(source, self.runtime_params)

        # blanket_points, div_points, num_inboard_points = mg.load_fw_points(
        #     self.tokamak_geometry,
        #     fw_deconstruction,
        #     divertor_wire,
        #     raw_uc(new_major_radius, "m", "cm"),
        #     new_aspect_ratio,
        #     new_elong,
        #     True,
        # )

        self.cells, self.universe = mcsg.make_geometry(
            self.tokamak_geometry,
            fw_deconstruction,
            divertor_wire,
            # num_inboard_points,
            self.material_lib,
        )

        # deduce source strength (self.src_rate) from the power of the reactor,
        # by assuming 100% of reactor power comes from DT fusion
        self.src_rate = n_DT_reactions(
            self.operation_variable.plasma_physics_units.reactor_power  # [MW]
        )

        # create_tallies(self.cells, self.material_lib)

        if plot_geometry:
            present.geometry_plotter(self.cells, self.tokamak_geometry)

    @staticmethod
    def run(*args, output=False, **kwargs) -> None:
        """Run the actual openmc simulation."""
        _timing(openmc.run, "Executed in", "Running OpenMC", debug_info_str=False)(
            *args, output=output, **kwargs
        )

    def get_result(self) -> present.OpenMCResult:
        """
        Create a summary object, attach it to self, and then return it.
        """
        if self.universe is None:
            raise RuntimeError(
                "The self.universe variable must first be populated by self.run()!"
            )
        return present.OpenMCResult.from_run(self.universe, self.src_rate)

    def calculate_volume_stochastically(self):
        """
        Using openmc's built-in stochastic volume calculation function to get the volume.
        """
        stochastic_volume_calculation(
            self.plasma_geometry,
            self.tokamak_geometry,
            self.cells,
            self.runtime_params.volume_calc_particles,
        )
