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
TODO:
[ ]Integration into our logging system (print should go through bluemira_print etc.)
[ ]Should we rename `quick_tbr_heating.py` to something else?
____
[ ]Tests?
"""
import openmc
from numpy import pi
from openmc.config import config
from pps_isotropic.source import create_parametric_plasma_source

import bluemira.neutronics.constants as neutronics_const
import bluemira.neutronics.make_geometry as mg
import bluemira.neutronics.result_presentation as present
from bluemira.base.constants import raw_uc
from bluemira.base.tools import _timing
from bluemira.geometry.wire import BluemiraWire
from bluemira.neutronics.make_materials import MaterialsLibrary
from bluemira.neutronics.params import (
    BreederTypeParameters,
    OpenMCSimulationRuntimeParameters,
    TokamakGeometry,
    TokamakGeometryCGS,
    TokamakOperationParameters,
    TokamakOperationParametersPPS,
)
from bluemira.neutronics.tallying import create_tallies
from bluemira.neutronics.volume_functions import stochastic_volume_calculation


def create_ring_source(tokamak_geometry: TokamakGeometry) -> openmc.Source:
    """
    Creating simple ring source.
    A more accurate source will slightly affect the wall loadings and dpa profiles.

    Parameters
    ----------
    tokamak_geometry: TokamakGeometry
        Only the
            - tokamak_geometry.cgs.major_r
            - tokamak_geometry.cgs.shaf_shift
        variables are used in this function.
    """
    ring_source = openmc.Source()
    source_radii = openmc.stats.Discrete(
        [tokamak_geometry.cgs.major_r + tokamak_geometry.cgs.shaf_shift], [1]
    )
    source_z_values = openmc.stats.Discrete([0], [1])
    source_angles = openmc.stats.Uniform(a=0.0, b=2 * pi)
    ring_source.space = openmc.stats.CylindricalIndependent(
        r=source_radii, phi=source_angles, z=source_z_values, origin=(0.0, 0.0, 0.0)
    )
    ring_source.angle = openmc.stats.Isotropic()
    ring_source.energy = openmc.stats.Discrete(
        [raw_uc(neutronics_const.dt_neutron_energy, "J", "eV")], [1]
    )

    return ring_source


def setup_openmc(
    plasma_source: openmc.Source,
    variables: OpenMCSimulationRuntimeParameters,
) -> None:
    """Configure openmc.Settings, so that it's ready for the run() step.

    Parameters
    ----------
    plasma_source: openmc.Source
        Openmc.Source used to emulate the neutron emission of the plasma.

    Notes
    -----
    Exports the settings to an xml file.

    We run the simulation with the assumption that temperature = 293K,
    as the nuclear cross-section values are evaluated at this temperature
    """
    config["cross_sections"] = variables.cross_section_xml
    settings = openmc.Settings()
    settings.source = plasma_source
    settings.particles = variables.particles
    settings.batches = variables.batches
    settings.photon_transport = variables.photon_transport
    settings.electron_treatment = variables.electron_treatment
    settings.run_mode = (
        variables.run_mode
        if isinstance(variables.run_mode, str)
        else variables.run_mode.value
    )
    settings.output = {"summary": variables.openmc_write_summary}
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
        runtime_variables: OpenMCSimulationRuntimeParameters,
        operation_variable: TokamakOperationParameters,
        breeder_materials: BreederTypeParameters,
        tokamak_geometry: TokamakGeometry,
    ):
        self.runtime_variables = runtime_variables
        self.operation_variable = TokamakOperationParametersPPS.from_si(
            operation_variable
        )
        self.breeder_materials = breeder_materials
        self.tokamak_geometry = TokamakGeometryCGS.from_si(tokamak_geometry)

        self.cells = None
        self.material_lib = None
        self.universe = None

    def setup(
        self,
        blanket_wire: BluemiraWire,
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
        blanket_wire:
            units: [m]
        divertor_wire:
            units: [m]
        new_major_radius:
            (new) major radius in SI units,
                separate to the one provided in TokamakGeometry
            unit: [m]
        new_aspect_ratio:
            scalar denoting the aspect ratio of the device (major/minor radius)
            unit: [dimensionless]
        new_elong:
            (new) elongation variable, separate to the one provided in TokamakGeometry
            unit: [dimensionless]
        plot_geometry:
            Should openmc plot the .png files or not.
        """
        material_lib = create_and_export_materials(self.breeder_materials)
        self.material_lib = material_lib
        mg.check_geometry(self.tokamak_geometry)
        if self.runtime_variables.parametric_source:
            source = create_parametric_plasma_source(
                # tokamak geometry
                major_r=self.tokamak_geometry.cgs.major_r,
                minor_r=self.tokamak_geometry.cgs.minor_r,
                elongation=self.tokamak_geometry.cgs.elong,
                triangulation=self.tokamak_geometry.cgs.triang,
                # plasma geometry
                peaking_factor=self.operation_variable.plasma_physics_units.peaking_factor,
                temperature=self.operation_variable.plasma_physics_units.temperature,
                radial_shift=self.operation_variable.plasma_physics_units.shaf_shift,
                vertical_shift=self.operation_variable.plasma_physics_units.vertical_shift,
            )
        else:
            source = create_ring_source(self.tokamak_geometry)

        setup_openmc(source, self.runtime_variables)

        blanket_points, div_points, num_inboard_points = mg.load_fw_points(
            self.tokamak_geometry,
            blanket_wire,
            divertor_wire,
            raw_uc(new_major_radius, "m", "cm"),
            new_aspect_ratio,
            new_elong,
            True,
        )
        self.cells, self.universe = mg.make_geometry(
            self.tokamak_geometry,
            blanket_points,
            div_points,
            num_inboard_points,
            self.material_lib,
        )

        # deduce source strength (self.src_rate) from the power of the reactor,
        # by assuming 100% of reactor power comes from DT fusion
        self.src_rate = self.operation_variable.calculate_total_neutron_rate()

        create_tallies(self.cells, self.material_lib, self.src_rate)

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

        Parameters
        ----------
        print_summary:
            print the summary to stdout or not.
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
            self.tokamak_geometry,
            self.cells,
            self.runtime_variables.volume_calc_particles,
        )
