"""
TODO:
[ ]compare the pps_api with open-radiation-source/parametric-plasma-source/.git
    [ ]find the units and documentations for creating source_params_dict
        - See details in PPS_OpenMC.so library
[ ]Integration into our logging system (print should go through bluemira_print etc.)
[ ]Use BluemiraWire instead of .npy files
[ ]Unit: cgs -> metric
    [ ]Check other files (other than quick_tbr_heating.py) as well
[ ]Find out from the author of plasma_lib.F90:
    [ ]What is a_array and s_array?
    [ ]What is with the 629 problem?
After talking w/ A. Davis:
[]Replace the following:
    - parametric-plasma-source/parametric_plasma_source/fortran_api/*src/
    - parametric-plasma-source/parametric_plasma_source/pps_api
    - `pip install
        git+https://github.com/open-radiation-source/parametric-plasma-source.git@main`
[ ]Some parameters are locked up inside functions:
    [ ]create_parametric_source
____
[ ]Tests?
"""
from dataclasses import dataclass
from typing import Literal

import openmc
from numpy import pi
from openmc.config import config

import bluemira.neutronics.make_geometry as mg
import bluemira.neutronics.make_materials as mm
import bluemira.neutronics.result_presentation as present

# Constants
from bluemira.base.constants import raw_uc
from bluemira.neutronics.constants import dt_neutron_energy_MeV
from bluemira.neutronics.params import (
    BreederTypeParameters,
    OpenMCSimulationRuntimeParameters,
    TokamakGeometry,
    TokamakOperationParameters,
)
from bluemira.neutronics.tallying import create_tallies
from bluemira.neutronics.volume_functions import stochastic_volume_calculation

config[
    "cross_sections"
] = "/home/ocean/Others/cross_section_data/cross_section_data/cross_sections.xml"
# cross_sections are probably downloaded from here
# https://github.com/openmc-dev/data/


# openmc source maker
def create_ring_source(tokamak_geometry: TokamakGeometry) -> openmc.Source:
    """
    Creating simple ring source.
    A more accurate source will slightly affect the wall loadings and dpa profiles.

    Parameters
    ----------
    tokamak_geometry: TokamakGeometry
        Only the
            - tokamak_geometry.major_r
            - tokamak_geometry.shaf_shift
        variables are used in this function.
    """
    ring_source = openmc.Source()
    source_radii = openmc.stats.Discrete(
        [tokamak_geometry.major_r + tokamak_geometry.shaf_shift], [1]
    )
    source_z_values = openmc.stats.Discrete([0], [1])
    source_angles = openmc.stats.Uniform(a=0.0, b=2 * pi)
    ring_source.space = openmc.stats.CylindricalIndependent(
        r=source_radii, phi=source_angles, z=source_z_values, origin=(0.0, 0.0, 0.0)
    )
    ring_source.angle = openmc.stats.Isotropic()
    ring_source.energy = openmc.stats.Discrete(
        [raw_uc(dt_neutron_energy_MeV, "MeV", "eV")], [1]
    )

    return ring_source


def create_parametric_source(tokamak_geometry: TokamakGeometry) -> openmc.Source:
    """
    Create a parametric plasma source using the PPS_OpenMC.so library
        and the relevant parameters.

    Parameters
    ----------
    tokamak_geometry: TokamakGeometry
    """
    source_params_dict = {
        "mode": 2,
        "temperature": 15.4,  # put this magic number in constants.py!
        "major_r": tokamak_geometry.major_r,
        "minor_r": tokamak_geometry.minor_r,
        "elongation": tokamak_geometry.elong,
        "triangulation": tokamak_geometry.triang,
        "radial_shift": tokamak_geometry.shaf_shift,
        "peaking_factor": 1.508,  # put this magic number in constants.py!
        "vertical_shift": 0.0,
        "start_angle": 0.0,
        "angle_range": 360.0,
    }
    # Parameter dictionary has to be formatted as string to be provided to the library.
    source_params_str = ",".join(f"{k}={v}" for k, v in source_params_dict.items())

    try:
        from parametric_plasma_source import PlasmaSource

        plasma = PlasmaSource(source_params_dict)
        parametric_source = openmc.Source(
            library=SOURCE_SAMPLING_PATH, parameters=plasma
        )
    except ImportError:
        parametric_source = openmc.Source(
            library="./PPS_OpenMC.so", parameters=source_params_str
        )

    return parametric_source


def setup_openmc(
    plasma_source: openmc.Source,
    particles: int,
    batches: int = 2,
    photon_transport=True,
    electron_treatment: Literal["ttb", "led"] = "ttb",
    run_mode="fixed source",
    output_summary=False,
) -> None:
    """Configure openmc.Settings, so that it's ready for the run() step.
    Assumptions
    -----------
    We run the simulation with the assumption that temperature = 293K,
    as the nuclear cross-section values are evaluated at this temperature

    Parameters (all of which are arguments parsed to openmc.Settings)
    ----------
    plasma_source:
        Openmc.Source used to emulate the neutron emission of the plasma.
    particles:
        Number of neutrons emitted by the plasma source per batch.
    batches: int, default=2
        How many batches to simulate.
    photon_transport: bool, default=True
        Whether to simulate the transport of photons (i.e. gamma-rays created) or not.
    electron_treatment: {'ttb', 'led'}
        The way in which OpenMC handles secondary charged particles.
        'thick-target bremsstrahlung' or 'local energy deposition'
        'thick-target bremsstrahlung' accounts for the energy carried away by
            bremsstrahlung photons and deposited elsewhere, whereas 'local energy
            deposition' assumes electrons deposit all energies locally.
        (the latter is expected to be computationally faster.)
    run_mode: {'fixed source', 'eigenvalue', 'plot', 'volume', 'particle restart'}
        see below for details:
        https://docs.openmc.org/en/stable/usersguide/settings.html#run-modes
    output_summary: whether a 'summary.h5' file is written or not.

    Returns
    -------
    Exports the settings to an xml file.
    """
    settings = openmc.Settings()
    settings.source = plasma_source
    settings.particles = particles
    settings.batches = batches
    settings.photon_transport = photon_transport
    settings.electron_treatment = electron_treatment
    settings.run_mode = run_mode
    settings.output = {"summary": output_summary}

    settings.export_to_xml()


def create_and_export_materials(
    breeder_materials: BreederTypeParameters,
) -> mm.MaterialsLibrary:
    """
    Parameters
    ----------
    breeder_materials:
        dataclass containing attributes: 'blanket_type', 'li_enrich_ao'
    """
    material_lib = mm.MaterialsLibrary.create_from_blanket_type(
        breeder_materials.blanket_type, breeder_materials.li_enrich_ao
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
        self.operation_variable = operation_variable
        self.breeder_materials = breeder_materials
        self.tokamak_geometry = tokamak_geometry

        self.cells = None
        self.material_lib = None
        self.universe = None

    def setup(self, plot_geometry: bool = True) -> None:
        """Plot the geometry and saving them as .png files with hard-coded names."""
        material_lib = create_and_export_materials(self.breeder_materials)
        self.material_lib = material_lib
        mg.check_geometry(self.tokamak_geometry)
        if self.runtime_variables.parametric_source:
            source = create_parametric_source(self.tokamak_geometry)
        else:
            source = create_ring_source(self.tokamak_geometry)

        setup_openmc(
            source,
            self.runtime_variables.particles,
            self.runtime_variables.batches,
            self.runtime_variables.photon_transport,
            self.runtime_variables.electron_treatment,
            self.runtime_variables.run_mode,
            self.runtime_variables.openmc_write_summary,
        )

        blanket_points, div_points, num_inboard_points = mg.load_fw_points(
            self.tokamak_geometry, True
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

    def run(self, *args, **kwargs) -> None:
        """Run the actual openmc simulation."""
        openmc.run(*args, **kwargs)

    def get_result(self, print_summary: bool) -> present.OpenMCResult:
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
        self.result = present.OpenMCResult(self.universe, self.src_rate)
        self.result.summarize(print_summary)
        return self.result

    def calculate_volume_stochastically(self):
        """
        Using openmc's built-in stochastic volume calculation function to get the volume.
        """
        stochastic_volume_calculation(
            self.tokamak_geometry,
            self.cells,
            self.runtime_variables.volume_calc_particles,
        )


if __name__ == "__main__":

    @dataclass
    class SimulatedBluemiraOutputVariables:
        """
        A quick dataclass consisting of two sub-dataclasses,
        both of which shares the commonality of being reactor-design specific.
        """

        breeder_materials: BreederTypeParameters
        tokamak_geometry: TokamakGeometry

    def get_preset_physical_properties(blanket_type: mm.BlanketType):
        """
        Works as a switch-case for choosing the tokamak geometry
            and blankets for a given blanket type.
        The allowed list of blanket types are specified in mm.BlanketType.
        Currently, the blanket types with pre-populated data in this function are:
            {'wcll', 'dcll', 'hcpb'}
        """
        if not isinstance(blanket_type, mm.BlanketType):
            raise KeyError(f"{blanket_type} is not an accepted blanket type.")
        breeder_materials = BreederTypeParameters(
            blanket_type=blanket_type,
            li_enrich_ao=60.0,  # atomic fraction percentage of lithium
        )

        # Geometry variables

        # Break down from here.
        # Paper inboard build ---
        # Nuclear analyses of solid breeder blanket options for DEMO:
        # Status,challenges and outlook,
        # Pereslavtsev, 2019
        #
        # 40.0,      # TF Coil inner
        # 20.0,      # gap                  from figures
        # 6.0,       # VV steel wall
        # 48.0,      # VV
        # 6.0,       # VV steel wall
        # 2.0,       # gap                  from figures
        # 35.0,      # Back Supporting Structure
        # 6.0,       # Back Wall and Gas Collectors   Back wall = 3.0
        # 35.0,      # breeder zone
        # 2.2        # fw and armour

        plasma_shape = {
            "minor_r": 288.3,
            "major_r": 893.8,
            "elong": 1.65,
            "shaf_shift": 0.0,
        }  # The shafranov shift of the plasma
        if blanket_type is mm.BlanketType.WCLL:
            tokamak_geometry = TokamakGeometry(
                **plasma_shape,
                inb_fw_thick=2.7,
                inb_bz_thick=37.8,
                inb_mnfld_thick=43.5,
                inb_vv_thick=60.0,
                tf_thick=40.0,
                outb_fw_thick=2.7,
                outb_bz_thick=53.8,
                outb_mnfld_thick=42.9,
                outb_vv_thick=60.0,
            )
        elif blanket_type is mm.BlanketType.DCLL:
            tokamak_geometry = TokamakGeometry(
                **plasma_shape,
                inb_fw_thick=2.2,
                inb_bz_thick=30.0,
                inb_mnfld_thick=17.8,
                inb_vv_thick=60.0,
                tf_thick=40.0,
                outb_fw_thick=2.2,
                outb_bz_thick=64.0,
                outb_mnfld_thick=24.8,
                outb_vv_thick=60.0,
            )
        elif blanket_type is mm.BlanketType.HCPB:
            # HCPB Design Report, 26/07/2019
            tokamak_geometry = TokamakGeometry(
                **plasma_shape,
                inb_fw_thick=2.7,
                inb_bz_thick=46.0,
                inb_mnfld_thick=56.0,
                inb_vv_thick=60.0,
                tf_thick=40.0,
                outb_fw_thick=2.7,
                outb_bz_thick=46.0,
                outb_mnfld_thick=56.0,
                outb_vv_thick=60.0,
            )

        return SimulatedBluemiraOutputVariables(breeder_materials, tokamak_geometry)

    # set up the variables to be used for the openmc simulation

    tokamak_properties = get_preset_physical_properties(mm.BlanketType.WCLL)
    # allowed blanket_type so far = {'WCLL', 'DCLL', 'HCPB'}
    breeder_materials = tokamak_properties.breeder_materials
    tokamak_geometry = tokamak_properties.tokamak_geometry

    runtime_variables = OpenMCSimulationRuntimeParameters(
        particles=16800,  # 16800 takes 5 seconds,  1000000 takes 280 seconds.
        batches=2,
        photon_transport=True,
        electron_treatment="ttb",
        run_mode="fixed source",
        openmc_write_summary=False,
        parametric_source=True,
        volume_calc_particles=4e8,  # only used if stochastic_volume_calculation is turned on.
    )

    operation_variable = TokamakOperationParameters(reactor_power_MW=1998.0)

    # set up a DEMO-like reactor, and run OpenMC simualtion
    tbr_heat_sim = TBRHeatingSimulation(
        runtime_variables, operation_variable, breeder_materials, tokamak_geometry
    )
    tbr_heat_sim.setup(True)
    tbr_heat_sim.run()
    # get the TBR, component heating, first wall dpa, and photon heat flux
    tbr_heat_sim.get_result(True)
    # tbr_heat_sim.calculate_volume_stochastically()
    # # don't do this because it takes a long time.
