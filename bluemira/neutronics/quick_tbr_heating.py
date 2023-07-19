"""
TODO:
[ ]Break quick_tbr_heating into multiple
    [x]get_dpa_coefs -> constants
    [x]quick_tbr_heating.PoloidalXSPlot -> result_presentation
    [x]quick_tbr_heating.print_df_decorator_with_title_string, quick_tbr_heating.OpenMCResult -> result_presentation
    [x]Normalize the methods in result_presentation.py
    [x]pandas_df_functions -> result_presentation
    [x]quick_tbr_heating.geometry_plotter -> result_presentation
    [ ]All openmc setting up -> stay here at quick_btr_heating.py
    [ ]quick_tbr_heating.filter_cells -> filter_cells.py
        [x]_load_fw_points -> somewhere??
        - create_tallies
        [x]stochastic_volume_calculation
[ ]compare the pps_api with open-radiation-source/parametric-plasma-source/.git
    [ ]find the units and documentations for creating source_params_dict
        - See details in PPS_OpenMC.so library
        - why parametric source mode=2: need to dig open the PPS_OpenMC.so
[]Unit: cgs -> metric
    [ ]Check other files (other than quick_tbr_heating.py) as well
[]Replace parametric-plasma-source/parametric_plasma_source/fortran_api/* and src/ vs pps_api
    - `pip install git+https://github.com/open-radiation-source/parametric-plasma-source.git@main`
[ ]Integration into our logging system (print should go through bluemira_print etc.)
[ ]Some parameters are locked up inside functions:
    [ ]create_parametric_source
____
[ ]Tests?
"""
import os
from dataclasses import dataclass

import numpy as np
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
from bluemira.neutronics.volume_functions import stochastic_volume_calculation

config[
    "cross_sections"
] = "/home/ocean/Others/cross_section_data/cross_section_data/cross_sections.xml"


# openmc source maker
def create_ring_source(tokamak_geometry):
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


def create_parametric_source(tokamak_geometry):
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

    parametric_source = openmc.Source(
        library="./PPS_OpenMC.so", parameters=source_params_str
    )

    return parametric_source


def setup_openmc(
    plasma_source,
    particles,
    batches=2,
    photon_transport=True,
    electron_treatment="ttb",
    run_mode="fixed source",
    output_summary=False,
):
    """Configure openmc.Settings, so that it's ready for the run() step.

    Parameters (all of which are arguments parsed to openmc.Settings)
    ----------
    plasma_source: openmc.Source
    particles: int
    batches: int, default=2
    photon_transport: bool, default=True
    electron_treatment: {'ttb', 'led'}
    run_mode: {'fixed source', 'eigenvalue', 'plot', 'volume', 'particle restart'}
    output_summary: bool , default=False
    """
    #######################
    ### OPENMC SETTINGS ###
    #######################

    # Assuming 293K temperature for nuclear cross-sections for calculation speed
    settings = openmc.Settings()
    settings.source = plasma_source
    settings.particles = particles
    settings.batches = batches
    settings.photon_transport = photon_transport
    settings.electron_treatment = electron_treatment
    settings.run_mode = run_mode
    settings.output = {"summary": output_summary}

    settings.export_to_xml()


def create_materials(breeder_materials: BreederTypeParameters):
    """
    Parameters
    ----------
    breeder_materials: BreederTypeParameters
        dataclass containing attributes: 'blanket_type', 'li_enrich_ao'
    """

    material_lib = mm.MaterialsLibrary.create_from_blanket_type(
        breeder_materials.blanket_type, breeder_materials.li_enrich_ao
    )
    material_lib.export()
    return material_lib


def filter_cells(cells_and_cell_lists, material_lib, src_rate):
    """
    Requests cells for scoring.
    Parameters
    ----------
    cells_and_cell_lists:
        dictionary where each item is either a single openmc.Cell,
            or a list of openmc.Cell.
    material_lib: (dict)
        A dictionary (or an instance of MaterialsLibrary,
            which is an offspring class of dict)
        with all of the material definitions stored.
    src_rate: float
        number of neutrons produced by the source (plasma) per second.
    """
    cell_filter = openmc.CellFilter(
        # the single cells
        [
            cells_and_cell_lists["tf_coil_cell"],
            cells_and_cell_lists["plasma_inner1"],
            cells_and_cell_lists["plasma_inner2"],
            cells_and_cell_lists["plasma_outer1"],
            cells_and_cell_lists["plasma_outer2"],
            cells_and_cell_lists["divertor_fw"],
            cells_and_cell_lists["divertor_fw_sf"],  # sf = surface
        ]
        # the cell lists
        + cells_and_cell_lists["inb_vv_cells"]
        + cells_and_cell_lists["inb_mani_cells"]
        + cells_and_cell_lists["inb_bz_cells"]
        + cells_and_cell_lists["inb_fw_cells"]
        + cells_and_cell_lists["inb_sf_cells"]  # sf = surface
        + cells_and_cell_lists["outb_vv_cells"]
        + cells_and_cell_lists["outb_mani_cells"]
        + cells_and_cell_lists["outb_bz_cells"]
        + cells_and_cell_lists["outb_fw_cells"]
        + cells_and_cell_lists["outb_sf_cells"]  # sf = surface
        + cells_and_cell_lists["divertor_cells"],
    )

    mat_filter = openmc.MaterialFilter(
        [
            material_lib.inb_fw_mat,
            material_lib.outb_fw_mat,
            material_lib.inb_bz_mat,
            material_lib.outb_bz_mat,
            material_lib.inb_mani_mat,
            material_lib.outb_mani_mat,
            material_lib.inb_vv_mat,
            material_lib.outb_vv_mat,
            material_lib.divertor_mat,
            material_lib.div_fw_mat,
            material_lib.tf_coil_mat,
            material_lib.inb_sf_mat,  # sf = surface
            material_lib.outb_sf_mat,  # sf = surface
            material_lib.div_sf_mat,  # sf = surface
        ]
    )

    fw_surf_filter = openmc.CellFilter(
        cells_and_cell_lists["inb_sf_cells"]  # sf = surface
        + cells_and_cell_lists["outb_sf_cells"]  # sf = surface
        + [cells_and_cell_lists["divertor_fw_sf"]]  # sf = surface
        + cells_and_cell_lists["inb_fw_cells"]
        + cells_and_cell_lists["outb_fw_cells"]
        + [cells_and_cell_lists["divertor_fw"]]
    )

    neutron_filter = openmc.ParticleFilter(["neutron"])
    photon_filter = openmc.ParticleFilter(["photon"])

    # eV per source particle to MW coefficients
    # SOMETHING SEEMS WRONG @ JAMES HAGUE (original file line L.313)
    eV_per_sp_to_MW = raw_uc(src_rate, "eV/s", "MW")

    MW_energy_bins = [0.0, 100.0e6]  # Up to 100 MeV
    MW_dose_coeffs = [eV_per_sp_to_MW, eV_per_sp_to_MW]
    # makes a flat line function
    MW_mult_filter = openmc.EnergyFunctionFilter(MW_energy_bins, MW_dose_coeffs)

    # photon heat flux coefficients (cm per source particle to MW cm)
    # Tally heat flux
    energy_bins = [0.0, 100.0e6]  # Up to 100 MeV
    dose_coeffs = [0.0 * eV_per_sp_to_MW, 100.0e6 * eV_per_sp_to_MW]
    # simply modify the energy by multiplying by the constant
    energy_mult_filter = openmc.EnergyFunctionFilter(energy_bins, dose_coeffs)

    cyl_mesh = openmc.CylindricalMesh(mesh_id=1)
    cyl_mesh.r_grid = np.linspace(400, 1400, 100 + 1)
    cyl_mesh.z_grid = np.linspace(-800.0, 800.0, 160 + 1)
    cyl_mesh_filter = openmc.MeshFilter(cyl_mesh)

    return (
        cell_filter,
        mat_filter,
        fw_surf_filter,
        neutron_filter,
        photon_filter,
        MW_mult_filter,
        energy_mult_filter,
        cyl_mesh_filter,
    )


def create_tallies(
    cell_filter,
    mat_filter,
    fw_surf_filter,
    neutron_filter,
    photon_filter,
    MW_mult_filter,
    energy_mult_filter,
    cyl_mesh_filter,
):
    """
    Produces tallies for OpenMC scoring.

    Parameters
    ----------
    cell_filter:        openmc.CellFilter
        tally binned by cell
    mat_filter:         openmc.MaterialFilter
        tally binned by materials
        # wait you should provide cells, not materials??!
    fw_surf_filter:     openmc.CellFilter
        tally binned by first wall surface
    neutron_filter:     openmc.ParticleFilter
        tally binned by neutron
    photon_filter:      openmc.ParticleFilter
        tally binned by photon
    MW_mult_filter:     openmc.EnergyFunctionFilter
        tally binned by energy so that it can be used to obtain the MW rate
    energy_mult_filter: openmc.EnergyFunctionFilter
        tally binned by energy so that it can calculate the spectrum
    cyl_mesh_filter:    openmc.MeshFilter
        tally binned spatially: the tokamak is cut into stacks of concentric rings
    """
    tally_tbr = openmc.Tally(name="TBR")
    tally_tbr.scores = ["(n,Xt)"]

    tally_heating = openmc.Tally(name="heating")  # eV per sp
    tally_heating.scores = ["heating"]
    tally_heating.filters = [mat_filter]

    tally_heating_MW = openmc.Tally(name="MW heating")  # MW
    tally_heating_MW.scores = ["heating"]
    tally_heating_MW.filters = [mat_filter, MW_mult_filter]

    tally_n_wall_load = openmc.Tally(name="neutron wall load")
    tally_n_wall_load.scores = ["damage-energy"]
    tally_n_wall_load.filters = [fw_surf_filter, neutron_filter]

    tally_p_heat_flux = openmc.Tally(name="photon heat flux")
    tally_p_heat_flux.scores = ["flux"]
    tally_p_heat_flux.filters = [fw_surf_filter, photon_filter, energy_mult_filter]

    tally_n_flux = openmc.Tally(name="neutron flux in every cell")
    tally_n_flux.scores = ["flux"]
    tally_n_flux.filters = [cell_filter, neutron_filter]

    tally_n_flux_mesh = openmc.Tally(name="neutron flux in 2d mesh")
    tally_n_flux_mesh.scores = ["flux"]
    tally_n_flux_mesh.filters = [cyl_mesh_filter, neutron_filter]

    tallies = openmc.Tallies(
        [
            tally_tbr,
            tally_heating,
            tally_heating_MW,
            tally_n_wall_load,
            tally_p_heat_flux,
            # tally_n_flux, # skipped
            # tally_n_flux_mesh, # skipped
        ]
    )
    tallies.export_to_xml()


class TBRHeatingSimulation:
    """
    Contains all the data necessary to run the openmc simulation of the tbr,
    and the relevant pre-and post-processing.
    """

    def __init__(
        self, runtime_variables, operation_variable, breeder_materials, tokamak_geometry
    ):
        self.runtime_variables = runtime_variables
        self.operation_variable = operation_variable
        self.breeder_materials = breeder_materials
        self.tokamak_geometry = tokamak_geometry

        self.cells = None
        self.material_lib = None
        self.universe = None

    def setup(self, plot_geometry=True):
        """plot the geometry and saving them as .png files with hard-coded names."""
        material_lib = create_materials(self.breeder_materials)
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

        create_tallies(*filter_cells(self.cells, self.material_lib, self.src_rate))

        if plot_geometry:
            present.geometry_plotter(self.cells, self.tokamak_geometry)
        return

    def run(self, *args, **kwargs):
        """Run the actual openmc simulation."""
        openmc.run(*args, **kwargs)

    def get_result(self, print_summary: bool):
        """
        Create a summary object, attach it to self, and then return it.
        Parameters
        ----------
        print_summary: bool
            print the summary to stdout or not.
        """
        assert (
            self.universe is not None
        ), "The self.universe variable must have been first populated by self.run()!"
        self.result = present.OpenMCResult(self.universe, self.src_rate)
        self.result.summarize(print_summary)
        return self.result

    def calculate_volume_stochastically(self):
        """Using openmc's built-in stochastic volume calculation function to calculate
        the volume
        """
        stochastic_volume_calculation(
            self.tokamak_geometry,
            self.cells,
            self.runtime_variables.volume_calc_particles,
        )


if __name__ == "__main__":

    @dataclass
    class SimulatedBluemiraOutputVariables:
        breeder_materials: BreederTypeParameters
        tokamak_geometry: TokamakGeometry

    def get_preset_physical_properties(blanket_type: str):
        """
        Works as a switch-case for choosing the tokamak geometry and blankets for a given blanket type.
        The allowed list of blanket types are specified in mm.BlanketType.
        Currently, the blanket types with pre-populated data in this function are:
            {'wcll', 'dcll', 'hcpb'}
        """
        breeder_materials = BreederTypeParameters(
            blanket_type=mm.BlanketType[blanket_type.upper()],
            li_enrich_ao=60.0,  # atomic fraction percentage of lithium
        )

        # Geometry variables

        # Break down from here.
        # Paper inboard build --- Nuclear analyses of solid breeder blanket options for DEMO: Status,challenges and outlook,
        #                         Pereslavtsev, 2019
        #                        40.0,      # TF Coil inner
        #                        20.0,      # gap                  from figures
        #                        6.0,       # VV steel wall
        #                        48.0,      # VV
        #                        6.0,       # VV steel wall
        #                        2.0,       # gap                  from figures
        #                        35.0,      # Back Supporting Structure
        #                        6.0,       # Back Wall and Gas Collectors   Back wall = 3.0
        #                        35.0,      # breeder zone
        #                        2.2        # fw and armour

        plasma_shape = {
            "minor_r": 288.3,
            "major_r": 893.8,
            "elong": 1.65,
            "shaf_shift": 0.0,
        }  # The shafranov shift of the plasma
        if breeder_materials.blanket_type is mm.BlanketType.WCLL:
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
        elif breeder_materials.blanket_type is mm.BlanketType.DCLL:
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
        elif breeder_materials.blanket_type is mm.BlanketType.HCPB:
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

    tokamak_properties = get_preset_physical_properties("wcll")
    # allowed blanket_type so far = {'wcll', 'dcll', 'hcpb'}
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
    # tbr_heat_sim.calculate_volume_stochastically() # don't do this because it takes a long time.
