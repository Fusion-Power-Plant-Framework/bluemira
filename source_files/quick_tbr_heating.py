import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import openmc
import pandas as pd
import pint
from periodictable import elements

import make_geometry as mg
import make_materials as mm
import pandas_df_functions as pdf

# Constants
pi = math.pi
MJ_per_MeV = pint.Quantity("MeV").to("MJ").magnitude
MJ_per_eV = pint.Quantity("eV").to("MJ").magnitude
s_in_yr = pint.Quantity("year").to("s").magnitude
per_cm2_to_per_m2 = pint.Quantity("1/cm^2").to("1/m^2").magnitude

avogadro = pint.Quantity("N_A").to_base_units().magnitude
fe_molar_mass_g = elements.isotope("Fe").mass
fe_density_g_cc = elements.isotope("Fe").density

# Manually set constants
energy_per_fusion_MeV = 17.58
dpa_fe_threshold_eV = 40


@dataclass
class TBRHeatingBase:
    def to_dict(self):
        pass


@dataclass
class TBRHeatingRuntimeParams:
    stochastic_vol_calc: bool
    plot_geometry: bool
    no_of_particles: int
    reactor_power_MW: float
    no_of_particles_stoch: int
    batches: int
    photon_transport: bool
    electron_treatment: str
    run_mode: str
    openmc_output_summary: str
    parametric_source: bool


@dataclass
class TBRHeatingMaterialParams:
    li_enrich_ao: float
    blanket_type: str


@dataclass
class TBRHeatingGeometryParams:
    minor_r: float
    major_r: float
    elong: float
    shaf_shift: float
    inb_fw_thick: float
    inb_bz_thick: float
    inb_mnfld_thick: float
    inb_vv_thick: float
    tf_thick: float
    outb_fw_thick: float
    outb_bz_thick: float
    outb_mnfld_thick: float
    outb_vv_thick: float
    divertor_width: float


def create_ring_source(geometry_variables):

    # Creating simple ring source
    # A more accurate source will slightly affect the wall loadings and dpa profiles
    
    my_source = openmc.Source()
    source_radii = openmc.stats.Discrete(
        [geometry_variables.major_r + geometry_variables.shaf_shift], [1]
    )
    source_z_values = openmc.stats.Discrete([0], [1])
    source_angles = openmc.stats.Uniform(a=0.0, b=2 * pi)
    my_source.space = openmc.stats.CylindricalIndependent(
        r=source_radii, phi=source_angles, z=source_z_values, origin=(0.0, 0.0, 0.0)
    )
    my_source.angle = openmc.stats.Isotropic()
    my_source.energy = openmc.stats.Discrete([14.06e6], [1])
    
    return my_source


def create_parametric_source(geometry_variables):
    source_params = "mode=2," + \
                "temperature=15.4," + \
                "major_r=" + str(geometry_variables.major_r) + ","\
                "minor_r=" + str(geometry_variables.minor_r) + "," \
                "elongation=" + str(geometry_variables.elong) + ","  \
                "triangulation=0.333," + \
                "radial_shift=" + str(geometry_variables.shaf_shift) + "," \
                "peaking_factor=1.508," + \
                "vertical_shift=0.0," + \
                "start_angle=0.0," + \
                "angle_range=360.0"

    my_source = openmc.Source(library='PPS_OpenMC.so', 
                              parameters=source_params )
    
    return my_source


def setup_openmc(
    my_source,
    no_of_particles,
    batches=2,
    photon_transport=True,
    electron_treatment="ttb",
    run_mode="fixed source",
    output_summary=False,
):
    #######################
    ### OPENMC SETTINGS ###
    #######################

    # Assuming 293K temperature for nuclear cross-sections for calculation speed
    settings = openmc.Settings()
    settings.source = my_source
    settings.particles = no_of_particles
    settings.batches = batches
    settings.photon_transport = photon_transport
    settings.electron_treatment = electron_treatment
    settings.run_mode = run_mode
    settings.output = {"summary": output_summary}

    settings.export_to_xml()


def create_materials(material_variables):

    if material_variables.blanket_type == "hcpb":
        material_lib = mm.make_hcpb_mats(material_variables.li_enrich_ao)
    elif material_variables.blanket_type == 'dcll':
        material_lib = mm.make_dcll_mats(material_variables.li_enrich_ao)
    elif material_variables.blanket_type == 'wcll':
        material_lib = mm.make_wcll_mats(material_variables.li_enrich_ao)
    else:
        raise ValueError("blanket_type must be either hcpb, dcll, or wcll")

    return material_lib


def filter_cells(cells, src_str):
    # Tallies
    cell_filter = openmc.CellFilter(
        [
            cells["tf_coil_cell"],
            cells["vv_inb_cell"],
            cells["manifold_inb_cell"],
            cells["bz_inb_cell"],
            cells["fw_inb_cell"],
            cells["divertor_cell"],
            # div_surf_cell,
            # inner_vessel_cell,
            cells["fw_cell"],
            cells["bz_cell"],
            cells["manifold_cell"],
            cells["vv_cell"],
        ]
    )

    fw_surf_filter = openmc.CellFilter(
        cells["fw_inb_surf_cells"][1:-1]
        + [cells["div_surf_cell"]]  # inb top and bottom divisions are behind blanket
        + cells["fw_outb_surf_cells"]
    )

    neutron_filter = openmc.ParticleFilter(["neutron"])
    photon_filter = openmc.ParticleFilter(["photon"])

    # eV per source particle to MW coefficients
    eV_per_sp_to_MW = src_str * MJ_per_eV

    MW_energy_bins = [0.0, 100.0e6]  # Up to 100 MeV
    MW_dose_coeffs = [eV_per_sp_to_MW, eV_per_sp_to_MW]
    MW_mult_filter = openmc.EnergyFunctionFilter(MW_energy_bins, MW_dose_coeffs)

    # photon heat flux coefficients (cm per source particle to MW cm)
    # Tally heat flux
    energy_bins = [0.0, 100.0e6]  # Up to 100 MeV
    dose_coeffs = [0.0 * eV_per_sp_to_MW, 100.0e6 * eV_per_sp_to_MW]
    energy_mult_filter = openmc.EnergyFunctionFilter(energy_bins, dose_coeffs)
    return (
        cell_filter,
        fw_surf_filter,
        neutron_filter,
        photon_filter,
        MW_mult_filter,
        energy_mult_filter,
    )


def create_tallies(
    cell_filter,
    fw_surf_filter,
    neutron_filter,
    photon_filter,
    MW_mult_filter,
    energy_mult_filter,
):
    #########################################
    # tallies
    #########################################

    tally_tbr = openmc.Tally(name="TBR")
    tally_tbr.scores = ["(n,Xt)"]

    tally_heating = openmc.Tally(name="heating")  # eV per sp
    tally_heating.scores = ["heating"]
    tally_heating.filters = [cell_filter]

    tally_heating_MW = openmc.Tally(name="MW heating")  # MW
    tally_heating_MW.scores = ["heating"]
    tally_heating_MW.filters = [cell_filter, MW_mult_filter]

    tally_n_wall_load = openmc.Tally(name="neutron wall load")
    tally_n_wall_load.scores = ["damage-energy"]
    tally_n_wall_load.filters = [fw_surf_filter, neutron_filter]

    tally_p_heat_flux = openmc.Tally(name="photon heat flux")
    tally_p_heat_flux.scores = ["flux"]
    tally_p_heat_flux.filters = [fw_surf_filter, photon_filter, energy_mult_filter]

    tallies = openmc.Tallies(
        [
            tally_tbr,
            tally_heating,
            tally_heating_MW,
            tally_n_wall_load,
            tally_p_heat_flux,
        ]
    )
    tallies.export_to_xml()


def geometry_plotter(cells, geometry_variables):

    color_assignment = {
        cells["tf_coil_cell"]: "brown",
        cells["vv_inb_cell"]: "orange",
        cells["manifold_inb_cell"]: "purple",
        cells["bz_inb_cell"]: "grey",
        cells["fw_inb_cell"]: "red",
        cells["inner_vessel_cell"]: "blue",
        cells["divertor_cell"]: "cyan",
        cells["fw_cell"]: "red",
        cells["bz_cell"]: "grey",
        cells["manifold_cell"]: "purple",
        cells["vv_cell"]: "orange",
        cells["outer_vessel_cell1"]: "white",
        cells["outer_vessel_cell2"]: "white",
        cells["outer_vessel_cell3"]: "white",
        cells["fw_outb_surf_cells"][0]: "red",
        cells["fw_outb_surf_cells"][1]: "orange",
    }

    plot_width = 2 * (
        geometry_variables.major_r
        + geometry_variables.minor_r
        + geometry_variables.outb_fw_thick
        + geometry_variables.outb_bz_thick
        + geometry_variables.outb_mnfld_thick
        + geometry_variables.outb_vv_thick
        + 100.0
    )  # margin

    plots = []
    for plot_no, basis in enumerate(["xz", "xy", "yz"]):
        plot = openmc.Plot()
        plot.basis = basis
        plot.width = (plot_width, plot_width)
        plot.colors = color_assignment
        plot.filename = f"./plots_{plot_no}"

        plots.append(plot)

    openmc.Plots(plots).export_to_xml()
    openmc.plot_geometry()


def summary(universe, src_str, statepoint_file="statepoint.2.h5"):

    #########################################
    # dpa coefficients
    #########################################

    # number of atoms in region = avogadro * density * volume / molecular mass
    # number of atoms in cc     = avogadro * density          / molecular mass
    # dpa_fpy = displacements / atoms * s_in_yr * src_str

    atoms_per_cc = avogadro * fe_density_g_cc / fe_molar_mass_g

    # dpa formula
    # TODO where does 0.8 come from?
    displacements_per_damage_eV = 0.8 / (2 * dpa_fe_threshold_eV)
    
    ##############################################
    ### Load statepoint file and print results ###
    ##############################################

    # Creating cell name dictionary to allow easy mapping to dataframe
    cell_names = {}
    for cell_id in universe.cells:
        cell_names[cell_id] = universe.cells[cell_id].name

    # Creating cell volume dictionary to allow easy mapping to dataframe
    cell_vols = {}
    for cell_id in universe.cells:
        cell_vols[cell_id] = universe.cells[cell_id].volume

    # Loads up the output file from the simulation
    statepoint = openmc.StatePoint(statepoint_file)

    tally = "TBR"
    tbr_df = statepoint.get_tally(name=tally).get_pandas_dataframe()
    tbr = "{:.2f}".format(tbr_df["mean"].sum())
    tbr_e = "{:.2f}".format(tbr_df["std. dev."].sum())
    print(f"\n{tally}\n{tbr} {tbr_e}")

    tally = "MW heating"  # 'mean' units are MW
    heating_df = statepoint.get_tally(name=tally).get_pandas_dataframe()
    heating_df["cell_name"] = heating_df["cell"].map(cell_names)
    heating_df["%err."] = heating_df.apply(pdf.get_percent_err, axis=1).apply(
        lambda x: "%.1f" % x
    )
    heating_df = heating_df[
        ["cell", "cell_name", "nuclide", "score", "mean", "std. dev.", "%err."]
    ]
    print("\nHeating (MW)\n", heating_df)

    tally = "neutron wall load"  # 'mean' units are eV per source particle
    n_wl_df = statepoint.get_tally(name=tally).get_pandas_dataframe()
    n_wl_df["cell_name"] = n_wl_df["cell"].map(cell_names)
    n_wl_df["vol(cc)"] = n_wl_df["cell"].map(cell_vols)
    n_wl_df["dpa/fpy"] = (
        n_wl_df["mean"]
        * displacements_per_damage_eV
        / (n_wl_df["vol(cc)"] * atoms_per_cc)
        * s_in_yr
        * src_str
    )
    n_wl_df["dpa/fpy"] = n_wl_df["dpa/fpy"].apply(lambda x: "%.1f" % x)
    n_wl_df["%err."] = n_wl_df.apply(pdf.get_percent_err, axis=1).apply(
        lambda x: "%.1f" % x
    )
    n_wl_df = n_wl_df[
        [
            "cell",
            "cell_name",
            "particle",
            "nuclide",
            "score",
            "vol(cc)",
            "mean",
            "std. dev.",
            "dpa/fpy",
            "%err.",
        ]
    ]
    print("\nNeutron Wall Load (eV)\n", n_wl_df)

    tally = "photon heat flux"  # 'mean' units are MW cm
    p_hf_df = statepoint.get_tally(name=tally).get_pandas_dataframe()
    p_hf_df["cell_name"] = p_hf_df["cell"].map(cell_names)
    p_hf_df["vol(cc)"] = p_hf_df["cell"].map(cell_vols)
    p_hf_df["MW_m-2"] = p_hf_df["mean"] / p_hf_df["vol(cc)"] * per_cm2_to_per_m2
    p_hf_df["%err."] = p_hf_df.apply(pdf.get_percent_err, axis=1).apply(
        lambda x: "%.1f" % x
    )
    p_hf_df = p_hf_df[
        [
            "cell",
            "cell_name",
            "particle",
            "nuclide",
            "score",
            "vol(cc)",
            "mean",
            "std. dev.",
            "MW_m-2",
            "%err.",
        ]
    ]
    print("\nPhoton Heat Flux MW m-2\n", p_hf_df)


def stochastic_volume_calculation(
    geometry_variables, cells, no_of_particles=4e7
):
    ##############################################
    ### Stochastic volume calculation ###
    ##############################################
    import os

    try:
        os.remove("summary.h5")
        os.remove("statepoint.1.h5")
    except OSError:
        pass

    maxr = (
        geometry_variables.major_r
        + geometry_variables.minor_r
        + geometry_variables.outb_fw_thick
        + geometry_variables.outb_bz_thick
    )
    maxz = (
        geometry_variables.elong * geometry_variables.minor_r
        + geometry_variables.outb_fw_thick
        + geometry_variables.outb_bz_thick
        + geometry_variables.outb_mnfld_thick
        + geometry_variables.outb_vv_thick
    )
    lower_left = (-maxr, -maxr, -maxz)
    upper_right = (maxr, maxr, maxz)
    cell_vol_calc = openmc.VolumeCalculation(
        cells["fw_inb_surf_cells"]
        + [cells["div_surf_cell"]]
        + cells["fw_outb_surf_cells"],
        int(no_of_particles),
        lower_left,
        upper_right,
    )
    settings = openmc.Settings()
    settings.volume_calculations = [cell_vol_calc]

    settings.export_to_xml()
    openmc.calculate_volumes()



def run_tbr_heating_calc(
    runtime_variables,
    material_variables,
    geometry_variables,
):
    """
    Runs OpenMC calculation to get TBR, component heating, first wall dpa, and photon
    heat flux for a DEMO-like reactor
    """

    material_lib = create_materials(material_variables)

    mg.check_geometry(geometry_variables)
    
    if runtime_variables.parametric_source:
        source = create_parametric_source(geometry_variables)
    else:
        source = create_ring_source(geometry_variables)

    setup_openmc(
        source,
        runtime_variables.no_of_particles,
        runtime_variables.batches,
        runtime_variables.photon_transport,
        runtime_variables.electron_treatment,
        runtime_variables.run_mode,
        runtime_variables.openmc_output_summary,
    )

    cells, universe = mg.make_geometry(geometry_variables, material_lib)

    # Calculating source strength
    src_str = runtime_variables.reactor_power_MW / (energy_per_fusion_MeV * MJ_per_MeV)

    create_tallies(*filter_cells(cells, src_str))

    if runtime_variables.plot_geometry:
        geometry_plotter(cells, geometry_variables)

    # Start the OpenMC calculation, run time calculated from here
    openmc.run()

    summary(universe, src_str)

    if runtime_variables.stochastic_vol_calc:
        stochastic_volume_calculation(
            geometry_variables,
            cells,
            runtime_variables.no_of_particles_stoch,
            runtime_variables.use_dagmc,
        )


################################################################################################################

if __name__ == "__main__":

    # Calculation runtime variables
    runtime_variables = TBRHeatingRuntimeParams(
        **{
            "stochastic_vol_calc": False,    # Do a stochastic volume calculation - this takes a long time!
            "plot_geometry": False,
            "reactor_power_MW": 1998.0,
            "no_of_particles": 18000,        # 18000 = 5 seconds,  1000000 = 259 seconds }
            "no_of_particles_stoch": 4e6,
            "batches": 2,
            "photon_transport": True,
            "electron_treatment": "ttb",
            "run_mode": "fixed source",
            "openmc_output_summary": False,
            "parametric_source": False
        }
    )

    material_variables = TBRHeatingMaterialParams(
        **{"blanket_type": "hcpb", 
           "li_enrich_ao": 60.0}
    )  # atomic fraction percentage of lithium

    # Geometry variables

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

    if material_variables.blanket_type == "wcll":
        geometry_variables = TBRHeatingGeometryParams(
            **{
                "minor_r": 288.3,
                "major_r": 893.8,
                "elong": 1.65,
                "shaf_shift": 60.0,  # The shafranov shift of the plasma
                "inb_fw_thick": 2.7,
                "inb_bz_thick": 37.8,
                "inb_mnfld_thick": 41.0,
                "inb_vv_thick": 60.0,
                "tf_thick": 40.0,
                "outb_fw_thick": 2.7,
                "outb_bz_thick": 53.8,
                "outb_mnfld_thick": 43.5,
                "outb_vv_thick": 60.0,
                "divertor_width": 267.0,
            }
        )
    elif material_variables.blanket_type == "dcll":
        geometry_variables = TBRHeatingGeometryParams(
            **{
                "minor_r": 288.3,
                "major_r": 893.8,
                "elong": 1.65,
                "shaf_shift": 60.0,  # The shafranov shift of the plasma
                "inb_fw_thick": 2.2,
                "inb_bz_thick": 30.0,
                "inb_mnfld_thick": 17.8,
                "inb_vv_thick": 60.0,
                "tf_thick": 40.0,
                "outb_fw_thick": 2.2,
                "outb_bz_thick": 64.0,
                "outb_mnfld_thick": 24.8,
                "outb_vv_thick": 60.0,
                "divertor_width": 200.0,
            }
        )
    else:
        geometry_variables = TBRHeatingGeometryParams(
            **{
                "minor_r": 288.3,
                "major_r": 893.8,
                "elong": 1.65,
                "shaf_shift": 60.0,  # The shafranov shift of the plasma
                "inb_fw_thick": 2.2,
                "inb_bz_thick": 35.0,
                "inb_mnfld_thick": 41.0,
                "inb_vv_thick": 60.0,
                "tf_thick": 40.0,
                "outb_fw_thick": 2.2,
                "outb_bz_thick": 57.0,
                "outb_mnfld_thick": 56.0,
                "outb_vv_thick": 60.0,
                "divertor_width": 200.0,
            }
        )
        

    run_tbr_heating_calc(
        runtime_variables,
        material_variables,
        geometry_variables
    )
