"""
TODO:
3. Integration into our logging system (print should go through bluemira_print etc.)
[ ]The rest of the *.py files too: but only those that actually will get reused (i.e. skip make_geometry.py)
[ ]Break quick_tbr_heating into multiple
    - keep openmc set-up here?
    - Ask how much time I should spend on this and future directions.
    [ ]filter_cells needs to be somewhere else. Definitely does not belong in quick_btr_heating.py
[ ]OOP-ize:
    i.e.same formatting to make it modular enough for re-testing
    [ ]_load_fw_points
[ ]find the units and documentations for creating source_params_dict (PPS_OpenMC.so library)
[ ]?Tests?
[ ]Unit: cgs -> metric
____Concerns
- Some parameters are locked up inside functions:
    - create_parametric_source
    - _load_fw_points
- why parametric source mode = 2: need to dig open the PPS_OpenMC.so
"""
import math
import dataclasses

import matplotlib.pyplot as plt
import openmc
import numpy as np
from numpy import pi
from periodictable import elements

import make_geometry as mg
import make_materials as mm
import pandas_df_functions as pdf
from collections import namedtuple

# Constants
from bluemira.base.constants import BMUnitRegistry

import os
os.environ['OPENMC_CROSS_SECTIONS'] = '/home/ocean/Others/cross_section_data/cross_section_data/cross_sections.xml'

MJ_per_MeV= BMUnitRegistry.Quantity("MeV").to("MJ").magnitude
MJ_per_eV = BMUnitRegistry.Quantity("eV").to("MJ").magnitude
eV_per_MeV= BMUnitRegistry.Quantity("MeV").to("eV").magnitude
s_in_yr =    BMUnitRegistry.Quantity("year").to("s").magnitude
per_cm2_to_per_m2 = BMUnitRegistry.Quantity("1/cm^2").to("1/m^2").magnitude
m_to_cm =   BMUnitRegistry.Quantity("m").to("cm").magnitude

avogadro =  BMUnitRegistry.Quantity("N_A").to_base_units().magnitude
fe_molar_mass_g = elements.isotope("Fe").mass
fe_density_g_cc = elements.isotope("Fe").density

# Manually set constants
energy_per_dt_MeV = 17.58
dpa_fe_threshold_eV = 40 # Energy required to displace an Fe atom in Fe. See docstring of get_dpa_coefs. Source cites 40 eV.

DPACoefficients = namedtuple('DPACoefficients', 'atoms_per_cc, displacements_per_damage_eV')

def get_dpa_coefs():
    """
    Get the coefficients required to convert the number of damage into the number of displacements.
    number of atoms in region = avogadro * density * volume / molecular mass
    number of atoms in cc     = avogadro * density          / molecular mass
    dpa_fpy = displacements / atoms * s_in_yr * src_rate

    taken from [1]_.
    .. [1] Shengli Chena, David Bernard
       On the calculation of atomic displacements using damage energy 
       Results in Physics 16 (2020) 102835
       https://doi.org/10.1016/j.rinp.2019.102835
    """
    atoms_per_cc = avogadro * fe_density_g_cc / fe_molar_mass_g
    displacements_per_damage_eV = 0.8 / (2 * dpa_fe_threshold_eV)
    return DPACoefficients(atoms_per_cc, displacements_per_damage_eV)
# ----------------------------------------------------------------------------------------
# classes to store parameters

class ParameterHolder:
    def to_dict(self):
        return dataclasses.asdict(self)


@dataclasses.dataclass
class OpenMCSimulationRuntimeParameters(ParameterHolder):
    """Parameters used in the actual simulation"""
    # parameters used in setup_openmc()
    particles: int # number of particles used in the neutronics simulation
    batches: int
    photon_transport: bool
    electron_treatment: str
    run_mode: str
    openmc_write_summary: str
    # Parameters used elsewhere
    parametric_source: bool
    volume_calc_particles: int # number of particles used in the volume calculation. 


@dataclasses.dataclass
class TokamakOperationParameters(ParameterHolder):
    """The tokamak's operational parameter, such as its power"""
    reactor_power_MW: float # MW

    def calculate_total_neutron_rate(self):
        return self.reactor_power_MW / (energy_per_dt_MeV * MJ_per_MeV)


@dataclasses.dataclass
class BreederTypeParameters(ParameterHolder):
    li_enrich_ao: float # [dimensionless]
    blanket_type: mm.BlanketType


@dataclasses.dataclass
class TokamakGeometry(ParameterHolder):
    minor_r:            float # [cm]
    major_r:            float # [cm]
    elong:              float # [dimensionless]
    shaf_shift:         float # [cm]
    inb_fw_thick:       float # [cm]
    inb_bz_thick:       float # [cm]
    inb_mnfld_thick:    float # [cm]
    inb_vv_thick:       float # [cm]
    tf_thick:           float # [cm]
    outb_fw_thick:      float # [cm]
    outb_bz_thick:      float # [cm]
    outb_mnfld_thick:   float # [cm]
    outb_vv_thick:      float # [cm]
    triang:             float = 0.333 # [dimensionless]
    inb_gap:            float = 20.0 # [cm]
        
# ----------------------------------------------------------------------------------------
# openmc source maker

def create_ring_source(tokamak_geometry):
    """
    Creating simple ring source.
    A more accurate source will slightly affect the wall loadings and dpa profiles.

    Parameters
    ----------
    tokamak_geometry: TokamakGeometry
        Only the tokamak_geometry.major_r & tokamak_geometry.shaf_shift variables are required.
    """
    ring_source = openmc.Source()
    source_radii =      openmc.stats.Discrete(
        [tokamak_geometry.major_r + tokamak_geometry.shaf_shift], [1]
    )
    source_z_values =   openmc.stats.Discrete([0], [1])
    source_angles =     openmc.stats.Uniform(a=0.0, b=2 * pi)
    ring_source.space = openmc.stats.CylindricalIndependent(
        r=source_radii, phi=source_angles, z=source_z_values, origin=(0.0, 0.0, 0.0)
    )
    ring_source.angle = openmc.stats.Isotropic()
    ring_source.energy = openmc.stats.Discrete([energy_per_dt_MeV * (4/5) * eV_per_MeV], [1])
    
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
        'mode'          : 2,
        'temperature'   : 15.4,
        'major_r'       : tokamak_geometry.major_r,
        'minor_r'       : tokamak_geometry.minor_r,
        'elongation'    : tokamak_geometry.elong,
        'triangulation' : tokamak_geometry.triang,
        'radial_shift'  : tokamak_geometry.shaf_shift,
        'peaking_factor': 1.508,
        'vertical_shift': 0.0,
        'start_angle'   : 0.0,
        'angle_range'   : 360.0,
    }
    # Parameter dictionary has to be formatted as string to be provided to the library.
    source_params_str = ",".join(f"{k}={v}" for k,v in source_params_dict.items())

    parametric_source = openmc.Source( library='./PPS_OpenMC.so', 
                              parameters=source_params_str )
    
    return parametric_source

# ----------------------------------------------------------------------------------------

class PoloidalXSPlot(object):
    """Context manager so that we can save the plot as soon as we exit.
    Using the 'with' statement (i.e. in the syntax of context manager in python)
    also improves readability, as the save_name is written at the top of the indented block,
    so it's obvious what's the indented block plotting."""
    def __init__(self, save_name, title=None):
        self.save_name = save_name
        self.ax = plt.subplot()
        self.ax.axis('equal')
        if title:
            self.ax.set_title(title)

    def __enter__(self):
        return self.ax

    def __exit__(self, exception_type, value, traceback):
        plt.savefig(self.save_name)
        # self.ax.cla()
        # self.ax.figure.clf()
        plt.close()


def _load_fw_points(tokamak_geometry, save_plots=True):
    """
    Load given first wall points,
        scale them according to the given major and minor radii,
        then downsample them so that a simplified geometry can be made.

    Currently these are full of magic numbers (e.g. manually chosen integer indices)
        because it was tailored for one specific model.

    Parameters
    ----------
    tokamak_geometry: TokamakGeometry

    Returns
    -------
    new_downsampled_fw: points belonging to the first wall
    new_downsampled_div: points belonging to the divertor
    num_inboard_points: Number of points in new_downsampled_fw that belongs to the inboard section.

    Dataflow diagram for this function
    ----------------------------------
    blanket_face.npy                divertor_face.npy
            ↓                           ↓
    full_blanket_2d_outline         divertor_2d_outline
            ↓                           ↓
    (select plasma-facing part)         ↓
            ↓                           ↓
    inner_blanket_face(ibf)             ↓
            ↓                           ↓
    (downsampling)                  (downsampling)
            ↓                           ↓
    downsampled_ibf                 divertor_2d_outline
            ↓                           ↓
            →           old_points      ←
                            ↓
        (expand points outwards using 'shift_cm')
                            ↓
                        new_points
                            ↓
                    (elongate, stretch)
                            ↓
                        new_points
                        ↓       ↓
    new_downsampled_fw  ←       →  new_downsampled_div

    Units
    -----
    All units for the diagram above are in cgs
    """

    ######## get data ########
    # Get the geometry from existing .npy files, each is an array of 
    # 3D coordinates of points sampled along the divertor first wall outline.
    full_blanket_2d_outline = np.load('blanket_face.npy')[0]
    divertor_2d_outline = np.load('divertor_face.npy')[0]
    ######## unfortunate <magic numbers and magic function> that fits only these npy model. ########
    # The plasma geometry
    ex_pts_maj_r = 900.
    ex_pts_min_r = 290.
    ex_pts_elong = 1.792
    # Specifying how many of the se the number of the selected points that define the inboard
    num_inboard_points = 6
    # indices
    selected_fw_samples = [0, 4, 8, 11, 14, 17, 21, 25, 28, 33, 39, 42, -1] # sample points
    selected_div_samples = [72, 77, 86]   # also going to use first and last points from first wall
    num_points_belongong_to_divertor = len(selected_div_samples)
    def _fix_downsampled_ibf(ds_ibf):
        """Move the point that is too close to plasma
        by moving it closer to the central column instead."""
        ds_ibf[-5][0] = ds_ibf[-5][0] - 25.
        return ds_ibf
    ######## </magic numbers and magic function> ########
    # we will get rid of this whole function later on anyways.
    
    
    ######## select the part of the outline facing the plasma ########
    ibf = inner_blanket_face = full_blanket_2d_outline[52:-2]

    ######## (down)sample existing data ########
    # blanket
    downsampled_ibf = ibf[ selected_fw_samples ] * m_to_cm
    downsampled_ibf = _fix_downsampled_ibf(downsampled_ibf)
    # divertor
    downsampled_divf = divertor_2d_outline[ selected_div_samples ] * m_to_cm
    
    ######## Create the full plasma-facing outline by concatenating existing var ########
    old_points = np.concatenate( (downsampled_ibf, downsampled_divf), axis=0 )
    
    print('FW points before adjustment\n', old_points )
    
    ######## rescale data to fit new geometry. ########
    # Expand point outwards according to new major radius
    shift_cm   = tokamak_geometry.major_r - ex_pts_maj_r
    new_points = mg.shift_points(old_points, shift_cm)
    
    # Adjusting points for elongation and minor radius
    # This elongation also include an allowance for the minor radius
    elong_w_minor_r =  tokamak_geometry.minor_r / ex_pts_min_r * tokamak_geometry.elong
    stretch_r_val  = tokamak_geometry.minor_r / ex_pts_min_r
    new_points = mg.elongate(new_points, elong_w_minor_r / ex_pts_elong)
    new_points = mg.stretch_r(new_points, tokamak_geometry, stretch_r_val)
    
    ######## split 'new_points' into new_downsampled_* variables ########
    new_downsampled_fw = new_points[:-num_points_belongong_to_divertor]
    new_downsampled_div           = np.concatenate( (new_points[-(num_points_belongong_to_divertor+1):],
                                                  new_points[:1]
                                                 ), axis=0 )
    
    ######## plotting. ########
    if save_plots:
        ######## create parametric variables for plotting smoother lines ########
        # https://hibp.ecse.rpi.edu/~connor/education/plasma/PlasmaEngineering/Miyamoto.pdf pg. 239
        # R = R0 + a cos(θ + δ sin θ)
        # where a = minor radius
        #       δ = triangularity
        u = tokamak_geometry.major_r                              # x-position of the center
        v = 0.0                                                   # y-position of the center
        a = tokamak_geometry.minor_r                              # radius on the x-axis
        b = tokamak_geometry.elong * tokamak_geometry.minor_r     # radius on the y-axis
        tri = tokamak_geometry.triang                             # triangularity
        t = np.linspace(0, 2 * pi, 100)

        with PoloidalXSPlot('blanket_face.svg', 'Blanket Face') as ax:
            ax.scatter(full_blanket_2d_outline[:,0], full_blanket_2d_outline[:,2])

        with PoloidalXSPlot('all_points_before_after.svg', 'Points sampled for making the MCNP model') as ax:
            ax.plot(old_points[:,0], old_points[:,2], label='Initial fw points')
            ax.plot(new_points[:,0], new_points[:,2], label='Adjusted fw points')
            ax.plot( u + a * np.cos( t + tri * np.sin(t) ), v + b * np.sin(t), label='Plasma envelope' ) # source envelope
            ax.legend(loc="upper right")
        
        with PoloidalXSPlot('selected_pts_inner_blanket_face.svg',
                'Selected points on the inner blanket') as ax:
            ax.scatter(new_downsampled_fw[:,0], new_downsampled_fw[:,2])
        
        with PoloidalXSPlot('selected_pts_divertor_face.svg',
                'Selected points on the divertor face') as ax:
            ax.scatter(new_downsampled_div[:,0], new_downsampled_div[:,2])

    return new_downsampled_fw, new_downsampled_div, num_inboard_points

# ----------------------------------------------------------------------------------------

def setup_openmc(
    plasma_source,
    particles,
    batches=2,
    photon_transport=True,
    electron_treatment="ttb",
    run_mode="fixed source",
    output_summary=False
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

# ----------------------------------------------------------------------------------------

def create_materials(breeder_materials):
    """
    Parameters
    ----------
    breeder_materials: BreederTypeParameters
        dataclass containing attributes: 'blanket_type', 'li_enrich_ao'
    """
    material_lib = mm.AutoPopulatingMaterialsLibrary.create_complete_material_library(breeder_materials.blanket_type, breeder_materials.li_enrich_ao)
    return material_lib
    # material_lib = mm.AutoPopulatingMaterialsLibrary.create_complete_material_library(breeder_materials.blanket_type, breeder_materials.li_enrich_ao)

# ----------------------------------------------------------------------------------------

def filter_cells(cells_and_cell_lists, material_lib, src_rate):
    """
    Requests cells for scoring.
    Parameters
    ----------
    cells_and_cell_lists:
        dictionary where each item is either a single openmc.Cell,
            or a list of openmc.Cell.
    material_lib: (dict)
        A dictionary (or an instance of MaterialsLibrary, which is an offspring class of dict)
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
            cells_and_cell_lists["divertor_fw_sf"] # sf = surface
            
        ]
        # the cell lists
          + cells_and_cell_lists["inb_vv_cells"]
          + cells_and_cell_lists["inb_mani_cells"] 
          + cells_and_cell_lists["inb_bz_cells"] 
          + cells_and_cell_lists["inb_fw_cells"]
          + cells_and_cell_lists["inb_sf_cells"] # sf = surface
        
          + cells_and_cell_lists["outb_vv_cells"] 
          + cells_and_cell_lists["outb_mani_cells"] 
          + cells_and_cell_lists["outb_bz_cells"] 
          + cells_and_cell_lists["outb_fw_cells"]  
          + cells_and_cell_lists["outb_sf_cells"] # sf = surface
        
          + cells_and_cell_lists["divertor_cells"],
    )
    
    mat_filter = openmc.MaterialFilter(
        [   
            material_lib['inb_fw_mat'],
            material_lib['outb_fw_mat'],
            material_lib['inb_bz_mat'],
            material_lib['outb_bz_mat'],
            material_lib['inb_mani_mat'],
            material_lib['outb_mani_mat'],
            material_lib['inb_vv_mat'],
            material_lib['outb_vv_mat'], 
            material_lib['divertor_mat'], 
            material_lib['div_fw_mat'], 
            material_lib['tf_coil_mat'], 
            material_lib['inb_sf_mat'], # sf = surface
            material_lib['outb_sf_mat'],   # sf = surface
            material_lib['div_sf_mat'] # sf = surface
        ]
    )

    fw_surf_filter = openmc.CellFilter(
        cells_and_cell_lists["inb_sf_cells"] # sf = surface
        + cells_and_cell_lists["outb_sf_cells"] # sf = surface
        + [cells_and_cell_lists["divertor_fw_sf"]] # sf = surface
        + cells_and_cell_lists["inb_fw_cells"]
        + cells_and_cell_lists["outb_fw_cells"]
        + [cells_and_cell_lists["divertor_fw"]]
    )

    neutron_filter = openmc.ParticleFilter(["neutron"])
    photon_filter = openmc.ParticleFilter(["photon"])

    # eV per source particle to MW coefficients
    eV_per_sp_to_MW = src_rate * MJ_per_eV

    MW_energy_bins = [0.0, 100.0e6]  # Up to 100 MeV
    MW_dose_coeffs = [eV_per_sp_to_MW, eV_per_sp_to_MW]
    MW_mult_filter = openmc.EnergyFunctionFilter(MW_energy_bins, MW_dose_coeffs)

    # photon heat flux coefficients (cm per source particle to MW cm)
    # Tally heat flux
    energy_bins = [0.0, 100.0e6]  # Up to 100 MeV
    dose_coeffs = [0.0 * eV_per_sp_to_MW, 100.0e6 * eV_per_sp_to_MW]
    energy_mult_filter = openmc.EnergyFunctionFilter(energy_bins, dose_coeffs)
    
    cyl_mesh = openmc.CylindricalMesh(mesh_id=1)
    cyl_mesh.r_grid = np.linspace(400, 1400, 100+1)
    cyl_mesh.z_grid = np.linspace(-800., 800., 160+1)   
    cyl_mesh_filter = openmc.MeshFilter(cyl_mesh)

    return (
        cell_filter,
        mat_filter,
        fw_surf_filter,
        neutron_filter,
        photon_filter,
        MW_mult_filter,
        energy_mult_filter,
        cyl_mesh_filter
    )

# ----------------------------------------------------------------------------------------

def create_tallies(
    cell_filter,
    mat_filter,
    fw_surf_filter,
    neutron_filter,
    photon_filter,
    MW_mult_filter,
    energy_mult_filter,
    cyl_mesh_filter
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

# ----------------------------------------------------------------------------------------
# result processing

def print_df_decorator_with_title_string(title_string):
    def print_dataframe_decorator(method):
        def dataframe_method_wrapper(self, print_df : bool=True):
            method_output = method(self)
            if print_df:
                print("\n{}\n".format(title_string)+str(method_output))
            return method_output
        return dataframe_method_wrapper
    return print_dataframe_decorator

class OpenMCResult():
    """
    Class that looks opens up the openmc universe from the statepoint file,
        so that the dataframes containing the relevant results
        can be generated and reformatted by its methods.
    """
    def __init__(self, universe, src_rate, statepoint_file="statepoint.2.h5"):
        self.universe = universe
        self.src_rate = src_rate
        self.statepoint_file = statepoint_file
        # Creating cell name dictionary to allow easy mapping to dataframe
        self.cell_names = {}
        for cell_id in self.universe.cells:
            self.cell_names[cell_id] = self.universe.cells[cell_id].name

        # Creating material dictionary to allow easy mapping to dataframe
        self.mat_names = {}
        for cell_id in self.universe.cells:
            try:
                self.mat_names[ self.universe.cells[cell_id].fill.id ] = self.universe.cells[cell_id].fill.name
            except:
                pass

        # Creating cell volume dictionary to allow easy mapping to dataframe
        self.cell_vols = {}
        for cell_id in self.universe.cells:
            self.cell_vols[cell_id] = self.universe.cells[cell_id].volume
        # Loads up the output file from the simulation
        self.statepoint = openmc.StatePoint(self.statepoint_file)
            
    @print_df_decorator_with_title_string("TBR")
    def load_tbr(self):
        """load the TBR value and uncertainty"""
        tally = "TBR"
        self.tbr_df = self.statepoint.get_tally(name=tally).get_pandas_dataframe()
        self.tbr = "{:.2f}".format(self.tbr_df["mean"].sum())
        self.tbr_e = "{:.2f}".format(self.tbr_df["std. dev."].sum())
        return self.tbr, self.tbr_e

    @print_df_decorator_with_title_string("Heating (MW)")
    def load_heating_in_MW(self):
        """load the heating (sorted by material) dataframe"""
        tally = "MW heating"  # 'mean' units are MW
        selected_columns = ["material", "material_name", "nuclide", "score", "mean", "std. dev.", "%err."]
        heating_df = self.statepoint.get_tally(name=tally).get_pandas_dataframe()
        heating_df["material_name"] = heating_df["material"].map(self.mat_names)
        heating_df["%err."] = heating_df.apply(pdf.get_percent_err, axis=1).apply(
            lambda x: "%.1f" % x
        )
        heating_df = heating_df[selected_columns] # rearrange dataframe into this desired order
        self.heating_df = heating_df
        return self.heating_df

    @print_df_decorator_with_title_string("Neutron Wall Load (eV)")
    def load_neutron_wall_loading(self):
        """load the neutron wall load dataframe"""
        dfa_coefs = get_dpa_coefs()
        tally = "neutron wall load"  # 'mean' units are eV per source particle
        n_wl_df = self.statepoint.get_tally(name=tally).get_pandas_dataframe()
        n_wl_df["cell_name"] = n_wl_df["cell"].map(self.cell_names)
        n_wl_df["vol(cc)"] = n_wl_df["cell"].map(self.cell_vols)
        n_wl_df["dpa/fpy"] = (
            n_wl_df["mean"]
            * dfa_coefs.displacements_per_damage_eV
            / (n_wl_df["vol(cc)"] * dfa_coefs.atoms_per_cc)
            * s_in_yr
            * self.src_rate
        )
        n_wl_df["dpa/fpy"] = n_wl_df["dpa/fpy"].apply(lambda x: "%.1f" % x)
        n_wl_df["%err."] = n_wl_df.apply(pdf.get_percent_err, axis=1).apply(
            lambda x: "%.1f" % x
        )
        n_wl_df = n_wl_df.drop(n_wl_df[~n_wl_df["cell_name"].str.contains("Surface")].index)  # ~ invert operator = doesnt contain
        n_wl_df = n_wl_df.reindex([12,0,1,2,3,4,5,6,7,8,9,10,11])
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

        self.neutron_wall_load = n_wl_df
        return self.neutron_wall_load

    @print_df_decorator_with_title_string("Photon Heat Flux MW m-2")
    def load_photon_heat_flux(self):
        """load the photon heaat flux dataframe"""
        tally = "photon heat flux"  # 'mean' units are MW cm
        p_hf_df = self.statepoint.get_tally(name=tally).get_pandas_dataframe()
        p_hf_df["cell_name"] = p_hf_df["cell"].map(self.cell_names)
        p_hf_df["vol(cc)"] = p_hf_df["cell"].map(self.cell_vols)
        p_hf_df["MW_m-2"] = p_hf_df["mean"] / p_hf_df["vol(cc)"] * per_cm2_to_per_m2
        p_hf_df["%err."] = p_hf_df.apply(pdf.get_percent_err, axis=1).apply(
            lambda x: "%.1f" % x
        )
        # Scaling first wall results by factor to surface results
        surface_total = p_hf_df.loc[ p_hf_df["cell_name"].str.contains("FW Surface"),     "MW_m-2"].sum()
        cell_total    = p_hf_df.loc[~p_hf_df["cell_name"].str.contains("FW Surface|PFC"), "MW_m-2"].sum()  # ~ invert operator = doesnt contain
        surface_factor = surface_total / cell_total
        p_hf_df["MW_m-2"] = np.where(~p_hf_df["cell_name"].str.contains("FW Surface|PFC"),
                                               p_hf_df["MW_m-2"] * surface_factor,
                                               p_hf_df["MW_m-2"])
        p_hf_df = p_hf_df.drop(p_hf_df[p_hf_df["cell_name"].str.contains("FW Surface")].index)
        p_hf_df = p_hf_df.drop(p_hf_df[p_hf_df["cell_name"] == "Divertor PFC"].index)
        p_hf_df = p_hf_df.replace('FW','FW Surface', regex=True) #df.replace('Py','Python with ', regex=True)
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

        self.photon_heat_flux = p_hf_df
        return self.photon_heat_flux

    def summarize(self, print_dfs):
        """Run all of the results formatting method available in this class."""
        self.load_tbr(print_dfs)
        self.load_heating_in_MW(print_dfs)
        self.load_neutron_wall_loading(print_dfs)
        self.load_photon_heat_flux(print_dfs)
    
# ----------------------------------------------------------------------------------------

def stochastic_volume_calculation(
    tokamak_geometry, cells_and_cell_lists, particles=4e7
):
    """
    Parameters
    ----------
    tokamak_geometry: TokamakGeometry
    cells_and_cell_lists: dict
        dictionary where each item is either a single openmc.Cell,
            or a list of openmc.Cell.
    particles:
        how many randomly generated particle to use
        for the stochastic volume calculation.
    """
    # Performs a stochastic volume calculation for the cells
    
    import os
    # quietly delete the unused .hf files
    try:
        os.remove("summary.h5")
        os.remove("statepoint.1.h5")
    except OSError:
        pass

    # maximum radii and heigth reached by all of the tokamak's breeder zone component initi
    maxr = (
        tokamak_geometry.major_r
        + tokamak_geometry.minor_r
        + tokamak_geometry.outb_fw_thick
        + tokamak_geometry.outb_bz_thick
    )
    maxz = (
        # height of plasma = 2 * elong * minor
        tokamak_geometry.elong * tokamak_geometry.minor_r
        + tokamak_geometry.outb_fw_thick
        + tokamak_geometry.outb_bz_thick
        + tokamak_geometry.outb_mnfld_thick
        + tokamak_geometry.outb_vv_thick
    )
    # draw the bounding box for the simulation.
    lower_left = (-maxr, -maxr, -maxz)
    upper_right = (maxr, maxr, maxz)
    cell_vol_calc = openmc.VolumeCalculation(
        cells_and_cell_lists["inb_fw_cells"]
        + [cells_and_cell_lists["divertor_fw"]]
        + cells_and_cell_lists["outb_fw_cells"],
        int(particles),
        lower_left,
        upper_right,
    )
    settings = openmc.Settings()
    settings.volume_calculations = [cell_vol_calc]

    settings.export_to_xml()
    # within this bounding box, use naive Monte Carlo to find
    # the volumes of the cells representing the tokamak components.
    openmc.calculate_volumes()
    
# ----------------------------------------------------------------------------------------

def geometry_plotter(cells_and_cell_lists, tokamak_geometry):
    """
    Uses the OpenMC plotter to produce an image of the modelled geometry

    Parameters
    ----------
    cells_and_cell_lists:
        dictionary where each item is either a single openmc.Cell,
            or a list of openmc.Cell.
    tokamak_geometry : TokamakGeometry
    """

    # Assigning colours for plots
    cell_color_assignment = {
        cells_and_cell_lists["tf_coil_cell"]: "brown",
        cells_and_cell_lists["plasma_inner1"]: "dimgrey",
        cells_and_cell_lists["plasma_inner2"]: "grey",
        cells_and_cell_lists["plasma_outer1"]: "darkgrey",
        cells_and_cell_lists["plasma_outer2"]: "dimgrey",
        cells_and_cell_lists["divertor_inner1"]: "grey",
        cells_and_cell_lists["divertor_inner2"]: "dimgrey",
        cells_and_cell_lists["outer_vessel_cell"]: "white",
        cells_and_cell_lists["inb_vv_cells"][0]: "red", 
        cells_and_cell_lists["outb_vv_cells"][1]:  "orange", 
        cells_and_cell_lists["outb_vv_cells"][2]:  "yellow", 
    }

    mat_color_assignment = {
        cells_and_cell_lists["bore_cell"]: "blue",
        cells_and_cell_lists["tf_coil_cell"]: "brown",
        cells_and_cell_lists["plasma_inner1"]: "white",
        cells_and_cell_lists["plasma_inner2"]: "white",
        cells_and_cell_lists["plasma_outer1"]: "white",
        cells_and_cell_lists["plasma_outer2"]: "white",
        cells_and_cell_lists["divertor_inner1"]: "white",
        cells_and_cell_lists["divertor_inner2"]: "white",
        cells_and_cell_lists["divertor_fw"]:     "red",
        cells_and_cell_lists["outer_vessel_cell"]: "white",
        cells_and_cell_lists["outer_container"]:   "darkgrey",
    }
    
    def color_cells(prefixed_cell_type, color):
        for i in range(len(cells_and_cell_lists[prefixed_cell_type+"_cells"])):
            mat_color_assignment[cells_and_cell_lists[prefixed_cell_type+"_cells"][i]] = color
    # first wall: red
    color_cells('outb_fw',   'red')
    color_cells( 'inb_fw',   'red')
    # breeding zone: yellow
    color_cells('outb_bz',   'yellow')
    color_cells( 'inb_bz',   'yellow')
    # manifold: green
    color_cells('outb_mani', 'green')
    color_cells( 'inb_mani', 'green')
    # vacuum vessel: grey
    color_cells('outb_vv',   'grey')
    color_cells( 'inb_vv',   'grey')
    # divertor: cyan
    color_cells('divertor',  'cyan')

    plot_width = 2 * (
        tokamak_geometry.major_r
        + tokamak_geometry.minor_r * tokamak_geometry.elong
        + tokamak_geometry.outb_fw_thick
        + tokamak_geometry.outb_bz_thick
        + tokamak_geometry.outb_mnfld_thick
        + tokamak_geometry.outb_vv_thick
        + 200.0                                 # margin
    )  

    plot_list = []
    for _, basis in enumerate(["xz", "xy", "yz"]):
        plot = openmc.Plot()
        plot.basis = basis
        plot.pixels = [400,400]
        plot.width = (plot_width, plot_width)
        if basis == "yz":
            plot.colors = cell_color_assignment
        else:
            plot.colors = mat_color_assignment          
        plot.filename = f"./plots_{basis}"

        plot_list.append(plot)

    openmc.Plots(plot_list).export_to_xml()
    openmc.plot_geometry()
    
# ----------------------------------------------------------------------------------------

class TBRHeatingSimulation():
    """
    Contains all the data necessary to run the openmc simulation of the tbr,
    and the relevant pre-and post-processing.
    """
    def __init__(self, runtime_variables, operation_variable, breeder_materials, tokamak_geometry):
        self.runtime_variables = runtime_variables
        self.operation_variable = operation_variable
        self.breeder_materials = breeder_materials
        self.tokamak_geometry = tokamak_geometry

        self.cells = None
        self.material_lib = None
        self.universe = None

    def setup(self, plot_geometry=True):
        """
        plot the geometry and saving them as hard-coded png names.
        """
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
            self.runtime_variables.openmc_write_summary
            )

        blanket_points, div_points, num_inboard_points = _load_fw_points(self.tokamak_geometry, True)
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
            geometry_plotter(self.cells, self.tokamak_geometry)
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
        assert self.universe is not None, "The self.universe variable must have been first populated by self.run()!"
        self.result = OpenMCResult(self.universe, self.src_rate)
        self.result.summarize(print_summary)
        return self.result

    def calculate_volume_stochastically(self):
        stochastic_volume_calculation(
            self.tokamak_geometry,
            self.cells,
            self.runtime_variables.volume_calc_particles,
        )

################################################################################################################

if __name__ == "__main__":
    import sys

    SimulatedBluemiraOutputVariables = namedtuple('SimulatedBluemiraOutputVariables', 'breeder_materials, tokamak_geometry')

    def get_preset_physical_properties(blanket_type):
        """
        Works as a switch-case for choosing the tokamak geometry and blankets for a given blanket type.
        The allowed list of blanket types are specified in mm.BlanketType.
        Currently, the blanket types with pre-populated data in this function are:
            {'wcll', 'dcll', 'hcpb'}
        """
        breeder_materials = BreederTypeParameters(
            blanket_type = blanket_type,
            li_enrich_ao = 60.0         # atomic fraction percentage of lithium
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

        plasma_shape = {"minor_r": 288.3,
                        "major_r": 893.8,
                        "elong": 1.65,
                        "shaf_shift": 0.0,} # The shafranov shift of the plasma
        if blanket_type == "wcll":
            tokamak_geometry = TokamakGeometry(
                **plasma_shape,
                inb_fw_thick =     2.7,
                inb_bz_thick =     37.8,
                inb_mnfld_thick =  43.5,
                inb_vv_thick =     60.0,
                tf_thick =         40.0,
                outb_fw_thick =    2.7,
                outb_bz_thick =    53.8,
                outb_mnfld_thick = 42.9,
                outb_vv_thick =    60.0,
            )
        elif blanket_type == "dcll":
            tokamak_geometry = TokamakGeometry(
                **plasma_shape,
                inb_fw_thick =     2.2,
                inb_bz_thick =     30.0,
                inb_mnfld_thick =  17.8,
                inb_vv_thick =     60.0,
                tf_thick =         40.0,
                outb_fw_thick =    2.2,
                outb_bz_thick =    64.0,
                outb_mnfld_thick = 24.8,
                outb_vv_thick =    60.0,
            )
        elif blanket_type == "hcpb":
            # HCPB Design Report, 26/07/2019
            tokamak_geometry = TokamakGeometry(
                **plasma_shape,
                inb_fw_thick =     2.7,
                inb_bz_thick =     46.0,
                inb_mnfld_thick =  56.0,
                inb_vv_thick =     60.0,
                tf_thick =         40.0,
                outb_fw_thick =    2.7,
                outb_bz_thick =    46.0,
                outb_mnfld_thick = 56.0,
                outb_vv_thick =    60.0,
            )
        elif blanket_type in mm.BlanketType.allowed_types():
            raise Exception(f"Programmer error: {blanket_type} is a valid BlanketType but is not implemented in {sys._getframe().f_code.co_name}.")
        else:
            raise ValueError(f"{blanket_type} is not a supported material!")

        return SimulatedBluemiraOutputVariables(breeder_materials, tokamak_geometry)

    breeder_materials, tokamak_geometry = get_preset_physical_properties('hcpb') # implemented blanket_type:{'wcll', 'dcll', 'hcpb'}

    runtime_variables = OpenMCSimulationRuntimeParameters(
            particles=16800,        # 16800 takes 5 seconds,  1000000 takes 280 seconds.
            batches=2,
            photon_transport=True,
            electron_treatment="ttb",
            run_mode="fixed source",
            openmc_write_summary=False,

            parametric_source=True,
            volume_calc_particles=4e8, # only used if stochastic_volume_calculation is turned on.
            )

    operation_variable = TokamakOperationParameters(reactor_power_MW=1998.0)

    # set up a DEMO-like reactor, and run OpenMC simualtion
    tbr_heat_sim = TBRHeatingSimulation(runtime_variables, operation_variable, breeder_materials, tokamak_geometry)
    # import sys; sys.exit()
    tbr_heat_sim.setup(True)
    tbr_heat_sim.run()
    # get the TBR, component heating, first wall dpa, and photon heat flux 
    tbr_heat_sim.get_result(True)
    # tbr_heat_sim.calculate_volume_stochastically() # don't do this because it takes a long time.