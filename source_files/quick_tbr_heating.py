import openmc
import math
import pandas as pd
import matplotlib.pyplot as plt
import make_materials as mm
import make_geometry as mg
import pandas_df_functions as pdf

pi = math.pi

def run_tbr_heating_calc(runtime_variables, reactor_power_MW, blanket_type, material_variables, geometry_variables):
    
    # Runs OpenMC calculation to get TBR, component heating, first wall dpa, and photon heat flux
    # for a DEMO-like reactor
    
    stochastic_vol_calc = runtime_variables['stochastic_vol_calc']   # Boolean for performing stochastic volume calc
    plot_geometry       = runtime_variables['plot_geometry']         # Boolean for plotting geometry
    no_of_particles     = runtime_variables['no_of_particles']       # Number of particles in each of two batches
    
    li_enrich_ao = material_variables['li_enrich_ao']
   

    #########################################
    # Calculating source strength
    #########################################

    energy_per_fusion_MeV = 17.58
    MJ_per_MeV = 1.6022e-19
    MJ_per_eV =  1.6022e-25

    src_str = reactor_power_MW / ( energy_per_fusion_MeV * MJ_per_MeV )

    s_in_yr = 365.25 * 24 * 60 * 60

    ##########################
    ### CREATING MATERIALS ###
    ##########################

    if blanket_type == 'hcpb':
        material_lib = mm.make_hcpb_mats(li_enrich_ao)
    else:
        raise ValueError("blanket_type must be either hcpb")
        
    #########################
    ### CREATING GEOMETRY ###
    #########################
    
    mg.check_geometry(geometry_variables)

    cells, universe = mg.make_geometry(geometry_variables, material_lib)

    #######################
    ### OPENMC SETTINGS ###
    #######################

    # Creating simple ring source
    # A more accurate source will slightly affect the wall loadings and dpa profiles       
    my_source = openmc.Source()
    source_radii = openmc.stats.Discrete([geometry_variables['major_r'] + geometry_variables['shaf_shift']], [1])
    source_z_values = openmc.stats.Discrete([0], [1])
    source_angles = openmc.stats.Uniform(a=0., b=2*pi)
    my_source.space = openmc.stats.CylindricalIndependent(r=source_radii, phi=source_angles, z=source_z_values, origin=(0.0, 0.0, 0.0))
    my_source.angle = openmc.stats.Isotropic()
    my_source.energy = openmc.stats.Discrete([14.06e6], [1])

    settings = openmc.Settings()
    settings.source = my_source
    settings.particles = no_of_particles
    settings.batches = 2
    settings.photon_transport = True
    settings.electron_treatment = 'ttb'
    settings.run_mode = 'fixed source'
    settings.output = {'summary' : False}             # Assuming 293K temperature for nuclear cross-sections for calculation speed

    settings.export_to_xml()


    ###############
    ### TALLIES ###
    ###############

    cell_filter = openmc.CellFilter([cells['tf_coil_cell'],
                                     cells['vv_inb_cell'],
                                     cells['manifold_inb_cell'],
                                     cells['bz_inb_cell'],
                                     cells['fw_inb_cell'],
                                     cells['divertor_cell'], 
                                     # div_surf_cell,
                                     # inner_vessel_cell, 
                                     cells['fw_cell'], 
                                     cells['bz_cell'], 
                                     cells['manifold_cell'],
                                     cells['vv_cell']
                                     ])

    fw_surf_cell_filter = openmc.CellFilter( cells['fw_inb_surf_cells'][1:-1] +   # inb top and bottom divisions are behind blanket
                                             [cells['div_surf_cell']] + 
                                             cells['fw_outb_surf_cells'] )

    neutron_filter = openmc.ParticleFilter(['neutron'])
    photon_filter  = openmc.ParticleFilter(['photon'])


    #############################################
    # eV per source particle to MW coefficients
    ##############################################
    eV_per_sp_to_MW = src_str * MJ_per_eV

    MW_energy_bins = [0.,              100.e6         ]             # Up to 100 MeV
    MW_dose_coeffs = [eV_per_sp_to_MW, eV_per_sp_to_MW] 
    MW_mult_filter = openmc.EnergyFunctionFilter( MW_energy_bins, MW_dose_coeffs )


    ######################################################################
    # photon heat flux coefficients (cm per source particle to MW cm)
    ######################################################################

    # Tally heat flux
    energy_bins = [0.,                   100.e6]         # Up to 100 MeV
    dose_coeffs = [0. * eV_per_sp_to_MW, 100.e6 * eV_per_sp_to_MW] 
    energy_mult_filter = openmc.EnergyFunctionFilter( energy_bins, dose_coeffs )


    #########################################
    # tallies
    #########################################

    tally_tbr = openmc.Tally(name='TBR')
    tally_tbr.scores = ['(n,Xt)']

    tally_heating = openmc.Tally(name='heating')     # eV per sp
    tally_heating.scores = ['heating']
    tally_heating.filters = [cell_filter]

    tally_heating_MW = openmc.Tally(name='MW heating')     # MW
    tally_heating_MW.scores = ['heating']
    tally_heating_MW.filters = [cell_filter, MW_mult_filter]

    tally_n_wall_load = openmc.Tally(name='neutron wall load')
    tally_n_wall_load.scores = ['damage-energy']
    tally_n_wall_load.filters = [fw_surf_cell_filter, neutron_filter]  

    tally_p_heat_flux = openmc.Tally(name='photon heat flux')
    tally_p_heat_flux.scores = ['flux']
    tally_p_heat_flux.filters = [fw_surf_cell_filter, photon_filter, energy_mult_filter]   


    tallies = openmc.Tallies( [tally_tbr, 
                               tally_heating, 
                               tally_heating_MW,
                               tally_n_wall_load,
                               tally_p_heat_flux] )
    tallies.export_to_xml()


    ####################
    ### Plot geometry
    ####################
    
    if plot_geometry:

        color_assignment = {cells['tf_coil_cell']: 'brown',
                            cells['vv_inb_cell']: 'orange',
                            cells['manifold_inb_cell']: 'purple',
                            cells['bz_inb_cell']: 'grey',
                            cells['fw_inb_cell']: 'red',
                            cells['inner_vessel_cell']: 'blue' ,
                            cells['divertor_cell']: 'cyan',
                            cells['fw_cell']: 'red', 
                            cells['bz_cell']: 'grey', 
                            cells['manifold_cell']: 'purple', 
                            cells['vv_cell']: 'orange',
                            cells['outer_vessel_cell1']: 'white',
                            cells['outer_vessel_cell2']: 'white',
                            cells['outer_vessel_cell3']: 'white',
                            cells['fw_outb_surf_cells'][0]: 'red',
                            cells['fw_outb_surf_cells'][1]: 'orange',
                            }

        plot_width = 2 * (geometry_variables['major_r'] + \
                          geometry_variables['minor_r'] + \
                          geometry_variables['outb_fw_thick'] + \
                          geometry_variables['outb_bz_thick'] + \
                          geometry_variables['outb_mnfld_thick'] + \
                          geometry_variables['outb_vv_thick'] + \
                          100.0 )       # margin

        plot1 = openmc.Plot()
        plot1.basis = 'xz'
        plot1.width = (plot_width, plot_width)
        plot1.colors = color_assignment
        plot1.filename = './plots_1'

        plot2 = openmc.Plot()
        plot2.basis = 'xy'
        plot2.width = (plot_width, plot_width)
        plot2.colors = color_assignment
        plot2.filename = './plots_2'

        plot3 = openmc.Plot()
        plot3.basis = 'yz'
        plot3.width = (plot_width, plot_width)
        plot3.colors = color_assignment
        plot3.filename = './plots_3'

        plots = openmc.Plots( [plot1, plot2, plot3] )
        plots.export_to_xml()

        openmc.plot_geometry()


    ################
    ### Run time ###
    ################

    # Start the OpenMC calculation
    openmc.run()


    #########################################
    # dpa coefficients
    #########################################

    # number of atoms in region = avogadro * density * volume / molecular mass
    # number of atoms in cc     = avogadro * density          / molecular mass
    # dpa_fpy = displacements / atoms * s_in_yr * src_str

    avogadro = 6.022e23          # atoms per mole
    fe_molar_mass_g = 55.845
    fe_density_g_cc = 7.78       # g/cc
    dpa_fe_threshold_eV = 40

    atoms_per_cc = avogadro * fe_density_g_cc / fe_molar_mass_g
    
    # dpa formula
    displacements_per_damage_eV = 0.8 / (2 * dpa_fe_threshold_eV )

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
    statepoint_file = "statepoint.2.h5"
    statepoint = openmc.StatePoint( statepoint_file )

    tally = "TBR" 
    tbr_df = statepoint.get_tally(name=tally).get_pandas_dataframe()
    tbr   = "{:.2f}".format( tbr_df['mean'].sum() )
    tbr_e = "{:.2f}".format( tbr_df['std. dev.'].sum() )
    print("\nTBR\n", tbr, tbr_e)

    tally = "MW heating"        # 'mean' units are MW
    heating_df = statepoint.get_tally(name=tally).get_pandas_dataframe()
    heating_df['cell_name'] = heating_df['cell'].map( cell_names )
    heating_df['%err.'] = heating_df.apply(pdf.get_percent_err, axis = 1).apply(lambda x: '%.1f' % x)
    heating_df = heating_df[[ 'cell', 'cell_name', 'nuclide', 'score', 'mean', 'std. dev.', '%err.' ]]
    print("\nHeating (MW)\n", heating_df)

    tally = "neutron wall load"  # 'mean' units are eV per source particle
    n_wl_df = statepoint.get_tally(name=tally).get_pandas_dataframe()
    n_wl_df['cell_name'] = n_wl_df['cell'].map( cell_names )
    n_wl_df['vol(cc)'] = n_wl_df['cell'].map( cell_vols )
    n_wl_df['dpa/fpy'] = n_wl_df['mean'] * displacements_per_damage_eV / ( n_wl_df['vol(cc)'] * atoms_per_cc ) * s_in_yr * src_str
    n_wl_df['dpa/fpy'] = n_wl_df['dpa/fpy'].apply(lambda x: '%.1f' % x)
    n_wl_df['%err.'] = n_wl_df.apply(pdf.get_percent_err, axis = 1).apply(lambda x: '%.1f' % x)
    n_wl_df = n_wl_df[[ 'cell', 'cell_name', 'particle', 'nuclide', 'score', 'vol(cc)', 'mean', 'std. dev.', 'dpa/fpy', '%err.' ]]
    print("\nNeutron Wall Load (eV)\n", n_wl_df)

    per_cm2_to_per_m2 = 1.e4

    tally = "photon heat flux"  # 'mean' units are MW cm
    p_hf_df = statepoint.get_tally(name=tally).get_pandas_dataframe()           
    p_hf_df['cell_name'] = p_hf_df['cell'].map( cell_names )
    p_hf_df['vol(cc)'] = p_hf_df['cell'].map( cell_vols )
    p_hf_df['MW_m-2'] = p_hf_df['mean'] / p_hf_df['vol(cc)'] * per_cm2_to_per_m2
    p_hf_df['%err.'] = p_hf_df.apply(pdf.get_percent_err, axis = 1).apply(lambda x: '%.1f' % x)
    p_hf_df = p_hf_df[[ 'cell', 'cell_name', 'particle', 'nuclide', 'score', 'vol(cc)', 'mean', 'std. dev.', 'MW_m-2', '%err.' ]]
    print("\nPhoton Heat Flux MW m-2\n", p_hf_df)


    ##############################################
    ### Stochastic volume calculation ###
    ##############################################

    if stochastic_vol_calc:

        import os

        try:
            os.remove('summary.h5')
            os.remove('statepoint.1.h5')
        except OSError:
            pass

        maxr = dummy_maj_r + dummy_min_r + outb_fw_thick + outb_bz_thick

        lower_left = (-maxr, -maxr, -maxz)
        upper_right = (maxr, maxr, maxz)
        no_of_particles_for_vol_calc = 40000000000
        cell_vol_calc = openmc.VolumeCalculation(fw_inb_surf_cells + [div_surf_cell] + fw_outb_surf_cells,
                                                 no_of_particles_for_vol_calc,
                                                 lower_left, 
                                                 upper_right)
        settings = openmc.Settings()
        settings.dagmc = True
        settings.volume_calculations = [cell_vol_calc]

        settings.export_to_xml()
        openmc.calculate_volumes()

        try:
            os.remove('summary.h5')
            os.remove('statepoint.2.h5')
        except OSError:
            pass
        
################################################################################################################
    
if __name__ == "__main__":
    
    # Calculation runtime variables
    runtime_variables = { 'stochastic_vol_calc': False,  # Do a stochastic volume calculation - this takes a long time!
                          'plot_geometry': True,
                          'no_of_particles': 18000 }     # 18000 = 5 seconds,  1000000 = 259 seconds }

    reactor_power_MW = 1998.
    
    blanket_type = 'hcpb' 
    
    material_variables = {'li_enrich_ao': 60.}   # atomic fraction percentage of lithium
    
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
    
    geometry_variables = {  'minor_r': 288.3,
                            'major_r': 893.8,
                            'elong':   1.65,
                            'shaf_shift': 60.0,       # The shafranov shift of the plasma

                            'inb_fw_thick': 2.2,
                            'inb_bz_thick': 35.,
                            'inb_mnfld_thick': 41.,
                            'inb_vv_thick': 60.,
                            'tf_thick': 40.,

                            'outb_fw_thick': 2.2,
                            'outb_bz_thick': 57.,
                            'outb_mnfld_thick': 56.,
                            'outb_vv_thick': 60.,

                            'divertor_width': 200.0 }

    run_tbr_heating_calc(runtime_variables, reactor_power_MW, blanket_type, material_variables, geometry_variables)
