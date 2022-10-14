import openmc

def make_common_mats():
    
    # Makes materials that are common to all blankets

    tungsten_mat = openmc.Material(name='tungsten')
    tungsten_mat.add_nuclide('W180', 0.266, percent_type='ao')
    tungsten_mat.add_nuclide('W182', 0.143, percent_type='ao')
    tungsten_mat.add_nuclide('W183', 0.307, percent_type='ao')
    tungsten_mat.add_nuclide('W184', 0.284, percent_type='ao')
    tungsten_mat.set_density('g/cm3', 19.3)

    eurofer_mat = openmc.Material(name='eurofer')
    eurofer_mat.add_element('Fe', 0.9006, percent_type='wo')
    eurofer_mat.add_element('Cr', 0.0886, percent_type='wo')
    eurofer_mat.add_nuclide('W180', 0.0108 * 0.266, percent_type='wo')
    eurofer_mat.add_nuclide('W182', 0.0108 * 0.143, percent_type='wo')
    eurofer_mat.add_nuclide('W183', 0.0108 * 0.307, percent_type='wo')
    eurofer_mat.add_nuclide('W184', 0.0108 * 0.284, percent_type='wo')
    eurofer_mat.set_density('g/cm3', 7.78)

    water_mat = openmc.Material(name='water')
    water_mat.add_nuclide('H1',  0.667, percent_type='ao')
    water_mat.add_nuclide('O16', 0.333, percent_type='ao')
    water_mat.set_density('g/cm3', 0.866)
    
    return tungsten_mat, eurofer_mat, water_mat


def make_hcpb_mats(li_enrich_ao):
    
    # This function creates openmc material definitions for an hcpb blanket
    
    tungsten_mat, eurofer_mat, water_mat = make_common_mats()

    he_cool_mat = openmc.Material(name='helium')
    he_cool_mat.add_nuclide('He4', 1.0, percent_type='ao')
    he_cool_mat.set_density('g/cm3', 0.008867)

    Be12Ti_mat = openmc.Material(name='Be12Ti')
    Be12Ti_mat.add_element('Be', 12.0, percent_type='ao')
    Be12Ti_mat.add_element('Ti', 1.0, percent_type='ao')
    Be12Ti_mat.set_density('g/cm3', 2.26)

    # Making enriched Li4SiO4 from elements with enrichment of Li6 enrichment
    Li4SiO4_mat = openmc.Material(name='lithium_orthosilicate')
    Li4SiO4_mat.add_element('Li', 4.0, percent_type='ao',
                            enrichment=li_enrich_ao,
                            enrichment_target='Li6',
                            enrichment_type='ao'
                            )
    Li4SiO4_mat.add_nuclide('Si28', 1.0, percent_type='ao')
    Li4SiO4_mat.add_nuclide('O16', 4.0, percent_type='ao')
    Li4SiO4_mat.set_density('g/cm3', 2.247 + 0.078 * (100. - li_enrich_ao) / 100.)  

    Li2TiO3_mat = openmc.Material(name='lithium_titanate')
    Li2TiO3_mat.add_element('Li', 2.0, percent_type='ao',
                            enrichment=li_enrich_ao,
                            enrichment_target='Li6',
                            enrichment_type='ao'
                            )
    Li2TiO3_mat.add_element('Ti', 1.0, percent_type='ao')
    Li2TiO3_mat.add_nuclide('O16', 3.0, percent_type='ao')
    Li2TiO3_mat.set_density('g/cm3', 3.28 + 0.06 * (100. - li_enrich_ao) / 100.) 

    KALOS_ACB_mat = openmc.Material.mix_materials(
        name='kalos_acb', # optional name of homogeneous material
        materials=[Li4SiO4_mat,
                   Li2TiO3_mat],
        fracs=[9*0.65 / (9*0.65 + 6*0.35), 
               6*0.35 / (9*0.65 + 6*0.35)],  # molar combination adjusted to atom fractions
        percent_type='ao')                   # combination fraction type is by atom fraction

    KALOS_ACB_mat.set_density('g/cm3', 2.52 * 0.642)    # applying packing fraction
                                                        # Ref: Current status and future perspectives 
                                                        #  of EU ceramic breeder development
            
    water_cooled_steel_mat = openmc.Material.mix_materials(
            name='water_cooled_steel',      # optional name of homogeneous material
            materials=[eurofer_mat,
                       water_mat],
            fracs=[0.6,
                   0.4],                    # molar combination adjusted to atom fractions
            percent_type='vo')              # combination fraction type is by atom fraction
    
    ### Making first wall
    fw_mat = openmc.Material.mix_materials(
        name='first_wall', # optional name of homogeneous material
        materials=[tungsten_mat,
                   eurofer_mat,
                   he_cool_mat],
        fracs=[2. / 22., 
               20. * 0.597 / 22.,
               20. * 0.403 / 22.],      # molar combination adjusted to atom fractions
        percent_type='vo')              # combination fraction type is by atom fraction


    ### Making blanket
    structural_fraction_vo = 0.128 
    helium_fraction_vo = 0.062
    breeder_fraction_vo = 0.163
    multiplier_fraction_vo = 0.647
    
    HCPB_BZ_mat = openmc.Material.mix_materials(
        name='hcpb_bz',                # optional name of homogeneous material
        materials=[eurofer_mat,
                   Be12Ti_mat,
                   KALOS_ACB_mat,
                   he_cool_mat],
        fracs=[structural_fraction_vo, 
               breeder_fraction_vo,
               multiplier_fraction_vo,
               helium_fraction_vo],    # molar combination adjusted to atom fractions
        percent_type='vo')             # combination fraction type is by atom fraction

    HCPB_manifold_mat = openmc.Material.mix_materials(
        name='hcpb_manifold',          # optional name of homogeneous material
        materials=[eurofer_mat,
                   KALOS_ACB_mat,
                   he_cool_mat],
        fracs=[0.4724, 
               0.0241,
               0.5035],                # molar combination adjusted to atom fractions
        percent_type='vo')             # combination fraction type is by atom fraction


    materials = openmc.Materials([fw_mat, HCPB_BZ_mat, HCPB_manifold_mat, water_cooled_steel_mat, eurofer_mat])

    materials.export_to_xml()
    
    material_lib = {'fw_mat':                 fw_mat, 
                    'bz_mat':                 HCPB_BZ_mat, 
                    'manifold_mat':           HCPB_manifold_mat, 
                    'water_cooled_steel_mat': water_cooled_steel_mat,
                    'eurofer_mat':            eurofer_mat}
    
    return material_lib

################################################################################################

def make_dcll_mats(li_enrich_ao):
    
    # This function creates openmc material definitions for a dcll blanket
    # 
    
    tungsten_mat, eurofer_mat, water_mat = make_common_mats()

    he_cool_mat = openmc.Material(name='helium')
    he_cool_mat.add_nuclide('He4', 1.0, percent_type='ao')
    he_cool_mat.set_density('g/cm3', 0.008867)

    al2o3_mat = openmc.Material(name='Aluminium Oxide')
    al2o3_mat.add_nuclide('Al27', 1.0, percent_type='ao')
    al2o3_mat.add_nuclide('O16',  1.0, percent_type='ao')
    al2o3_mat.set_density('g/cm3', 3.95)

    PbLi_mat = openmc.Material(name='PbLi')
    PbLi_mat.add_element('Pb', 0.83, percent_type='ao')
    PbLi_mat.add_element('Li', 0.17, percent_type='ao',
                            enrichment=li_enrich_ao,
                            enrichment_target='Li6',
                            enrichment_type='ao'
                            )
    PbLi_mat.set_density('g/cm3', 9.4)
    
    lined_euro_mat = openmc.Material.mix_materials(
        name='Eurofer with Al2O3 lining',              # optional name of homogeneous material
        materials=[eurofer_mat,
                   al2o3_mat],
        fracs=[2. / 2.4, 
               0.4 / 2.4 ],                            # molar combination adjusted to atom fractions
        percent_type='vo')                             # combination fraction type is by atom fraction    

    # Divertor definition from Neutronic analyses of the preliminary 
    #  design of a DCLL blanket for the EUROfusion DEMO power, 24 March 2016 
    # Using Eurofer instead of SS316LN
    water_cooled_steel_mat = openmc.Material.mix_materials(
            name='water_cooled_steel',      # optional name of homogeneous material
            materials=[eurofer_mat,
                       water_mat],
            fracs=[0.8,
                   0.2],                    # molar combination adjusted to atom fractions
            percent_type='vo')              # combination fraction type is by atom fraction
    
    ### Making first wall
    fw_mat = openmc.Material.mix_materials(
        name='first_wall',              # optional name of homogeneous material
        materials=[tungsten_mat,
                   eurofer_mat,
                   he_cool_mat,
                   lined_euro_mat],
        fracs=[2. / 27., 
               1.5 / 27.,
               12. / 27.,
               11.5 / 27.],             # molar combination adjusted to atom fractions
        percent_type='vo')              # combination fraction type is by atom fraction


    ### Making blanket 
    DCLL_BZ_mat = openmc.Material.mix_materials(
        name='dcll_bz',                # optional name of homogeneous material
        materials=[lined_euro_mat,
                   PbLi_mat],
        fracs=[0.0605 + 0.9395 * 0.05,
               0.9395 * 0.95],         # molar combination adjusted to atom fractions
        percent_type='vo')             # combination fraction type is by atom fraction

    DCLL_manifold_mat = openmc.Material.mix_materials(
        name='dcll_manifold',          # optional name of homogeneous material
        materials=[eurofer_mat,
                   DCLL_BZ_mat],
        fracs=[0.573, 
               0.426],                 # molar combination adjusted to atom fractions
        percent_type='vo')             # combination fraction type is by atom fraction


    materials = openmc.Materials([fw_mat, DCLL_BZ_mat, DCLL_manifold_mat, water_cooled_steel_mat, eurofer_mat])

    materials.export_to_xml()
    
    material_lib = {'fw_mat':                 fw_mat, 
                    'bz_mat':                 DCLL_BZ_mat, 
                    'manifold_mat':           DCLL_manifold_mat, 
                    'water_cooled_steel_mat': water_cooled_steel_mat,
                    'eurofer_mat':            eurofer_mat}
    
    return material_lib


################################################################################################

def make_wcll_mats(li_enrich_ao):
    
    # This function creates openmc material definitions for a wcll blanket
    # Ref. D. Nevo and M. Oron-Carl, WCLL Design Report 2018, Eurofusion, WPBB-DEL-BB-3.2.1-T005-D001, June 2019. 
    
    tungsten_mat, eurofer_mat, water_mat = make_common_mats()

    PbLi_mat = openmc.Material(name='PbLi')
    PbLi_mat.add_element('Pb', 0.83, percent_type='ao')
    PbLi_mat.add_element('Li', 0.17, percent_type='ao',
                            enrichment=li_enrich_ao,
                            enrichment_target='Li6',
                            enrichment_type='ao'
                            )
    PbLi_mat.set_density('g/cm3', 9.538)

    # Divertor definition from Neutronic analyses of the preliminary 
    #  design of a DCLL blanket for the EUROfusion DEMO power, 24 March 2016 
    # Using Eurofer instead of SS316LN
    water_cooled_steel_mat = openmc.Material.mix_materials(
            name='water_cooled_steel',      # optional name of homogeneous material
            materials=[eurofer_mat,
                       water_mat],
            fracs=[0.6,
                   0.4],                    # molar combination adjusted to atom fractions
            percent_type='vo')              # combination fraction type is by atom fraction
    
    ### Making first wall
    fw_mat = openmc.Material.mix_materials(
        name='first_wall',              # optional name of homogeneous material
        materials=[tungsten_mat,
                   water_mat,
                   eurofer_mat],
        fracs=[0.0766, 
               0.1321,
               0.7913],                
        percent_type='vo')          

    ### Making blanket 
    WCLL_BZ_mat = openmc.Material.mix_materials(
        name='wcll_bz',              # optional name of homogeneous material
        materials=[tungsten_mat,
                   PbLi_mat,
                   water_mat,
                   eurofer_mat],
        fracs=[0.0004, 
               0.8238,
               0.0176,
               0.1582],               
        percent_type='vo')          

    WCLL_manifold_mat = openmc.Material.mix_materials(
        name='wcll_manifold',              # optional name of homogeneous material
        materials=[PbLi_mat,
                   water_mat,
                   eurofer_mat],
        fracs=[0.2129,
               0.2514,
               0.5357],               
        percent_type='vo')          


    materials = openmc.Materials([fw_mat, WCLL_BZ_mat, WCLL_manifold_mat, water_cooled_steel_mat, eurofer_mat])

    materials.export_to_xml()
    
    material_lib = {'fw_mat':                 fw_mat, 
                    'bz_mat':                 WCLL_BZ_mat, 
                    'manifold_mat':           WCLL_manifold_mat, 
                    'water_cooled_steel_mat': water_cooled_steel_mat,
                    'eurofer_mat':            eurofer_mat}
    
    return material_lib



