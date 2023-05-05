import openmc
import copy
from enum import Enum, EnumMeta, auto

class BlanketTypeEnumMeta(EnumMeta):
    """
    Override KeyError message string
    """

    def __getitem__(self, name):
        try:
            return super().__getitem__(name)
        except KeyError as err:
            raise KeyError(f"BlanketType {name} hasn't been implemented") from None


class BlanketType(Enum, metaclass=BlanketTypeEnumMeta):
    """
    Acceptable blanket types
    """
    hcpb = auto()
    dcll = auto()
    wcll = auto()

    @classmethod
    def allowed_types(cls):
        return cls._member_map_

material_lib = {} # BAD! TODO: Make into dict(), or completely redo!

def make_common_mats():
    
    # Makes materials that are common to all blankets

    tungsten_mat = openmc.Material(name='tungsten')
    tungsten_mat.add_nuclide('W182', 0.266, percent_type='ao')
    tungsten_mat.add_nuclide('W183', 0.143, percent_type='ao')
    tungsten_mat.add_nuclide('W184', 0.307, percent_type='ao')
    tungsten_mat.add_nuclide('W186', 0.284, percent_type='ao')
    tungsten_mat.set_density('g/cm3', 19.3)
    
    material_lib['tungsten_mat'] = tungsten_mat

    eurofer_mat = openmc.Material(name='eurofer')
    eurofer_mat.add_element('Fe', 0.9006, percent_type='wo')
    eurofer_mat.add_element('Cr', 0.0886, percent_type='wo')
    eurofer_mat.add_nuclide('W182', 0.0108 * 0.266, percent_type='wo')
    eurofer_mat.add_nuclide('W183', 0.0108 * 0.143, percent_type='wo')
    eurofer_mat.add_nuclide('W184', 0.0108 * 0.307, percent_type='wo')
    eurofer_mat.add_nuclide('W186', 0.0108 * 0.284, percent_type='wo')
    eurofer_mat.set_density('g/cm3', 7.78)
    
    material_lib['eurofer_mat'] = eurofer_mat
    
    water_mat = openmc.Material(name='water')
    water_mat.add_nuclide('H1',  0.667, percent_type='ao')
    water_mat.add_nuclide('O16', 0.333, percent_type='ao')
    water_mat.set_density('g/cm3', 0.866)
    
    material_lib['water_mat'] = water_mat
    
    return

# ----------------------------------------------------------------------------------------

def clone_and_rename_mat(mat_to_clone, new_id, new_name):
    
    # Clones and renames an OpenMC material
    
    new_mat    = mat_to_clone.clone()
    new_mat.id = new_id
    new_mat.name = new_name
    
    return new_mat

# ----------------------------------------------------------------------------------------

def export_materials():
    
    # Duplicates and exports material defintions to xml
    
    # Need to duplicate materials as using material filter for scoring heating
    material_lib['outb_fw_mat']   = clone_and_rename_mat( material_lib['inb_fw_mat'],   201, 'outb_first_wall')
    material_lib['outb_bz_mat']   = clone_and_rename_mat( material_lib['inb_bz_mat'],   202, 'outb_breeder_zone')
    material_lib['outb_mani_mat'] = clone_and_rename_mat( material_lib['inb_mani_mat'], 203, 'outb_manifold')
    material_lib['outb_vv_mat']   = clone_and_rename_mat( material_lib['inb_vv_mat'],   204, 'outb_vacuum_vessel')
    material_lib['tf_coil_mat']   = clone_and_rename_mat( material_lib['eurofer_mat'],  401, 'tf_coil')
    material_lib['container_mat'] = clone_and_rename_mat( material_lib['inb_vv_mat'],   501, 'container')
    material_lib['inb_sf_mat']    = clone_and_rename_mat( material_lib['eurofer_mat'],  601, 'inb_sf')
    material_lib['outb_sf_mat']   = clone_and_rename_mat( material_lib['eurofer_mat'],  602, 'outb_sf')
    material_lib['div_sf_mat']    = clone_and_rename_mat( material_lib['eurofer_mat'],  603, 'div_sf')

    materials = openmc.Materials( material_lib.values() )

    materials.export_to_xml()

    return
    

# ----------------------------------------------------------------------------------------

def make_hcpb_mats(li_enrich_ao):
    
    # This function creates openmc material definitions for an hcpb blanket
    
    # HCPB Design Report, 26/07/2019
    # WPBB-DEL-BB-1.2.1-T005-D001
    
    make_common_mats()

    he_cool_mat = openmc.Material(name='helium')
    he_cool_mat.add_nuclide('He4', 1.0, percent_type='ao')
    he_cool_mat.set_density('g/cm3', 0.008867)

    Be12Ti_mat = openmc.Material(name='Be12Ti')
    Be12Ti_mat.add_element('Be', 12.0, percent_type='ao')
    Be12Ti_mat.add_element('Ti', 1.0, percent_type='ao')
    Be12Ti_mat.set_density('g/cm3', 2.25)

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

    KALOS_ACB_mat.set_density('g/cm3', 2.52 * 0.642 )   #  applying packing fraction
                                                        # Ref: Current status and future perspectives 
                                                        #  of EU ceramic breeder development
            
    material_lib['inb_vv_mat'] = openmc.Material().mix_materials(
            name='inb_vacuum_vessel',      # optional name of homogeneous material
            materials=[material_lib['eurofer_mat'],
                       material_lib['water_mat']],
            fracs=[0.6,
                   0.4],                  
            percent_type='vo')           
    material_lib['inb_vv_mat'].id = 104
    
    ### Making first wall
    material_lib['inb_fw_mat'] = openmc.Material().mix_materials(
        name='inb_first_wall', # optional name of homogeneous material
        materials=[material_lib['tungsten_mat'],
                   material_lib['eurofer_mat'],
                   he_cool_mat],
        fracs=[2. / 27., 
               25. * 0.573 / 27.,
               25. * 0.427 / 27.],    
        percent_type='vo')            
    material_lib['inb_fw_mat'].id = 101

    ### Making blanket
    structural_fraction_vo = 0.128 
    multiplier_fraction_vo = 0.493  # 0.647
    breeder_fraction_vo = 0.103 # 0.163
    helium_fraction_vo = 0.276 # 0.062
    
    material_lib['inb_bz_mat'] = openmc.Material( material_id=102 ).mix_materials(
        name='inb_breeder_zone',        
        materials=[material_lib['eurofer_mat'],
                   Be12Ti_mat,
                   KALOS_ACB_mat,
                   he_cool_mat],
        fracs=[structural_fraction_vo, 
               multiplier_fraction_vo,
               breeder_fraction_vo,
               helium_fraction_vo],    
        percent_type='vo')             
    material_lib['inb_bz_mat'].id = 102

    material_lib['inb_mani_mat'] = openmc.Material( material_id=103 ).mix_materials(
        name='inb_manifold',      
        materials=[material_lib['eurofer_mat'],
                   KALOS_ACB_mat,
                   he_cool_mat],
        fracs=[0.4724, 
               0.0241,
               0.5035],                
        percent_type='vo')            
    material_lib['inb_mani_mat'].id = 103
    
    # Making divertor
    material_lib['divertor_mat'] = clone_and_rename_mat( material_lib['inb_vv_mat'], 301, 'divertor')
    material_lib['div_fw_mat'] = openmc.Material().mix_materials(
            name='div_first_wall', 
            materials=[material_lib['tungsten_mat'],
                       material_lib['water_mat'],
                       material_lib['eurofer_mat']],
            fracs=[16. / 25.,
                   4.5 / 25.,
                   4.5 / 25.],             
            percent_type='vo')             
    material_lib['div_fw_mat'].id = 302
    
    # Exporting materials
    export_materials()
    
    return

################################################################################################

def make_dcll_mats(li_enrich_ao):
    
    # This function creates openmc material definitions for a dcll blanket
    # 
    
    make_common_mats()

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
        name='Eurofer with Al2O3 lining',             
        materials=[material_lib['eurofer_mat'],
                   al2o3_mat],
        fracs=[2. / 2.4, 
               0.4 / 2.4 ],                           
        percent_type='vo')                              

    # Divertor definition from Neutronic analyses of the preliminary 
    #  design of a DCLL blanket for the EUROfusion DEMO power, 24 March 2016 
    # Using Eurofer instead of SS316LN
    material_lib['inb_vv_mat'] = openmc.Material.mix_materials(
            name='inb_vacuum_vessel',     
            materials=[material_lib['eurofer_mat'],
                       material_lib['water_mat']],
            fracs=[0.8,
                   0.2],                    
            percent_type='vo')              
    material_lib['inb_vv_mat'].id = 104
    
    ### Making first wall
    material_lib['inb_fw_mat'] = openmc.Material.mix_materials(
        name='inb_first_wall',             
        materials=[material_lib['tungsten_mat'],
                   material_lib['eurofer_mat'],
                   he_cool_mat,
                   lined_euro_mat],
        fracs=[2. / 27., 
               1.5 / 27.,
               12. / 27.,
               11.5 / 27.],           
        percent_type='vo')            
    material_lib['inb_fw_mat'].id = 101

    ### Making blanket 
    material_lib['inb_bz_mat'] = openmc.Material.mix_materials(
        name='inb_breeder_zone',            
        materials=[lined_euro_mat,
                   PbLi_mat],
        fracs=[0.0605 + 0.9395 * 0.05,
               0.9395 * 0.95],         
        percent_type='vo')             
    material_lib['inb_bz_mat'].id = 102

    material_lib['inb_mani_mat'] = openmc.Material.mix_materials(
        name='inb_manifold',         
        materials=[material_lib['eurofer_mat'],
                   material_lib['inb_bz_mat']],
        fracs=[0.573, 
               0.426],                
        percent_type='vo')            
    material_lib['inb_mani_mat'].id = 103
    
    # Making divertor
    material_lib['divertor_mat']  = clone_and_rename_mat( material_lib['inb_vv_mat'], 301, 'divertor')
    material_lib['div_fw_mat']    = clone_and_rename_mat( material_lib['inb_fw_mat'], 302, 'div_first_wall')
    
    # Exporting materials
    export_materials()
    
    return


################################################################################################

def make_wcll_mats(li_enrich_ao):
    
    # This function creates openmc material definitions for a wcll blanket
    # Ref. D. Nevo and M. Oron-Carl, WCLL Design Report 2018, Eurofusion, WPBB-DEL-BB-3.2.1-T005-D001, June 2019. 
    
    make_common_mats()

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
    material_lib['inb_vv_mat'] = openmc.Material.mix_materials(
            name='inb_vacuum_vessel',      
            materials=[material_lib['eurofer_mat'],
                       material_lib['water_mat']],
            fracs=[0.6,
                   0.4],                    
            percent_type='vo')              
    material_lib['inb_vv_mat'].id = 104
    
    ### Making first wall
    material_lib['inb_fw_mat']= openmc.Material.mix_materials(
        name='inb_first_wall',              
        materials=[material_lib['tungsten_mat'],
                   material_lib['water_mat'],
                   material_lib['eurofer_mat']],
        fracs=[0.0766, 
               0.1321,
               0.7913],                
        percent_type='vo')          
    material_lib['inb_fw_mat'].id = 101
    
    ### Making blanket 
    material_lib['inb_bz_mat'] = openmc.Material.mix_materials(
        name='inb_breeder_zone',           
        materials=[material_lib['tungsten_mat'],
                   PbLi_mat,
                   material_lib['water_mat'],
                   material_lib['eurofer_mat']],
        fracs=[0.0004, 
               0.8238,
               0.0176,
               0.1582],               
        percent_type='vo') 
    material_lib['inb_bz_mat'].id = 102

    material_lib['inb_mani_mat'] = openmc.Material.mix_materials(
        name='inb_manifold',            
        materials=[PbLi_mat,
                   material_lib['water_mat'],
                   material_lib['eurofer_mat']],
        fracs= [0.2129,
                0.2514,
                0.5357],               
        percent_type='vo') 
    material_lib['inb_mani_mat'].id =103

    # Making divertor
    material_lib['divertor_mat']  = clone_and_rename_mat( material_lib['eurofer_mat'], 301, 'divertor')
    material_lib['div_fw_mat']    = clone_and_rename_mat( material_lib['inb_fw_mat'],  302, 'div_first_wall')

    # Exporting materials
    export_materials()
    
    return

for blanket_name in [blanket_type.name for blanket_type in list(BlanketType)]:
    # make sure all of the allowed blanket types has a defined make_{}_mats function.
    assert f"make_{blanket_name}_mats" in vars().keys()