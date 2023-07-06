import copy
import enum

import openmc
from openmc import Material
from openmc.checkvalue import PathLike, check_type


def export_dict_to_xml(material_dict, path: PathLike = "materials.xml"):
    """Save a dictionary using the openmc export_to_xml function"""
    return openmc.Materials(material_dict.values()).export_to_xml(path)


def duplicate_mat_as(mat_to_clone, new_id, new_name):
    """Clones and renames an OpenMC material"""
    new_mat = mat_to_clone.clone()
    new_mat.id = new_id
    new_mat.name = new_name

    return new_mat


@dataclass
class ReactorMaterials:
    inb_vv_mat: Material
    inb_fw_mat: Material
    inb_bz_mat: Material
    inb_mani_mat: Material
    divertor_mat: Material
    div_fw_mat: Material

    def str_to_componet(self, s: str):
        """Created to solve the ...+'_mats' problem. Probably discardible."""
        return getattr(self, s)


class MaterialsLibrary(dict):
    """A dictionary of materials that auto-complete the materials
    according to the type of blanket used
    """

    def __init__(self, blanket_type: str):
        if blanket_type == ...:
            self._make_dcll_mats()

        super().__init__(common_mats)

    def populate_materials(self):
        _populate_materials(self)

    #############################################################################################

    def _make_dcll_mats(self, li_enrich_ao) -> ReactorMaterials:
        """This function creates openmc material definitions for a dcll blanket.
        # Divertor definition from Neutronic analyses of the preliminary
        #  design of a DCLL blanket for the EUROfusion DEMO power, 24 March 2016
        # Using Eurofer instead of SS316LN
        """

        inb_vv_mat = Material.mix_materials(
            name="inb_vacuum_vessel",
            materials=[md.eurofer_mat, md.water_mat],
            fracs=[0.8, 0.2],
            percent_type="vo",
        )
        inb_vv_mat.id = 104

        # Making first wall
        self["inb_vv_mat"] = inb_vv_mat
        self["inb_fw_mat"] = Material.mix_materials(
            name="inb_first_wall",
            materials=[
                md.tungsten_mat,
                md.eurofer_mat,
                md.he_cool_mat,
                md.lined_euro_mat,
            ],
            fracs=[2.0 / 27.0, 1.5 / 27.0, 12.0 / 27.0, 11.5 / 27.0],
            percent_type="vo",
        )
        self["inb_fw_mat"].id = 101

        # Making blanket
        self["inb_bz_mat"] = Material.mix_materials(
            name="inb_breeder_zone",
            materials=[md.lined_euro_mat, md.PbLi_mat],
            fracs=[0.0605 + 0.9395 * 0.05, 0.9395 * 0.95],
            percent_type="vo",
        )
        self["inb_bz_mat"].id = 102

        self["inb_mani_mat"] = Material.mix_materials(
            name="inb_manifold",
            materials=[md.eurofer_mat, self["inb_bz_mat"]],
            fracs=[0.573, 0.426],
            percent_type="vo",
        )
        self["inb_mani_mat"].id = 103

        # Making divertor
        self["divertor_mat"] = duplicate_mat_as(self["inb_vv_mat"], 301, "divertor")
        self["div_fw_mat"] = duplicate_mat_as(self["inb_fw_mat"], 302, "div_first_wall")

    def _make_hcpb_mats(self, li_enrich_ao):
        """creates openmc material definitions for an hcpb blanket.
        HCPB Design Report, 26/07/2019
        WPBB-DEL-BB-1.2.1-T005-D001
        """
        self["inb_vv_mat"] = Material().mix_materials(
            name="inb_vacuum_vessel",  # optional name of homogeneous material
            materials=[md.eurofer_mat, md.water_mat],
            fracs=[0.6, 0.4],
            percent_type="vo",
        )
        self["inb_vv_mat"].id = 104

        # Making first wall
        self["inb_fw_mat"] = Material().mix_materials(
            name="inb_first_wall",  # optional name of homogeneous material
            materials=[md.tungsten_mat, md.eurofer_mat, md.he_cool_mat],
            fracs=[2.0 / 27.0, 25.0 * 0.573 / 27.0, 25.0 * 0.427 / 27.0],
            percent_type="vo",
        )
        self["inb_fw_mat"].id = 101

        # Making blanket
        structural_fraction_vo = 0.128
        multiplier_fraction_vo = 0.493  # 0.647
        breeder_fraction_vo = 0.103  # 0.163
        helium_fraction_vo = 0.276  # 0.062

        self["inb_bz_mat"] = Material(material_id=102).mix_materials(
            name="inb_breeder_zone",
            materials=[
                md.eurofer_mat,
                md.Be12Ti_mat,
                md.make_KALOS_ACB_mat(li_enrich_ao),
                md.he_cool_mat,
            ],
            fracs=[
                structural_fraction_vo,
                multiplier_fraction_vo,
                breeder_fraction_vo,
                helium_fraction_vo,
            ],
            percent_type="vo",
        )
        self["inb_bz_mat"].id = 102

        self["inb_mani_mat"] = Material(material_id=103).mix_materials(
            name="inb_manifold",
            materials=[
                md.eurofer_mat,
                md.make_KALOS_ACB_mat(li_enrich_ao),
                md.he_cool_mat,
            ],
            fracs=[0.4724, 0.0241, 0.5035],
            percent_type="vo",
        )
        self["inb_mani_mat"].id = 103

        # Making divertor
        self["divertor_mat"] = duplicate_mat_as(self["inb_vv_mat"], 301, "divertor")
        self["div_fw_mat"] = Material().mix_materials(
            name="div_first_wall",
            materials=[md.tungsten_mat, md.water_mat, md.eurofer_mat],
            fracs=[16.0 / 25.0, 4.5 / 25.0, 4.5 / 25.0],
            percent_type="vo",
        )
        self["div_fw_mat"].id = 302

    def _make_wcll_mats(self, li_enrich_ao):
        """
        This function creates openmc material definitions for a wcll blanket
        Ref. D. Nevo and M. Oron-Carl, WCLL Design Report 2018, Eurofusion,
            WPBB-DEL-BB-3.2.1-T005-D001, June 2019.
        """

        PbLi_mat = make_PbLi_mat(li_enrich_ao)

        # Divertor definition from Neutronic analyses of the preliminary
        #  design of a DCLL blanket for the EUROfusion DEMO power, 24 March 2016
        # Using Eurofer instead of SS316LN
        self["inb_vv_mat"] = Material.mix_materials(
            name="inb_vacuum_vessel",
            materials=[md.eurofer_mat, md.water_mat],
            fracs=[0.6, 0.4],
            percent_type="vo",
        )
        self["inb_vv_mat"].id = 104

        # Making first wall
        self["inb_fw_mat"] = Material.mix_materials(
            name="inb_first_wall",
            materials=[md.tungsten_mat, md.water_mat, md.eurofer_mat],
            fracs=[0.0766, 0.1321, 0.7913],
            percent_type="vo",
        )
        self["inb_fw_mat"].id = 101

        # Making blanket
        self["inb_bz_mat"] = Material.mix_materials(
            name="inb_breeder_zone",
            materials=[
                md.tungsten_mat,
                md.PbLi_mat,
                md.water_mat,
                md.eurofer_mat,
            ],
            fracs=[0.0004, 0.8238, 0.0176, 0.1582],
            percent_type="vo",
        )
        self["inb_bz_mat"].id = 102

        self["inb_mani_mat"] = Material.mix_materials(
            name="inb_manifold",
            materials=[md.PbLi_mat, md.water_mat, md.eurofer_mat],
            fracs=[0.2129, 0.2514, 0.5357],
            percent_type="vo",
        )
        self["inb_mani_mat"].id = 103

        # Making divertor
        self["divertor_mat"] = duplicate_mat_as(eurofer_mat, 301, "divertor")
        self["div_fw_mat"] = duplicate_mat_as(self["inb_fw_mat"], 302, "div_first_wall")

    def export_materials(self):
        """Exports material defintions to xml"""
        material_list = openmc.Materials(self.values())
        return material_list.export_to_xml()

    @classmethod
    def create_complete_material_library(
        cls, blanket_type, li_enrich_ao, export_to_xml=True
    ):
        make_method = getattr(cls, "make_{}_mats".format(blanket_type), None)
        if make_method is None:
            raise KeyError(
                "Not a valid blanket_type; See BlanketType for allowed blanket_type."
            )
        self = AutoPopulatingMaterialsLibrary()
        make_method(self, li_enrich_ao)
        self.populate_materials()
        if export_to_xml:
            self.export_to_xml()
        return self

    # def create_complete_material_library(self,
    #                                      blanket_type,
    #                                      li_enrich_ao,
    #                                      export_to_xml=True):
    #     make_method = getattr(self, "make_{}_mats".format(blanket_type), None)
    #     if make_method is None:
    #         msg = "Not a valid blanket_type; See BlanketType for allowed blanket_type."
    #         raise KeyError(msg)
    #     make_method(li_enrich_ao)
    #     self.populate_materials()
    #     if export_to_xml:
    #         self.export_to_xml()


def _make_common_mats():
    """Makes materials that are common to all blankets"""
    material_lib = dict()
    material_lib["tungsten_mat"] = tungsten_mat
    material_lib["eurofer_mat"] = eurofer_mat
    material_lib["water_mat"] = water_mat
    return material_lib


def _populate_materials(material_lib):
    """
    Need to duplicate materials as using material filter for scoring heating.
    Given a half-defined material_lib dictionary, populate the rest.
    """
    material_lib["outb_fw_mat"] = duplicate_mat_as(
        material_lib["inb_fw_mat"], 201, "outb_first_wall"
    )
    material_lib["outb_bz_mat"] = duplicate_mat_as(
        material_lib["inb_bz_mat"], 202, "outb_breeder_zone"
    )
    material_lib["outb_mani_mat"] = duplicate_mat_as(
        material_lib["inb_mani_mat"], 203, "outb_manifold"
    )
    material_lib["outb_vv_mat"] = duplicate_mat_as(
        material_lib["inb_vv_mat"], 204, "outb_vacuum_vessel"
    )
    material_lib["tf_coil_mat"] = duplicate_mat_as(
        material_lib["eurofer_mat"], 401, "tf_coil"
    )
    material_lib["container_mat"] = duplicate_mat_as(
        material_lib["inb_vv_mat"], 501, "container"
    )
    # sf = surface
    material_lib["inb_sf_mat"] = duplicate_mat_as(
        material_lib["eurofer_mat"], 601, "inb_sf"
    )
    material_lib["outb_sf_mat"] = duplicate_mat_as(
        material_lib["eurofer_mat"], 602, "outb_sf"
    )
    material_lib["div_sf_mat"] = duplicate_mat_as(
        material_lib["eurofer_mat"], 603, "div_sf"
    )
    return material_lib


BlanketType = enum.Enum(
    "BlanketType",
    [
        method_name[5:-5]
        for method_name in dir(AutoPopulatingMaterialsLibrary)
        if method_name.startswith("make_") and method_name.endswith("_mats")
    ],
)

# class BlanketType(BlanketTypeRaw):
# @classmethod
# def get_allowed_types(cls):
#     """Give an alias to _member_map_ so that the enumerated members are more explicit and easier to extract."""
#     return cls._member_map_
