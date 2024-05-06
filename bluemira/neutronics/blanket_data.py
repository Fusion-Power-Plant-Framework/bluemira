# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Create specific materials from known blanket data."""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto

from bluemira.base.constants import raw_uc
from bluemira.materials.material import MassFractionMaterial
from bluemira.materials.mixtures import HomogenisedMixture, MixtureFraction
from bluemira.neutronics.materials import NeutronicsMaterials

# Elements
he_cool_mat = MassFractionMaterial(
    name="helium", nuclides={"He4": 1.0}, density=0.008867, percent_type="ao"
)

tungsten_mat = MassFractionMaterial(
    name="tungsten",
    nuclides={
        "W182": 0.266,
        "W183": 0.143,
        "W184": 0.307,
        "W186": 0.284,
    },
    percent_type="ao",
    density=19.3,
    density_unit="g/cm3",
)

water_mat = MassFractionMaterial(
    name="water",
    nuclides={"H1": 2, "O16": 1},
    percent_type="ao",
    density=0.866,
    density_unit="g/cm3",
)

al2o3_mat = MassFractionMaterial(
    name="Aluminium Oxide",
    nuclides={"Al27": 2, "O16": 3},
    percent_type="ao",
    density=3.95,
    density_unit="g/cm3",
)


# alloys
eurofer_mat = MassFractionMaterial(
    name="eurofer",
    elements={"Fe": 0.9006, "Cr": 0.0886},
    nuclides={
        "W182": 0.0108 * 0.266,
        "W183": 0.0108 * 0.143,
        "W184": 0.0108 * 0.307,
        "W186": 0.0108 * 0.284,
    },
    percent_type="wo",
    density=7.78,
    density_unit="g/cm3",
)


Be12Ti_mat = MassFractionMaterial(
    name="Be12Ti",
    elements={"Be": 12, "Ti": 1},
    percent_type="ao",
    density=2.25,
    density_unit="g/cm3",
)


# Lithium-containing materials
def make_PbLi_mat(li_enrich_ao) -> MassFractionMaterial:
    """Make PbLi according to the enrichment fraction inputted."""
    return MassFractionMaterial(
        name="PbLi",
        elements={"Pb": 0.83, "Li": 0.17},
        percent_type="ao",
        enrichment=li_enrich_ao,
        enrichment_target="Li6",
        enrichment_type="ao",
        density=9.4,
        density_unit="g/cm3",
    )


def make_Li4SiO4_mat(li_enrich_ao) -> MassFractionMaterial:
    """Making enriched Li4SiO4 from elements with enrichment of Li6 enrichment"""
    return MassFractionMaterial(
        name="lithium_orthosilicate",
        elements={"Li": 4},
        nuclides={"Si28": 1, "O16": 4},
        percent_type="ao",
        enrichment=li_enrich_ao,
        enrichment_target="Li6",
        enrichment_type="ao",
        density=2.247 + 0.078 * (100.0 - li_enrich_ao) / 100.0,
        density_unit="g/cm3",
    )


def make_Li2TiO3_mat(li_enrich_ao) -> MassFractionMaterial:
    """Make Li2TiO3 according to the enrichment fraction inputted."""
    return MassFractionMaterial(
        name="lithium_titanate",
        elements={"Li": 2, "Ti": 1},
        nuclides={"O16": 3},
        percent_type="ao",
        enrichment=li_enrich_ao,
        enrichment_target="Li6",
        enrichment_type="ao",
        density=3.28 + 0.06 * (100.0 - li_enrich_ao) / 100.0,
        density_unit="g/cm3",
    )


# mixture of existing materials
lined_euro_mat = HomogenisedMixture(
    name="Eurofer with Al2O3 lining",
    materials=[
        MixtureFraction(eurofer_mat, 2.0 / 2.4),
        MixtureFraction(al2o3_mat, 0.4 / 2.4),
    ],
    percent_type="vo",
)


# Lithium-containing material that is also a mixture of existing materials
def make_KALOS_ACB_mat(li_enrich_ao) -> HomogenisedMixture:
    """Ref: Current status and future perspectives of EU ceramic breeder development"""
    return HomogenisedMixture(
        name="kalos_acb",  # optional name of homogeneous material
        materials=[  # molar combination adjusted to atom fractions
            MixtureFraction(
                make_Li4SiO4_mat(li_enrich_ao), 9 * 0.65 / (9 * 0.65 + 6 * 0.35)
            ),
            MixtureFraction(
                make_Li2TiO3_mat(li_enrich_ao), 6 * 0.35 / (9 * 0.65 + 6 * 0.35)
            ),
        ],
        percent_type="ao",
        packing_fraction=0.642,
    )  # combination fraction type is by atom fraction
    # todo: check if this packing fraction is correct (as set above)
    # KALOS_ACB_mat.set_density("g/cm3", 2.52 * 0.642)  # applying packing fraction


def duplicate_mat_as(
    mat_to_clone: MassFractionMaterial | HomogenisedMixture,
    new_name: str,
    new_id: int | None = None,
) -> MassFractionMaterial | HomogenisedMixture:
    """Clones and renames an OpenMC material"""
    new_mat = deepcopy(mat_to_clone)
    new_mat.material_id = new_id
    new_mat.name = new_name

    return new_mat


class BlanketType(Enum):
    """Types of allowed blankets, named by their acronyms."""

    DCLL = auto()
    HCPB = auto()
    WCLL = auto()

    @classmethod
    def _missing_(cls, value: str):
        try:
            return cls[value.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid query: {value}. Choose from: {(*cls._member_names_,)}"
            ) from None


@dataclass
class ReactorBaseMaterials:
    """Minimum set of materials that can create a tokamak.
    The rest can be populated by duplication using a priori knowledge,
    e.g. inboard material = outboard material etc.
    """

    inb_vv_mat: HomogenisedMixture
    inb_fw_mat: HomogenisedMixture
    inb_bz_mat: HomogenisedMixture
    inb_mani_mat: HomogenisedMixture
    divertor_mat: HomogenisedMixture
    div_fw_mat: HomogenisedMixture


@dataclass
class BreederTypeParameters:
    """Dataclass to hold information about the breeder blanket material
    and design choices.
    """

    enrichment_fraction_Li6: float
    blanket_type: BlanketType


def _make_dcll_mats(li_enrich_ao: float) -> ReactorBaseMaterials:
    """Creates openmc material definitions for a dcll blanket.

    Parameters
    ----------
    --_-------
    li_enrich_ao: float
        Enrichment of Li-6 as a percentage
        to be parsed as argument to openmc.Material.add_element

    Notes
    -----
    Divertor definition from Neutronic analyses of the preliminary
    design of a DCLL blanket for the EUROfusion DEMO power, 24 March 2016
    Using Eurofer instead of SS316LN
    """
    inb_vv_mat = HomogenisedMixture(
        name="inb_vacuum_vessel",
        material_id=104,
        materials=[
            MixtureFraction(eurofer_mat, 0.8),
            MixtureFraction(water_mat, 0.2),
        ],
        percent_type="vo",
    )

    # Making first wall
    inb_fw_mat = HomogenisedMixture(
        name="inb_first_wall",
        material_id=101,
        materials=[
            MixtureFraction(tungsten_mat, 2.0 / 27.0),
            MixtureFraction(eurofer_mat, 1.5 / 27.0),
            MixtureFraction(he_cool_mat, 12.0 / 27.0),
            MixtureFraction(lined_euro_mat, 11.5 / 27.0),
        ],
        percent_type="vo",
    )

    # Making blanket
    _PbLi_mat = make_PbLi_mat(li_enrich_ao)
    inb_bz_mat = HomogenisedMixture(
        name="inb_breeder_zone",
        material_id=102,
        materials=[
            MixtureFraction(lined_euro_mat, 0.0605 + 0.9395 * 0.05),
            MixtureFraction(_PbLi_mat, 0.9395 * 0.95),
        ],
        percent_type="vo",
    )

    inb_mani_mat = HomogenisedMixture(
        name="inb_manifold",
        material_id=103,
        materials=[
            MixtureFraction(eurofer_mat, 0.573),
            MixtureFraction(inb_bz_mat, 0.426),
        ],  # 1% void
        percent_type="vo",
    )

    # Making divertor
    divertor_mat = duplicate_mat_as(inb_vv_mat, "divertor", 301)
    div_fw_mat = duplicate_mat_as(inb_fw_mat, "div_first_wall", 302)
    return ReactorBaseMaterials(
        inb_vv_mat=inb_vv_mat,
        inb_fw_mat=inb_fw_mat,
        inb_bz_mat=inb_bz_mat,
        inb_mani_mat=inb_mani_mat,
        divertor_mat=divertor_mat,
        div_fw_mat=div_fw_mat,
    )


def _make_hcpb_mats(li_enrich_ao: float) -> ReactorBaseMaterials:
    """Creates openmc material definitions for an hcpb blanket.

    Parameters
    ----------
    li_enrich_ao:
        Enrichment of Li-6 as a percentage
        to be parsed as argument to openmc.Material.add_element

    Notes
    -----
    HCPB Design Report, 26/07/2019
    WPBB-DEL-BB-1.2.1-T005-D001
    """
    inb_vv_mat = HomogenisedMixture(
        name="inb_vacuum_vessel",  # optional name of homogeneous material
        material_id=104,
        materials=[
            MixtureFraction(eurofer_mat, 0.6),
            MixtureFraction(water_mat, 0.4),
        ],
        percent_type="vo",
    )

    # Making first wall
    inb_fw_mat = HomogenisedMixture(
        name="inb_first_wall",  # optional name of homogeneous material
        material_id=101,
        materials=[
            MixtureFraction(tungsten_mat, 2.0 / 27.0),
            MixtureFraction(eurofer_mat, 25.0 * 0.573 / 27.0),
            MixtureFraction(he_cool_mat, 25.0 * 0.427 / 27.0),
        ],
        percent_type="vo",
    )

    # Making blanket
    structural_fraction_vo = 0.128
    multiplier_fraction_vo = 0.493  # 0.647
    breeder_fraction_vo = 0.103  # 0.163
    helium_fraction_vo = 0.276  # 0.062

    inb_bz_mat = HomogenisedMixture(
        name="inb_breeder_zone",
        material_id=102,
        materials=[
            MixtureFraction(eurofer_mat, structural_fraction_vo),
            MixtureFraction(Be12Ti_mat, multiplier_fraction_vo),
            MixtureFraction(make_KALOS_ACB_mat(li_enrich_ao), breeder_fraction_vo),
            MixtureFraction(he_cool_mat, helium_fraction_vo),
        ],
        percent_type="vo",
    )

    inb_mani_mat = HomogenisedMixture(
        name="inb_manifold",
        material_id=103,
        materials=[
            MixtureFraction(eurofer_mat, 0.4724),
            MixtureFraction(make_KALOS_ACB_mat(li_enrich_ao), 0.0241),
            MixtureFraction(he_cool_mat, 0.5035),
        ],
        percent_type="vo",
    )

    # Making divertor
    divertor_mat = duplicate_mat_as(inb_vv_mat, "divertor", 301)
    div_fw_mat = HomogenisedMixture(
        name="div_first_wall",
        material_id=302,
        materials=[
            MixtureFraction(tungsten_mat, 16.0 / 25.0),
            MixtureFraction(water_mat, 4.5 / 25.0),
            MixtureFraction(eurofer_mat, 4.5 / 25.0),
        ],
        percent_type="vo",
    )
    return ReactorBaseMaterials(
        inb_vv_mat=inb_vv_mat,
        inb_fw_mat=inb_fw_mat,
        inb_bz_mat=inb_bz_mat,
        inb_mani_mat=inb_mani_mat,
        divertor_mat=divertor_mat,
        div_fw_mat=div_fw_mat,
    )


def _make_wcll_mats(li_enrich_ao: float) -> ReactorBaseMaterials:
    """Creates openmc material definitions for a wcll blanket

    Parameters
    ----------
    li_enrich_ao:
        Enrichment of Li-6 as a percentage
        to be parsed as argument to openmc.Material.add_element

    Notes
    -----
    Ref. D. Nevo and M. Oron-Carl, WCLL Design Report 2018, Eurofusion,
    WPBB-DEL-BB-3.2.1-T005-D001, June 2019.
    """
    _PbLi_mat = make_PbLi_mat(li_enrich_ao)

    # Divertor definition from Neutronic analyses of the preliminary
    #  design of a DCLL blanket for the EUROfusion DEMO power, 24 March 2016
    # Using Eurofer instead of SS316LN
    inb_vv_mat = HomogenisedMixture(
        name="inb_vacuum_vessel",
        material_id=104,
        materials=[
            MixtureFraction(eurofer_mat, 0.6),
            MixtureFraction(water_mat, 0.4),
        ],
        percent_type="vo",
    )

    # Making first wall
    inb_fw_mat = HomogenisedMixture(
        name="inb_first_wall",
        material_id=101,
        materials=[
            MixtureFraction(tungsten_mat, 0.0766),
            MixtureFraction(water_mat, 0.1321),
            MixtureFraction(eurofer_mat, 0.7913),
        ],
        percent_type="vo",
    )

    # Making blanket
    inb_bz_mat = HomogenisedMixture(
        name="inb_breeder_zone",
        material_id=102,
        materials=[
            MixtureFraction(tungsten_mat, 0.0004),
            MixtureFraction(_PbLi_mat, 0.8238),
            MixtureFraction(water_mat, 0.0176),
            MixtureFraction(eurofer_mat, 0.1582),
        ],
        percent_type="vo",
    )

    inb_mani_mat = HomogenisedMixture(
        name="inb_manifold",
        material_id=103,
        materials=[
            MixtureFraction(_PbLi_mat, 0.2129),
            MixtureFraction(water_mat, 0.2514),
            MixtureFraction(eurofer_mat, 0.5357),
        ],
        percent_type="vo",
    )

    # Making divertor
    divertor_mat = duplicate_mat_as(eurofer_mat, "divertor", 301)
    div_fw_mat = duplicate_mat_as(inb_fw_mat, "div_first_wall", 302)
    return ReactorBaseMaterials(
        inb_vv_mat=inb_vv_mat,
        inb_fw_mat=inb_fw_mat,
        inb_bz_mat=inb_bz_mat,
        inb_mani_mat=inb_mani_mat,
        divertor_mat=divertor_mat,
        div_fw_mat=div_fw_mat,
    )


@dataclass(frozen=True)
class TokamakGeometry:
    """The thickness measurements for all of the generic components of the tokamak.

    Parameters
    ----------
    inb_fw_thick:
        inboard first wall thickness [m]
    inb_bz_thick:
        inboard breeding zone thickness [m]
    inb_mnfld_thick:
        inboard manifold thickness [m]
    inb_vv_thick:
        inboard vacuum vessel thickness [m]
    tf_thick:
        toroidal field coil thickness [m]
    outb_fw_thick:
        outboard first wall thickness [m]
    outb_bz_thick:
        outboard breeding zone thickness [m]
    outb_mnfld_thick:
        outboard manifold thickness [m]
    outb_vv_thick:
        outboard vacuum vessel thickness [m]
    inb_gap:
        inboard gap [m]
    """

    inb_fw_thick: float
    inb_bz_thick: float
    inb_mnfld_thick: float
    inb_vv_thick: float
    tf_thick: float
    outb_fw_thick: float
    outb_bz_thick: float
    outb_mnfld_thick: float
    outb_vv_thick: float
    inb_gap: float


def get_preset_physical_properties(
    blanket_type: str | BlanketType,
) -> tuple[BreederTypeParameters, TokamakGeometry]:
    """
    Works as a switch-case for choosing the tokamak geometry
    and blankets for a given blanket type.
    The allowed list of blanket types are specified in BlanketType.
    Currently, the blanket types with pre-populated data in this function are:
    {'wcll', 'dcll', 'hcpb'}
    """
    blanket_type = BlanketType(blanket_type)

    breeder_materials = BreederTypeParameters(
        blanket_type=blanket_type,
        enrichment_fraction_Li6=0.60,
    )

    # Geometry variables

    # Break down from here.
    # Paper inboard build ---
    # Nuclear analyses of solid breeder blanket options for DEMO:
    # Status,challenges and outlook,
    # Pereslavtsev, 2019
    #
    # 0.400,      # TF Coil inner
    # 0.200,      # gap                  from figures
    # 0.060,       # VV steel wall
    # 0.480,      # VV
    # 0.060,       # VV steel wall
    # 0.020,       # gap                  from figures
    # 0.350,      # Back Supporting Structure
    # 0.060,       # Back Wall and Gas Collectors   Back wall = 3.0
    # 0.350,      # breeder zone
    # 0.022        # fw and armour

    shared_building_geometry = {  # that are identical in all three types of reactors.
        "inb_gap": 0.2,  # [m]
        "inb_vv_thick": 0.6,  # [m]
        "tf_thick": 0.4,  # [m]
        "outb_vv_thick": 0.6,  # [m]
    }
    if blanket_type is BlanketType.WCLL:
        tokamak_geometry = TokamakGeometry(
            **shared_building_geometry,
            inb_fw_thick=0.027,  # [m]
            inb_bz_thick=0.378,  # [m]
            inb_mnfld_thick=0.435,  # [m]
            outb_fw_thick=0.027,  # [m]
            outb_bz_thick=0.538,  # [m]
            outb_mnfld_thick=0.429,  # [m]
        )
    elif blanket_type is BlanketType.DCLL:
        tokamak_geometry = TokamakGeometry(
            **shared_building_geometry,
            inb_fw_thick=0.022,  # [m]
            inb_bz_thick=0.300,  # [m]
            inb_mnfld_thick=0.178,  # [m]
            outb_fw_thick=0.022,  # [m]
            outb_bz_thick=0.640,  # [m]
            outb_mnfld_thick=0.248,  # [m]
        )
    elif blanket_type is BlanketType.HCPB:
        # HCPB Design Report, 26/07/2019
        tokamak_geometry = TokamakGeometry(
            **shared_building_geometry,
            inb_fw_thick=0.027,  # [m]
            inb_bz_thick=0.460,  # [m]
            inb_mnfld_thick=0.560,  # [m]
            outb_fw_thick=0.027,  # [m]
            outb_bz_thick=0.460,  # [m]
            outb_mnfld_thick=0.560,  # [m]
        )
    return breeder_materials, tokamak_geometry


def create_materials(
    breeder_materials: BreederTypeParameters,
) -> NeutronicsMaterials:
    """
    Parameters
    ----------
    breeder_materials:
        dataclass containing attributes: 'blanket_type', 'enrichment_fraction_Li6'
    """
    return _create_from_blanket_type(
        breeder_materials.blanket_type,
        raw_uc(breeder_materials.enrichment_fraction_Li6, "", "%"),
    )


def _create_from_blanket_type(
    blanket_type: BlanketType, li_enrich_ao: float
) -> NeutronicsMaterials:
    """Create Materials Library by specifying just the blanket type

    Parameters
    ----------
    blanket_type:
        the blanket type
    li_enrich_ao:
        PERCENTAGE enrichment of Li6 (float between 0 - 100)
    """
    if blanket_type is BlanketType.DCLL:
        base_materials = _make_dcll_mats(li_enrich_ao)
    elif blanket_type is BlanketType.HCPB:
        base_materials = _make_hcpb_mats(li_enrich_ao)
    elif blanket_type is BlanketType.WCLL:
        base_materials = _make_wcll_mats(li_enrich_ao)
    return NeutronicsMaterials(
        inb_vv_mat=base_materials.inb_vv_mat,
        inb_fw_mat=base_materials.inb_fw_mat,
        inb_bz_mat=base_materials.inb_bz_mat,
        inb_mani_mat=base_materials.inb_mani_mat,
        divertor_mat=base_materials.divertor_mat,
        div_fw_mat=base_materials.div_fw_mat,
        outb_fw_mat=duplicate_mat_as(base_materials.inb_fw_mat, "outb_first_wall", 201),
        outb_bz_mat=duplicate_mat_as(
            base_materials.inb_bz_mat, "outb_breeder_zone", 202
        ),
        outb_mani_mat=duplicate_mat_as(
            base_materials.inb_mani_mat, "outb_manifold", 203
        ),
        outb_vv_mat=duplicate_mat_as(
            base_materials.inb_vv_mat, "outb_vacuum_vessel", 204
        ),
        tf_coil_mat=duplicate_mat_as(eurofer_mat, "tf_coil", 401),
        container_mat=duplicate_mat_as(base_materials.inb_vv_mat, "container", 501),
        # surfaces
        inb_sf_mat=duplicate_mat_as(eurofer_mat, "inb_sf", 601),
        outb_sf_mat=duplicate_mat_as(eurofer_mat, "outb_sf", 602),
        div_sf_mat=duplicate_mat_as(eurofer_mat, "div_sf", 603),
    )
