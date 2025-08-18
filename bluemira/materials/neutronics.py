# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Create specific materials from known blanket data."""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto

from matproplib.converters.neutronics import OpenMCNeutronicsConfig
from matproplib.library.berylium import Be12Ti
from matproplib.material import Material, material, mixture
from matproplib.properties.groups import props

# Elements
he_cool_mat = material(
    name="helium",
    elements={"He4": 1.0},
    properties=props(density=(0.008867, "g/cm^3")),
    converters=OpenMCNeutronicsConfig(),
)()

tungsten_mat = material(
    name="tungsten",
    elements={
        "W182": 0.266,
        "W183": 0.143,
        "W184": 0.307,
        "W186": 0.284,
    },
    properties=props(density=(19.3, "g/cm3")),
    converters=OpenMCNeutronicsConfig(),
)()

water_mat = material(
    name="water",
    elements={"H1": 2, "O16": 1},
    properties=props(density=(0.866, "g/cm3")),
    converters=OpenMCNeutronicsConfig(),
)()

al2o3_mat = material(
    name="Aluminium Oxide",
    elements={"Al27": 2, "O16": 3},
    properties=props(density=(3.95, "g/cm3")),
    converters=OpenMCNeutronicsConfig(),
)()


# alloys
eurofer_mat = material(
    name="eurofer",
    elements={
        "Fe": 0.9006,
        "Cr": 0.0886,
        "W182": 0.0108 * 0.266,
        "W183": 0.0108 * 0.143,
        "W184": 0.0108 * 0.307,
        "W186": 0.0108 * 0.284,
        "fraction_type": "mass",
    },
    properties=props(density=(7.78, "g/cm^3")),
    converters=OpenMCNeutronicsConfig(),
)()


# Lithium-containing materials
def make_PbLi_mat(li_enrich_ao) -> Material:
    """
    Make PbLi according to the enrichment fraction inputted.

    Parameters
    ----------
    li_enrich_ao:
        The fraction of enrichment of the lithium-6.

    Returns
    -------
    :
        PbLi material with the specified Li-6 enrichment.
    """
    return material(
        name="PbLi",
        elements={"Pb": 0.83, "Li": 0.17},
        properties=props(density=(9.4, "g/cm^3")),
        converters=OpenMCNeutronicsConfig(
            enrichment=li_enrich_ao, enrichment_target="Li6", enrichment_type="ao"
        ),
    )()


def make_Li4SiO4_mat(li_enrich_ao) -> Material:
    """
    Making enriched Li4SiO4 from elements with enrichment of Li6 enrichment

    Parameters
    ----------
    li_enrich_ao:
        The fraction of enrichment of the lithium-6.

    Returns
    -------
    :
        Li4SiO4 material with the specified Li-6 enrichment.
    """
    return material(
        name="lithium_orthosilicate",
        elements={"Li": 4, "Si28": 1, "O16": 4},
        properties=props(
            density=(2.247 + 0.078 * (100.0 - li_enrich_ao) / 100.0, "g/cm^3")
        ),
        converters=OpenMCNeutronicsConfig(
            enrichment=li_enrich_ao, enrichment_target="Li6", enrichment_type="ao"
        ),
    )()


def make_Li2TiO3_mat(li_enrich_ao) -> Material:
    """
    Make Li2TiO3 according to the enrichment fraction inputted.

    Parameters
    ----------
    li_enrich_ao:
        The fraction of enrichment of the lithium-6.

    Returns
    -------
    :
        Li2TiO3 material with the specified Li-6 enrichment.
    """
    return material(
        name="lithium_titanate",
        elements={"Li": 2, "Ti": 1, "O16": 3},
        properties=props(
            density=(
                3.28 + 0.06 * (100.0 - li_enrich_ao) / 100.0,
                "g/cm^3",
            )
        ),
        converters=OpenMCNeutronicsConfig(
            enrichment=li_enrich_ao, enrichment_target="Li6", enrichment_type="ao"
        ),
    )()


# mixture of existing materials
lined_euro_mat = mixture(
    name="Eurofer with Al2O3 lining",
    materials=[
        (eurofer_mat, 2.0 / 2.4),
        (al2o3_mat, 0.4 / 2.4),
    ],
    fraction_type="volume",
    converters=OpenMCNeutronicsConfig(),
)


# Lithium-containing material that is also a mixture of existing materials
def make_KALOS_ACB_mat(li_enrich_ao) -> Material:
    """
    Parameters
    ----------
    li_enrich_ao:
        The fraction of enrichment of the lithium-6.

    Returns
    -------
    :
        the KALOS_ACB material with the specified Li-6 enrichment.

    Notes
    -----
    Ref: Current status and future perspectives of EU ceramic breeder development
    (Fusion Eng. Des., 164, 112171)
    """
    return mixture(
        name="kalos_acb",  # optional name of homogeneous material
        materials=[  # molar combination adjusted to atom fractions
            (make_Li4SiO4_mat(li_enrich_ao), 9 * 0.65 / (9 * 0.65 + 6 * 0.35)),
            (make_Li2TiO3_mat(li_enrich_ao), 6 * 0.35 / (9 * 0.65 + 6 * 0.35)),
        ],
        fraction_type="ao",
        converters=OpenMCNeutronicsConfig(
            packing_fraction=0.642,  # Fusion Eng. Des., 164, 112171. See issue #3657
        ),
    )  # combination fraction type is by atom fraction
    # KALOS_ACB_mat.set_density("g/cm3", 2.52 * 0.642)  # applying packing fraction
    # 3657


def duplicate_mat_as(
    mat_to_clone: Material,
    new_name: str,
    new_id: int | None = None,
) -> Material:
    """
    Clones and renames an OpenMC material

    Parameters
    ----------
    mat_to_clone:
        parent material to be cloned from.
    new_name:
        new name to be given to the material
    new_id:
        new id to be given to the material.

    Returns
    -------
    new_mat:
        New copy of the material.
    """
    new_mat = deepcopy(mat_to_clone)
    new_mat.converters["openmc"].material_id = new_id
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

    inb_vv_mat: Material
    inb_fw_mat: Material
    inb_bz_mat: Material
    inb_mani_mat: Material
    divertor_mat: Material
    div_fw_mat: Material


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
    li_enrich_ao: float
        Enrichment of Li-6 as a percentage
        to be parsed as argument to openmc.Material.add_element

    Returns
    -------
    :
        The set of materials used to create a DCLL reactor.

    Notes
    -----
    Divertor definition from Neutronic analyses of the preliminary
    design of a DCLL blanket for the EUROfusion DEMO power, 24 March 2016
    Using Eurofer instead of SS316LN
    """
    inb_vv_mat = mixture(
        name="inb_vacuum_vessel",
        materials=[
            (eurofer_mat, 0.8),
            (water_mat, 0.2),
        ],
        fraction_type="vo",
        converters=OpenMCNeutronicsConfig(
            material_id=104,
        ),
    )

    # Making first wall
    inb_fw_mat = mixture(
        name="inb_first_wall",
        materials=[
            (tungsten_mat, 2.0 / 27.0),
            (eurofer_mat, 1.5 / 27.0),
            (he_cool_mat, 12.0 / 27.0),
            (lined_euro_mat, 11.5 / 27.0),
        ],
        fraction_type="vo",
        converters=OpenMCNeutronicsConfig(
            material_id=101,
        ),
    )

    # Making blanket
    inb_bz_mat = mixture(
        name="inb_breeder_zone",
        materials=[
            (lined_euro_mat, 0.0605 + 0.9395 * 0.05),
            (make_PbLi_mat(li_enrich_ao), 0.9395 * 0.95),
        ],
        fraction_type="vo",
        converters=OpenMCNeutronicsConfig(
            material_id=102,
        ),
    )

    return ReactorBaseMaterials(
        inb_vv_mat=inb_vv_mat,
        inb_fw_mat=inb_fw_mat,
        inb_bz_mat=inb_bz_mat,
        inb_mani_mat=mixture(
            name="inb_manifold",
            materials=[
                (eurofer_mat, 0.573),
                (inb_bz_mat, 0.426),
            ],  # 1% void
            fraction_type="vo",
            converters=OpenMCNeutronicsConfig(material_id=103),
        ),
        divertor_mat=duplicate_mat_as(inb_vv_mat, "divertor", 301),
        div_fw_mat=duplicate_mat_as(inb_fw_mat, "div_first_wall", 302),
    )


def _make_hcpb_mats(li_enrich_ao: float) -> ReactorBaseMaterials:
    """Creates openmc material definitions for an hcpb blanket.

    Parameters
    ----------
    li_enrich_ao:
        Enrichment of Li-6 as a percentage
        to be parsed as argument to openmc.Material.add_element

    Returns
    -------
    :
        The set of materials used to create DCLL reactors with the specified Li-6
        enrichment.

    Notes
    -----
    HCPB Design Report, 26/07/2019
    WPBB-DEL-BB-1.2.1-T005-D001
    """
    inb_vv_mat = mixture(
        name="inb_vacuum_vessel",  # optional name of homogeneous material
        materials=[(eurofer_mat, 0.6), (water_mat, 0.4)],
        fraction_type="vo",
        converters=OpenMCNeutronicsConfig(material_id=104),
    )

    # Making blanket
    structural_fraction_vo = 0.128
    multiplier_fraction_vo = 0.493  # 0.647
    breeder_fraction_vo = 0.103  # 0.163
    helium_fraction_vo = 0.276  # 0.062

    return ReactorBaseMaterials(
        inb_vv_mat=inb_vv_mat,
        inb_fw_mat=mixture(
            name="inb_first_wall",  # optional name of homogeneous material
            materials=[
                (tungsten_mat, 2.0 / 27.0),
                (eurofer_mat, 25.0 * 0.573 / 27.0),
                (he_cool_mat, 25.0 * 0.427 / 27.0),
            ],
            fraction_type="vo",
            converters=OpenMCNeutronicsConfig(material_id=101),
        ),
        inb_bz_mat=mixture(
            name="inb_breeder_zone",
            materials=[
                (eurofer_mat, structural_fraction_vo),
                (Be12Ti, multiplier_fraction_vo),
                (make_KALOS_ACB_mat(li_enrich_ao), breeder_fraction_vo),
                (he_cool_mat, helium_fraction_vo),
            ],
            fraction_type="vo",
            converters=OpenMCNeutronicsConfig(material_id=102),
        ),
        inb_mani_mat=mixture(
            name="inb_manifold",
            materials=[
                (eurofer_mat, 0.4724),
                (make_KALOS_ACB_mat(li_enrich_ao), 0.0241),
                (he_cool_mat, 0.5035),
            ],
            fraction_type="vo",
            converters=OpenMCNeutronicsConfig(material_id=103),
        ),
        divertor_mat=duplicate_mat_as(inb_vv_mat, "divertor", 301),
        div_fw_mat=mixture(
            name="div_first_wall",
            materials=[
                (tungsten_mat, 16.0 / 25.0),
                (water_mat, 4.5 / 25.0),
                (eurofer_mat, 4.5 / 25.0),
            ],
            fraction_type="vo",
            converters=OpenMCNeutronicsConfig(material_id=302),
        ),
    )


def _make_wcll_mats(li_enrich_ao: float) -> ReactorBaseMaterials:
    """Creates openmc material definitions for a wcll blanket

    Parameters
    ----------
    li_enrich_ao:
        Enrichment of Li-6 as a percentage
        to be parsed as argument to openmc.Material.add_element

    Returns
    -------
    :
        The set of materials used to create WCLL reactors with the specified Li-6
        enrichment.

    Notes
    -----
    Ref. D. Nevo and M. Oron-Carl, WCLL Design Report 2018, Eurofusion,
    WPBB-DEL-BB-3.2.1-T005-D001, June 2019.
    """
    PbLi_mat = make_PbLi_mat(li_enrich_ao)

    # Divertor definition from Neutronic analyses of the preliminary
    #  design of a DCLL blanket for the EUROfusion DEMO power, 24 March 2016
    # Using Eurofer instead of SS316LN
    inb_fw_mat = mixture(
        name="inb_first_wall",
        materials=[
            (tungsten_mat, 0.0766),
            (water_mat, 0.1321),
            (eurofer_mat, 0.7913),
        ],
        fraction_type="vo",
        converters=OpenMCNeutronicsConfig(material_id=101),
    )

    return ReactorBaseMaterials(
        inb_vv_mat=mixture(
            name="inb_vacuum_vessel",
            materials=[
                (eurofer_mat, 0.6),
                (water_mat, 0.4),
            ],
            fraction_type="vo",
            converters=OpenMCNeutronicsConfig(material_id=104),
        ),
        inb_fw_mat=inb_fw_mat,
        inb_bz_mat=mixture(
            name="inb_breeder_zone",
            materials=[
                (tungsten_mat, 0.0004),
                (PbLi_mat, 0.8238),
                (water_mat, 0.0176),
                (eurofer_mat, 0.1582),
            ],
            fraction_type="vo",
            converters=OpenMCNeutronicsConfig(material_id=102),
        ),
        inb_mani_mat=mixture(
            name="inb_manifold",
            materials=[
                (PbLi_mat, 0.2129),
                (water_mat, 0.2514),
                (eurofer_mat, 0.5357),
            ],
            fraction_type="vo",
            converters=OpenMCNeutronicsConfig(material_id=103),
        ),
        divertor_mat=duplicate_mat_as(eurofer_mat, "divertor", 301),
        div_fw_mat=duplicate_mat_as(inb_fw_mat, "div_first_wall", 302),
    )
