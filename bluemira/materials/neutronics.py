# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Create specific materials from known blanket data."""

from copy import deepcopy
from dataclasses import dataclass
from enum import Enum, auto

from matproplib.conditions import OperationalConditions
from matproplib.converters.neutronics import OpenMCNeutronicConfig
from matproplib.library.beryllium import Be12Ti
from matproplib.library.fluids import Helium, Water
from matproplib.library.tungsten import PlanseeTungsten
from matproplib.material import Material, material, mixture
from matproplib.properties.group import props

from bluemira.base.look_and_feel import bluemira_warn

try:
    from eurofusion_materials.library.steel import EUROfer97
    from eurofusion_materials.library.tungsten import Tungsten

    EUROFER_MAT = EUROfer97()
    TUNGSTEN_MAT = Tungsten()
    WATER_MAT = Water()
    HELIUM_MAT = Helium()
    raise ImportError
except ImportError:
    bluemira_warn(
        "You do have eurofusion_materials installed, or do not have access. "
        "We're going to use some representative imitation materials instead, "
        "as opposed to the official, material descriptions."
    )
    EUROFER_MAT = material(
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
        converters=OpenMCNeutronicConfig(),
    )()
    TUNGSTEN_MAT = PlanseeTungsten()

    # Debugging replacements (to be removed)
    TUNGSTEN_MAT = material(
        name="tungsten",
        elements={
            "W182": 0.266,
            "W183": 0.143,
            "W184": 0.307,
            "W186": 0.284,
            "fraction_type": "atomic",
        },
        properties=props(density=(19.3, "g/cm^3")),
        converters=OpenMCNeutronicConfig(),
    )()

    Be12Ti = material(
        "Be12Ti",
        elements={"Be": 12.0 / 13, "Ti": 1.0 / 13, "fraction_type": "atomic"},
        converters=OpenMCNeutronicConfig(),
        properties=props(density=2250.0),
    )
    WATER_MAT = material(
        "water",
        elements={"H1": 2 / 3, "O16": 1 / 3, "fraction_type": "atomic"},
        properties=props(density=866.0),  # WTF
        converters=OpenMCNeutronicConfig(),
    )()

    HELIUM_MAT = material(
        "He",
        elements={"He4": 1.0},
        converters=OpenMCNeutronicConfig(),
        properties=props(density=0.008867),
    )()


al2o3_mat = material(
    name="Aluminium Oxide",
    elements={"Al27": 2 / 5, "O16": 3 / 5},
    properties=props(density=(3.95, "g/cm^3")),
    converters=OpenMCNeutronicConfig(),
)()

# Be12Ti = material(
#     "Be12Ti",
#     elements="Be12Ti",
#     converters=OpenMCNeutronicConfig(),
#     properties=props(density=2250.0),
# )


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
        converters=OpenMCNeutronicConfig(
            enrichment=li_enrich_ao, enrichment_target="Li6", enrichment_type="atomic"
        ),
    )()


def make_Li4SiO4_mat(li_enrich_ao, packing_fraction=1.0) -> Material:
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

    Notes
    -----
    packing_fraction=0.642 Fusion Eng. Des., 164, 112171. See issue #3657
    """
    return material(
        name="lithium_orthosilicate",
        elements={"Li": 4 / 9, "Si28": 1 / 9, "O16": 4 / 9},
        properties=props(
            density=(packing_fraction * (2.247 + 0.078 * (1.0 - li_enrich_ao)), "g/cm^3")
        ),
        converters=OpenMCNeutronicConfig(
            enrichment=li_enrich_ao * 100,
            enrichment_target="Li6",
            enrichment_type="atomic",
        ),
    )()


def make_Li2TiO3_mat(li_enrich_ao, packing_fraction=1.0) -> Material:
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

    Notes
    -----
    packing_fraction=0.642 Fusion Eng. Des., 164, 112171. See issue #3657
    """
    return material(
        name="lithium_titanate",
        elements={"Li": 2 / 6, "Ti": 1 / 6, "O16": 3 / 6},
        properties=props(
            density=(
                packing_fraction * (3.28 + 0.06 * (1.0 - li_enrich_ao)),
                "g/cm^3",
            )
        ),
        converters=OpenMCNeutronicConfig(
            enrichment=li_enrich_ao * 100,
            enrichment_target="Li6",
            enrichment_type="atomic",
        ),
    )()


# mixture of existing materials
lined_euro_mat = mixture(
    name="Eurofer with Al2O3 lining",
    materials=[
        (EUROFER_MAT, 2.0 / 2.4),
        (al2o3_mat, 0.4 / 2.4),
    ],
    fraction_type="volume",
    volume_conditions=OperationalConditions(temperature=673.15),
    converters=OpenMCNeutronicConfig(),
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
        fraction_type="atomic",
        converters=OpenMCNeutronicConfig(
            # packing_fraction=0.642,  # Fusion Eng. Des., 164, 112171. See issue #3657
            enrichment=li_enrich_ao * 100,
            enrichment_target="Li6",
            enrichment_type="atomic",
        ),
    )  # combination fraction type is by atom fraction
    # KALOS_ACB_mat.set_density("g/cm^3", 2.52 * 0.642)  # applying packing fraction
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
        materials=[(EUROFER_MAT, 0.8), (WATER_MAT, 0.2)],
        fraction_type="volume",
        volume_conditions=OperationalConditions(temperature=673.15, pressure=1e5),
        converters=OpenMCNeutronicConfig(material_id=104),
    )

    # Making first wall
    inb_fw_mat = mixture(
        name="inb_first_wall",
        materials=[
            (TUNGSTEN_MAT, 2.0 / 27.0),
            (EUROFER_MAT, 1.5 / 27.0),
            (HELIUM_MAT, 12.0 / 27.0),
            (lined_euro_mat, 11.5 / 27.0),
        ],
        fraction_type="volume",
        volume_conditions=OperationalConditions(temperature=673.15, pressure=8e6),
        converters=OpenMCNeutronicConfig(material_id=101),
    )

    # Making blanket
    inb_bz_mat = mixture(
        name="inb_breeder_zone",
        materials=[
            (lined_euro_mat, 0.0605 + 0.9395 * 0.05),
            (make_PbLi_mat(li_enrich_ao), 0.9395 * 0.95),
        ],
        fraction_type="volume",
        volume_conditions=OperationalConditions(temperature=673.15),
        converters=OpenMCNeutronicConfig(material_id=102),
    )

    return ReactorBaseMaterials(
        inb_vv_mat=inb_vv_mat,
        inb_fw_mat=inb_fw_mat,
        inb_bz_mat=inb_bz_mat,
        inb_mani_mat=mixture(
            name="inb_manifold",
            materials=[(EUROFER_MAT, 0.573), (inb_bz_mat, 0.426)],  # 1% void
            fraction_type="volume",
            volume_conditions=OperationalConditions(temperature=673.15),
            converters=OpenMCNeutronicConfig(material_id=103),
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
        materials=[(EUROFER_MAT, 0.6), (WATER_MAT, 0.4)],
        fraction_type="volume",
        mix_conditions=OperationalConditions(temperature=373.15, pressure=1e5),
        converters=OpenMCNeutronicConfig(material_id=104),
    )

    # Making blanket
    structural_fraction_vo = 0.128
    multiplier_fraction_vo = 0.493  # 0.647
    breeder_fraction_vo = 0.103  # 0.163
    helium_fraction_vo = 0.276  # 0.062

    KALOS_ACB_MAT = make_KALOS_ACB_mat(li_enrich_ao)

    return ReactorBaseMaterials(
        inb_vv_mat=inb_vv_mat,
        inb_fw_mat=mixture(
            name="inb_first_wall",  # optional name of homogeneous material
            materials=[
                (TUNGSTEN_MAT, 2.0 / 27.0),
                (EUROFER_MAT, 25.0 * 0.573 / 27.0),
                (HELIUM_MAT, 25.0 * 0.427 / 27.0),
            ],
            fraction_type="volume",
            volume_conditions=OperationalConditions(temperature=673.15, pressure=8e6),
            converters=OpenMCNeutronicConfig(material_id=101),
        ),
        inb_bz_mat=mixture(
            name="inb_breeder_zone",
            materials=[
                (EUROFER_MAT, structural_fraction_vo),
                (Be12Ti(), multiplier_fraction_vo),
                (KALOS_ACB_MAT, breeder_fraction_vo),
                (HELIUM_MAT, helium_fraction_vo),
            ],
            fraction_type="volume",
            mix_conditions=OperationalConditions(temperature=673.15, pressure=8e6),
            converters=OpenMCNeutronicConfig(
                material_id=102,
                enrichment=li_enrich_ao * 100,
                enrichment_target="Li6",
                enrichment_type="atomic",
            ),
        ),
        inb_mani_mat=mixture(
            name="inb_manifold",
            materials=[
                (EUROFER_MAT, 0.4724),
                (KALOS_ACB_MAT, 0.0241),
                (HELIUM_MAT, 0.5035),
            ],
            fraction_type="volume",
            mix_conditions=OperationalConditions(temperature=673.15, pressure=8e6),
            converters=OpenMCNeutronicConfig(
                material_id=103,
                enrichment=li_enrich_ao * 100,
                enrichment_target="Li6",
                enrichment_type="atomic",
            ),
        ),
        divertor_mat=duplicate_mat_as(inb_vv_mat, "divertor", 301),
        div_fw_mat=mixture(
            name="div_first_wall",
            materials=[
                (TUNGSTEN_MAT, 16.0 / 25.0),
                (WATER_MAT, 4.5 / 25.0),
                (EUROFER_MAT, 4.5 / 25.0),
            ],
            fraction_type="volume",
            volume_conditions=OperationalConditions(temperature=673.15, pressure=1e5),
            converters=OpenMCNeutronicConfig(material_id=302),
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
        materials=[(TUNGSTEN_MAT, 0.0766), (WATER_MAT, 0.1321), (EUROFER_MAT, 0.7913)],
        fraction_type="volume",
        volume_conditions=OperationalConditions(temperature=673.15, pressure=1e5),
        converters=OpenMCNeutronicConfig(material_id=101),
    )

    return ReactorBaseMaterials(
        inb_vv_mat=mixture(
            name="inb_vacuum_vessel",
            materials=[(EUROFER_MAT, 0.6), (WATER_MAT, 0.4)],
            fraction_type="volume",
            volume_conditions=OperationalConditions(temperature=673.15, pressure=1e5),
            converters=OpenMCNeutronicConfig(material_id=104),
        ),
        inb_fw_mat=inb_fw_mat,
        inb_bz_mat=mixture(
            name="inb_breeder_zone",
            materials=[
                (TUNGSTEN_MAT, 0.0004),
                (PbLi_mat, 0.8238),
                (WATER_MAT, 0.0176),
                (EUROFER_MAT, 0.1582),
            ],
            fraction_type="volume",
            volume_conditions=OperationalConditions(temperature=673.15, pressure=1e5),
            converters=OpenMCNeutronicConfig(material_id=102),
        ),
        inb_mani_mat=mixture(
            name="inb_manifold",
            materials=[(PbLi_mat, 0.2129), (WATER_MAT, 0.2514), (EUROFER_MAT, 0.5357)],
            fraction_type="volume",
            volume_conditions=OperationalConditions(temperature=673.15, pressure=1e5),
            converters=OpenMCNeutronicConfig(material_id=103),
        ),
        divertor_mat=duplicate_mat_as(EUROFER_MAT, "divertor", 301),
        div_fw_mat=duplicate_mat_as(inb_fw_mat, "div_first_wall", 302),
    )


if __name__ == "__main__":
    m = _make_hcpb_mats(0.6)
    from matproplib import OperationalConditions

    r = repr(
        m.inb_bz_mat.convert(
            "openmc", OperationalConditions(temperature=300, pressure=8e6)
        )
    )

    true_output = """
    Material
	ID             =	102
	Name           =	inb_breeder_zone
	Temperature    =	None
	Density        =	2.273067751637386 [g/cm3]
	Volume         =	None [cm^3]
	Depletable     =	False
	S(a,b) Tables
	Nuclides
	Be9            =	0.6998631398987396 [ao]
	Cr50           =	0.0006047907018436486 [ao]
	Cr52           =	0.011662786678199645 [ao]
	Cr53           =	0.001322466388542349 [ao]
	Cr54           =	0.0003291898756870492 [ao]
	Fe54           =	0.0076998749495311705 [ao]
	Fe56           =	0.12087156990920156 [ao]
	Fe57           =	0.002791451671181617 [ao]
	Fe58           =	0.0003714909727575347 [ao]
	He4            =	5.015408434749478e-06 [ao]
	Li6            =	0.023828836409492977 [ao]
	Li7            =	0.01588589093966199 [ao]
	O16            =	0.04392689540133806 [ao]
	Si28           =	0.00782259781119719 [ao]
	Ti46           =	0.0051590629511089415 [ao]
	Ti47           =	0.004652536770454608 [ao]
	Ti48           =	0.04610013584918195 [ao]
	Ti49           =	0.003383094613999923 [ao]
	Ti50           =	0.0032392661923326435 [ao]
	W182           =	0.00012897639723287904 [ao]
	W183           =	6.895717680765253e-05 [ao]
	W184           =	0.0001472355767323238 [ao]
	W186           =	0.0001347374563400298 [ao]
    """

    import re
    from math import isclose

    def compare_materials(str1: str, str2: str, tol: float = 1e-8):
        """
        Compare two material definition strings, using str1 as the reference.
        Shows absolute and relative (to str1) differences.
        """

        def parse_material(s: str):
            pattern = re.compile(r"(\w+)\s*=\s*([^\s]+)")
            data = {}
            for key, value in pattern.findall(s):
                v = re.sub(r"\[.*?\]", "", value).strip()
                try:
                    data[key] = float(v)
                except ValueError:
                    data[key] = v
            return data

        ref = parse_material(str1)
        new = parse_material(str2)

        ref_keys, new_keys = set(ref), set(new)

        only_in_ref = sorted(ref_keys - new_keys)
        only_in_new = sorted(new_keys - ref_keys)
        both = sorted(ref_keys & new_keys)

        diffs = []

        for key in both:
            v1, v2 = ref[key], new[key]
            if isinstance(v1, float) and isinstance(v2, float):
                if not isclose(v1, v2, rel_tol=tol, abs_tol=tol):
                    rel_diff = (v2 - v1) / v1 if v1 != 0 else float("inf")
                    diffs.append((key, v1, v2, v2 - v1, rel_diff))
            elif v1 != v2:
                diffs.append((key, v1, v2, None, None))

        # --- Print summary ---
        print("🔹 Only in reference (missing in second):")
        for k in only_in_ref:
            print(f"  {k} = {ref[k]}")

        print("\n🔹 Only in second (not in reference):")
        for k in only_in_new:
            print(f"  {k} = {new[k]}")

        print("\n🔹 Differences beyond tolerance (relative to reference):")
        for key, v1, v2, delta, rel in diffs:
            if rel is None:
                print(f"  {key}: '{v1}' != '{v2}'")
            else:
                print(
                    f"  {key}: {v1:.6g} → {v2:.6g}  "
                    f"(Δ={delta:.3g}, rel={rel * 100:.3f}%)"
                )

        return {
            "only_in_ref": only_in_ref,
            "only_in_new": only_in_new,
            "diffs": diffs,
        }

    compare_materials(true_output, r)
