# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.
"""Create specific materials that will be imported by other modules."""

from bluemira.materials.material import MassFractionMaterial
from bluemira.materials.mixtures import HomogenisedMixture, MixtureFraction

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
    KALOS_ACB_mat = HomogenisedMixture(
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
    return KALOS_ACB_mat  # noqa: RET504 for now
