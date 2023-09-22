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
from openmc import Material

# Elements
he_cool_mat = Material(name="helium")
he_cool_mat.add_nuclide("He4", 1.0, percent_type="ao")
he_cool_mat.set_density("g/cm3", 0.008867)

tungsten_mat = Material(name="tungsten")
tungsten_mat.add_nuclide("W182", 0.266, percent_type="ao")
tungsten_mat.add_nuclide("W183", 0.143, percent_type="ao")
tungsten_mat.add_nuclide("W184", 0.307, percent_type="ao")
tungsten_mat.add_nuclide("W186", 0.284, percent_type="ao")
tungsten_mat.set_density("g/cm3", 19.3)

# simple compounds (excluding alloys)
water_mat = Material(name="water")
water_mat.add_nuclide("H1", 0.667, percent_type="ao")
water_mat.add_nuclide("O16", 0.333, percent_type="ao")
water_mat.set_density("g/cm3", 0.866)

water_mat = Material(name="water")
water_mat.add_nuclide("H1", 0.667, percent_type="ao")
water_mat.add_nuclide("O16", 0.333, percent_type="ao")
water_mat.set_density("g/cm3", 0.866)

al2o3_mat = Material(name="Aluminium Oxide")
al2o3_mat.add_nuclide("Al27", 1.0, percent_type="ao")
al2o3_mat.add_nuclide("O16", 1.0, percent_type="ao")
al2o3_mat.set_density("g/cm3", 3.95)

# alloys
eurofer_mat = Material(name="eurofer")
eurofer_mat.add_element("Fe", 0.9006, percent_type="wo")
eurofer_mat.add_element("Cr", 0.0886, percent_type="wo")
eurofer_mat.add_nuclide("W182", 0.0108 * 0.266, percent_type="wo")
eurofer_mat.add_nuclide("W183", 0.0108 * 0.143, percent_type="wo")
eurofer_mat.add_nuclide("W184", 0.0108 * 0.307, percent_type="wo")
eurofer_mat.add_nuclide("W186", 0.0108 * 0.284, percent_type="wo")
eurofer_mat.set_density("g/cm3", 7.78)

Be12Ti_mat = Material(name="Be12Ti")
Be12Ti_mat.add_element("Be", 12.0, percent_type="ao")
Be12Ti_mat.add_element("Ti", 1.0, percent_type="ao")
Be12Ti_mat.set_density("g/cm3", 2.25)


# Lithium-containing materials
def make_PbLi_mat(li_enrich_ao) -> Material:
    """Make PbLi according to the enrichment fraction inputted."""
    PbLi_mat = Material(name="PbLi")
    PbLi_mat.add_element("Pb", 0.83, percent_type="ao")
    PbLi_mat.add_element(
        # 17% lithium,
        "Li",
        0.17,
        percent_type="ao",
        # enriched to the desired fraction
        enrichment=li_enrich_ao,
        enrichment_target="Li6",
        enrichment_type="ao",
    )
    PbLi_mat.set_density("g/cm3", 9.4)
    return PbLi_mat


def make_Li4SiO4_mat(li_enrich_ao) -> Material:
    """Making enriched Li4SiO4 from elements with enrichment of Li6 enrichment"""
    Li4SiO4_mat = Material(name="lithium_orthosilicate")
    Li4SiO4_mat.add_element(
        "Li",
        4.0,
        percent_type="ao",
        enrichment=li_enrich_ao,
        enrichment_target="Li6",
        enrichment_type="ao",
    )
    Li4SiO4_mat.add_nuclide("Si28", 1.0, percent_type="ao")
    Li4SiO4_mat.add_nuclide("O16", 4.0, percent_type="ao")
    Li4SiO4_mat.set_density("g/cm3", 2.247 + 0.078 * (100.0 - li_enrich_ao) / 100.0)
    return Li4SiO4_mat


def make_Li2TiO3_mat(li_enrich_ao) -> Material:
    """Make Li2TiO3 according to the enrichment fraction inputted."""
    Li2TiO3_mat = Material(name="lithium_titanate")
    Li2TiO3_mat.add_element(
        "Li",
        2.0,
        percent_type="ao",
        enrichment=li_enrich_ao,
        enrichment_target="Li6",
        enrichment_type="ao",
    )
    Li2TiO3_mat.add_element("Ti", 1.0, percent_type="ao")
    Li2TiO3_mat.add_nuclide("O16", 3.0, percent_type="ao")
    Li2TiO3_mat.set_density("g/cm3", 3.28 + 0.06 * (100.0 - li_enrich_ao) / 100.0)
    return Li2TiO3_mat


# mixture of existing materials
lined_euro_mat = Material.mix_materials(
    name="Eurofer with Al2O3 lining",
    materials=[eurofer_mat, al2o3_mat],
    fracs=[2.0 / 2.4, 0.4 / 2.4],
    percent_type="vo",
)


# Lithium-containing material that is also a mixture of existing materials
def make_KALOS_ACB_mat(li_enrich_ao) -> Material:
    """Ref: Current status and future perspectives of EU ceramic breeder development"""
    KALOS_ACB_mat = Material.mix_materials(
        name="kalos_acb",  # optional name of homogeneous material
        materials=[make_Li4SiO4_mat(li_enrich_ao), make_Li2TiO3_mat(li_enrich_ao)],
        fracs=[
            9 * 0.65 / (9 * 0.65 + 6 * 0.35),
            6 * 0.35 / (9 * 0.65 + 6 * 0.35),
        ],  # molar combination adjusted to atom fractions
        percent_type="ao",
    )  # combination fraction type is by atom fraction
    KALOS_ACB_mat.set_density("g/cm3", 2.52 * 0.642)  # applying packing fraction
    return KALOS_ACB_mat
