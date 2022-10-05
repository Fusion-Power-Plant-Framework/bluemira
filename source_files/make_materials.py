import openmc


def make_hcpb_mats(li_enrich_ao):

    # This function creates openmc material definitions for an hcpb blanket

    tungsten_mat = openmc.Material(name="tungsten")
    tungsten_mat.add_nuclide("W180", 0.266, percent_type="ao")
    tungsten_mat.add_nuclide("W182", 0.143, percent_type="ao")
    tungsten_mat.add_nuclide("W183", 0.307, percent_type="ao")
    tungsten_mat.add_nuclide("W184", 0.284, percent_type="ao")
    tungsten_mat.set_density("g/cm3", 19.3)

    eurofer_mat = openmc.Material(name="eurofer")
    eurofer_mat.add_element("Fe", 0.9006, percent_type="wo")
    eurofer_mat.add_element("Cr", 0.0886, percent_type="wo")
    eurofer_mat.add_nuclide("W180", 0.0108 * 0.266, percent_type="wo")
    eurofer_mat.add_nuclide("W182", 0.0108 * 0.143, percent_type="wo")
    eurofer_mat.add_nuclide("W183", 0.0108 * 0.307, percent_type="wo")
    eurofer_mat.add_nuclide("W184", 0.0108 * 0.284, percent_type="wo")
    eurofer_mat.set_density("g/cm3", 7.78)

    he_cool_mat = openmc.Material(name="helium")
    he_cool_mat.add_nuclide("He4", 1.0, percent_type="ao")
    he_cool_mat.set_density("g/cm3", 0.008867)

    Be12Ti_mat = openmc.Material(name="Be12Ti")
    Be12Ti_mat.add_element("Be", 12.0, percent_type="ao")
    Be12Ti_mat.add_element("Ti", 1.0, percent_type="ao")
    Be12Ti_mat.set_density("g/cm3", 2.26)

    # Making enriched Li4SiO4 from elements with enrichment of Li6 enrichment
    Li4SiO4_mat = openmc.Material(name="lithium_orthosilicate")
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

    Li2TiO3_mat = openmc.Material(name="lithium_titanate")
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

    KALOS_ACB_mat = openmc.Material.mix_materials(
        name="kalos_acb",  # optional name of homogeneous material
        materials=[Li4SiO4_mat, Li2TiO3_mat],
        fracs=[
            9 * 0.65 / (9 * 0.65 + 6 * 0.35),
            6 * 0.35 / (9 * 0.65 + 6 * 0.35),
        ],  # molar combination adjusted to atom fractions
        percent_type="ao",
    )  # combination fraction type is by atom fraction

    KALOS_ACB_mat.set_density("g/cm3", 2.52 * 0.642)  # applying packing fraction
    # Ref: Current status and future perspectives
    #  of EU ceramic breeder development

    ### Making first wall
    fw_mat = openmc.Material.mix_materials(
        name="first_wall",  # optional name of homogeneous material
        materials=[tungsten_mat, eurofer_mat, he_cool_mat],
        fracs=[
            2.0 / 22.0,
            20.0 * 0.597 / 22.0,
            20.0 * 0.403 / 22.0,
        ],  # molar combination adjusted to atom fractions
        percent_type="vo",
    )  # combination fraction type is by atom fraction

    ### Making blanket
    structural_fraction_vo = 0.128
    helium_fraction_vo = 0.062
    breeder_fraction_vo = 0.163
    multiplier_fraction_vo = 0.647

    HCPB_BZ_mat = openmc.Material.mix_materials(
        name="hcpb_bz",  # optional name of homogeneous material
        materials=[eurofer_mat, Be12Ti_mat, KALOS_ACB_mat, he_cool_mat],
        fracs=[
            structural_fraction_vo,
            breeder_fraction_vo,
            multiplier_fraction_vo,
            helium_fraction_vo,
        ],  # molar combination adjusted to atom fractions
        percent_type="vo",
    )  # combination fraction type is by atom fraction

    HCPB_manifold_mat = openmc.Material.mix_materials(
        name="hcpb_manifold",  # optional name of homogeneous material
        materials=[eurofer_mat, KALOS_ACB_mat, he_cool_mat],
        fracs=[0.4724, 0.0241, 0.5035],  # molar combination adjusted to atom fractions
        percent_type="vo",
    )  # combination fraction type is by atom fraction

    materials = openmc.Materials([fw_mat, HCPB_BZ_mat, HCPB_manifold_mat, eurofer_mat])

    materials.export_to_xml()

    material_lib = {
        "fw_mat": fw_mat,
        "BZ_mat": HCPB_BZ_mat,
        "manifold_mat": HCPB_manifold_mat,
        "eurofer_mat": eurofer_mat,
    }

    return material_lib


################################################################################################
