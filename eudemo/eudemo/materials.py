# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Materials for the LAR example."""

from matproplib.conditions import OperationalConditions
from matproplib.converters.neutronics import OpenMCNeutronicConfig
from matproplib.library.beryllium import Be12Ti
from matproplib.library.concrete import HeavyConcrete
from matproplib.library.fluids import Helium, Water
from matproplib.library.steel import SS316_L
from matproplib.library.tungsten import PlanseeTungsten
from matproplib.material import material, mixture
from matproplib.properties.group import props

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.materials.neutronics import make_KALOS_ACB_mat

try:
    from eurofusion_materials.library.steel import SS316_LN, EUROfer97
    from eurofusion_materials.library.tungsten import Tungsten

    EUROFER_MAT = EUROfer97()
    SS316_LN_MAT = SS316_LN()
    TUNGSTEN_MAT = Tungsten()

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
    SS316_LN_MAT = SS316_L()
    TUNGSTEN_MAT = PlanseeTungsten()

WATER_MAT = Water()
HELIUM_MAT = Helium()
HeavyConcrete = HeavyConcrete()

VV_MATERIAL = mixture(
        name="inb_vacuum_vessel",
        materials=[(EUROFER_MAT, 0.8), (WATER_MAT, 0.2)],
        fraction_type="volume",
        mix_condition=OperationalConditions(temperature=673.15, pressure=1e5),
        converters=OpenMCNeutronicConfig(material_id=104),
    )


structural_fraction_vo = 0.128
multiplier_fraction_vo = 0.493
breeder_fraction_vo = 0.103
helium_fraction_vo = 0.276
li6_enrich_atomic = 0.6

KALOS_ACB_MATERIAL = make_KALOS_ACB_mat(li6_enrich_atomic)

BB_FW_MATERIAL = mixture(
    name="FW_material",
    materials=[
        (TUNGSTEN_MAT, 2.0 / 27.0),
        (EUROFER_MAT, 25.0 * 0.573 / 27.0),
        (HELIUM_MAT, 25.0 * 0.427 / 27.0),
    ],
    fraction_type="volume",
    mix_condition=OperationalConditions(temperature=673.15, pressure=8e6),
    converters=OpenMCNeutronicConfig(material_id=101),
)

BB_BZ_MATERIAL = mixture(
    name="BZ_material",
    materials=[
        (EUROFER_MAT, structural_fraction_vo),
        (Be12Ti(), multiplier_fraction_vo),
        (KALOS_ACB_MATERIAL, breeder_fraction_vo),
        (HELIUM_MAT, helium_fraction_vo),
    ],
    fraction_type="volume",
    mix_condition=OperationalConditions(temperature=673.15, pressure=8e6),
    converters=OpenMCNeutronicConfig(
        material_id=192,
        enrichment=li6_enrich_atomic * 100,
        enrichment_target="Li6",
        enrichment_type="atomic",
    ),
)

BB_MANI_MATERIAL = mixture(
    name="Manifold_material",
    materials=[
        (EUROFER_MAT, 0.4724),
        (KALOS_ACB_MATERIAL, 0.0241),
        (HELIUM_MAT, 0.5035),
    ],
    fraction_type="volume",
    mix_condition=OperationalConditions(temperature=673.15, pressure=8e6),
    converters=OpenMCNeutronicConfig(
        material_id=103,
        enrichment=li6_enrich_atomic * 100,
        enrichment_target="Li6",
        enrichment_type="atomic",
    ),
)

DIV_FW_MATERIAL = mixture(
    name="Divertor FW material",
    materials=[
        (TUNGSTEN_MAT, 16.0 / 25.0),
        (WATER_MAT, 4.5 / 25.0),
        (EUROFER_MAT, 4.5 / 25.0),
    ],
    fraction_type="volume",
    mix_condition=OperationalConditions(temperature=673.15, pressure=1e5),
    converters=OpenMCNeutronicConfig(material_id=302),
)
