# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""Materials for design examples"""

from matproplib.conditions import OperationalConditions
from matproplib.converters.neutronics import OpenMCNeutronicConfig
from matproplib.library.beryllium import Be12Ti
from matproplib.library.fluids import Helium
from matproplib.library.tungsten import PlanseeTungsten
from matproplib.material import material, mixture
from matproplib.properties.group import props

from bluemira.materials.neutronics import make_KALOS_ACB_mat

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
HELIUM_MAT = Helium()

TUNGSTEN_MAT = PlanseeTungsten()

BB_FW_MATERIAL = mixture(
    name="FW material",
    materials=[
        (TUNGSTEN_MAT, 2.0 / 27.0),
        (EUROFER_MAT, 25.0 * 0.573 / 27.0),
        (HELIUM_MAT, 25.0 * 0.427 / 27.0),
    ],
    fraction_type="volume",
    mix_condition=OperationalConditions(temperature=673.15, pressure=8e6),
    converters=OpenMCNeutronicConfig(material_id=101),
)

structural_fraction_vo = 0.128
multiplier_fraction_vo = 0.493
breeder_fraction_vo = 0.103
helium_fraction_vo = 0.276
li6_enrich_atomic = 0.6
KALOS_ACB_MATERIAL = make_KALOS_ACB_mat(li6_enrich_atomic)

BB_BZ_MATERIAL = mixture(
    name="BZ material",
    materials=[
        (EUROFER_MAT, structural_fraction_vo),
        (Be12Ti(), multiplier_fraction_vo),
        (KALOS_ACB_MATERIAL, breeder_fraction_vo),
        (HELIUM_MAT, helium_fraction_vo),
    ],
    fraction_type="volume",
    mix_condition=OperationalConditions(temperature=673.15, pressure=8e6),
    converters=OpenMCNeutronicConfig(
        material_id=102,
        enrichment=li6_enrich_atomic * 100,
        enrichment_target="Li6",
        enrichment_type="atomic",
    ),
)
