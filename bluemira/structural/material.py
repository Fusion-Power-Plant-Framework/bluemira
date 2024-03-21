# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Simple structural material representations
"""

from dataclasses import dataclass, field

from bluemira.materials.material import MassFractionMaterial
from bluemira.materials.mixtures import HomogenisedMixture

MaterialType = MassFractionMaterial | HomogenisedMixture


@dataclass
class StructuralMaterial:
    """
    Dataclass for a structural representation of a material.

    Parameters
    ----------
    E:
        Youngs modulus [Pa]
    nu:
        Poisson ratio
    rho:
        Density [kg/m^3]
    sigma_y:
        Yield stress [Pa]
    description:
        A description of the material
    """

    E: float
    nu: float
    rho: float
    alpha: float
    sigma_y: float
    G: float = field(init=False, repr=True)
    description: str | None = field(default="", repr=True)

    def __post_init__(self):
        """
        Shear modulus for isotropic materials
        """
        self.G = self.E / (0.5 + 0.5 * self.nu)


def make_structural_material(
    material: MaterialType, temperature: float
) -> StructuralMaterial:
    """
    Make a structural representation of a material.

    Parameters
    ----------
    material:
        Material type to create a structural representation for
    temperature:
        Temperature at which to make a structural representation of a material [K]

    Returns
    -------
    Structural representation of a material
    """
    description = f"{material.name} at {temperature:.2f} K"
    return StructuralMaterial(
        material.E(temperature),
        material.mu(temperature),
        material.rho(temperature),
        material.CTE(temperature),
        material.Sy(temperature),
        description,
    )


class Material(dict):
    """
    A simple material property dictionary (keep small for speed and memory)
    """

    __slots__ = ()

    def __init__(self, e_modulus, nu, rho, alpha, sigma_y):
        self["E"] = e_modulus
        self["nu"] = nu
        self["alpha"] = alpha
        self["rho"] = rho
        self["G"] = e_modulus / (1 + nu) / 2
        self["sigma_y"] = sigma_y


# Just some simple materials to play with during tests and the like

SS316 = StructuralMaterial(
    200e9,
    0.33,
    8910,
    18e-6,
    360e6,
    "Typical stainless steel properties at room temperature",
)

FORGED_SS316LN = StructuralMaterial(
    205e9,
    0.29,
    8910,
    10.36e-6,
    800e6,
    "Forged SS316LN plates: OIS structural material as defined in 2MBS88 and"
    "ITER SDC-MC DRG1 Annex A (values at 4K).",
)

FORGED_JJ1 = StructuralMaterial(
    205e9,
    0.29,
    8910,
    10.38e-6,
    1000e6,
    "Forged EK1/JJ1 strengthened austenitic steel plates: TF inner leg material"
    " as defined in 2MBS88 and ITER SDC-MC DRG1 Annex A (values at 4K).",
)

CAST_EC1 = StructuralMaterial(
    190e9,
    0.29,
    8910,
    10.38e-6,
    750e6,
    " Cast EC1 strengthened austenitic steel castings: TF outer leg material as"
    " defined in 2MBS88 and ITER SDC-MC DRG1 Annex A (values at 4K).",
)

CONCRETE = StructuralMaterial(
    40e9, 0.3, 2400, 12e-6, 40e6, "Typical concrete properties at room temperature"
)
