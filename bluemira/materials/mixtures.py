# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Material mixture utility classes
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Union

if TYPE_CHECKING:
    from bluemira.materials.material import MassFractionMaterial


import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.materials.constants import T_DEFAULT
from bluemira.materials.error import MaterialsError


class HomogenisedMixture:
    """
    Inherits and does some dropping of 0 fractions (avoid touching nmm)
    """

    def __init__(
        self,
        name: str,
        materials: List[Union[MassFractionMaterial, HomogenisedMixture]],
        fracs: List[float],
        material_id: Optional[int] = None,
        percent_type="vo",
        packing_fraction: Optional[float] = 1.0,
    ):
        self.name = name
        self.material_id = material_id
        self.materials = materials
        self.fracs = fracs
        self.percent_type = percent_type
        self.packing_fraction = packing_fraction

    def __str__(self) -> str:
        """
        Get the name of the mixture.
        """
        return self.name

    def to_openmc_material(self, temperature: float = T_DEFAULT):
        """
        Convert the mixture to an openmc material.
        """
        # Convert the constituent materials to openmc materials
        openmc_materials = [
            mat.to_openmc_material(temperature) for mat in self.materials
        ]

        from neutronics_material_maker import Material  # noqa: PLC0415

        # Create the mixture
        return Material.from_mixture(
            name=self.name,
            material_id=self.material_id,
            materials=openmc_materials,
            fracs=self.fracs,
            percent_type=self.percent_type,
            packing_fraction=self.packing_fraction,
            temperature=temperature,
        ).openmc_material

    def _calc_homogenised_property(self, prop: str, temperature: float) -> float:
        """
        Calculate an mass-fraction-averaged property for the homogenised mixture.
        """
        warn = []
        values, fractions = [], []
        # Calculate property mixtures, ignoring liquids and voids
        # for certain properties
        for mat, vf in zip(self.materials, self.fracs, strict=False):
            try:
                v = getattr(mat, prop)(temperature)
                values.append(v)
                fractions.append(vf)
            except (  # noqa: PERF203
                NotImplementedError,
                AttributeError,
                MaterialsError,
            ):
                warn.append([mat, prop])

        f = np.array(fractions) / sum(fractions)  # Normalised
        value = np.dot(values, f)

        if warn:
            txt = (
                f"Materials::{self.__class__.__name__}: The following "
                "mat.prop calls failed:\n"
            )
            for w in warn:
                txt += f"{w[0]}: {w[1]}" + "\n"
            bluemira_warn(txt)

        return value

    def E(self, temperature: float) -> float:  # noqa: N802
        """
        Young's modulus.

        Parameters
        ----------
        temperature:
            The optional temperature [K].

        Returns
        -------
        The Young's modulus of the material at the given temperature.
        """
        return self._calc_homogenised_property("E", temperature)

    def mu(self, temperature: float) -> float:
        """
        Poisson's ratio.

        Parameters
        ----------
        temperature:
            The optional temperature [K].

        Returns
        -------
        Poisson's ratio for the material at the given temperature.
        """
        return self._calc_homogenised_property("mu", temperature)

    def CTE(self, temperature: float) -> float:  # noqa: N802
        """
        Mean coefficient of thermal expansion in 10**-6/T

        Parameters
        ----------
        temperature:
            The temperature in Kelvin

        Returns
        -------
        Mean coefficient of thermal expansion in 10**-6/T at the given temperature.
        """
        return self._calc_homogenised_property("CTE", temperature)

    def rho(self, temperature: float) -> float:
        """
        Density.

        Parameters
        ----------
        temperature:
            The optional temperature [K].

        Returns
        -------
        The density of the material at the given temperature.
        """
        return self._calc_homogenised_property("rho", temperature)

    def Sy(self, temperature: float) -> float:  # noqa: N802
        """
        Minimum yield stress in MPa

        Parameters
        ----------
        temperature:
            The temperature in Kelvin

        Returns
        -------
        Minimum yield stress in MPa at the given temperature.
        """
        return self._calc_homogenised_property("Sy", temperature)

    # @classmethod
    # def from_dict(
    #     cls, name: str, material_dict: Dict[str, Any], material_cache: MaterialCache
    # ) -> SerialisedMaterial:
    #     """
    #     Generate an instance of the mixture from a dictionary of materials.

    #     Parameters
    #     ----------
    #     name:
    #         The name of the mixture
    #     materials_dict:
    #         The dictionary defining this and any additional mixtures
    #     material_cache:
    #         The cache to load the constituent materials from

    #     Returns
    #     -------
    #     The mixture
    #     """
    #     mat_dict = copy.deepcopy(material_dict[name])
    #     if "materials" not in material_dict[name]:
    #         raise MaterialsError("Mixture must define constituent materials.")

    #     for mat in material_dict[name]["materials"]:
    #         if isinstance(mat, str):
    #             del mat_dict["materials"][mat]
    #             material_inst = material_cache.get_material(mat, False)
    #             material_value = material_dict[name]["materials"][mat]
    #             mat_dict["materials"][material_inst] = material_value

    #     return super().from_dict(name, {name: mat_dict})
