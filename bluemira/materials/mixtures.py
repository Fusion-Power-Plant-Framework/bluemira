# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Material mixture utility classes
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.materials.error import MaterialsError
from bluemira.materials.tools import to_openmc_material_mixture

if TYPE_CHECKING:
    from bluemira.materials.cache import MaterialCache
    from bluemira.materials.material import MassFractionMaterial


@dataclass
class MixtureFraction:
    """Material mixture fraction"""

    material: Union[MassFractionMaterial, HomogenisedMixture]
    fraction: float

    def __post_init__(self):
        """Set name"""
        self.name = self.material.name


@dataclass
class HomogenisedMixture:
    """
    Inherits and does some dropping of 0 fractions (avoid touching nmm)
    """

    name: str
    materials: List[MixtureFraction]
    material_id: Optional[int] = None
    percent_type: str = "vo"
    packing_fraction: float = 1.0
    enrichment: Optional[float] = None
    temperature: Optional[float] = None

    def __str__(self) -> str:
        """
        Get the name of the mixture.
        """
        return self.name

    def to_openmc_material(self, temperature: Optional[float] = None):
        """
        Convert the mixture to an openmc material.
        """
        temperature = self.temperature if temperature is None else temperature
        return to_openmc_material_mixture(
            [mat.material.to_openmc_material(temperature) for mat in self.materials],
            [mat.fraction for mat in self.materials],
            self.name,
            self.material_id,
            temperature,
            percent_type=self.percent_type,
            packing_fraction=self.packing_fraction,
        )

    def _calc_homogenised_property(self, prop: str, temperature: float):
        """
        Calculate an mass-fraction-averaged property for the homogenised mixture.
        """
        warn = []
        values, fractions = [], []
        # Calculate property mixtures, ignoring liquids and voids
        # for certain properties
        for mat in self.materials:
            try:
                v = getattr(mat.material, prop)(temperature)
                if v is None:
                    warn.append([mat.name, prop])
                else:
                    values.append(v)
                    fractions.append(mat.fraction)
            except AttributeError:  # noqa: PERF203
                warn.append([mat.name, prop])

        f = np.array(fractions) / sum(fractions)  # Normalised
        value = np.dot(values, f)

        if warn:
            txt = (
                f"Materials::{type(self).__name__}: The following "
                "mat.prop calls failed:\n"
            )
            for w in warn:
                txt += f"{w[0]}: {w[1]}\n"
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

    @classmethod
    def from_dict(
        cls, name: str, material_dict: Dict[str, Any], material_cache: MaterialCache
    ) -> HomogenisedMixture:
        """
        Generate an instance of the mixture from a dictionary of materials.

        Parameters
        ----------
        name:
            The name of the mixture
        materials_dict:
            The dictionary defining this and any additional mixtures
        material_cache:
            The cache to load the constituent materials from

        Returns
        -------
        The mixture
        """
        mat_dict = copy.deepcopy(material_dict[name])
        if "materials" not in material_dict[name]:
            raise MaterialsError("Mixture must define constituent materials.")

        materials = []
        for mat in material_dict[name]["materials"]:
            if isinstance(mat, str):
                material_inst = material_cache.get_material(mat, False)
                material_value = material_dict[name]["materials"][mat]
                materials.append(MixtureFraction(material_inst, material_value))
        mat_dict["materials"] = materials
        return cls(name, **mat_dict)
