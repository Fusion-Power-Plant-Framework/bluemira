# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.

"""
Material mixture utility classes
"""
import copy
import numpy as np
import typing
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)
    import neutronics_material_maker as nmm

from BLUEPRINT.base.lookandfeel import bpwarn
from BLUEPRINT.materials.constants import MATERIAL_BEAM_MAP, T_DEFAULT
from BLUEPRINT.materials.material import (
    MaterialsError,
    SerialisedMaterial,
)


class HomogenisedMixture(SerialisedMaterial, nmm.MultiMaterial):
    """
    Inherits and does some dropping of 0 fractions (avoid touching nmm)
    """

    materials: typing.Dict[str, float]
    temperature_in_K: float  # noqa (N815)
    enrichment: float

    default_temperature = T_DEFAULT
    _material_classes = []

    def __init__(
        self,
        name,
        materials,
        temperature_in_K=None,  # noqa(N803)
        enrichment=None,
        zaid_suffix=None,
        material_id=None,
    ):
        if temperature_in_K is None:
            temperature_in_K = self.default_temperature  # noqa(N803)

        mats = []
        for mat in materials.keys():
            mat.temperature = temperature_in_K
            if "enrichment" in mat.__class__.__annotations__:
                mat.enrichment = enrichment
            mats += [mat]

        super().__init__(
            material_tag=name,
            materials=mats,
            fracs=list(materials.values()),
            percent_type="vo",
            temperature_in_K=temperature_in_K,
            zaid_suffix=zaid_suffix,
            material_id=material_id,
        )

        self.name = name

    def __str__(self):
        """
        Get the name of the mixture.
        """
        return self.name

    def make_mat_dict(self, temperature):
        """
        Makes a material dictionary for use in simple beam FE solver

        Parameters
        ----------
        temperature: float
            The temperature in Kelvin

        Returns
        -------
        mat_dict: dict
            The simplified dictionary of material properties
        """
        mat_dict = {}

        for prop in ["E", "mu", "rho", "CTE", "Sy"]:
            if hasattr(self, prop):
                # Override mixture property calculation if one has been specced
                value = getattr(self, prop)(temperature)
            else:
                warn = []
                values, fractions = [], []
                # Calculate property mixtures, ignoring liquids and voids
                # for certain properties
                for mat, vf in zip(self.materials, self.fracs):
                    try:
                        v = getattr(mat, prop)(temperature)
                        values.append(v)
                        fractions.append(vf)
                    except (NotImplementedError, AttributeError, MaterialsError):
                        warn.append([mat, prop])

                vals = np.array(values)
                f = np.array(fractions) / sum(fractions)  # Normalised
                value = np.dot(vals, f)

                if warn:
                    txt = (
                        f"Materials::{self.__class__.__name__}: The following "
                        + "mat.prop calls failed:\n"
                    )
                    for w in warn:
                        txt += f"{w[0]}: {w[1]}" + "\n"
                    bpwarn(txt)

            key = MATERIAL_BEAM_MAP[prop]
            if prop == "E":
                value *= 1e9
            mat_dict[key] = value

        return mat_dict

    @classmethod
    def from_dict(cls, name, materials_dict, materials_cache):
        """
        Generate an instance of the mixture from a dictionary of materials.

        Parameters
        ----------
        name : str
            The name of the mixture.
        materials_dict: Dict[str, Any]
            The dictionary defining this and any additional mixtures.
        materials_cache: MaterialCache
            The cache to load the constituent materials from.

        Returns
        -------
        mixture : SerialisedMaterial
            The mixture.
        """
        mat_dict = copy.deepcopy(materials_dict[name])
        if "materials" not in materials_dict[name].keys():
            raise MaterialsError("Mixture must define constituent materials.")

        for mat in materials_dict[name]["materials"]:
            if isinstance(mat, str):
                del mat_dict["materials"][mat]
                material_inst = materials_cache.get_material(mat, False)
                material_value = materials_dict[name]["materials"][mat]
                mat_dict["materials"][material_inst] = material_value

        return super().from_dict(name, {name: mat_dict})


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
