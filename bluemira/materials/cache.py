# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Classes and methods to load, store, and retrieve materials.
"""

from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any, ClassVar

from bluemira.materials.material import (
    BePebbleBed,
    CopperRRR,
    Liquid,
    MassFractionMaterial,
    Material,
    MaterialsError,
    NbSnSuperconductor,
    NbSnSuperconductorMag,
    NbTiSuperconductor,
    Plasma,
    UnitCellCompound,
    Void,
)
from bluemira.materials.mixtures import HomogenisedMixture

if TYPE_CHECKING:
    from pathlib import Path


class MaterialCache:
    """
    A helper class for loading and caching materials.

    Notes
    -----
    Extend the `available_classes` attribute to load custom classes.
    """

    _material_dict: ClassVar = {}

    default_classes = (
        HomogenisedMixture,
        BePebbleBed,
        Liquid,
        MassFractionMaterial,
        Material,
        MaterialsError,
        NbSnSuperconductor,
        NbSnSuperconductorMag,
        NbTiSuperconductor,
        Plasma,
        UnitCellCompound,
        Void,
        CopperRRR,
    )

    def __init__(self):
        self.available_classes = {
            mat_class.__name__: mat_class for mat_class in self.default_classes
        }

    _instance: MaterialCache | None = None

    @classmethod
    def get_instance(cls) -> MaterialCache:
        """
        Get the singleton instance of the MaterialCache.

        Returns
        -------
        MaterialCache
            The singleton instance of the MaterialCache.
        """
        if cls._instance is None:
            cls._instance = MaterialCache()
        return cls._instance

    def __getattr__(self, value: str):
        """Allow attribute access to cached materials

        Returns
        -------
        :
            The requested Material

        Raises
        ------
        AttributeError
            No attribute found
        """
        try:
            super().__getattribute__(value)
        except AttributeError:
            if value in self._material_dict:
                return self._material_dict[value]
            raise

    def load_from_file(self, path: str) -> dict[str, Any]:
        """
        Load materials from a file.

        Parameters
        ----------
        path:
            The path to the file from which to load the materials.
        """
        with open(path) as fh:
            mats_dict = json.load(fh)
        for name in mats_dict:
            self.load_from_dict(name, mats_dict)

    def load_from_dict(
        self, mat_name: str, mats_dict: dict[str, Any], *, overwrite: bool = True
    ):
        """
        Load a material or mixture from a dictionary.

        Parameters
        ----------
        mat_name:
            The name of the material or mixture.
        mat_dict:
            The dictionary containing the material or mixture attributes to be loaded.

        Raises
        ------
        MaterialsError
            Unknown material class
        """
        if (
            material_class := mats_dict[mat_name]["material_class"]
        ) not in self.available_classes:
            raise MaterialsError(
                f"Request to load unknown material class {material_class}"
            )

        if issubclass(self.available_classes[material_class], HomogenisedMixture):
            self.mixture_from_dict(mat_name, mats_dict, overwrite=overwrite)
        else:
            self.material_from_dict(mat_name, mats_dict, overwrite=overwrite)

    def mixture_from_dict(
        self, mat_name: str, mats_dict: dict[str, Any], *, overwrite: bool = True
    ):
        """
        Load a mixture from a dictionary.

        Parameters
        ----------
        mat_name:
            The name of the mixture.
        mat_dict:
            The dictionary containing the mixture attributes to be loaded.
        """
        mat_class = self.available_classes[mats_dict[mat_name].pop("material_class")]
        self._update_cache(
            mat_name, mat_class.from_dict(mat_name, mats_dict, self), overwrite=overwrite
        )

    def material_from_dict(
        self, mat_name: str, mats_dict: dict[str, Any], *, overwrite: bool = True
    ):
        """
        Load a material from a dictionary.

        Parameters
        ----------
        mat_name:
            The name of the material.
        mat_dict:
            The dictionary containing the material attributes to be loaded.
        """
        mat_class = self.available_classes[mats_dict[mat_name].pop("material_class")]
        self._update_cache(
            mat_name, mat_class(mat_name, **mats_dict[mat_name]), overwrite=overwrite
        )

    def get_material(self, name: str, *, clone: bool = True):
        """
        Get the named material from the material dictionary

        Parameters
        ----------
        name:
            The name of the material to retrieve from the dictionary
        clone:
            If True, get a clone (deepcopy) of the material, else get the actual material
            as stored in the material dictionary. By default True.

        Returns
        -------
        The requested material.
        """
        if clone:
            return copy.deepcopy(self._material_dict[name])
        return self._material_dict[name]

    def _update_cache(self, mat_name: str, mat, *, overwrite: bool = True):
        if not overwrite and mat_name in self._material_dict:
            raise MaterialsError(
                f"Attempt to load material {mat_name}, which already "
                "exists in the cache."
            )
        self._material_dict[mat_name] = mat


def establish_material_cache(materials_json_paths: list[Path | str]):
    """
    Load the material data from the provided json files into the global material cache
    instance.

    This instance can be accessed using the `MaterialCache.get_instance()`
    function.

    Parameters
    ----------
    materials_json_paths:
        A list of paths to the data files to load into the material cache.

    Returns
    -------
    The material cache.
    """
    cache = MaterialCache.get_instance()
    for path in materials_json_paths:
        cache.load_from_file(path)
    return cache


def get_cached_material(
    material_name: str, cache: MaterialCache | None = None
) -> Material | None:
    """
    Get the named material from the MaterialCache.

    If cache is None, the global cache instance is used.

    Parameters
    ----------
    material_name:
        The name of the material to retrieve from the dictionary
    cache:
        The material cache to retrieve the material from. By default the global cache.

    Returns
    -------
    :
        The requested material.

    Raises
    ------
    MaterialsError
        If the material name is not a string.
    """
    if not (material_name and isinstance(material_name, str)):
        raise MaterialsError("Material name must be a non-empty string.")
    if cache is None:
        cache = MaterialCache.get_instance()
    return cache.get_material(material_name)
