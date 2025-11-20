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
import types

from matproplib.library.fluids import Void
from matproplib.material import Material

from bluemira.materials.error import MaterialsError
from bluemira.utilities.tools import get_module

vacuum_void = Void(name="Vacuum")


class MaterialCache:
    """
    A helper class for loading and caching materials.

    Notes
    -----
    Extend the `available_classes` attribute to load custom classes.
    """

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
            if value in self._material_package:
                return self._material_package[value]
            raise

    def load_from_package(self, package):
        """Load material package"""
        self._material_packages = [get_module(p) for p in package]

    def _get_material(self, name):
        name = name.replace("-", "_")
        name = name.split(".") if "." in name else [name]

        def _d(p, name, initial_p):
            if name in dir(p):
                return getattr(p, name)
            for dp in filter(lambda s: not s.startswith("__"), dir(p)):
                attr = getattr(p, dp)
                if dp == name:
                    return attr
                if (
                    isinstance(attr, types.ModuleType)
                    and initial_p in repr(attr)
                    and (attr := _d(attr, name, initial_p)) is not None
                ):
                    return attr
            return None

        for p in self._material_packages:
            initial_p = p.__name__.replace(".", "/")
            mod = p
            for n in name:
                mod = _d(mod, n, initial_p)
                if isinstance(mod, Material):
                    return mod
        raise AttributeError(f"No such material: {name}")

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
            return copy.deepcopy(self._get_material(name))
        return self._get_material(name)


def establish_material_cache(materials_package: str | object):
    """
    Load the material data from the provided json files into the global material cache
    instance.

    This instance can be accessed using the `MaterialCache.get_instance()`
    function.

    Parameters
    ----------
    materials_package:
        A list of paths to the data files to load into the material cache.

    Returns
    -------
    The material cache.
    """
    cache = MaterialCache.get_instance()
    cache.load_from_package(materials_package)
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
