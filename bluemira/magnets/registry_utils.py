# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Generic class and instance registration utilities.

This module provides:
- RegistrableMeta: A metaclass that automatically registers classes into a specified
registry.
- InstanceRegistrable: A mixin that automatically registers instances into a specified
global cache.

Intended for use in frameworks where automatic discovery of classes
and instances is required, such as for strands, cables, conductors, or other physical
models.

Usage
-----
Classes intended to be registered must:
- Define a class-level `_registry_` dictionary (for class registration).
- Optionally set a `_name_in_registry_` string (custom name for registration).

Instances intended to be globally tracked must:
- Inherit from InstanceRegistrable.
- Provide a unique `name` attribute at creation.
"""

from abc import ABCMeta
from typing import ClassVar

from bluemira.base.look_and_feel import bluemira_debug


# ------------------------------------------------------------------------------
# RegistrableMeta
# ------------------------------------------------------------------------------
class RegistrableMeta(ABCMeta):
    """
    Metaclass for automatic class registration into a registry.

    Enforces that:
    - '_name_in_registry_' must be explicitly defined in every class body (no
    inheritance allowed).
    - '_registry_' can be inherited if not redefined.
    """

    def __new__(mcs, name, bases, namespace):
        """
        Create and register a new class instance using the RegistrableMeta metaclass.

        This method:
        - Automatically registers concrete (non-abstract) classes into a specified
        registry.
        - Enforces that concrete classes explicitly declare a '_name_in_registry_'.
        - Allows '_registry_' to be inherited from base classes if not redefined.

        Parameters
        ----------
        mcs : type
            The metaclass (usually RegistrableMeta itself).
        name : str
            The name of the class being created.
        bases : tuple of type
            The base classes of the class being created.
        namespace : dict
            The attribute dictionary of the class.

        Returns
        -------
        type
            The newly created class.

        Raises
        ------
        TypeError
            If a concrete class does not define a '_name_in_registry_'.
            If no '_registry_' can be found (either defined or inherited).
        ValueError
            If a duplicate '_name_in_registry_' is detected within the registry.

        Notes
        -----
        Abstract base classes (ABCs) are exempted from registration requirements.

        Registration process:
        - If the class is abstract, skip registration.
        - Otherwise:
            - Check for existence of '_registry_' (allow inheritance).
            - Require explicit '_name_in_registry_' (must be defined in the class body).
            - Insert the class into the registry under the specified name.
        """
        bluemira_debug(f"Registering {name}...")  # Debug print

        cls = super().__new__(mcs, name, bases, namespace)

        is_abstract = bool(getattr(cls, "__abstractmethods__", False))

        if not is_abstract:
            # Only enforce _name_in_registry_ and _registry_ for concrete classes
            # _registry_ can be inherited
            registry = getattr(cls, "_registry_", None)

            # _name_in_registry_ must be explicit in the class body
            register_name = namespace.get("_name_in_registry_", None)

            # Checks
            if registry is None:
                raise TypeError(
                    f"Class {name} must define or inherit a '_registry_' for "
                    f"registration."
                )

            if register_name is None:
                raise TypeError(
                    f"Class {name} must explicitly define a '_name_in_registry_' for "
                    f"registration."
                )

            # Registration
            if register_name:
                if register_name in registry:
                    raise ValueError(
                        f"Duplicate registration for class '{register_name}'."
                    )
                registry[register_name] = cls
                cls._name_in_registry_ = register_name  # Optional: for introspection

        return cls


# ------------------------------------------------------------------------------
# InstanceRegistrable
# ------------------------------------------------------------------------------
class InstanceRegistrable:
    """
    Mixin to provide automatic instance registration into a global instance cache.

    Classes using this mixin must define:
    - _global_instance_cache_: dict
        The dictionary where instances are registered.

    Provides:
    - Automatic registration and unregistration when 'name' is set.
    - Unique name generation to avoid collisions in the registry.
    """

    _global_instance_cache_: ClassVar[dict] = {}  # Must be overridden in subclasses.

    def _register_self(self):
        """
        Register this instance into the global instance cache.

        Raises
        ------
        AttributeError
            If the instance has no 'name' attribute.
        ValueError
            If an instance with the same name is already registered.
        """
        if not hasattr(self, "name") or self._name is None:
            raise AttributeError("Instance must have a 'name' attribute to register.")

        if self._name in self._global_instance_cache_:
            raise ValueError(f"Instance with name '{self._name}' already registered.")

        self._global_instance_cache_[self._name] = self

    def _unregister_self(self):
        """
        Unregister this instance from the global instance cache.

        Does nothing if the instance is not registered.
        """
        if hasattr(self, "_name") and self._name in self._global_instance_cache_:
            del self._global_instance_cache_[self._name]

    @property
    def name(self) -> str:
        """
        Get the name of the instance.

        Returns
        -------
        str
            Current name of the instance.
        """
        return self._name

    @name.setter
    def name(self, value: str):
        """
        Set a new name for the instance and update the registry accordingly.

        Parameters
        ----------
        value : str
            New name for the instance.

        Raises
        ------
        ValueError
            If the new name is already registered.
        """
        if hasattr(self, "_name"):
            self._unregister_self()

        self._name = value

        if value is not None:
            self._register_self()

    @classmethod
    def generate_unique_name(cls, base_name: str) -> str:
        """
        Generate a unique name based on a given base name.

        If the base name already exists in the instance cache,
        appends an incremental suffix (_1, _2, etc.) until a unique name is found.

        Parameters
        ----------
        base_name : str
            Proposed base name.

        Returns
        -------
        str
            A unique name not already registered.
        """
        if base_name not in cls._global_instance_cache_:
            return base_name

        counter = 1
        while True:
            candidate = f"{base_name}_{counter}"
            if candidate not in cls._global_instance_cache_:
                return candidate
            counter += 1

    @classmethod
    def get_registered_instance(cls, name: str):
        """
        Retrieve a registered instance by name.

        Parameters
        ----------
        name : str
            Name of the instance to retrieve.

        Returns
        -------
        InstanceRegistrable or None
            The instance if found, otherwise None.
        """
        return cls._global_instance_cache_.get(name)

    @classmethod
    def list_registered_instances(cls) -> list[str]:
        """
        List all currently registered instance names.

        Returns
        -------
        list of str
            List of registered instance names.
        """
        return list(cls._global_instance_cache_.keys())

    @classmethod
    def clear_registered_instances(cls):
        """
        Clear all instances from the global instance cache.

        Useful for testing or reloading.
        """
        cls._global_instance_cache_.clear()
