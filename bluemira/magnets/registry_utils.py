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
    Mixin class to automatically register instances into a global instance cache.

    This class provides:
    - Automatic instance registration into a global cache.
    - Optional control over registration (register or not).
    - Optional automatic generation of unique names to avoid conflicts.

    Attributes
    ----------
    _global_instance_cache_ : dict
        Class-level cache shared among all instances for lookup by name.
    _do_not_register : bool
        If True, the instance will not be registered.
    _unique : bool
        If True, automatically generate a unique name if the desired name already exists.
    """

    _global_instance_cache_: ClassVar[dict] = {}

    def __init__(self, name: str, *, unique_name: bool = False):
        """
        Initialize an instance and optionally register it.

        Parameters
        ----------
        name : str
            Desired name of the instance.
        unique_name : bool, optional
            If True, generate a unique name if the given name already exists.
            If False (strict mode), raise a ValueError on duplicate names.
        """
        self._unique_name = None
        self.unique_name = unique_name

        # Setting the name will trigger registration (unless do_registration is False)
        self._name = None
        self.name = name

    @property
    def unique_name(self) -> bool:
        """
        Flag indicating whether to automatically generate a unique name on conflict.

        Returns
        -------
        bool
            True if automatic unique name generation is enabled (the passed name is
            neglected)
            False if strict name checking is enforced.
        """
        return self._unique_name

    @unique_name.setter
    def unique_name(self, value: bool):
        """
        Set whether automatic unique name generation should be enabled.

        Parameters
        ----------
        value : bool
            If True, automatically generate a unique name if the desired name
            is already registered.
            If False, raise a ValueError if the name already exists.
        """
        self._unique_name = value

    @property
    def name(self) -> str:
        """Return the instance name."""
        return self._name

    @name.setter
    def name(self, value: str):
        """
        Set the instance name and (re)register it according to registration rules.

        Behavior
        --------
        - If `_do_not_register` is True, just assign the name without caching.
        - If `unique_name` is True and the name already exists, automatically generate
        a unique name.
        - If `unique_name` is False and the name already exists, raise ValueError.

        Parameters
        ----------
        value : str
            Desired instance name.

        Raises
        ------
        ValueError
            If `unique_name` is False and the name is already registered.
        """
        if hasattr(self, "_name") and self._name is not None:
            self._unregister_self()

        if value is None:
            self._name = None
            return

        if value in self._global_instance_cache_:
            if self.unique_name:
                value = self.generate_unique_name(value)
            else:
                raise ValueError(f"Instance with name '{value}' already registered.")

        self._name = value
        self._register_self()

    def _register_self(self):
        """
        Register this instance into the global instance cache.

        Raises
        ------
        AttributeError
            If the instance does not have a 'name' attribute.
        ValueError
            If an instance with the same name already exists and unique is False.
        """
        if getattr(self, "_do_not_register", False):
            return  # Skip registration if explicitly disabled

        if not hasattr(self, "name") or self.name is None:
            raise AttributeError("Instance must have a 'name' attribute to register.")

        if self.name in self._global_instance_cache_:
            if self.unique_name:
                self.name = self.generate_unique_name(self.name)
            else:
                raise ValueError(f"Instance with name '{self.name}' already registered.")

        self._global_instance_cache_[self.name] = self

    def _unregister_self(self):
        """
        Unregister this instance from the global instance cache.
        """
        if hasattr(self, "name") and self.name in self._global_instance_cache_:
            del self._global_instance_cache_[self.name]

    @classmethod
    def get_registered_instance(cls, name: str):
        """
        Retrieve a registered instance by name.

        Parameters
        ----------
        name : str
            Name of the registered instance.

        Returns
        -------
        InstanceRegistrable or None
            The registered instance, or None if not found.
        """
        return cls._global_instance_cache_.get(name)

    @classmethod
    def list_registered_instances(cls) -> list[str]:
        """
        List names of all registered instances.

        Returns
        -------
        list of str
            List of names of registered instances.
        """
        return list(cls._global_instance_cache_.keys())

    @classmethod
    def clear_registered_instances(cls):
        """
        Clear all registered instances from the global cache.
        """
        cls._global_instance_cache_.clear()

    @classmethod
    def generate_unique_name(cls, base_name: str) -> str:
        """
        Generate a unique name by appending a numeric suffix if necessary.

        Parameters
        ----------
        base_name : str
            Desired base name.

        Returns
        -------
        str
            Unique name guaranteed not to conflict with existing instances.
        """
        if base_name not in cls._global_instance_cache_:
            return base_name

        i = 1
        while f"{base_name}_{i}" in cls._global_instance_cache_:
            i += 1
        return f"{base_name}_{i}"
