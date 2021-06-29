# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
Module containing the base Component class.
"""

import anytree
from anytree import NodeMixin, RenderTree
import copy
from typing import Any, Dict, List, Optional, Type, Union

from bluemira.base.parameter import Parameter, ParameterFrame, ParameterMapping
from bluemira.components.error import ComponentError


class Component(NodeMixin):
    """
    The Component is the fundamental building block for a bluemira reactor design. It
    encodes the way that the corresponding part of the reactor will be built, along with
    any other derived properties that relate to that component.

    Components define a tree structure, based on the parent and children properties. This
    allows the nodes on that tree to be passed around within bluemira so that
    operations can be performed on the child branches of that structure.

    For example, a reactor design including just a TFCoilSystem may look as below:

    .. digraph:: base_component_tree

      "FusionPowerPlant" -> "TFCoilSystem" -> {"TFWindingPack" "TFCasing"}

    A Component cannot be used directly - only subclasses should be instantiated.
    """

    name: str
    params: ParameterFrame
    _subsystem_base_classes: Dict[str, Type["Component"]] = {}
    __available_classes: Dict[str, Type["Component"]] = {}

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        inputs: Dict[str, Any],
        parent: Optional["Component"] = None,
        children: Optional[List["Component"]] = None,
    ):
        self.name = name
        self.params = ParameterFrame(self.default_params.to_records())
        self.params.update_kw_parameters(config)
        self.inputs = inputs
        self.parent = parent
        if children:
            self.children = children

    def __new__(cls, *args, **kwargs) -> Type["Component"]:
        """
        Constructor for Component class.
        """
        if cls is Component:
            raise ComponentError(
                "Component cannot be initialised directly - use a subclass."
            )
        return super().__new__(cls)

    def __init_subclass__(cls):
        """
        Initialise a Component subclass.
        """
        super().__init_subclass__()

        # Get the default_params (or those of the parent if not overridden)
        default_params = getattr(cls, "default_params", [])
        if isinstance(default_params, ParameterFrame):
            default_params = default_params.to_records()
        cls.default_params = ParameterFrame(default_params)

        # Build a registry of unique ReactorSystem classes for access by name.
        if cls.__name__ not in cls.__available_classes:
            cls.__available_classes[cls.__name__] = cls
        else:
            raise ComponentError(
                f"A class with name {cls.__name__} is already defined in bluemira as a "
                f"ReactorSystem : {cls.__available_classes[cls.__name__]} from "
                f"{cls.__available_classes[cls.__name__].__module__}. You tried to "
                f"specify {cls} from {cls.__module__}."
            )

        # Get the subsystem base classes from the parent class
        cls._subsystem_base_classes = {}
        for parent_class in cls.__bases__:
            if issubclass(parent_class, Component):
                cls._subsystem_base_classes.update(parent_class._subsystem_base_classes)

        # Add any new subsystem base classes (or perform any overrides)
        if hasattr(cls, "__annotations__"):
            for name, ty in cls.__annotations__.items():
                if hasattr(ty, "__origin__") and (
                    ty.__origin__ is Type or ty.__origin__ is type
                ):
                    # Only handle single base classes for now.
                    if len(ty.__args__) == 1 and issubclass(ty.__args__[0], Component):
                        cls._subsystem_base_classes[name] = ty.__args__[0]

    def __repr__(self) -> str:
        """
        The string representation of the instance
        """
        return self.name + " (" + self.__class__.__name__ + ")"

    def tree(self) -> str:
        """
        Get the tree of descendants of this instance.
        """
        return str(RenderTree(self))

    @classmethod
    def get_class(cls, name: str) -> "Component":
        """
        Get the class with the provided name.

        The requested class must be either the class itself or one of its subclasses.

        Parameters
        ----------
        name: str
            The class name.

        Returns
        -------
        the_class: Component
            The class with the specified name.
        """
        if name not in cls.__available_classes:
            raise ComponentError(
                f"{name} is not known as a bluemira Component. Either ensure that  the "
                "class inherits from a Component or check that you have imported the "
                "required module."
            )

        the_class = cls.__available_classes[name]
        if issubclass(the_class, cls):
            return the_class
        else:
            raise ComponentError(
                f"Unable to find {name} as a subclass of {cls.__name__}"
            )

    def _generate_subsystem_classes(self, config: Dict[str, str]):
        """
        Generate a dictionary of classes to be used in the build, and validate them.

        Parameters
        ----------
        config: Dict[str, str]
            The build configuration for the Component, containing a key specifying the
            class names for the Components.
        """
        self._subsystem_classes = {}
        for name, ty in self._subsystem_base_classes.items():
            config_key = f"{name.lower()}_class_name"
            class_name = config.get(config_key, ty.__name__)
            self._subsystem_classes[name] = ty.get_class(class_name)

    def get_subsystem_class(self, key: str) -> Type["Component"]:
        """
        Get the subsystem class corresponding to the key.

        Parameters
        ----------
        key: str
            The subsystem key.

        Returns
        -------
        system_class: Type[Component]
            The subsystem class corresponding to the provided key.
        """
        try:
            return self._subsystem_classes[key]
        except KeyError:
            raise ComponentError(f"Unknown subsystem key {key} requested.")

    def get_component(
        self, name: str, first: bool = True
    ) -> Union["Component", List["Component"]]:
        """
        Find the components with the specified name.

        Parameters
        ----------
        name: str
            The name of the component to search for.
        first: bool
            If True, only the first element is returned, by default True.

        Returns
        -------
        found_components: Union[Component, List[Component]]
            The first component of the search if first is True, else all components
            matching the search.

        Notes
        -----
            This function is just a wrapper of the anytree.search.findall_by_attr
            function.
        """
        found_components = anytree.search.findall_by_attr(self.root, name)
        if len(found_components) == 0:
            return None
        if first:
            return found_components[0]
        return found_components

    def add_parameter(
        self,
        var: str,
        name: Optional[str] = None,
        value: Any = None,
        unit: Optional[str] = None,
        description: Optional[str] = None,
        source: Optional[str] = None,
        mapping: Optional[Dict[str, ParameterMapping]] = None,
        value_history: Optional[List[Any]] = None,
        source_history: Optional[List[str]] = None,
    ):
        """
        Takes a list or Parameter object and adds it to the ParameterFrame
        Handles updates if existing parameter (Var_name sorted).

        Parameters
        ----------
        var: str
            The short parameter name
        name: Union[str, None]
            The long parameter name, by default None.
        value: Union[str, float, int, None]
            The value of the parameter, by default None.
        unit: Union[str, None]
            The unit of the parameter, by default None.
        description: Union[str, None]
            The long description of the parameter, by default None.
        source: Union[str, None]
            The source (reference and/or code) of the parameter, by default None.
        mapping: Union[Dict[str, ParameterMapping], None]
            The names used for this parameter in external software, and whether
            that parameter should be written to and/or read from the external tool,
            by default, None.
        value_history: Union[list, None]
            History of the value
        source_history: Union[list, None]
            History
        """
        self.params.add_parameter(
            var,
            name,
            value,
            unit,
            description,
            source,
            mapping,
            value_history,
            source_history,
        )

    def add_parameters(
        self,
        record_list: Union[Dict[str, Any], List[List[Any]], List[Parameter]],
        source=None,
    ):
        """
        Handles a record_list for ParameterFrames and updates accordingly.
        Items in record_list may be Parameter objects or lists in the following format:

        [var, name, value, unit, description, source]

        If a record_list is a dict, it is passed to update_kw_parameters
        with the specified source.

        Parameters
        ----------
        record_list: Union[dict, list, Parameter]
            Container of individual Parameters
        source: str
            Updates the source parameter for each item in record_list with the
            specified value, by default None (i.e. the value is left unchanged).
        """
        self.params.add_parameters(record_list, source=source)

    def copy(self):
        """
        Provides a deep copy of the Component

        Returns
        -------
        copy: Component
            The copy of the Component
        """
        return copy.deepcopy(self)


class GroupingComponent(Component):
    """
    A Component that groups other Components.
    """

    pass


class PhysicalComponent(Component):
    """
    A physical component. It includes shape and materials.
    """

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        inputs: Dict[str, Any],
        shape: Any,
        material: Any = None,
        parent: Component = None,
        children: Component = None,
    ):
        super().__init__(name, config, inputs, parent, children)
        self.shape = shape
        self.material = material

    @property
    def shape(self):
        """
        The geometric shape of the Component.
        """
        return self._shape

    @shape.setter
    def shape(self, value):
        self._shape = value

    @property
    def material(self):
        """
        The material that the Component is built from.
        """
        return self._material

    @material.setter
    def material(self, value):
        self._material = value


class MagneticComponent(PhysicalComponent):
    """
    A magnetic component. It includes a shape, a material, and a source conductor.
    """

    def __init__(
        self,
        name: str,
        config: Dict[str, Any],
        inputs: Dict[str, Any],
        shape: Any,
        material: Any = None,
        conductor: Any = None,
        parent: Component = None,
        children: Component = None,
    ):
        super().__init__(name, config, inputs, shape, material, parent, children)
        self.conductor = conductor

    @property
    def conductor(self):
        """
        The conductor used by current-carrying filaments.
        """
        return self._conductor

    @conductor.setter
    def conductor(self, value):
        self._conductor = value
