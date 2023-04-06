# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
Interfaces for builder classes.
"""

from __future__ import annotations

import abc
from typing import Dict, List, Optional, Type, Union

from bluemira.base.components import Component
from bluemira.base.parameter_frame import ParameterFrame, make_parameter_frame
from bluemira.base.reactor_config import ConfigParams
from bluemira.base.tools import timing
from bluemira.utilities.plot_tools import set_component_view

BuildConfig = Dict[str, Union[int, float, str, "BuildConfig"]]
"""
Type alias for representing nested build configuration information.
"""


def _remove_suffix(s: str, suffix: str) -> str:
    # Python 3.9 has str.removesuffix()
    if suffix and s.endswith(suffix):
        return s[: -len(suffix)]
    return s


class ComponentManager(abc.ABC):
    """
    A wrapper around a component tree.

    The purpose of the classes deriving from this is to abstract away
    the structure of the component tree and provide access to a set of
    its features. This way a reactor build procedure can be completely
    agnostic of the structure of component trees, relying instead on
    a set of methods implemented on concrete `ComponentManager`
    instances.

    This class can also be used to hold 'construction geometry' that may
    not be part of the component tree, but was useful in construction
    of the tree, and could be subsequently useful (e.g., an equilibrium
    can be solved to get a plasma shape, the equilibrium is not
    derivable from the plasma component tree, but can be useful in
    other stages of a reactor build procedure).

    Parameters
    ----------
    component_tree: Component
        The component tree this manager should wrap.
    """

    def __init__(self, component_tree: Component) -> None:
        super().__init__()
        self._component = component_tree

    def component(self) -> Component:
        """
        Return the component tree wrapped by this manager.
        """
        return self._component


class Builder(abc.ABC):
    """
    Base class for component builders.

    Parameters
    ----------
    params
        The parameters required by the builder.
    build_config
        The build configuration for the builder.
    verbose
        control how much logging the designer will output

    Notes
    -----
    If there are no parameters associated with a concrete builder, set
    `param_cls` to `None` and pass `None` into this class's constructor.
    """

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, ConfigParams, None],
        build_config: Optional[Dict] = None,
        *,
        verbose=True,
    ):
        super().__init__()
        self.params = make_parameter_frame(params, self.param_cls)
        self.build_config = build_config if build_config is not None else {}
        self.name = self.build_config.get(
            "name", _remove_suffix(self.__class__.__name__, "Builder")
        )
        self.build = timing(
            self.build, "Built in", f"Building {self.name}", print_name=verbose
        )

    @abc.abstractproperty
    def param_cls(self) -> Union[Type[ParameterFrame], None]:
        """The class to hold this Builders's parameters."""
        pass

    @abc.abstractmethod
    def build(self) -> Component:
        """Build the component."""
        pass

    def component_tree(
        self, xz: List[Component], xy: List[Component], xyz: List[Component]
    ) -> Component:
        """
        Adds views of components to an overall component tree.

        Parameters
        ----------
        xz: List[Component]
            xz view of component
        xy: List[Component]
            xy view of component
        xyz: List[Component]
            xyz view of component

        Returns
        -------
        component

        """
        component = Component(self.name)
        component.add_child(Component("xz", children=xz))
        component.add_child(Component("xy", children=xy))
        component.add_child(Component("xyz", children=xyz))

        set_component_view(component.get_component("xz"), "xz")
        set_component_view(component.get_component("xy"), "xy")

        return component
