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
from typing import Dict, List, Optional, Tuple, Type, Union
from warnings import warn

from bluemira.base.components import Component
from bluemira.base.error import ComponentError
from bluemira.base.parameter_frame import ParameterFrame, make_parameter_frame
from bluemira.base.reactor_config import ConfigParams
from bluemira.display.displayer import ComponentDisplayer
from bluemira.display.plotter import ComponentPlotter
from bluemira.utilities.plot_tools import set_component_view

BuildConfig = Dict[str, Union[int, float, str, "BuildConfig"]]
"""
Type alias for representing nested build configuration information.
"""


_PLOT_DIMS = ["xy", "xz"]
_CAD_DIMS = ["xy", "xz", "xyz"]


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
        self._component = component_tree

    def component(self) -> Component:
        """
        Return the component tree wrapped by this manager.
        """
        return self._component

    def tree(self) -> str:
        """
        Get the component tree
        """
        return self.component().tree()

    def _validate_cad_dims(self, *dims: str, **kwargs) -> Tuple[str, ...]:
        """
        Validate showable CAD dimensions
        """
        # give dims_to_show a default value
        dims_to_show = ("xyz",) if len(dims) == 0 else dims

        # if a kw "dim" is given, it is only used
        if kw_dim := kwargs.pop("dim", None):
            warn(
                "Using kwarg 'dim' is no longer supported. "
                "Simply pass in the dimensions you would like to show, e.g. show_cad('xz')",
                category=DeprecationWarning,
            )
            dims_to_show = (kw_dim,)
        for dim in dims_to_show:
            if dim not in _CAD_DIMS:
                raise ComponentError(
                    f"Invalid plotting dimension '{dim}'."
                    f"Must be one of {str(_CAD_DIMS)}"
                )

        return dims_to_show

    def _validate_plot_dims(self, *dims) -> Tuple[str, ...]:
        """
        Validate showable plot dimensions
        """
        # give dims_to_show a default value
        dims_to_show = ("xz",) if len(dims) == 0 else dims

        for dim in dims_to_show:
            if dim not in _PLOT_DIMS:
                raise ComponentError(
                    f"Invalid plotting dimension '{dim}'."
                    f"Must be one of {str(_PLOT_DIMS)}"
                )

        return dims_to_show

    def _filter_tree(self, comp: Component, dims_to_show: Tuple[str, ...]) -> Component:
        """
        Filter a component tree

        Notes
        -----
        A copy of the component tree must be made
        as filtering would mutate the ComponentMangers' underlying component trees
        """
        comp_copy = comp.copy()
        comp_copy.filter_components(dims_to_show)
        return comp_copy

    def show_cad(
        self,
        *dims: str,
        **kwargs,
    ):
        """
        Show the CAD build of the reactor.

        Parameters
        ----------
        *dims
            The dimension of the reactor to show, typically one of
            'xz', 'xy', or 'xyz'. (default: 'xyz')
        """
        ComponentDisplayer().show_cad(
            self._filter_tree(
                self.component(), self._validate_cad_dims(*dims, **kwargs)
            ),
            **kwargs,
        )

    def _plot_dims(self, comp: Component, dims_to_show: Tuple[str, ...]):
        for i, dim in enumerate(dims_to_show):
            ComponentPlotter(view=dim).plot_2d(
                self._filter_tree(comp, dims_to_show),
                show=i == len(dims_to_show) - 1,
            )

    def plot(self, *dims: str, with_components: Optional[List[ComponentManager]] = None):
        """
        Plot the reactor.

        Parameters
        ----------
        *dims:
            The dimension(s) of the reactor to show, 'xz' and/or 'xy'.
            (default: 'xz')
        with_components:
            The components to construct when displaying CAD for xyz.
            Defaults to None, which means show "all" components.
        """
        self._plot_dims(self.component(), self._validate_plot_dims(dims))


class Builder(abc.ABC):
    """
    Base class for component builders.

    Parameters
    ----------
    params: Union[Dict, ParameterFrame]
        The parameters required by the builder.
    build_config: Dict
        The build configuration for the builder.
    designer: Optional[Designer]
        A designer to solve a design problem required by the builder.

    Notes
    -----
    If there are no parameters associated with a concrete builder, set
    `param_cls` to `None` and pass `None` into this class's constructor.
    """

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, ConfigParams, None],
        build_config: Optional[Dict] = None,
    ):
        super().__init__()
        self.params = make_parameter_frame(params, self.param_cls)
        self.build_config = build_config if build_config is not None else {}
        self.name = self.build_config.get(
            "name", _remove_suffix(self.__class__.__name__, "Builder")
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
