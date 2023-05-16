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
"""Base class for a Bluemira reactor."""

import abc
import time
from typing import Any, Callable, List, Optional, Tuple, Type, Union
from warnings import warn

import anytree
from rich.progress import track

from bluemira.base.components import Component
from bluemira.base.error import ComponentError
from bluemira.base.look_and_feel import bluemira_print
from bluemira.builders.tools import circular_pattern_component
from bluemira.display.displayer import ComponentDisplayer
from bluemira.display.plotter import ComponentPlotter
from bluemira.materials.material import SerialisedMaterial, Void

_PLOT_DIMS = ["xy", "xz"]
_CAD_DIMS = ["xy", "xz", "xyz"]


class BaseManager(abc.ABC):
    """
    A base wrapper around a component tree or component trees.

    The purpose of the classes deriving from this is to abstract away
    the structure of the component tree and provide access to a set of
    its features. This way a reactor build procedure can be completely
    agnostic of the structure of component trees.

    """

    @abc.abstractmethod
    def component(self) -> Component:
        """
        Return the component tree wrapped by this manager.
        """

    @abc.abstractmethod
    def show_cad(
        self,
        *dims: str,
        filter_: Union[Callable[[Component], bool], None],
        **kwargs,
    ):
        """
        Show the CAD build of the component.

        Parameters
        ----------
        *dims:
            The dimension of the reactor to show, typically one of
            'xz', 'xy', or 'xyz'. (default: 'xyz')
        filter_:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        """

    @abc.abstractmethod
    def plot(self, *dims: str, filter_: Union[Callable[[Component], bool], None]):
        """
        Plot the component.

        Parameters
        ----------
        *dims:
            The dimension(s) of the reactor to show, 'xz' and/or 'xy'.
            (default: 'xz')
        filter_:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        """

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

    def _filter_tree(
        self,
        comp: Component,
        dims_to_show: Tuple[str, ...],
        filter_: Union[Callable[[Component], bool], None],
    ) -> Component:
        """
        Filter a component tree

        Notes
        -----
        A copy of the component tree must be made
        as filtering would mutate the ComponentMangers' underlying component trees
        """
        comp_copy = comp.copy()
        comp_copy.filter_components(dims_to_show, filter_)
        return comp_copy

    def _plot_dims(
        self,
        comp: Component,
        dims_to_show: Tuple[str, ...],
        filter_: Union[Callable[[Component], bool], None],
    ):
        for i, dim in enumerate(dims_to_show):
            ComponentPlotter(view=dim).plot_2d(
                self._filter_tree(comp, dims_to_show, filter_),
                show=i == len(dims_to_show) - 1,
            )


class FilterMaterial:
    """
    Filter nodes by material

    Parameters
    ----------
    keep_material:
       materials to include
    reject_material:
       materials to exclude

    """

    def __init__(
        self,
        keep_material: Union[
            Type[SerialisedMaterial], Tuple[Type[SerialisedMaterial]], None
        ] = None,
        reject_material: Union[
            Type[SerialisedMaterial], Tuple[Type[SerialisedMaterial]], None
        ] = Void,
    ):
        super().__setattr__("keep_material", keep_material)
        super().__setattr__("reject_material", reject_material)

    def __call__(self, node: anytree.Node) -> bool:
        """Filter node based on material include and exclude rules"""
        if not hasattr(node, "material"):
            return True
        return self._apply_filters(node.material)

    def __setattr__(self, name: str, value: Any):
        """
        Override setattr to force immutability

        This method makes the class nearly immutable as no new attributes
        can be modified or added by standard methods.

        See #2236 discussion_r1191246003 for further details
        """
        raise AttributeError(f"{type(self).__name__} is immutable")

    def _apply_filters(
        self, material: Union[SerialisedMaterial, Tuple[SerialisedMaterial]]
    ) -> bool:
        bool_store = True

        if self.keep_material is not None:
            bool_store = isinstance(material, self.keep_material)

        if self.reject_material is not None:
            bool_store = not isinstance(material, self.reject_material)

        return bool_store


class ComponentManager(BaseManager):
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
    component_tree:
        The component tree this manager should wrap.
    """

    def __init__(self, component_tree: Component) -> None:
        self._component = component_tree

    def component(self) -> Component:
        """
        Return the component tree wrapped by this manager.
        """
        return self._component

    def show_cad(
        self,
        *dims: str,
        filter_: Union[Callable[[Component], bool], None] = FilterMaterial(),
        **kwargs,
    ):
        """
        Show the CAD build of the component.

        Parameters
        ----------
        *dims:
            The dimension of the reactor to show, typically one of
            'xz', 'xy', or 'xyz'. (default: 'xyz')
        filter_:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        """
        ComponentDisplayer().show_cad(
            self._filter_tree(
                self.component(), self._validate_cad_dims(*dims, **kwargs), filter_
            ),
            **kwargs,
        )

    def plot(
        self,
        *dims: str,
        filter_: Union[Callable[[Component], bool], None] = FilterMaterial(),
    ):
        """
        Plot the component.

        Parameters
        ----------
        *dims:
            The dimension(s) of the reactor to show, 'xz' and/or 'xy'.
            (default: 'xz')
        filter_:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        """
        self._plot_dims(self.component(), self._validate_plot_dims(*dims), filter_)


class Reactor(BaseManager):
    """
    Base class for reactor definitions.

    Assign :obj:`bluemira.base.builder.ComponentManager` instances to
    fields defined on the reactor, and this class acts as a container
    to group those components' trees. It is also a place to define any
    methods to calculate/derive properties that require information
    about multiple reactor components.

    Components should be defined on the reactor as class properties
    annotated with a type (similar to a ``dataclass``). A type that
    subclasses ``ComponentManager`` must be given, or it will not be
    recognised as part of the reactor tree. Note that a declared
    component is not required to be set for the reactor to be valid.
    So it is possible to just add a reactor's plasma, but not its
    TF coils, for example.

    Parameters
    ----------
    name:
        The name of the reactor. This will be the label for the top
        level :obj:`bluemira.base.components.Component` in the reactor
        tree.
    n_sectors:
        Number of sectors in a reactor

    Example
    -------

    .. code-block:: python

        class MyReactor(Reactor):
            '''An example of how to declare a reactor structure.'''

            plasma: MyPlasma
            tf_coils: MyTfCoils

            def get_ripple(self):
                '''Calculate the ripple in the TF coils.'''

        reactor = MyReactor("My Reactor")
        reactor.plasma = build_plasma()
        reactor.tf_coils = build_tf_coils()
        reactor.show_cad()

    """

    def __init__(self, name: str, n_sectors: int):
        self.name = name
        self.n_sectors = n_sectors
        self.start_time = time.perf_counter()

    def component(
        self,
        with_components: Optional[List[ComponentManager]] = None,
    ) -> Component:
        """Return the component tree."""
        return self._build_component_tree(with_components)

    def time_since_init(self) -> float:
        """
        Get time since initialisation
        """
        return time.perf_counter() - self.start_time

    def _build_component_tree(
        self,
        with_components: Optional[List[ComponentManager]] = None,
    ) -> Component:
        """Build the component tree from this class's annotations."""
        component = Component(self.name)
        comp_type: Type
        for comp_name, comp_type in self.__annotations__.items():
            if not issubclass(comp_type, ComponentManager):
                continue
            try:
                component_manager = getattr(self, comp_name)
                if (
                    with_components is not None
                    and component_manager not in with_components
                ):
                    continue
            except AttributeError:
                # We don't mind if a reactor component is not set, it
                # just won't be part of the tree
                continue

            component.add_child(component_manager.component())
        return component

    def _construct_xyz_cad(
        self,
        reactor_component: Component,
        with_components: Optional[List[ComponentManager]] = None,
        n_sectors: int = 1,
    ):
        xyzs = reactor_component.get_component(
            "xyz",
            first=False,
        )
        xyzs = [xyzs] if isinstance(xyzs, Component) else xyzs

        comp_names = (
            "all"
            if not with_components
            else ", ".join([cm.component().name for cm in with_components])
        )
        bluemira_print(
            f"Constructing xyz CAD for display with {n_sectors} sectors and components: {comp_names}"
        )
        for xyz in track(xyzs):
            xyz.children = circular_pattern_component(
                list(xyz.children),
                n_sectors,
                degree=(360 / self.n_sectors) * n_sectors,
            )

    def show_cad(
        self,
        *dims: str,
        with_components: Optional[List[ComponentManager]] = None,
        n_sectors: Optional[int] = None,
        filter_: Union[Callable[[Component], bool], None] = FilterMaterial(),
        **kwargs,
    ):
        """
        Show the CAD build of the reactor.

        Parameters
        ----------
        *dims:
            The dimension of the reactor to show, typically one of
            'xz', 'xy', or 'xyz'. (default: 'xyz')
        with_components:
            The components to construct when displaying CAD for xyz.
            Defaults to None, which means show "all" components.
        n_sectors:
            The number of sectors to construct when displaying CAD for xyz
            Defaults to None, which means show "all" sectors.
        filter_:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        """
        dims_to_show = self._validate_cad_dims(*dims, **kwargs)

        # We filter because self.component (above) only creates
        # a new root node for this reactor, not a new component tree.
        comp_copy = self._filter_tree(
            self.component(with_components), dims_to_show, filter_
        )
        # if "xyz" is requested, construct the 3d cad
        # from each xyz component in the tree,
        # as it's assumed that the cad is only built for 1 sector
        # and is sector symmetric, therefore can be patterned
        if "xyz" in dims_to_show:
            self._construct_xyz_cad(
                comp_copy,
                with_components,
                self.n_sectors if n_sectors is None else n_sectors,
            )

        ComponentDisplayer().show_cad(comp_copy, **kwargs)

    def plot(
        self,
        *dims: str,
        with_components: Optional[List[ComponentManager]] = None,
        filter_: Union[Callable[[Component], bool], None] = FilterMaterial(),
    ):
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
        filter_:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        """
        self._plot_dims(
            self.component(with_components), self._validate_plot_dims(*dims), filter_
        )
