# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Base class for a bluemira reactor."""

from __future__ import annotations

import abc
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, get_args, get_type_hints

from rich.progress import track

from bluemira.base.components import (
    Component,
    PhysicalComponent,
)
from bluemira.base.error import ComponentError
from bluemira.base.tools import (
    CADConstructionType,
    build_comp_manager_save_xyz_cad_tree,
    build_comp_manager_show_cad_tree,
    copy_and_filter_component,
    plot_component_dim,
    save_components_cad,
    show_components_cad,
)
from bluemira.display.displayer import ComponentDisplayer
from bluemira.display.plotter import ComponentPlotter
from bluemira.materials.material import Material, Void

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable
    from os import PathLike

    import bluemira.codes._freecadapi as cadapi
    from bluemira.base.components import ComponentT


DIM_2D = Literal["xy", "xz"]
DIM_3D = Literal["xyz"]

_CAD_DIMS_T = DIM_2D | DIM_3D


class BaseManager(abc.ABC):
    """
    A base wrapper around a component tree or component trees.

    The purpose of the classes deriving from this is to abstract away
    the structure of the component tree and provide access to a set of
    its features. This way a reactor build procedure can be completely
    agnostic of the structure of component trees.

    """

    @abc.abstractmethod
    def component(self) -> ComponentT:
        """
        Return the component tree wrapped by this manager.
        """

    @abc.abstractmethod
    def save_cad(
        self,
        dim: DIM_3D | DIM_2D,
        components: ComponentT | Iterable[ComponentT],
        filename: str,
        cad_format: str | cadapi.CADFileType = "stp",
        **kwargs,
    ):
        """
        Save the CAD build of the component.

        Parameters
        ----------
        components:
            components to save
        filename:
            the filename to save
        cad_format:
            CAD file format
        """

    @abc.abstractmethod
    def show_cad(
        self,
        dim: DIM_3D | DIM_2D,
        component_filter: Callable[[ComponentT], bool] | None,
        **kwargs,
    ):
        """
        Show the CAD build of the component.

        Parameters
        ----------
        *dims:
            The dimension of the reactor to show, typically one of
            'xz', 'xy', or 'xyz'. (default: 'xyz')
        component_filter:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        """

    @abc.abstractmethod
    def plot(
        self,
        dim: DIM_2D,
        component_filter: Callable[[ComponentT], bool] | None,
    ):
        """
        Plot the component.

        Parameters
        ----------
        *dims:
            The dimension(s) of the reactor to show, 'xz' and/or 'xy'.
            (default: 'xz')
        component_filter:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        """

    def tree(self) -> str:
        """
        Get the component tree.

        Returns
        -------
        :
            The component tree as a string.
        """
        return self.component().tree()

    @staticmethod
    def _validate_cad_dim(dim: DIM_3D | DIM_2D) -> None:
        """
        Validate showable CAD dimensions.

        Raises
        ------
        ComponentError
            Unknown plot dimension
        """
        # give dims_to_show a default value
        cad_dims = get_args(DIM_2D) + get_args(DIM_3D)
        if dim not in cad_dims:
            raise ComponentError(
                f"Invalid plotting dimension '{dim}'. Must be one of {cad_dims!s}"
            )

    @staticmethod
    def _validate_plot_dims(dim: DIM_2D) -> None:
        """
        Validate showable plot dimensions.

        Raises
        ------
        ComponentError
            Unknown plot dimension
        """
        plot_dims = get_args(DIM_2D)
        if dim not in plot_dims:
            raise ComponentError(
                f"Invalid plotting dimension '{dim}'. Must be one of {plot_dims!s}"
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

    __slots__ = ("keep_material", "reject_material")

    def __init__(
        self,
        keep_material: type[Material] | tuple[type[Material]] | None = None,
        reject_material: type[Material] | tuple[type[Material]] | None = Void,
    ):
        super().__setattr__("keep_material", keep_material)
        super().__setattr__("reject_material", reject_material)

    def __call__(self, node: ComponentT) -> bool:
        """Filter node based on material include and exclude rules.

        Parameters
        ----------
        node:
            The node to filter.

        Returns
        -------
        :
            True if the node should be kept, False otherwise.
        """
        if hasattr(node, "material"):
            return self._apply_filters(node.material)
        return True

    def __setattr__(self, name: str, value: Any):
        """
        Override setattr to force immutability

        This method makes the class nearly immutable as no new attributes
        can be modified or added by standard methods.

        See #2236 discussion_r1191246003 for further details

        Raises
        ------
        AttributeError
            FilterMaterial is immutable
        """
        raise AttributeError(f"{type(self).__name__} is immutable")

    def _apply_filters(self, material: Material | tuple[Material]) -> bool:
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

    def __init__(self, component: ComponentT) -> None:
        self._component = component

    def cad_construction_type(self) -> CADConstructionType | PhysicalComponent:  # noqa: PLR6301
        """
        Return the construction type of the component tree wrapped by this manager.

        Returns
        -------
        :
            The construction type of the component managed by this.
        """
        return CADConstructionType.PATTERN

    def component(self) -> ComponentT:
        """
        Return the component tree wrapped by this manager.

        Returns
        -------
        :
            The underlying component, with all descendants.
        """
        return self._component

    def save_cad(
        self,
        dim: DIM_3D | DIM_2D = "xyz",
        component_filter: Callable[[ComponentT], bool] | None = FilterMaterial(),
        filename: str | None = None,
        cad_format: str | cadapi.CADFileType = "stp",
        directory: str | PathLike = "",
        **kwargs,
    ):
        """
        Save the CAD build of the component.

        Parameters
        ----------
        *dims:
            The dimension of the reactor to show, typically one of
            'xz', 'xy', or 'xyz'. (default: 'xyz')
        component_filter:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        filename:
            the filename to save, will default to the component name
        cad_format:
            CAD file format
        directory:
            Directory to save into, defaults to the current directory
        kwargs:
            passed to the :func:`bluemira.geometry.tools.save_cad` function
        """
        comp = copy_and_filter_component(self.component(), dim, component_filter)
        if filename is None:
            filename = comp.name
        save_components_cad(
            comp,
            filename=Path(directory, filename).as_posix(),
            cad_format=cad_format,
            **kwargs,
        )

    def show_cad(
        self,
        dim: DIM_3D | DIM_2D = "xyz",
        component_filter: Callable[[ComponentT], bool] | None = FilterMaterial(),
        **kwargs,
    ):
        """
        Show the CAD build of the component.

        Parameters
        ----------
        *dims:
            The dimension of the reactor to show, typically one of
            'xz', 'xy', or 'xyz'. (default: 'xyz')
        component_filter:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        kwargs:
            passed to the `~bluemira.display.displayer.show_cad` function
        """
        ComponentDisplayer().show_cad(
            copy_and_filter_component(self.component(), dim, component_filter),
            **kwargs,
        )

    def plot(
        self,
        dim: DIM_2D = "xz",
        component_filter: Callable[[ComponentT], bool] | None = FilterMaterial(),
    ):
        """
        Plot the component.

        Parameters
        ----------
        *dims:
            The dimension(s) of the reactor to show, 'xz' and/or 'xy'.
            (default: 'xz')
        component_filter:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        """
        ComponentPlotter(view=dim).plot_2d(
            copy_and_filter_component(self.component(), dim, component_filter)
        )


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

        reactor = MyReactor("My Reactor", n_sectors=1)
        reactor.plasma = build_plasma()
        reactor.tf_coils = build_tf_coils()
        reactor.show_cad()

    """

    def __init__(self, name: str, n_sectors: int):
        self.name = name
        self.n_sectors = n_sectors
        self.start_time = time.perf_counter()

    def component(self) -> Component:
        """Return the component tree.

        Returns
        -------
        :
            The reactor component tree.
        """
        return self._build_component_tree(None)

    def time_since_init(self) -> float:
        """
        Get time since initialisation.

        Returns
        -------
        :
            The time since initialisation.
        """
        return time.perf_counter() - self.start_time

    def _component_managers(
        self,
        with_components: list[ComponentManager] | None = None,
    ) -> list[ComponentManager]:
        """
        Get the component managers for the reactor.

        Parameters
        ----------
        with_components:
            The components to include. Defaults to None, which means
            include all components.

        Returns
        -------
        :
            A list of initialised component managers.

        Raises
        ------
        ComponentError
            Initialising Reactor directly
        """
        if not hasattr(self, "__annotations__"):
            raise ComponentError(
                "This reactor is ill-defined. "
                "Make sure you have sub-classed Reactor and "
                "correctly defined component managers for it. "
                "Please see the examples for a template Reactor."
            )
        return [
            getattr(self, comp_name)
            for comp_name, comp_type in get_type_hints(type(self)).items()
            if issubclass(comp_type, ComponentManager)
            and getattr(self, comp_name, None) is not None
            and (with_components is None or getattr(self, comp_name) in with_components)
        ]

    def _build_component_tree(
        self,
        dim: str | None,
        with_components: list[ComponentManager] | None = None,
        component_filter: Callable[[ComponentT], bool] | None = None,
        *,
        n_sectors: int | None = None,
        for_save: bool = False,
    ) -> Component:
        reactor_component = Component(self.name)
        for comp_manager in track(self._component_managers(with_components)):
            # if dim is None, return the raw, underlying component tree
            if dim:
                n_secs = n_sectors or self.n_sectors
                sec_degrees = int((360 / self.n_sectors) * n_secs)
                if dim == "xyz" and for_save:
                    comp = build_comp_manager_save_xyz_cad_tree(
                        comp_manager, component_filter, n_secs, sec_degrees
                    )
                else:
                    # all other cases, can use the show CAD tree
                    comp = build_comp_manager_show_cad_tree(
                        comp_manager, dim, component_filter, n_secs, sec_degrees
                    )
            reactor_component.add_child(comp)
        return reactor_component

    def save_cad(
        self,
        dim: DIM_3D | DIM_2D = "xyz",
        with_components: list[ComponentManager] | None = None,
        n_sectors: int | None = None,
        component_filter: Callable[[ComponentT], bool] | None = FilterMaterial(),
        filename: str | None = None,
        cad_format: str | cadapi.CADFileType = "stp",
        directory: str | PathLike = "",
        **kwargs,
    ):
        """
        Save the CAD build of the reactor.

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
        component_filter:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        filename:
            the filename to save, will default to the component name
        cad_format:
            CAD file format
        directory:
            Directory to save into, defaults to the current directory
        kwargs:
            passed to the :func:`bluemira.geometry.tools.save_cad` function
        """
        if filename is None:
            filename = self.name
        save_components_cad(
            self._build_component_tree(
                dim,
                with_components,
                component_filter,
                n_sectors=n_sectors,
                for_save=True,
            ),
            Path(directory, filename).as_posix(),
            cad_format,
            **kwargs,
        )

    def show_cad(
        self,
        dim: DIM_3D | DIM_2D = "xyz",
        with_components: list[ComponentManager] | None = None,
        n_sectors: int | None = None,
        component_filter: Callable[[ComponentT], bool] | None = FilterMaterial(),
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
        component_filter:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        kwargs:
            passed to the `~bluemira.display.displayer.show_cad` function
        """
        show_components_cad(
            self._build_component_tree(
                dim,
                with_components,
                component_filter,
                n_sectors=n_sectors,
                for_save=False,
            ),
            **kwargs,
        )

    def plot(
        self,
        dim: DIM_2D = "xz",
        with_components: list[ComponentManager] | None = None,
        component_filter: Callable[[ComponentT], bool] | None = FilterMaterial(),
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
        component_filter:
            A callable to filter Components from the Component tree,
            returning True keeps the node False removes it
        """
        plot_component_dim(
            dim,
            self._build_component_tree(
                dim, with_components, component_filter, n_sectors=1, for_save=False
            ),
        )
