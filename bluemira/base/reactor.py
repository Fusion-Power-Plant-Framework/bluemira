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

from typing import List, Optional, Type
from warnings import warn

from bluemira.base.builder import ComponentManager
from bluemira.base.components import Component
from bluemira.base.error import ReactorError
from bluemira.builders.tools import circular_pattern_component
from bluemira.display.displayer import ComponentDisplayer

_PLOT_DIMS = ["xy", "xz", "xyz"]


class Reactor:
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
    name: str
        The name of the reactor. This will be the label for the top
        level :obj:`bluemira.base.components.Component` in the reactor
        tree.

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

    def component(
        self,
        with_components: Optional[List[ComponentManager]] = None,
    ) -> Component:
        """Return the component tree."""
        return self._build_component_tree(with_components)

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

    def show_cad(
        self,
        *dims,
        with_components: Optional[List[ComponentManager]] = None,
        with_n_sectors: Optional[int] = None,
        **kwargs,
    ):
        """
        Show the CAD build of the reactor.

        Parameters
        ----------
        dim: str
            The dimension of the reactor to show, typically one of
            'xz', 'xy', or 'xyz'. (default: 'xyz')
        """
        if with_n_sectors is None:
            with_n_sectors = self.n_sectors

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
            if dim not in _PLOT_DIMS:
                raise ReactorError(
                    f"Invalid plotting dimension '{dim}'."
                    f"Must be one of {str(_PLOT_DIMS)}"
                )

        comp = self.component(with_components)

        # A copy of the component tree must be made
        # as filtering would mutate the underlying component tree,
        # that exists statically in memory.
        # self.component (above) only creates a new root node for this reactor,
        # not a new component tree.
        comp_copy = comp.copy()
        comp_copy.filter_components(dims_to_show)

        if "xyz" in dims_to_show:

            def create_patterned_component(c: Component):
                print(f" --> {c.name}")
                return Component(
                    c.name,
                    children=circular_pattern_component(
                        c,
                        with_n_sectors,
                        degree=(360 / self.n_sectors) * with_n_sectors,
                    ),
                )

            xyzs = comp_copy.get_component(
                "xyz",
                first=False,
            )
            print("Building xyz CAD")
            if isinstance(xyzs, Component):
                patterned_children = [
                    create_patterned_component(c) for c in xyzs.children
                ]
                xyzs.children = patterned_children
            else:
                for i, xyz in enumerate(xyzs):
                    print(f"{i+1}/{len(xyzs)}:")
                    patterned_children = [
                        create_patterned_component(c) for c in xyz.children
                    ]
                    xyz.children = patterned_children

        ComponentDisplayer().show_cad(comp_copy, **kwargs)
