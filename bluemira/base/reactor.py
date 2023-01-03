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

from typing import Type

from bluemira.base.builder import ComponentManager
from bluemira.base.components import Component
from bluemira.base.error import ReactorError
from bluemira.display.displayer import ComponentDisplayer

_PLOT_DIMS = ["xyz", "xz", "xyz"]


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

    def __init__(self, name: str):
        self.name = name

    def component(self) -> Component:
        """Return the component tree."""
        return self._build_component_tree()

    def _build_component_tree(self) -> Component:
        """Build the component tree from this class's annotations."""
        component = Component(self.name)
        comp_type: Type
        for comp_name, comp_type in self.__annotations__.items():
            if not issubclass(comp_type, ComponentManager):
                continue
            try:
                component_manager = getattr(self, comp_name)
            except AttributeError:
                # We don't mind if a reactor component is not set, it
                # just won't be part of the tree
                continue

            component.add_child(component_manager.component())
        return component

    def show_cad(self, dim: str = "xyz", **kwargs):
        """
        Show the CAD build of the reactor.

        Parameters
        ----------
        dim: str
            The dimension of the reactor to show, typically one of
            'xz', 'xy', or 'xyz'. (default: 'xyz')
        """
        if dim not in _PLOT_DIMS:
            raise ReactorError(
                f"Invalid plotting dimension '{dim}'. "
                f"Must be one of {str(_PLOT_DIMS)[1:-1]}"
            )
        comp = self.component()
        ComponentDisplayer().show_cad(comp.get_component(dim, first=False), **kwargs)
