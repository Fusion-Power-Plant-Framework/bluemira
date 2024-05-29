# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Interfaces for builder classes.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, TypeAlias, Union

from bluemira.base.components import Component
from bluemira.base.parameter_frame import ParameterFrame, make_parameter_frame
from bluemira.base.tools import _timing
from bluemira.utilities.plot_tools import set_component_view

if TYPE_CHECKING:
    from bluemira.base.components import ComponentT
    from bluemira.base.parameter_frame.typing import ParameterFrameLike

BuildConfig: TypeAlias = dict[str, Union[int, float, str, "BuildConfig"]]
"""
Type alias for representing nested build configuration information.
"""


class Builder(abc.ABC):
    """
    Base class for component builders.

    Parameters
    ----------
    params:
        The parameters required by the builder.
    build_config:
        The build configuration for the builder.
    verbose:
        control how much logging the designer will output

    Notes
    -----
    If there are no parameters associated with a concrete builder, set
    `param_cls` to `None` and pass `None` into this class's constructor.
    """

    def __init__(
        self,
        params: ParameterFrameLike,
        build_config: dict | None = None,
        *,
        verbose=True,
    ):
        super().__init__()
        self.params = make_parameter_frame(params, self.param_cls)
        self.build_config = build_config if build_config is not None else {}
        self.name = self.build_config.get(
            "name", self.__class__.__name__.removesuffix("Builder")
        )
        self.build = _timing(
            self.build, "Built in", f"Building {self.name}", debug_info_str=not verbose
        )

    @abc.abstractproperty
    def param_cls(self) -> type[ParameterFrame] | None:
        """The class to hold this Builders's parameters."""

    @abc.abstractmethod
    def build(self) -> Component:
        """Build the component."""

    def component_tree(
        self,
        xz: list[ComponentT] | None,
        xy: list[ComponentT] | None,
        xyz: list[ComponentT] | None,
    ) -> Component:
        """
        Adds views of components to an overall component tree.

        Parameters
        ----------
        xz:
            xz view of component
        xy:
            xy view of component
        xyz:
            xyz view of component
        """
        component = Component(self.name)
        component.add_child(Component("xz", children=xz))
        component.add_child(Component("xy", children=xy))
        component.add_child(Component("xyz", children=xyz))

        set_component_view(component.get_component("xz"), "xz")
        set_component_view(component.get_component("xy"), "xy")

        return component
