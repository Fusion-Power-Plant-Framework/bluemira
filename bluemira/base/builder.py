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
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame import ParameterFrame, make_parameter_frame
from bluemira.base.tools import _timing
from bluemira.materials.cache import get_cached_material
from bluemira.materials.error import MaterialsError
from bluemira.utilities.plot_tools import set_component_view

if TYPE_CHECKING:
    from bluemira.base.components import ComponentT
    from bluemira.base.parameter_frame.typed import ParameterFrameLike
    from bluemira.materials.material import Material

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

    def get_material(self, component_name: str | None = None) -> Material | None:
        """
        Get the material for a component from the build config.

        This will lookup the component_name in the "material" section
        of the build config to get the material name.
        It will then use that to get the corresponding material from the
        material cache.

        If no component_name is given, it's assumed the material name is
        directly given by the "material" key in the build config.

        See `establish_material_cache` and `get_cached_material` for more
        information on how the material cache is used.

        If no material is found or there is no "material" key in the build_config,
        a warning is raised and None returned.

        Parameters
        ----------
        component_name:
            The name of the component.

        Returns
        -------
        :
            The material for the component.

        Raises
        ------
        MaterialsError
            If the material build config is not a string when no component name is given.
        """
        mats = self.build_config.get("material")
        if not mats:
            bluemira_warn("No 'material' found in build_config, returning None")
            return None

        if not component_name:
            if not isinstance(mats, str):
                raise MaterialsError(
                    "'material' build_config must be a string when "
                    "no component name is given."
                )
            mat_name = mats
        else:
            if not isinstance(mats, dict):
                raise MaterialsError(
                    "'material' build_config must be a dictionary when "
                    "a component name is given."
                )
            mat_name: str = mats.get(component_name)

        mat = get_cached_material(mat_name)
        if mat is None:
            bluemira_warn(
                f"No corresponding material found for {component_name} in {mats}"
            )

        return mat

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

        Returns
        -------
        component:
            The component tree
        """
        component = Component(self.name)
        component.add_child(Component("xz", children=xz))
        component.add_child(Component("xy", children=xy))
        component.add_child(Component("xyz", children=xyz))

        set_component_view(component.get_component("xz"), "xz")
        set_component_view(component.get_component("xy"), "xy")

        return component
