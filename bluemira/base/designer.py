# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Interface for designer classes.
"""

from __future__ import annotations

import abc
from typing import TYPE_CHECKING, Generic, TypeVar

from bluemira.base.parameter_frame import make_parameter_frame
from bluemira.base.tools import _timing

if TYPE_CHECKING:
    from collections.abc import Callable

    from bluemira.base.builder import BuildConfig
    from bluemira.base.parameter_frame import ParameterFrameLike, ParameterFrameT

_DesignerReturnT = TypeVar("_DesignerReturnT")


class Designer(abc.ABC, Generic[_DesignerReturnT]):
    """
    Base class for 'Designers' that solver design problems as part of
    building a reactor component.

    Parameters
    ----------
    params:
        The parameters required by the designer.
    build_config:
        The build configuration options for the designer.
    verbose:
        control how much logging the designer will output

    Notes
    -----
    If there are no parameters associated with a concrete builder, set
    `param_cls` to `None` and pass `None` into this class's constructor.
    """

    KEY_RUN_MODE = "run_mode"

    def __init__(
        self,
        params: ParameterFrameLike,
        build_config: BuildConfig | None = None,
        *,
        verbose: bool = True,
    ):
        self.params = make_parameter_frame(params, self.param_cls)
        self.build_config = build_config if build_config is not None else {}
        self._verbose = verbose

    def execute(self) -> _DesignerReturnT:
        """
        Execute the designer with the run mode specified by the build config.

        By default the run mode is 'run'.
        """
        return _timing(
            self._get_run_func(self.run_mode),
            "Executed in",
            f"Executing {type(self).__name__}",
            debug_info_str=not self._verbose,
        )()

    @abc.abstractmethod
    def run(self) -> _DesignerReturnT:
        """Run the design problem."""

    def mock(self) -> _DesignerReturnT:
        """
        Return a mock of a design.

        Optionally implemented.
        """
        raise NotImplementedError

    def read(self) -> _DesignerReturnT:
        """
        Read a design from a file.

        The file path should be specified in the build config.

        Optionally implemented.
        """
        raise NotImplementedError

    @abc.abstractproperty
    def param_cls(self) -> type[ParameterFrameT]:
        """The ParameterFrame class defining this designer's parameters."""
        ...

    @property
    def run_mode(self) -> str:
        """Get the run mode of this designer."""
        return self.build_config.get(self.KEY_RUN_MODE, "run")

    def _get_run_func(self, mode: str) -> Callable:
        """Retrieve the function corresponding to the given run mode.

        Raises
        ------
        ValueError
            Run mode doesnt exist
        """
        try:
            return getattr(self, mode)
        except AttributeError:
            raise ValueError(
                f"{type(self).__name__} has no run mode '{mode}'."
            ) from None


def run_designer(
    designer_cls: type[Designer[_DesignerReturnT]],
    params: ParameterFrameLike,
    build_config: dict,
    **kwargs,
) -> _DesignerReturnT:
    """Make and run a designer, returning the result."""
    designer = designer_cls(params, build_config, **kwargs)
    return designer.execute()
