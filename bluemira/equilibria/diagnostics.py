# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Diagnostic options for use in the equilibria module.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, Flag, auto
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import try_get_bluemira_path
from bluemira.equilibria.constants import DPI_GIF, PLT_PAUSE
from bluemira.utilities.plot_tools import make_gif, save_figure

if TYPE_CHECKING:
    from os import PathLike

    from bluemira.equilibria.equilibrium import Equilibrium
    from bluemira.equilibria.solve import ConvergenceCriterion


class GridPlotType(Flag):
    """
    Whether to plot grid lines, grid edge or both.
    """

    GRID = auto()
    EDGE = auto()
    GRID_WITH_EDGE = auto()


class EqSubplots(Enum):
    """
    Type of plot axes.
    Determines number of subplots, axes label, etc.
    """

    XZ = auto()
    """Plot x vs z"""
    VS_PSI_NORM = auto()
    """Plot vs normalised psi."""
    XZ_COMPONENT_PSI = auto()
    """PLot x vs z for different psi components."""
    VS_PSI_NORM_STACK = auto()
    """Plot parameters (numbers of which can vary) against the normalised psi."""
    VS_X = auto()
    """Plot parameters (numbers of which can vary) against x."""


class PsiPlotType(Flag):
    """
    For use with psi comparison plotter.
    """

    PSI = auto()
    """Plot equilibrium psi."""
    PSI_DIFF = auto()
    """
    Plot the difference between a reference equilibrium psi
    and the equilibrium psi.
    """
    PSI_ABS_DIFF = auto()
    """
    Plot the absolute difference between a reference equilibrium psi
    and the equilibrium psi.
    """
    PSI_REL_DIFF = auto()
    """
    Plot the relative difference between a reference equilibrium psi
    and the equilibrium psi.
    """
    DIFF = PSI_DIFF | PSI_ABS_DIFF | PSI_REL_DIFF


class DivLegsToPlot(Flag):
    """Which divertor legs to create plots for."""

    UP = auto()
    """Upper pair of legs."""
    LW = auto()
    """Lower pair of legs."""
    ALL = auto()
    """All available legs."""
    PAIR = UP | LW


class EqPlotMask(Flag):
    """
    For LCFS masking in plots.
    Block the area within or outside of the reference LCFS.
    """

    NONE = auto()
    "No mask."
    IN_LCFS = auto()
    """Mask out values inside chosen LCFS."""
    OUT_LCFS = auto()
    """Mask out values outside chosen LCFS."""
    IN_REF_LCFS = auto()
    """Mask out values inside reference LCFS."""
    OUT_REF_LCFS = auto()
    """Mask out values outside reference LCFS."""
    IN_COMBO_LCFS = auto()
    """Mask out values inside chosen and reference LCFS."""
    OUT_COMBO_LCFS = auto()
    """Mask out values outside chosen and reference LCFS."""
    DIV_AREA = (
        auto()
    )  # NOTE: add new mask types, this currently raises a not implemented error
    """Mask out the values outside divertor area."""
    POLYGON = (
        auto()
    )  # NOTE: add new mask types, this currently raises a not implemented error
    """Mask out the values outside a chosen polygon area."""
    INPUT = IN_LCFS | OUT_LCFS
    REF = IN_REF_LCFS | OUT_REF_LCFS
    COMBO = IN_COMBO_LCFS | OUT_COMBO_LCFS
    LCFS = (
        IN_REF_LCFS
        | OUT_REF_LCFS
        | IN_LCFS
        | OUT_LCFS
        | IN_COMBO_LCFS
        | OUT_COMBO_LCFS
        | NONE
    )
    IN = IN_LCFS | IN_REF_LCFS | IN_COMBO_LCFS
    OUT = OUT_LCFS | OUT_REF_LCFS | OUT_COMBO_LCFS | DIV_AREA | POLYGON


class InterpGrid(Enum):
    """Specify how the interpolated grid is sized."""

    OVERLAP = auto()
    """
    Make a new grid for interpolation using the
    overlapping areas of the old grids.
    """
    BOTH = auto()
    """
    Make a new grid for interpolation using an area
    which includes both of the old grids.
    """


class CSData(Enum):
    """
    For the coilset comparison tables.
    Value to be compared: current, x-position, z-position,
    field, and force.
    """

    CURRENT = "I [MA]"
    XLOC = "x [m]"
    ZLOC = "z [m]"
    B = "B [T]"
    F = "F [GN]"


class FixedOrFree(Enum):
    """
    For use in select_eq - to create an appropriate
    Equilibrium or FixedPlasmaEquilibrium object.
    Fixed or free boundary equilibrium.
    """

    FIXED = auto()
    FREE = auto()


class EqBPlotParam(Flag):
    """
    The parameter to plot for an equilibria xz plot.
    """

    PSI = auto()
    """Poloidal Magnetic Flux"""
    BP = auto()
    """Poloidal Field"""
    BT = auto()
    """Toroidal Field"""
    FIELD = BP | BT


class FluxSurfaceType(Enum):
    """
    For flux surface comparison plotting.
    Compare LCFSs, separatricies or flux surfaces with a given normalised psi.
    """

    LCFS = auto()
    SEPARATRIX = auto()
    PSI_NORM = auto()


@dataclass
class EqDiagnosticOptions:
    """Diagnostic plotting options for Equilibrium."""

    psi_diff: PsiPlotType = PsiPlotType.PSI
    split_psi_plots: EqSubplots = EqSubplots.XZ
    plot_mask: EqPlotMask = EqPlotMask.NONE
    interpolation_grid: InterpGrid = InterpGrid.OVERLAP
    plot_name: str = "default_0"
    folder: str | PathLike | None = None
    save: bool = False

    def __post_init__(self):
        """Post init folder definition"""
        self.folder = Path.cwd() if self.folder is None else Path(self.folder)


class PicardDiagnostic(Flag):
    """Type of plot to view during optimisation."""

    EQ = auto()
    """Plot the equilibrium"""
    CONVERGENCE = auto()
    """Plot the convergence"""
    EQ_AND_CONVERGENCE = EQ | CONVERGENCE
    """Plot both equilibrium and convergence"""
    NO_PLOT = auto()


class _PicardDiagnosticDescriptor:
    """Descriptor for Picard diagnostics"""

    def __init__(self):
        self._default = PicardDiagnostic.NO_PLOT

    def __set_name__(self, _, name: str):
        """Set the attribute name from a dataclass"""
        self._name = "_" + name

    def __get__(self, obj: Any, _) -> PicardDiagnostic:
        """Get the Diagnostic type

        Returns
        -------
        :
            The diagnostic
        """
        if obj is None:
            return self._default

        return getattr(obj, self._name, self._default)

    def __set__(
        self,
        obj: Any,
        value: bool | str | int | list[str | int] | PicardDiagnostic,  # noqa: FBT001
    ):
        """
        Set the diagnostic
        """

        def _setup(value):
            if isinstance(value, str):
                return PicardDiagnostic[value.upper()]
            if isinstance(value, bool):
                return PicardDiagnostic.EQ if value else PicardDiagnostic.NO_PLOT
            return PicardDiagnostic(value)

        if isinstance(value, list):
            fv = list({_setup(v) for v in value})
            if PicardDiagnostic.NO_PLOT in fv:
                value = PicardDiagnostic.NO_PLOT
            else:
                value = fv[0]
                for v in fv[1:]:
                    value |= v
        else:
            value = _setup(value)

        setattr(obj, self._name, value)


@dataclass
class PicardDiagnosticOptions:
    """
    Diagnostic plotting options for the Picard Iterator

    plot:
        What type of plot to produce. None for no plotting.
    gif:
        Whether or not to make a GIF
    plot_name:
        GIF plot file base-name
    figure_folder:
        The path where figures will be saved. If the input value is None (e.g. default)
        then this will be reinterpreted as the path data/plots/equilibria under the
        bluemira root folder, if that path is available.
    """

    plot: PicardDiagnostic = _PicardDiagnosticDescriptor()
    gif: bool = False
    plot_name: str = "default_0"
    figure_folder: str | PathLike | None = None

    # TODO @oliverfunk: Use of genereated_data folder to be reviewed.
    # 3806
    def __post_init__(self):
        """Post init folder definition"""
        if self.figure_folder is None:
            figure_folder = try_get_bluemira_path(
                "", subfolder="generated_data", allow_missing=not self.gif
            )
        self.figure_folder = figure_folder

        if self.plot is PicardDiagnostic.NO_PLOT:

            def noop(*args, **kwargs):  # noqa: ARG001
                return

            self.update_figure = self.make_gif = self.finalise_plots = noop
        else:
            self.f, ax = plt.subplots(
                ncols=2 if self.plot is PicardDiagnostic.EQ_AND_CONVERGENCE else 1
            )
            self.ax = ax if isinstance(ax, np.ndarray) else [ax]
            self.ax_eq = self.ax[0]
            self.ax_con = self.ax[1] if len(self.ax) == 2 else self.ax[0]  # noqa: PLR2004

    @staticmethod
    def finalise_plots():
        """Finalise plotting showing"""
        plt.show(block=False)

    def make_gif(self):
        """Make gif of iterator plot"""
        if self.gif:
            make_gif(self.figure_folder, self.plot_name)

    def update_figure(self, eq: Equilibrium, convergence: ConvergenceCriterion, i: int):
        """
        Updates the figure if plotting is used
        """
        if PicardDiagnostic.EQ in self.plot:
            self.ax_eq.clear()
            eq.plot(ax=self.ax_eq)
        if PicardDiagnostic.CONVERGENCE in self.plot:
            self.ax_con.clear()
            convergence.plot(ax=self.ax_con)
        plt.pause(PLT_PAUSE)
        save_figure(
            self.f,
            f"{self.plot_name}{i}",
            save=self.gif,
            folder=self.figure_folder,
            dpi=DPI_GIF,
        )
