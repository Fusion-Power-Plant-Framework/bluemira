# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Equilibria and Equilibria optimisation analysis tools"""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.gridspec import GridSpec
from tabulate import tabulate

from bluemira.base.constants import CoilType, raw_uc
from bluemira.base.error import BluemiraError
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter_frame import tabulate_values_from_multiple_frames
from bluemira.equilibria.coils._tools import rename_coilset
from bluemira.equilibria.constants import BLUEMIRA_DEFAULT_COCOS
from bluemira.equilibria.diagnostics import (
    CSData,
    DivLegsToPlot,
    EqBPlotParam,
    EqDiagnosticOptions,
    FixedOrFree,
    FluxSurfaceType,
)
from bluemira.equilibria.equilibrium import Equilibrium, FixedPlasmaEquilibrium
from bluemira.equilibria.find import find_flux_surface_through_point, find_flux_surfs
from bluemira.equilibria.find_legs import (
    LegFlux,
    get_legs_length_and_angle,
)
from bluemira.equilibria.flux_surfaces import analyse_plasma_core
from bluemira.equilibria.plotting import (
    CorePlotter,
    EquilibriumComparisonPostOptPlotter,
    EquilibriumPlotter,
)
from bluemira.geometry.coordinates import Coordinates
from bluemira.utilities.tools import is_num

if TYPE_CHECKING:
    from bluemira.equilibria.coils import CoilSet
    from bluemira.equilibria.flux_surfaces import CoreResults
    from bluemira.equilibria.physics import EqSummary


def select_eq(
    file_path: str,
    fixed_or_free: FixedOrFree = FixedOrFree.FREE,
    dummy_coils: CoilSet | None = None,
    from_cocos: int = BLUEMIRA_DEFAULT_COCOS,
    qpsi_positive: bool = False,  # noqa: FBT001, FBT002
    control: CoilType | list[str] | None = None,
) -> FixedPlasmaEquilibrium | Equilibrium:
    """
    Return an Equilibrium or FixedPlasmaEquilibrium object given a particular file name.

    Cocos indices and qpsi sign are set to Bluemira Defaults unless specified.

    Parameters
    ----------
    file_path:
        file path to chosen equilibria
    fixed_or_free:
        whether or not it is for a fixed plasma boundary
    dummy_coils:
        coilset if none in equilibria file
        (a default coilset is used, and a warning message prints if none is provided)
    from_cocos:
        The COCOS index of the EQDSK file. Used when the determined
        COCOS is ambiguous. Will raise if given and not one of
        the determined COCOS indices.
    qpsi_positive:
        Whether or not qpsi is positive, required for identification
        when qpsi is not present in the file.
    control:
        Set which coils are control coils.
        Can be a coil type, a list of coil names,
        or None for all coils.
        N.B. many equilibrium objects already have coilsets with
        control coils set, and this input may not be necessary.

    Returns
    -------
    :
        Equilibrium or FixedPlasmaEquilibrium
    """
    if fixed_or_free == FixedOrFree.FREE:
        eq = Equilibrium.from_eqdsk(
            file_path,
            from_cocos=from_cocos,
            user_coils=dummy_coils,
            qpsi_positive=qpsi_positive,
        )
        eq.coilset = rename_coilset(eq.coilset)
        if isinstance(control, Iterable):
            eq.coilset.control = control
        if isinstance(control, CoilType):
            eq.coilset.control = eq.coilset.get_coiltype(control).name
        return eq
    return FixedPlasmaEquilibrium.from_eqdsk(
        file_path,
        from_cocos=from_cocos,
        qpsi_positive=qpsi_positive,
    )


def select_multi_eqs(
    equilibrium_input: str | Equilibrium | Iterable[str | Equilibrium],
    fixed_or_free: FixedOrFree = FixedOrFree.FREE,
    equilibrium_names: str | Iterable[str] | None = None,
    dummy_coils=None,
    from_cocos: int | Iterable[int] = BLUEMIRA_DEFAULT_COCOS,
    *,
    qpsi_positive: bool | Iterable[bool] = False,
    control_coils: CoilType | list[str] | None = None,
):
    """
    Put information needed to load eq into a dictionary.

    Cocos indices and qpsi sign are set to Bluemira Defaults unless specified.

    Parameters
    ----------
    equilibrium_input:
        List of chosen equilibria or file paths to chosen equilibria
    fixed_or_free:
        whether or not it is for a fixed plasma boundary
    equilibrium_names:
        Names of chosen equilibrium
    dummy_coils:
        coilset if none in equilibria file
        (a default coilset is used, and a warning message prints if none is provided)
    from_cocos:
        The COCOS index of the EQDSK file. Used when the determined
        COCOS is ambiguous. Will raise if given and not one of
        the determined COCOS indices.
    qpsi_positive:
        Whether or not qpsi is positive, required for identification
        when qpsi is not present in the file.
    control_coils:
        Set which coils are control coils.
        Can be a coil type, a list of coil names,
        or None for all coils.
        N.B. many equilibrium objects already have coilsets with
        control coils set, and this input may not be necessary.

    Returns
    -------
    equilibria_dict:
        Dictionary of load info

    Raises
    ------
    ValueError
        If list of input values is not the same length as the input equilibria
    """
    if not isinstance(equilibrium_input, Iterable):
        equilibrium_input = [equilibrium_input]
    if not isinstance(fixed_or_free, Iterable):
        fixed_or_free = [fixed_or_free] * len(equilibrium_input)
    elif len(fixed_or_free) != len(equilibrium_input):
        raise ValueError(
            "FixedOrFree list length not equal to the number of equilibria."
        )
    if not isinstance(dummy_coils, Iterable):
        dummy_coils = [dummy_coils] * len(equilibrium_input)
    elif len(dummy_coils) != len(equilibrium_input):
        raise ValueError(
            "dummy_coils list length not equal to the number of equilibria."
        )
    if is_num(from_cocos):
        from_cocos = np.ones(len(equilibrium_input)) * from_cocos
    if isinstance(qpsi_positive, bool):
        qpsi_positive = len(equilibrium_input) * [qpsi_positive]
    if equilibrium_names is None:
        equilibrium_names = [
            "Eq_" + str(x) for x in range(1, len(equilibrium_input) + 1)
        ]
    elif len(equilibrium_names) != len(equilibrium_input):
        raise ValueError(
            "equilibrium_names length not equal to the number of equilibria."
        )
    if not isinstance(control_coils, Iterable):
        control_coils = [control_coils] * len(equilibrium_input)

    if isinstance(equilibrium_input[0], Equilibrium | FixedPlasmaEquilibrium):
        equilibrium_paths = ["no path used"] * len(equilibrium_input)
    else:
        equilibrium_paths = equilibrium_input

    equilibria_dict = {}
    for name, file, eq_type, dc, fc, qp, cc in zip(
        equilibrium_names,
        equilibrium_paths,
        fixed_or_free,
        dummy_coils,
        from_cocos,
        qpsi_positive,
        control_coils,
        strict=False,
    ):
        equilibria_dict.update({
            name: {
                "filepath": file,
                "fixed_or_free": eq_type,
                "dummy_coils": dc,
                "from_cocos": fc,
                "qpsi_positive": qp,
                "control_coils": cc,
            }
        })

    if not isinstance(equilibrium_input[0], Path):
        for eq, equilibrium_dict in zip(
            equilibrium_input, equilibria_dict.values(), strict=False
        ):
            equilibrium_dict.update({"eq": eq})
            equilibrium_dict.update({"profiles": eq.profiles})
    return equilibria_dict


def get_eqs(equilibria_dict):
    """
    Use equilibrium load dictionaries to load equilibrium objects
    and add the objects to the dictionary.

    Parameters
    ----------
    equilibria_dict:
        Dictionary of load information for multiple equilibria.

    Returns
    -------
    equilibria_dict:
        Dictionary updated with equilibrium and profile objects.
    """
    for equilibrium_dict in equilibria_dict.values():
        eq = select_eq(
            equilibrium_dict["filepath"],
            fixed_or_free=equilibrium_dict["fixed_or_free"],
            dummy_coils=equilibrium_dict["dummy_coils"],
            from_cocos=equilibrium_dict["from_cocos"],
            qpsi_positive=equilibrium_dict["qpsi_positive"],
            control=equilibrium_dict["control_coils"],
        )
        equilibrium_dict.update({"eq": eq})
        equilibrium_dict.update({"profiles": eq.profiles})
    return equilibria_dict


def get_leg_flux_info(
    eq_list,
    n_layers=10,
    dx_off=0.1,
    plasma_facing_boundary_list=None,
    legs_to_plot=DivLegsToPlot.ALL,
) -> tuple[dict[float | np.ndarray], dict[float | np.ndarray], dict[float | np.ndarray]]:
    """
    Get the divertor leg length and grazing angle (used in divertor comparison plotting).

    Parameters
    ----------
    eq_list:
        List of Equilibrium objects to compare
    n_layers:
        Number of flux surfaces to extract for each leg
    dx_off:
        Total span in radial space of the flux surfaces to extract
    plasma_facing_boundary_list:
        List of associated plasma facing boundary coordinates
    legs_to_plot:
        Which legs to plot i.e, upper, lower or all.

    Returns
    -------
    lengths:
        Dictionary of connection length values for each named leg.
    angles:
        Dictionary of grazing values for each named leg.

    Raises
    ------
    BluemiraError
        If leg pair is chosen that does not exist in chosen eq.

    """
    lengths = []
    angles = []
    if plasma_facing_boundary_list is None:
        plasma_facing_boundary_list = [None] * len(eq_list)
    for eq, pfb in zip(eq_list, plasma_facing_boundary_list, strict=False):
        legflux = LegFlux(eq)
        legs = legflux.get_legs(n_layers=n_layers, dx_off=dx_off)
        lgth, ang = get_legs_length_and_angle(eq, legs, pfb)
        if legs_to_plot in DivLegsToPlot.PAIR:
            location = "lower" if legs_to_plot is DivLegsToPlot.LW else "upper"
            if f"{location}_inner" not in lgth:
                raise BluemiraError(
                    f"One of your chosen equilibria does not have {location}_inner legs."
                )
            if f"{location}_outer" not in lgth:
                raise BluemiraError(
                    f"One of your chosen equilibria does not have {location}_outer legs."
                )
            lgth = {k: lgth[k] for k in (f"{location}_inner", f"{location}_outer")}
            ang = {k: ang[k] for k in (f"{location}_inner", f"{location}_outer")}
        else:
            # sort alphabetically
            lgth = {k: lgth[k] for k in sorted(lgth)}
            ang = {k: ang[k] for k in sorted(ang)}
        lengths.append(lgth)
        angles.append(ang)
    return lengths, angles


def get_target_flux(eq, target, target_coords, n_layers, vertical=False):  # noqa: FBT002
    """
    Get a selection flux surfaces that cross the divertor target.

    eq:
        chosen Equilibrium
    target:
        Location of divertor target, lower_outer, lower_inner, upper_outer or upper_inner
    target_coords:
        Coordinates of divertor target
    n_layers:
        Number of flux surfaces to extract for each leg
    vertical:
        Set to True if using a vertical target.
        This boolean determines whether to get flux surfaces across
        the target x-range (horizontal) or z-range (vertical).

    Raises
    ------
    BluemiraError
        If the target is set to vertical or horizontal incorrectly.

    Returns
    -------
    fs_list:
        List of flux surface coordinates.

    """
    if eq._o_points is None:
        _, _ = eq.get_OX_points()

    if not vertical:
        x = np.min(target_coords.x)
        z = target_coords.z[np.argmin(target_coords.x)]
        target_size = np.abs(np.max(target_coords.x) - np.min(target_coords.x))
        if target_size == 0:
            raise BluemiraError(
                "No x-range found for target coords,"
                " perhaps you are using a vertical target (set vertical=True)."
            )
        target_offsets = np.linspace(0, target_size, n_layers)[1:]
        dx = x + target_offsets
        dz = np.full(n_layers, z)
    else:
        x = target_coords.x[np.argmin(target_coords.z)]
        z = np.min(target_coords.z)
        target_size = np.abs(np.max(target_coords.z) - np.min(target_coords.z))
        if target_size == 0:
            raise BluemiraError(
                "No z-range found for target coords,"
                " perhaps you are using a horizontal target (set vertical=False)."
            )
        target_offsets = np.linspace(0, target_size, n_layers)[1:]
        dx = np.full(n_layers, x)
        dz = z + target_offsets

    fs_list = []
    for x, z in zip(dx, dz, strict=False):
        fs_x, fs_z = find_flux_surface_through_point(
            eq.x,
            eq.z,
            eq.psi(),
            x,
            z,
            eq.psi(x, z),
        )
        # Only need to plot from midpoint
        select_idx = (fs_z >= 0) if target.find("lower") == -1 else (fs_z <= 0)
        if all(fs_z >= 0) or all(fs_z <= 0):
            select_idx_x = (
                (fs_x <= eq._x_points[0].x)
                if target.find("inner") != -1
                else (fs_x >= eq._x_points[0].x)
            )
            select_idx &= select_idx_x
        fs_list.append(Coordinates({"x": fs_x[select_idx], "z": fs_z[select_idx]}))

    return fs_list


class EqAnalysis:
    """
    Equilibria analysis toolbox for selected Equilibrium.

    Can  compare selected equilibrium to a reference equilibrium.
    The input and reference equilibria be different types (i.e, fixed or free),
    and have different grid sizes and grid resolutions.

    Parameters
    ----------
    diag_ops:
        Diagnostic plotting options
    input_eq:
        input equilibrium
    reference_eq:
        reference equilibrium
    """

    def __init__(
        self,
        input_eq: Equilibrium | FixedPlasmaEquilibrium | None = None,
        reference_eq: Equilibrium | FixedPlasmaEquilibrium | None = None,
        diag_ops: EqDiagnosticOptions | None = None,
    ):
        self.input = input_eq
        self.reference = reference_eq
        self.diag_ops = diag_ops or EqDiagnosticOptions()

    def set_input(self, input_eq: Equilibrium | FixedPlasmaEquilibrium):
        """Sets input equilibrium."""
        self.input = input_eq

    def set_reference(self, reference_eq: Equilibrium | FixedPlasmaEquilibrium):
        """Sets reference equilibrium."""
        self.reference = reference_eq

    def _get_input(self):
        try:
            return self.input
        except AttributeError:
            raise AttributeError("Input equilibrium is not set.") from None

    def _get_reference(self):
        try:
            return self.reference
        except AttributeError:
            raise AttributeError("Reference equilibrium is not set.") from None

    def plot(self, ax=None):
        """
        Plot input equilibria.

        Returns
        -------
        ax:
            Matplotlib Axes object
        """
        eq = self._get_input()
        plotter = eq.plot(ax=ax)
        plotter.ax.set_title(eq.label)
        plt.show()
        return plotter.ax

    def plot_field(self, ax=None):
        """
        Plot poloidal and toroidal field for input equilibria.

        Raises
        ------
        BluemiraError
            if wrong number of axes is input

        Returns
        -------
        ax:
            Matplotlib Axes object
        """
        eq = self._get_input()
        n_ax = 2
        if ax is not None:
            if len(ax) != n_ax:
                raise BluemiraError(
                    f"There are 2 subplots, you have provided settings for {len(ax)}."
                )
            ax = ax[0], ax[1]
        else:
            _, ax = plt.subplots(1, 2)
        ax[0].set_xlabel("$x$ [m]")
        ax[0].set_ylabel("$z$ [m]")
        ax[0].set_title("Poloidal")
        ax[0].set_aspect("equal")
        ax[1].set_xlabel("$x$ [m]")
        ax[1].set_ylabel("$z$ [m]")
        ax[1].set_title("Toroidal")
        ax[1].set_aspect("equal")
        plt.suptitle(eq.label)

        EquilibriumPlotter(
            eq, ax=ax[0], plasma=False, show_ox=True, field=EqBPlotParam.BP
        )
        EquilibriumPlotter(
            eq, ax=ax[1], plasma=False, show_ox=True, field=EqBPlotParam.BT
        )
        return ax

    def plot_profiles(self, ax=None):
        """
        Plot profiles for input equilibria.

        Returns
        -------
        ax:
            Matplotlib Axes object
        """
        eq = self._get_input()
        plotter = eq.profiles.plot(ax=ax)
        plotter.ax.set_title(eq.label)
        plt.show()
        return plotter.ax

    def plot_eq_core_analysis(self, ax=None) -> CoreResults:
        """
        Plot characteristics of the plasma core for input equilibria and return results.
        Currently only works for free boundary equilibria.

        Returns
        -------
        core_results:
            Dataclass for core results.
        ax:
            Matplotlib Axes object

        Raises
        ------
        BluemiraError
            If the equilibrium is fixed boundary.
        """
        eq = self._get_input()
        if isinstance(eq, FixedPlasmaEquilibrium):
            raise BluemiraError(
                "This function can only be used for Free Boundary Equilbria."
            )
        core_results, ax = eq.analyse_core(ax=ax)
        plt.suptitle(eq.label)
        return core_results, ax

    def plot_eq_core_mag_axis(self, ax=None):
        """
        Plot a 1-D section through the magnetic axis of the input equilibria.
        Currently only works for free boundary equilibria.

        Returns
        -------
        ax:
            Matplotlib Axes object

        Raises
        ------
        BluemiraError
            If the equilibrium is fixed boundary.
        """
        eq = self._get_input()
        if isinstance(eq, FixedPlasmaEquilibrium):
            raise BluemiraError(
                "This function can only be used for Free Boundary Equilbria."
            )
        plotter = eq.plot_core(ax=ax)
        plt.suptitle(eq.label)
        return plotter.ax

    def physics_info_table(self) -> EqSummary:
        """
        Create a table with the physics information
        from the input equilibria.
        Not for use with FixedPlasmaEquilibrium.

        Returns
        -------
        eq_summary:
            Dataclass of physics information.

        Raises
        ------
        BluemiraError
            If the equilibrium is fixed boundary.

        """
        eq = self._get_input()
        if isinstance(eq, FixedPlasmaEquilibrium):
            raise BluemiraError(
                "This function can only be used for Free Boundary Equilbria."
            )
        eq_summary = eq.analyse_plasma()
        print(  # noqa: T201
            eq_summary.tabulate(
                ["Parameter", "value"], tablefmt="simple", value_label=eq.label
            )
        )
        return eq_summary

    def control_coil_table(self, control: list | None = None):
        """
        Create a table with the control coil information
        from the input equilibria.
        Not for use with FixedPlasmaEquilibrium.

        Parameters
        ----------
        control:
            Set the control coils with a coil type or list of coil names.

        Returns
        -------
        table:
            Table with summary of control coil information.

        Raises
        ------
        BluemiraError
            If the equilibrium is fixed boundary.

        """
        eq = deepcopy(self._get_input())
        if isinstance(eq, FixedPlasmaEquilibrium):
            raise BluemiraError(
                "This function can only be used for Free Boundary Equilbria."
            )
        if isinstance(control, list):
            eq.coilset.control = control
        if isinstance(control, CoilType):
            eq.coilset.control = eq.coilset.get_coiltype(control).name
        table, _fz_c_stot, _fsep = eq.analyse_coils()
        return table

    def plot_equilibria_with_profiles(
        self,
        ax=None,
        show=True,  # noqa: FBT002
    ):
        """
        Plot input equilibria alongside profiles.

        Parameters
        ----------
        title:
            Title to be added at top of figure
        ax:
            List of Matplotlib Axes objects set by user
        show:
            Whether or not to display the plot

        Returns
        -------
        ax1, ax2:
            The Matplotlib Axes objects for each subplot.

        Raises
        ------
        BluemiraError
            if the wrong number of axes is provided

        """
        eq = self._get_input()
        n_ax = 2
        if ax is not None:
            if len(ax) != n_ax:
                raise BluemiraError(
                    f"There are 2 subplots, you have provided settings for {len(ax)}."
                )
            ax1, ax2 = ax[0], ax[1]
        else:
            _, (ax1, ax2) = plt.subplots(1, 2)

        eq.plot(ax=ax1)
        ax1.set_title(
            f"R_0 = {eq.profiles.R_0} m \n "
            f"B_0 = {eq.profiles._B_0} T \n "
            f"I_p = {eq.profiles.I_p / 1e6:.2f} MA \n"
        )
        eq.profiles.plot(ax=ax2)
        ax2.set_title("Profiles")
        plt.suptitle(eq.label)
        if show:
            plt.show()
        return ax1, ax2

    def plot_compare_separatrix(
        self,
        title=None,
        ax=None,
        show=True,  # noqa: FBT002
    ):
        """
        Plot separatricies of input equilibria and reference equilibria.
        N.B. Plots LCFS if a fixed boundary is used.

        Parameters
        ----------
        title:
            Title to be added at top of figure
        ax:
            Matplotlib Axes objects set by user
        show:
            Whether or not to display the plot
        equilibrium_name:
            Name used in plot label

        Returns
        -------
        ax:
            Matplotlib Axes object

        """
        eq = self._get_input()
        ref = self._get_reference()
        if ax is None:
            _, ax = plt.subplots()

        eq_fs = eq.get_separatrix() if isinstance(eq, Equilibrium) else eq.get_LCFS()

        if isinstance(ref, Equilibrium):
            ref_eq_fs = ref.get_separatrix()
        else:
            ref_eq_fs = ref.get_LCFS()

        if isinstance(eq_fs, list):
            ax.plot(eq_fs[0].x, eq_fs[0].z, color="red", label=eq.label)
            ax.plot(eq_fs[1].x, eq_fs[1].z, color="red")
        else:
            ax.plot(eq_fs.x, eq_fs.z, color="red", label=eq.label)

        if isinstance(ref_eq_fs, list):
            ax.plot(
                ref_eq_fs[0].x,
                ref_eq_fs[0].z,
                color="blue",
                linestyle="--",
                label=ref.label,
            )
            ax.plot(ref_eq_fs[1].x, ref_eq_fs[1].z, color="blue", linestyle="--")
        else:
            ax.plot(
                ref_eq_fs.x,
                ref_eq_fs.z,
                color="blue",
                linestyle="--",
                label=ref.label,
            )

        ax.legend(loc="best")
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$z$ [m]")
        ax.set_aspect("equal")

        if title is not None:
            plt.suptitle(title)
        if show:
            plt.show()
        return ax

    def plot_compare_psi(
        self,
        diag_ops: EqDiagnosticOptions | None = None,
        ax=None,
    ):
        """
        Plot Psi comparison.

        Parameters
        ----------
        diag_ops:
            Diagnostic plotting options
        ax:
            List of Matplotlib Axes objects set by user

        Raises
        ------
        BluemiraError
            if no reference equilibrium is provided

        Returns
        -------
        :
            plotting class

        """
        eq = self._get_input()
        ref = self._get_reference()
        diag_ops = diag_ops or self.diag_ops
        return EquilibriumComparisonPostOptPlotter(
            equilibrium=eq,
            reference_equilibrium=ref,
            diag_ops=diag_ops,
            ax=ax,
        ).plot_compare_psi()

    def plot_compare_profiles(
        self,
        reference_profile_sign=None,
        ax=None,
        diff=True,  # noqa: FBT002
    ):
        """
        Plot equilibria reference and input profiles. The diff option can be used to plot
        the difference between the reference and input profiles on the same axis.

        Parameters
        ----------
        reference_profile_sign:
            To be used with the diff option if the profile convention of the compared
            equilibria is different.
        ax:
            Matplotlib axes object.
        diff:
            If two equilibria are being compared then we have the option of also
            plotting the difference between them.

        Returns
        -------
        ax:
            Matplotlib Axes object

        Raises
        ------
        BluemiraError
            if no reference equilibrium or equilibrium file path is provided
        ValueError
            if the profile sign array provided is an incorrect length
        """
        eq = self._get_input()
        ref = self._get_reference()

        n_prof = 5
        if reference_profile_sign is None:
            reference_profile_sign = np.ones(n_prof)
        elif is_num(reference_profile_sign):
            reference_profile_sign *= np.ones(n_prof)
        elif len(reference_profile_sign) != n_prof:
            raise ValueError("profile_sign length not equal to 5.")

        shape_ax = (2, 3)
        if ax is not None:
            if np.shape(ax) != shape_ax:
                raise BluemiraError(
                    "Subplot shape is (2,3), "
                    f"you have provided settings for {np.shape(ax)}."
                )
        else:
            _, ax = plt.subplots(2, 3)

        ref_profs = [
            ref.profiles.pprime,
            ref.profiles.ffprime,
            ref.profiles.fRBpol,
            ref.profiles.pressure,
            ref.profiles.shape,
        ]
        profs = [
            eq.profiles.pprime,
            eq.profiles.ffprime,
            eq.profiles.fRBpol,
            eq.profiles.pressure,
            eq.profiles.shape,
        ]
        ax_titles = ["pprime", "ffprime", "fRBpol", "pressure [Pa]", "shape"]
        axes = [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1], ax[0, 2]]

        x = np.linspace(0, 1, 50)

        for ref_prof, prof, sign, axs, a_title in zip(
            ref_profs, profs, reference_profile_sign, axes, ax_titles, strict=False
        ):
            axs.plot(x, sign * ref_prof(x), marker=".", label=ref.label)
            axs.plot(x, prof(x), marker=".", label=eq.label)
            if diff:
                axs.plot(
                    x,
                    sign * ref_prof(x) - prof(x),
                    marker=".",
                    label=ref.label + " - " + eq.label,
                )
            axs.set_title(a_title)
            axs.legend(loc="best")

        ax[1, 2].axis("off")
        plt.suptitle("Profile Comparison")
        plt.show()
        return ax

    def plot_target_flux(
        self,
        target: str,
        target_coords: Coordinates,
        use_reference=True,  # noqa: FBT002
        n_layers=10,
        vertical=False,  # noqa: FBT002
        ax=None,
        show=True,  # noqa: FBT002
    ):
        """
        Plot the divertor leg flux. Will find flux surfaces at evenly spaced points
        in x- or z- direction for a set of target coordinates.

        Parameters
        ----------
        target:
            Location of divertor target, i.e.,
            lower_outer, lower_inner, upper_outer or upper_inner
        target_coords:
            Coordinates of divertor target
        use_reference:
            Plot reference as well as input
        n_layers:
            Number of flux surfaces to extract for each leg
        vertical:
            Set to True if using a vertical target.
            This boolean determines whether to get flux surfaces across
            the target x-range (horizontal) or z-range (vertical).
        ax:
            Matplotlib Axes objects set by user
        show:
            Whether or not to display the plot

        Returns
        -------
        ax:
            Matplotlib Axes object
        target_flux:
            Coordinates of flux surfaces crossing target for input equilibria
        ref_target_flux:
            Coordinates of flux surfaces crossing target for reference equilibria

        """
        if ax is None:
            _, ax = plt.subplots()
            ax.set_aspect("equal")

        if use_reference:
            ref = self._get_reference()
            ref_target_flux = get_target_flux(
                ref, target, target_coords, n_layers, vertical
            )

            if isinstance(ref_target_flux, list):
                for i in np.arange(len(ref_target_flux)):
                    if i == 0:
                        ax.plot(
                            ref_target_flux[i].x,
                            ref_target_flux[i].z,
                            color="blue",
                            linestyle="--",
                            label=ref.label + " LCFS",
                        )
                    else:
                        ax.plot(
                            ref_target_flux[i].x,
                            ref_target_flux[i].z,
                            color="blue",
                            linestyle="--",
                        )
            elif ref_target_flux is not None:
                ax.plot(
                    ref_target_flux.x,
                    ref_target_flux.z,
                    color="blue",
                    linestyle="--",
                    label=ref.label + " LCFS",
                )

        eq = self._get_input()

        target_flux = get_target_flux(eq, target, target_coords, n_layers, vertical)

        if isinstance(target_flux, list):
            for i in np.arange(len(target_flux)):
                if i == 0:
                    ax.plot(
                        target_flux[i].x,
                        target_flux[i].z,
                        color="red",
                        label=eq.label + " LCFS",
                    )
                else:
                    ax.plot(target_flux[i].x, target_flux[i].z, color="red")
        elif target_flux is not None:
            ax.plot(
                target_flux.x,
                target_flux.z,
                color="red",
                label=eq.label + " LCFS",
            )
        else:
            bluemira_warn("No flux found crossing target coordinates.")

        ax.plot(target_coords.x, target_coords.z, color="black", linewidth=5)

        ax.legend(loc="best")
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$z$ [m]")
        ax.set_aspect("equal")
        if show:
            plt.show()
        return ax, target_flux, ref_target_flux


@dataclass
class MultiEqProfiles:
    """Profile dataclass for plotting."""

    pprime: list[npt.NDArray[np.float64]] | None = None
    ffprime: list[npt.NDArray[np.float64]] | None = None
    fRBpol: list[npt.NDArray[np.float64]] | None = None
    pressure: list[npt.NDArray[np.float64]] | None = None
    shape: list[npt.NDArray[np.float64]] | None = None

    def add_profile(self, profile, n_points):
        """Add profile to dataclass."""
        x = np.linspace(0, 1, n_points)
        dataclass_dict = self.__dataclass_fields__
        for key in dataclass_dict:
            p_list = [] if getattr(self, key) is None else getattr(self, key)
            p = getattr(profile, key)
            p_list.append(p(x))
            setattr(self, key, p_list)


class MultiEqAnalysis:
    """
    Equilibria analysis toolbox for multiple Equilibrium.

    Parameters
    ----------
    equilibria_dict:
        Dictionary of equilibria load information.
        Can be created using select_multi_eqs function.

    """

    def __init__(
        self,
        equilibria_dict: dict,
        n_points: int = 50,
    ):
        self.equilibria_dict = equilibria_dict
        self.set_equilibria(equilibria_dict, n_points)

    def set_equilibria(self, equilibria_dict: dict, n_points: int = 50):
        """
        Use dictionary of equilibria information to get values for plotting.

        Parameters
        ----------
        equilibria_dict:
            Dictionary of equilibria load information.
            Can be created using select_multi_eqs function.
        n_points:
            Number of profile points to plot.
        """
        check_eq = next(iter(equilibria_dict))
        if "eq" not in equilibria_dict[check_eq]:
            self.equilibria_dict = get_eqs(equilibria_dict)
        self.equilibria_dict = equilibria_dict
        self.equilibria = []
        self.profiles = []
        for equilibrium_dict in equilibria_dict.values():
            self.equilibria.append(equilibrium_dict["eq"])
            self.profiles.append(equilibrium_dict["profiles"])
        self.plotting_profiles = MultiEqProfiles()
        self.fill_plotting_profiles(n_points)

    def make_eq_dataclass_list(self, method, *args):
        """
        Create a list of dataclasses from a function with
        equilibrium as the first argument.

        Parameters
        ----------
        method:
            Dataclass creation method.
        args:
            Arguments needed for chosen method.

        Returns
        -------
        results:
            list of dataclass objects
        """
        results = []
        for eq in self.equilibria:
            results.append(method(eq, *args))  # noqa: PERF401
        return results

    def fill_plotting_profiles(self, n_points):
        """
        Calculate profile values for plotting. This is done when a MultiEqAnalysis
        class initiated but can be reset with different n_points using this function.

        Parameters
        ----------
        n_points:
            number of normalised psi points

        """
        for profile in self.profiles:
            self.plotting_profiles.add_profile(profile, n_points)

    def physics_info_table(self):
        """
        Create a table with the the physics information
        from all listed Equilibria.
        Not for use with FixedPlasmaEquilibrium.

        Returns
        -------
        table:
            Table with equilibria physics information.

        """
        eq_summaries = self.make_eq_dataclass_list(Equilibrium.analyse_plasma)

        table = tabulate_values_from_multiple_frames(
            eq_summaries, self.equilibria_dict.keys()
        )

        print(table)  # noqa: T201
        return table

    def plot_core_physics(
        self,
        title="Physics Parameters",
        n_points: int = 50,
        ax=None,
        show=True,  # noqa: FBT002
    ):
        """
        Plot physics parameters for the core plasma of each equilibria
        (i.e., plot across the normalised 1-D flux coordinate).

        title:
            Title to be added at top of figure
        n_points:
            number of normalised psi points
        ax:
            Matplotlib Axes objects set by user
        show:
            Whether or not to display the plot

        Returns
        -------
        core_results:
            List[dataclass] of equilibria core physics results.
        ax:
            Matplotlib Axes object

        """
        core_results = self.make_eq_dataclass_list(analyse_plasma_core, n_points)

        if ax is None:
            r, c = int((len(core_results[0].__dict__) - 1) / 2) + 1, 2
            gs = GridSpec(r, c)
            ax = [plt.subplot(gs[i]) for i in range(r * c)]

        for res, name in zip(core_results, self.equilibria_dict.keys(), strict=False):
            CorePlotter(res, ax, eq_name=name)

        plt.suptitle(title)
        if show:
            plt.show()
        return core_results, ax

    def coilset_info_table(self, value_type=CSData.CURRENT):
        """
        Create a table with the the control coil information
        from all listed Equilibria.

        Parameters
        ----------
        equilibria_dict:
            Dictionary of equilibria load information.
            Can be created using select_multi_eqs function.
            Will set or reset the values used by MultiEqAnalysis.
        value_type:
            Choose the type of coilset data to be printed,
            default is current values.

        Returns
        -------
        table:
            table with equilibria control coil information.

        """
        n_cc = [
            len(eq.coilset.get_control_coils().name)
            for eq in self.equilibria
            if isinstance(eq, Equilibrium)
        ]
        max_n_cc = np.max(n_cc)
        cc_tab_list, cc_eq_names = [], []
        for name, eq_dict in self.equilibria_dict.items():
            if isinstance(eq_dict["eq"], Equilibrium):
                n = eq_dict["eq"].coilset.get_control_coils().name
                cc_tab_list.append(n + [None] * (max_n_cc - len(n)))
                coil_dict, _, _ = eq_dict["eq"].analyse_coils(print_table=False)
                cc_tab_list.append(
                    coil_dict[value_type.value].tolist() + [None] * (max_n_cc - len(n))
                )
            else:
                cc_tab_list.extend([[None] * max_n_cc, [None] * (max_n_cc - len(n))])
            cc_eq_names.extend([name, value_type.value])

        table = tabulate(
            list(zip(*cc_tab_list, strict=False)),
            headers=cc_eq_names,
            tablefmt="simple",
            showindex=False,
            numalign="right",
        )
        print(table)  # noqa: T201
        return table

    def plot_compare_profiles(
        self,
        ax=None,
        header=None,
        n_points=None,
    ):
        """
        Plot the profiles of all the listed equilibria.

        Parameters
        ----------
        ax:
            List of Matplotlib Axes objects set by user
        header:
            Text to be added at the top of the figure
        show:
            Whether or not to display the plot
        n_points:
            number of normalised psi points

        Returns
        -------
        ax1, ax2, ax3, ax4, ax5, ax6:
            The Matplotlib Axes objects for each subplot.

        Raises
        ------
        BluemiraError
            if the axes provided are the incorrect shape

        """
        if n_points is not None:
            self.fill_plotting_profiles(n_points)

        x = np.linspace(0, 1, n_points or 50)

        shape_ax = (2, 3)
        if ax is not None:
            if np.shape(ax) != shape_ax:
                raise BluemiraError(
                    "Subplot shape is (2,3), you have provided "
                    f"settings for {np.shape(ax)}."
                )
        else:
            _, ax = plt.subplots(2, 3)

        ax_list = [ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1], ax[0, 2]]
        ax_titles = ["pprime", "ffprime", "fRBpol", "pressure [Pa]", "shape"]

        for axs, key, title in zip(
            ax_list, self.plotting_profiles.__dataclass_fields__, ax_titles, strict=False
        ):
            for profile, name in zip(
                getattr(self.plotting_profiles, key),
                self.equilibria_dict.keys(),
                strict=False,
            ):
                axs.plot(x, profile, marker=".", label=name)
                axs.legend(loc="best")
                axs.set_title(title)

        ax[1, 2].axis("off")

        if header is not None:
            plt.suptitle(header)
        plt.show()
        return ax[0, 0], ax[0, 1], ax[1, 0], ax[1, 1], ax[0, 2], ax[1, 2]

    def plot_compare_flux_surfaces(
        self,
        flux_surface=FluxSurfaceType.LCFS,
        plot_fixed=True,  # noqa: FBT002
        psi_norm=0.98,
        title=None,
        ax=None,
        show=True,  # noqa: FBT002
    ):
        """
        Plot a flux surface of choice for each equilibrium on the same axes.
        User can select either the LCFS, separatrix or a flux surface with
        a given psi_norm.

        Always plots a LCFS for a fixed boundary equilibrium,
        unless plot_fixed is set to false, in which case no flux surfaces
        are plotted for any fixed boundary equilibria.

        Parameters
        ----------
        equilibria_dict:
            Dictionary of equilibria load information.
            Can be created using select_multi_eqs function.
            Will set or reset the values used by MultiEqAnalysis.
        flux_surface:
            Type of flux surface to be plotted
        plot_fixed:
            Weather or not to plot the LCFS for any fixed boundary equilibria.
        psi_norm:
            The normalised psi value to use, if the user selects
            the psi norm comparison option.
        title:
            Title to be added at top of figure
        ax:
            Matplotlib Axes objects set by user
        show:
            Whether or not to display the plot

        Returns
        -------
        ax:
            Matplotlib Axes object

        """
        if ax is None:
            _, ax = plt.subplots()

        fs_list = []
        ccycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        for eq, label in zip(self.equilibria, self.equilibria_dict.keys(), strict=False):
            free = isinstance(eq, Equilibrium)
            if (flux_surface is FluxSurfaceType.SEPARATRIX) and free:
                fs = eq.get_separatrix()
            elif (flux_surface is FluxSurfaceType.PSI_NORM) and free:
                if psi_norm <= 1.0:
                    fs = eq.get_flux_surface(psi_norm)
                else:
                    coords = find_flux_surfs(
                        eq.x,
                        eq.z,
                        eq.psi(),
                        psi_norm,
                        o_points=eq._o_points,
                        x_points=eq._x_points,
                    )
                    loops = [Coordinates({"x": c.T[0], "z": c.T[1]}) for c in coords]
                    loops.sort(key=lambda loop: -loop.length)
                    fs = loops[:2]
            else:
                fs = eq.get_LCFS()
            fs_list.append(fs)

            color = next(ccycle)
            if free or plot_fixed:
                if isinstance(fs, list):
                    ax.plot(fs[0].x, fs[0].z, label=label, color=color)
                    ax.plot(fs[1].x, fs[1].z, color=color)
                else:
                    ax.plot(fs.x, fs.z, label=label, color=color)

        ax.legend(loc="best")

        if title is not None:
            plt.suptitle(title)
        if show:
            plt.show()
        return ax

    def plot_divertor_length_angle(
        self,
        n_layers=10,
        dx_off=0.10,
        plasma_facing_boundary_list=None,
        legs_to_plot=DivLegsToPlot.ALL,
        ax=None,
        show=True,  # noqa: FBT002
        radian=True,  # noqa: FBT002
    ):
        """
        Plot the divertor leg length and grazing angle for the equilibria
        divertor legs.

        Parameters
        ----------
        n_layers:
            Number of flux surfaces to extract for each leg
        dx_off:
            Total span in radial space of the flux surfaces to extract
        plasma_facing_boundary_list:
            List of associated plasma facing boundary coordinates
        legs_to_plot:
            Which legs to plot i.e, upper, lower or all.
        ax:
            Matplotlib Axes objects set by user
        show:
            Whether or not to display the plot
        radian:
            True for plot angles in units of radians
            False for plot angles in units of degrees

        Returns
        -------
        ax:
            Matplotlib Axes object

        """
        lengths, angles = get_leg_flux_info(
            self.equilibria,
            n_layers=n_layers,
            dx_off=dx_off,
            plasma_facing_boundary_list=plasma_facing_boundary_list,
            legs_to_plot=legs_to_plot,
        )

        # convert angles to degrees if required
        if not radian:
            for ang_dict in angles:
                for key, ang_value in ang_dict.items():
                    ang_dict[key] = raw_uc(ang_value, "radian", "degree")

        if ax is None:
            if legs_to_plot is DivLegsToPlot.ALL:
                _, ax = plt.subplots(2, 4, sharex="col")
            elif legs_to_plot in DivLegsToPlot.PAIR:
                _, ax = plt.subplots(2, 2, sharex="col")

        def plot_div_value(i, value, ax, name, n_layers=n_layers, dx_off=dx_off):
            for j, (k, v) in enumerate(value.items()):
                ax[i, j].plot(
                    np.arange(n_layers) * (dx_off / n_layers), v, label=name, marker="."
                )
                ax[i, j].set_title(k)
                ax[i, j].legend()

        for lngth, ang, name in zip(
            lengths, angles, self.equilibria_dict.keys(), strict=False
        ):
            plot_div_value(0, lngth, ax, name)
            plot_div_value(1, ang, ax, name)

        ax[0, 0].set_ylabel("length [m]")
        ax[1, 0].set_ylabel("angle [rad]") if radian else ax[1, 0].set_ylabel(
            "angle [deg]"
        )
        ax[1, 1].set_xlabel("Cumulative offset from seperatrix at midpoint [m]")
        if show:
            plt.show()
        return ax
