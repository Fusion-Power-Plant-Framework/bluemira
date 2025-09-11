# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Toroidal Field (TF) Coil 2D Case Class.

This class models and optimises the cross-sectional layout of the inboard leg of a TF
coil.
It is designed to define and adjust the distribution of structural materials and
winding pack arrangement to achieve optimal performance and mechanical robustness.

Note:
- Focused on the two-dimensional analysis of the inboard leg.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from bluemira.base.look_and_feel import (
    bluemira_debug,
    bluemira_error,
    bluemira_print,
    bluemira_warn,
)
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.tools import make_polygon
from bluemira.magnets.utils import reciprocal_summation, summation
from bluemira.magnets.winding_pack import WindingPack, create_wp_from_dict
from bluemira.utilities.opt_variables import OptVariable, OptVariablesFrame, VarDictT, ov

if TYPE_CHECKING:
    from matproplib import OperationalConditions
    from matproplib.material import Material

    from bluemira.geometry.wire import BluemiraWire


def _dx_at_radius(radius: float, rad_theta: float) -> float:
    """
    Compute the toroidal half-width at a given radial position.

    Parameters
    ----------
    radius:
        Radial position at which to compute the toroidal width [m].
    rad_theta:
        Toroidal angular span of the TF coil [radians].

    Returns
    -------
    :
        Toroidal width [m] at the given radius.
    """
    return 2 * radius * np.tan(rad_theta / 2)


@dataclass
class CaseGeometryOptVariables(OptVariablesFrame):
    """Optimisiation variables for Trapezoidal Geometry."""

    Ri: OptVariable = ov(
        "Ri",
        0,
        lower_bound=0,
        upper_bound=np.inf,
        description="External radius of the TF coil case [m].",
    )
    Rk: OptVariable = ov(
        "Rk",
        0,
        lower_bound=0,
        upper_bound=np.inf,
        description="Internal radius of the TF coil case [m].",
    )
    theta_TF: OptVariable = ov(
        "theta_TF",
        0,
        lower_bound=0,
        upper_bound=360,
        description="Toroidal angular span of the TF coil [degrees].",
    )


class TrapezoidalGeometry(GeometryParameterisation[CaseGeometryOptVariables]):
    """
    Geometry of a Toroidal Field (TF) coil case with trapezoidal cross-section.

    The coil cross-section has a trapezoidal shape: wider at the outer radius (Ri)
    and narrower at the inner radius (Rk), reflecting typical TF coil designs
    for magnetic and mechanical optimisation.
    """

    def __init__(self, var_dict: VarDictT | None = None):
        variables = CaseGeometryOptVariables()
        variables.adjust_variables(var_dict, strict_bounds=False)
        super().__init__(variables)

    @property
    def rad_theta(self) -> float:
        """
        Compute the Toroidal angular span of the TF coil in radians
        """
        return np.radians(self.variables.theta_TF.value)

    @property
    def area(self) -> float:
        """
        Compute the cross-sectional area of the trapezoidal TF case.

        The area is calculated as the average of the toroidal widths at Ri and Rk,
        multiplied by the radial height (Ri - Rk).

        Returns
        -------
        :
            Cross-sectional area [m²].
        """
        return (
            0.5
            * (self.variables.Ri.value - self.variables.Rk.value)
            * (
                _dx_at_radius(self.variables.Ri.value, self.rad_theta)
                + _dx_at_radius(self.variables.Rk.value, self.rad_theta)
            )
        )

    def create_shape(self, label: str = "") -> BluemiraWire:
        """
        Construct the (x, r) coordinates of the trapezoidal cross-section polygon.

        Returns
        -------
        :
            Array of shape (4, 2) representing the corners of the trapezoid.
            Coordinates are ordered counterclockwise starting from the top-left corner:
            [(-dx_outer, Ri), (dx_outer, Ri), (dx_inner, Rk), (-dx_inner, Rk)].
        """
        dx_outer = _dx_at_radius(self.variables.Ri.value, self.rad_theta) / 2
        dx_inner = _dx_at_radius(self.variables.Rk.value, self.rad_theta) / 2

        return make_polygon(
            [
                [-dx_outer, 0.0, self.variables.Ri.value],
                [dx_outer, 0.0, self.variables.Ri.value],
                [dx_inner, 0.0, self.variables.Rk.value],
                [-dx_inner, 0.0, self.variables.Rk.value],
            ],
            closed=True,
            label=label,
        )


class WedgedGeometry(GeometryParameterisation[CaseGeometryOptVariables]):
    """
    TF coil case shaped as a sector of an annulus (wedge with arcs).

    The geometry consists of two circular arcs (inner and outer radii)
    connected by radial lines, forming a wedge-like shape.
    """

    def __init__(self, var_dict: VarDictT | None = None):
        variables = CaseGeometryOptVariables()
        variables.adjust_variables(var_dict, strict_bounds=False)
        super().__init__(variables)

    @property
    def rad_theta(self) -> float:
        """
        Compute the Toroidal angular span of the TF coil in radians
        """
        return np.radians(self.variables.theta_TF.value)

    def area(self) -> float:
        """
        Compute the cross-sectional area of the wedge geometry.

        Returns
        -------
        :
            Cross-sectional area [m²] defined by the wedge between outer radius Ri
            and inner radius Rk over the toroidal angle theta_TF.
        """
        return (
            0.5
            * self.rad_theta
            * (self.variables.Ri.value**2 - self.variables.Rk.value**2)
        )

    def create_shape(self, label: str = "", n_points: int = 50) -> BluemiraWire:
        """
        Build the polygon representing the wedge shape.

        The polygon is created by discretizing the outer and inner arcs
        into a series of points connected sequentially.

        Parameters
        ----------
        n_points:
            Number of points to discretize each arc. Default is 50.

        Returns
        -------
        :
            Array of (x, y) coordinates [m] describing the wedge polygon.
        """
        theta1 = -self.rad_theta / 2
        theta2 = -theta1

        angles_outer = np.linspace(theta1, theta2, n_points)
        angles_inner = np.linspace(theta2, theta1, n_points)

        arc_outer = np.column_stack((
            self.variables.Ri.value * np.sin(angles_outer),
            self.variables.Ri.value * np.cos(angles_outer),
        ))
        arc_inner = np.column_stack((
            self.variables.Rk.value * np.sin(angles_inner),
            self.variables.Rk.value * np.cos(angles_inner),
        ))

        return make_polygon(np.vstack((arc_outer, arc_inner)), label=label)


class CaseTF(ABC):
    """
    Abstract Base Class for Toroidal Field Coil Case configurations.

    Defines the universal properties common to all TF case geometries.
    """

    def __init__(
        self,
        Ri: float,
        theta_TF: float,
        dy_ps: float,
        dy_vault: float,
        mat_case: Material,
        WPs: list[WindingPack],  # noqa: N803
        geometry: GeometryParameterisation,
        name: str = "BaseCaseTF",
    ):
        """
        Initialize a BaseCaseTF instance.

        Parameters
        ----------
        Ri:
            External radius of the TF coil case [m].
        theta_TF:
            Toroidal angular span of the TF coil [degrees].
        dy_ps:
            Radial thickness of the poloidal support region [m].
        dy_vault:
            Radial thickness of the vault support region [m].
        mat_case:
            Structural material assigned to the TF coil case.
        WPs:
            List of winding pack objects embedded inside the TF case.
        name:
            String identifier for the TF coil case instance (default is "BaseCaseTF").
        """
        self.geometry = geometry
        self.geometry.variables.Ri.value = Ri
        self.geometry.variables.theta_TF.value = theta_TF
        self.dy_ps = dy_ps
        self.dy_vault = dy_vault
        self.mat_case = mat_case
        self.WPs = WPs
        self.name = name
        # sets Rk
        self.update_dy_vault(self.dy_vault)

    @property
    def name(self) -> str:
        """
        Name identifier of the TF case.

        Returns
        -------
        :
            Human-readable label for the coil case instance.
        """
        return self._name

    @name.setter
    def name(self, value: str):
        """
        Set the name of the TF case.

        Parameters
        ----------
        value:
            Case name.

        Raises
        ------
        TypeError
            If value is not a string.
        """
        if not isinstance(value, str):
            raise TypeError("name must be a string.")
        self._name = value

    @property
    def rad_theta(self) -> float:
        """
        Compute the Toroidal angular span of the TF coil in radians
        """
        return np.radians(self.geometry.variables.theta_TF.value)

    def update_dy_vault(self, value: float):
        """
        Update the value of the vault support region thickness

        Parameters
        ----------
        value:
            Vault thickness [m].
        """
        self.dy_vault = value
        self.geometry.variables.Rk.value = self.R_wp_k[-1] - self.dy_vault

    # jm -  not used but replaces functionality of original Rk setter
    #       can't find when (if) it was used originally
    def update_Rk(self, value: float):  # noqa: N802
        """
        Set or update the internal (innermost) radius of the TF case.
        """
        self.geometry.variables.Rk.value = value
        self.dy_vault = self.R_wp_k[-1] - self.geometry.variables.Rk.value

    @property
    @abstractmethod
    def dx_vault(self):
        """
        Average toroidal length of the vault.

        Returns
        -------
        :
            Average length of the vault in the toroidal direction [m].
        """

    @property
    def mat_case(self) -> Material:
        """
        Structural material assigned to the TF case.

        Returns
        -------
        :
            Material object providing mechanical and thermal properties.
        """
        return self._mat_case

    @mat_case.setter
    def mat_case(self, value: Material):
        """
        Set the structural material assigned to the TF case.

        Parameters
        ----------
        value:
            Material object.

        Raises
        ------
        TypeError
            If value is not a Material instance.
        """
        # Optional: check type here if you want
        self._mat_case = value

    @property
    def WPs(self) -> list[WindingPack]:  # noqa: N802
        """
        List of winding pack (WP) objects embedded inside the TF case.

        Returns
        -------
        :
            Winding pack instances composing the internal coil layout.
        """
        return self._WPs

    @WPs.setter
    def WPs(self, value: list[WindingPack]):  # noqa: N802
        """
        Set the winding pack objects list.

        Parameters
        ----------
        value:
            List containing only WindingPack objects.

        Raises
        ------
        TypeError
            If value is not a list of WindingPack instances.
        """
        if not isinstance(value, list):
            raise TypeError("WPs must be a list of WindingPack objects.")
        if not all(isinstance(wp, WindingPack) for wp in value):
            raise TypeError("All elements of WPs must be WindingPack instances.")
        self._WPs = value

        # fix dy_vault (this will recalculate Rk)
        self.update_dy_vault(self.dy_vault)

    @property
    def dx_i(self):
        """Toroidal length of the coil case at its maximum radial position [m]"""
        return _dx_at_radius(self.geometry.variables.Ri.value, self.rad_theta)

    @property
    def dx_ps(self):
        """Average toroidal length of the ps plate [m]"""
        return (
            self.geometry.variables.Ri.value
            + (self.geometry.variables.Ri.value - self.dy_ps)
        ) * np.tan(self.rad_theta / 2)

    @property
    def n_conductors(self) -> int:
        """Total number of conductors in the winding pack."""
        return sum(w.n_conductors for w in self.WPs)

    @property
    def dy_wp_i(self) -> np.ndarray:
        """
        Computes the radial thickness of each winding pack.

        Returns
        -------
        :
            Array containing the radial thickness [m] of each Winding Pack.
            Each element corresponds to one WP in the self.WPs list.
        """
        return np.array([wp.dy for wp in self.WPs])

    @property
    def dy_wp_tot(self) -> float:
        """
        Computes the total radial thickness occupied by all winding packs.

        Returns
        -------
        :
            Total radial thickness [m] summed over all winding packs.
        """
        return sum(self.dy_wp_i)

    @property
    def R_wp_i(self) -> np.ndarray:  # noqa: N802
        """
        Compute the radial positions for the outer edge (start) of each winding pack.

        Returns
        -------
        :
            Array of radial positions [m] corresponding to the outer edge of each WP.
        """
        dy_wp_cumsum = np.cumsum(np.concatenate(([0.0], self.dy_wp_i)))
        result_initial = self.geometry.variables.Ri.value - self.dy_ps
        if len(dy_wp_cumsum) == 1:
            result = np.array([result_initial])
        else:
            result = result_initial - dy_wp_cumsum[:-1]

        if len(result) != len(self.WPs):
            bluemira_error(f"Mismatch: {len(result)} R_wp_i vs {len(self.WPs)} WPs!")

        return result

    @property
    def R_wp_k(self):  # noqa: N802
        """
        Compute the minimum radial positions for each winding pack.

        Returns
        -------
        :
            Array of radial positions [m] corresponding to the outer edge of
            each winding pack.
        """
        return self.R_wp_i - self.dy_wp_i

    def plot(
        self,
        ax: plt.Axes | None = None,
        *,
        show: bool = False,
        homogenised: bool = False,
    ) -> plt.Axes:
        """
        Schematic plot of the TF case cross-section including winding packs.

        Parameters
        ----------
        ax:
            Axis on which to draw the figure. If `None`, a new figure and axis will be
            created.
        show:
            If `True`, displays the plot immediately using `plt.show()`.
            Default is `False`.
        homogenised:
            If `True`, plots winding packs as homogenised blocks.
            If `False`, plots individual conductors inside WPs.
            Default is `False`.

        Returns
        -------
        :
            The axis object containing the rendered plot.
        """
        if ax is None:
            _, ax = plt.subplots()
            ax.set_aspect("equal", adjustable="box")

        self.geometry.plot(ax=ax)

        # Plot winding packs
        for i, wp in enumerate(self.WPs):
            xc_wp = 0.0
            yc_wp = self.R_wp_i[i] - wp.dy / 2
            ax = wp.plot(xc=xc_wp, yc=yc_wp, ax=ax, homogenised=homogenised)

        # Finalize plot
        ax.set_xlabel("Toroidal direction [m]")
        ax.set_ylabel("Radial direction [m]")
        ax.set_title(f"TF Case Cross Section: {self.name}")

        if show:
            plt.show()

        return ax

    @property
    def area_case_jacket(self) -> float:
        """
        Area of the case jacket (excluding winding pack regions).

        Returns
        -------
            Case jacket area [m²], computed as total area minus total WP area.
        """
        return self.geometry.area - self.area_wps

    @property
    def area_wps(self) -> float:
        """
        Total area occupied by all winding packs.

        Returns
        -------
        :
            Combined area of the winding packs [m²].
        """
        return np.sum([w.area for w in self.WPs])

    @property
    def area_wps_jacket(self) -> float:
        """
        Total jacket area of all winding packs.

        Returns
        -------
        :
            Combined area of conductor jackets in all WPs [m²].
        """
        return np.sum([w.jacket_area for w in self.WPs])

    @property
    def area_jacket_total(self) -> float:
        """
        Total structural material area of the TF coil case, including:

        - The case jacket area (structural material surrounding the winding packs).
        - The conductor jackets area (jackets enclosing the individual conductors
        inside the WPs).

        Returns
        -------
        :
            Combined area of the case structure and the conductor jackets [m²].

        Notes
        -----
        - This represents the total metallic structural material in the TF case
        cross-section.
        """
        return self.area_case_jacket + self.area_wps_jacket

    @abstractmethod
    def rearrange_conductors_in_wp(
        self,
        n_conductors: int,
        wp_reduction_factor: float,
        min_gap_x: float,
        n_layers_reduction: int,
        layout: str = "auto",
    ):
        """
        Abstract method to rearrange the total number of conductors into winding packs.

        Parameters
        ----------
        n_conductors:
            Total number of conductors to distribute.
        wp_reduction_factor:
            Fractional reduction of available toroidal space for WPs.
        min_gap_x:
            Minimum gap between the WP and the case boundary in toroidal direction [m].
        n_layers_reduction:
            Number of layers to remove after each WP.
        layout:
            Layout strategy ("auto", "layer", "pancake").
        """

    @staticmethod
    def enforce_wp_layout_rules(
        n_conductors: int,
        dx_WP: float,  # noqa: N803
        dx_cond: float,
        dy_cond: float,
        layout: str,
    ) -> tuple[int, int]:
        """
        Compute the maximum number of horizontal layers (n_layers_max)
        and vertical turns (n_turns_max) for a winding pack (WP),
        based on available toroidal width, conductor size, layout rules,
        and number of conductors to allocate.

        Parameters
        ----------
        n_conductors:
            Number of conductors to allocate.
        dx_WP:
            Available toroidal half-width for the winding pack [m].
        dx_cond:
            Toroidal half-width of a single conductor [m].
        dy_cond:
            Radial half-height of a single conductor [m].
        layout:
            Layout type:
            - "auto"    : no constraints
            - "layer"   : enforce even number of turns (ny % 2 == 0)
            - "pancake" : enforce even number of layers (nx % 2 == 0)

        Returns
        -------
        :
            n_layers_max (nx)
        :
            n_turns_max (ny)

        Raises
        ------
        ValueError
            If layout is unknown or insufficient space.
        ValueError
            if dx_WP is not big enough to allocate the conductor layers
        """
        if dx_cond <= 0 or dy_cond <= 0:
            raise ValueError("Conductor dimensions must be positive.")

        if n_conductors <= 0:
            raise ValueError("Number of conductors must be positive.")

        # --- Step 1: Compute maximum layers (horizontal) ---
        n_layers_max = int(dx_WP // dx_cond)

        if layout == "pancake" and n_layers_max % 2 != 0:
            n_layers_max = max(2, n_layers_max - 1)

        if dx_WP < n_layers_max * dx_cond:
            raise ValueError(
                f"Adjusted number of layers ({n_layers_max}) does not fit in available "
                f"width (dx_WP={dx_WP})."
            )

        # How many vertical turns are needed to allocate all conductors
        n_turns_max = int(np.ceil(n_conductors / n_layers_max))

        if layout == "layer" and n_turns_max % 2 != 0:
            n_turns_max = max(2, n_turns_max - 1)

        return n_layers_max, n_turns_max

    @abstractmethod
    def optimise_vault_radial_thickness(
        self,
        pm: float,
        fz: float,
        T: float,  # noqa: N803
        B: float,
        allowable_sigma: float,
        bounds: np.ndarray = None,
    ):
        """
        Abstract method to optimise the radial thickness of the vault support region.

        Parameters
        ----------
        pm:
            Radial magnetic pressure [Pa].
        fz:
            Axial electromagnetic force [N].
        T:
            Operating temperature [K].
        B:
            Magnetic field strength [T].
        allowable_sigma:
            Allowable maximum stress [Pa].
        bounds:
            Optimisation bounds for vault thickness [m].
        """

    def to_dict(self) -> dict[str, float | str | list[dict[str, float | str | Any]]]:
        """
        Serialize the BaseCaseTF instance into a dictionary.

        Returns
        -------
        dict
            Serialized data representing the TF case, including geometry and material
            information.
        """
        return {
            "name": self.name,
            "Ri": self.geometry.variables.Ri.value,
            "dy_ps": self.dy_ps,
            "dy_vault": self.dy_vault,
            "theta_TF": self.geometry.variables.theta_TF.value,
            "mat_case": self.mat_case.name,  # Assume Material has 'name' attribute
            "WPs": [wp.to_dict() for wp in self.WPs],
            # Assume each WindingPack implements to_dict()
        }

    @classmethod
    def from_dict(cls, case_dict: dict, name: str | None = None) -> CaseTF:
        """
        Deserialize a BaseCaseTF instance from a dictionary.

        Parameters
        ----------
        case_dict:
            Dictionary containing serialized TF case data.
        name:
            Optional name override for the new instance.

        Returns
        -------
        :
            Reconstructed TF case instance.

        Raises
        ------
        ValueError
            If the 'name_in_registry' field does not match this class.
        """
        WPs = [create_wp_from_dict(wp_dict) for wp_dict in case_dict["WPs"]]  # noqa:N806

        return cls(
            Ri=case_dict["Ri"],
            dy_ps=case_dict["dy_ps"],
            dy_vault=case_dict["dy_vault"],
            theta_TF=case_dict["theta_TF"],
            mat_case=case_dict["mat_case"],
            WPs=WPs,
            name=name or case_dict.get("name"),
        )

    def __str__(self) -> str:
        """
        Generate a human-readable summary of the TF case.

        Returns
        -------
        :
            Multiline string summarizing key properties of the TF case.
        """
        return (
            f"CaseTF '{self.name}'\n"
            f"  - Ri: {self.geometry.variables.Ri.value:.3f} m\n"
            f"  - Rk: {self.geometry.variables.Rk.value:.3f} m\n"
            f"  - dy_ps: {self.dy_ps:.3f} m\n"
            f"  - dy_vault: {self.dy_vault:.3f} m\n"
            f"  - theta_TF: {self.geometry.variables.theta_TF.value:.2f}°\n"
            f"  - Material: {self.mat_case.name}\n"
            f"  - Winding Packs: {len(self.WPs)} packs\n"
        )


class TrapezoidalCaseTF(CaseTF):
    """
    Toroidal Field Coil Case with Trapezoidal Geometry.
    Note: this class considers a set of Winding Pack with the same conductor (instance).
    """

    def __init__(
        self,
        Ri: float,
        theta_TF: float,
        dy_ps: float,
        dy_vault: float,
        mat_case: Material,
        WPs: list[WindingPack],  # noqa: N803
        name: str = "TrapezoidalCaseTF",
    ):
        self._check_WPs(WPs)

        geom = TrapezoidalGeometry()
        super().__init__(
            Ri=Ri,
            theta_TF=theta_TF,
            dy_ps=dy_ps,
            dy_vault=dy_vault,
            mat_case=mat_case,
            WPs=WPs,
            name=name,
            geometry=geom,
        )

    def _check_WPs(  # noqa: PLR6301, N802
        self,
        WPs: list[WindingPack],  # noqa:N803
    ):
        """
        Validate that the provided winding packs (WPs) are non-empty and share the
        same conductor.

        Parameters
        ----------
        WPs:
            List of winding pack objects to validate.

        Raises
        ------
        ValueError
            If no winding packs are provided.
        ValueError
            If winding packs have different conductor instances.
        """
        if not WPs:
            raise ValueError("At least one non-empty winding pack must be provided.")

        first_conductor = WPs[0].conductor
        for i, wp in enumerate(WPs[1:], start=1):
            if wp.conductor is not first_conductor:
                bluemira_warn(
                    f"[Winding pack at index {i} uses a different conductor object "
                    f"than the first one. This module requires all WPs to "
                    f"share the same conductor instance."
                    f"Please verify the inputs or unify the conductor assignment."
                )

    @property
    def dx_vault(self):
        """
        Average toroidal length of the vault.

        Returns
        -------
        :
            Average length of the vault in the toroidal direction [m].
        """
        return (self.R_wp_k[-1] + self.geometry.variables.Rk.value) * np.tan(
            self.rad_theta / 2
        )

    def Kx_ps(self, op_cond: OperationalConditions):  # noqa: N802
        """
        Compute the equivalent radial stiffness of the poloidal support (PS) region.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        float
            Equivalent radial stiffness of the poloidal support [Pa].
        """
        return self.mat_case.youngs_modulus(op_cond) * self.dy_ps / self.dx_ps

    def Kx_lat(self, op_cond: OperationalConditions):  # noqa: N802
        """
        Compute the equivalent radial stiffness of the lateral case sections.

        These are the mechanical links between each winding pack and the outer case.
        Each lateral segment is approximated as a rectangular element.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        np.ndarray
            Array of radial stiffness values for each lateral segment [Pa].
        """
        dx_lat = np.array([
            (self.R_wp_i[i] + self.R_wp_k[i]) / 2 * np.tan(self.rad_theta / 2) - w.dx / 2
            for i, w in enumerate(self.WPs)
        ])
        dy_lat = np.array([w.dy for w in self.WPs])
        return self.mat_case.youngs_modulus(op_cond) * dy_lat / dx_lat

    def Kx_vault(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Compute the equivalent radial stiffness of the vault region.

        Parameters
        ----------
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Equivalent radial stiffness of the vault [Pa].
        """
        return self.mat_case.youngs_modulus(op_cond) * self.dy_vault / self.dx_vault

    def Kx(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Compute the total equivalent radial stiffness of the entire case structure.

        Combines:
        - Two lateral case elements (left and right) per winding pack in series
            with the WP
        - All such combined units in parallel with the PS and vault regions

        Parameters
        ----------
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Total equivalent radial stiffness of the TF case [Pa].
        """
        kx_lat = self.Kx_lat(op_cond)
        temp = [
            reciprocal_summation([
                kx_lat[i],
                w.Kx(op_cond),
                kx_lat[i],
            ])
            for i, w in enumerate(self.WPs)
        ]
        return summation([self.Kx_ps(op_cond), self.Kx_vault(op_cond), *temp])

    def Ky_ps(self, op_cond: OperationalConditions):  # noqa: N802
        """
        Compute the equivalent toroidal stiffness of the poloidal support (PS) region.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Equivalent toroidal stiffness of the PS region [Pa].
        """
        return self.mat_case.youngs_modulus(op_cond) * self.dx_ps / self.dy_ps

    def Ky_lat(self, op_cond: OperationalConditions):  # noqa: N802
        """
        Compute the equivalent toroidal stiffness of lateral case sections
        per winding pack.

        Each lateral piece is treated as a rectangular beam in the toroidal direction.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        np.ndarray
            Array of toroidal stiffness values for each lateral segment [Pa].
        """
        dx_lat = np.array([
            (self.R_wp_i[i] + self.R_wp_k[i]) / 2 * np.tan(self._rad_theta / 2)
            - w.dx / 2
            for i, w in enumerate(self.WPs)
        ])
        dy_lat = np.array([w.dy for w in self.WPs])
        return self.mat_case.youngs_modulus(op_cond) * dx_lat / dy_lat

    def Ky_vault(self, op_cond: OperationalConditions):  # noqa: N802
        """
        Compute the equivalent toroidal stiffness of the vault region.

        Parameters
        ----------
        op_cond: OperationalConditions
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        float
            Equivalent toroidal stiffness of the vault [Pa].
        """
        return self.mat_case.youngs_modulus(op_cond) * self.dx_vault / self.dy_vault

    def Ky(self, op_cond: OperationalConditions) -> float:  # noqa: N802
        """
        Compute the total equivalent toroidal stiffness of the entire case structure.

        Combines:
        - Each winding pack and its adjacent lateral case sections in parallel
        - These parallel combinations are arranged in series with the PS and
            vault regions

        Parameters
        ----------
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Total equivalent toroidal stiffness of the TF case [Pa].
        """
        ky_lat = self.Ky_lat(op_cond)
        temp = [
            summation([
                ky_lat[i],
                w.Ky(op_cond),
                ky_lat[i],
            ])
            for i, w in enumerate(self.WPs)
        ]
        return reciprocal_summation([self.Ky_ps(op_cond), self.Ky_vault(op_cond), *temp])

    def rearrange_conductors_in_wp(
        self,
        n_conductors: int,
        wp_reduction_factor: float,
        min_gap_x: float,
        n_layers_reduction: int,
        layout: str = "auto",
    ):
        """
        Rearrange the total number of conductors into winding packs (WPs)
        within the TF coil case geometry using enforce_wp_layout_rules.

        Parameters
        ----------
        n_conductors:
            Total number of conductors to be allocated.
        wp_reduction_factor:
            Fractional reduction of the total available toroidal space for WPs.
        min_gap_x:
            Minimum allowable toroidal gap between WP and boundary [m].
        n_layers_reduction:
            Number of horizontal layers to reduce after each WP.
        layout:
            Layout type ("auto", "layer", "pancake").

        Raises
        ------
        ValueError
            If there is not enough space to allocate all the conductors.
        """
        debug_msg = ["Method rearrange_conductors_in_wp"]
        conductor = self.WPs[0].conductor
        R_wp_i = self.R_wp_i[0]  # noqa: N806

        dx_WP = self.dx_i * wp_reduction_factor  # noqa: N806
        debug_msg.extend([
            f"dx_WP = {dx_WP}",
            f"self.dx_i = {self.dx_i}",
            f"wp_reduction_factor = {wp_reduction_factor}",
            f"min_gap_x = {min_gap_x}",
            f"n_layers_reduction = {n_layers_reduction}",
            f"layout = {layout}",
            f"n_conductors = {n_conductors}",
        ])

        WPs = []  # noqa: N806
        # number of conductors to be allocated
        remaining_conductors = n_conductors
        # maximum number of winding packs in WPs
        i_max = 50
        i = 0
        while i < i_max and remaining_conductors > 0:
            i += 1

            # maximum number of turns on the considered WP
            if i == 1:
                n_layers_max = math.floor(dx_WP / conductor.dx)
                if layout == "pancake":
                    n_layers_max = math.floor(dx_WP / conductor.dx / 2) * 2
                    if n_layers_max == 0:
                        n_layers_max = 2
            else:
                n_layers_max -= n_layers_reduction

            if n_layers_max < 1:
                raise ValueError(
                    f"n_layers_max: {n_layers_max} < 1. There is not enough space to "
                    f"allocate all the conductors"
                )

            if n_layers_max >= remaining_conductors:
                WPs.append(
                    WindingPack(conductor=conductor, nx=remaining_conductors, ny=1)
                )
                remaining_conductors = 0
            else:
                dx_WP = n_layers_max * conductor.dx  # noqa: N806

                gap_0 = R_wp_i * np.tan(self.rad_theta / 2) - dx_WP / 2
                gap_1 = min_gap_x

                max_dy = (gap_0 - gap_1) / np.tan(self.rad_theta / 2)
                n_turns_max = min(
                    int(np.floor(max_dy / conductor.dy)),
                    int(np.ceil(remaining_conductors / n_layers_max)),
                )
                if layout == "layer":
                    n_turns_max = min(
                        int(np.floor(max_dy / conductor.dy / 2.0) * 2),
                        int(np.ceil(remaining_conductors / n_layers_max / 2.0) * 2),
                    )
                    if n_turns_max == 0:
                        n_turns_max = 2

                if n_turns_max < 1:
                    raise ValueError(
                        f"n_turns_max: {n_turns_max} < 1. There is not enough space to "
                        f"allocate all the conductors"
                    )

                if n_layers_max * n_turns_max > remaining_conductors:
                    n_turns_max -= 1
                    WPs.append(
                        WindingPack(conductor=conductor, nx=n_layers_max, ny=n_turns_max)
                    )
                    remaining_conductors -= n_layers_max * n_turns_max
                    WPs.append(
                        WindingPack(conductor=conductor, nx=remaining_conductors, ny=1)
                    )
                    remaining_conductors = 0
                else:
                    WPs.append(
                        WindingPack(conductor=conductor, nx=n_layers_max, ny=n_turns_max)
                    )
                    remaining_conductors -= n_layers_max * n_turns_max

                if remaining_conductors < 0:
                    bluemira_warn(
                        f"{abs(remaining_conductors)}/{n_layers_max * n_turns_max}"
                        f"have been added to complete the last winding pack (nx"
                        f"={n_layers_max}, ny={n_turns_max})."
                    )

                R_wp_i -= n_turns_max * conductor.dy  # noqa: N806
                debug_msg.append(
                    f"n_layers_max: {n_layers_max}, n_turns_max: {n_turns_max}"
                )

        bluemira_debug("\n".join(debug_msg))
        self.WPs = WPs

        # just a final check
        if self.n_conductors != n_conductors:
            msg = (
                f"Mismatch in conductor count after rearrangement:\n "
                f"Expected: {n_conductors}, Obtained: {self.n_conductors}\n"
                f"Check winding pack construction and available space constraints."
            )
            bluemira_error(msg)
            raise ValueError(msg)

    def _tresca_stress(
        self, pm: float, fz: float, op_cond: OperationalConditions
    ) -> float:
        """
        Estimate the maximum principal (Tresca) stress on the inner case of the TF coil.

        The stress is computed as the sum of:
        - The hoop (circumferential) compressive stress due to radial magnetic pressure.
        - The vertical tensile stress due to axial electromagnetic forces on the inner
            leg.

        The hoop stress is derived using classical shell theory for isotropic continuous
        cylindrical shells, corrected by the stiffness distribution in the structure.

        Parameters
        ----------
        pm:
            Radial magnetic pressure acting on the case [Pa].
        fz:
            Vertical force acting on the inner leg of the case [N].
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material property.

        Returns
        -------
        :
            Estimated maximum stress [Pa] acting on the case nose (hoop + vertical
            contribution).
        """
        # The maximum principal stress acting on the case nose is the compressive
        # hoop stress generated in the equivalent shell from the magnetic pressure. From
        # the Shell theory, for an isotropic continuous shell with a thickness ratio:
        beta = self.geometry.variables.Rk.value / (
            self.geometry.variables.Rk.value + self.dy_vault
        )
        # the maximum hoop stress, corrected to account for the presence of the WP, is
        # placed at the innermost radius of the case as:
        sigma_theta = (
            2.0 / (1 - beta**2) * pm * self.Kx_vault(op_cond) / self.Kx(op_cond)
        )

        # In addition to the radial centripetal force, the second in-plane component
        # to be accounted is the vertical force acting on the TFC inner-leg.
        # t_z = 0.5*np.log(self.Ri / Re) * MU_0_4PI * (360. / self.theta_TF) * I ** 2

        # As conservative approximation, the vertical force is considered to act only
        # on jackets and vault
        sigma_z = fz / (self.area_case_jacket + self.area_wps_jacket)
        return sigma_theta + sigma_z

    def optimise_vault_radial_thickness(
        self,
        pm: float,
        fz: float,
        op_cond: OperationalConditions,
        allowable_sigma: float,
        bounds: np.array = None,
    ):
        """
        Optimise the vault radial thickness of the case

        Parameters
        ----------
        pm:
            The magnetic pressure applied along the radial direction (Pa).
        f_z:
            The force applied in the z direction, perpendicular to the case
            cross-section (N).
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material properties.
        allowable_sigma:
            The allowable stress (Pa) for the jacket material.
        bounds:
            Optional bounds for the jacket thickness optimisation (default is None).

        Returns
        -------
        :
            The result of the optimisation process containing information about the
            optimal vault thickness.

        Raises
        ------
        ValueError
            If the optimisation process did not converge.
        """
        method = None
        if bounds is not None:
            method = "bounded"

        result = minimize_scalar(
            fun=self._sigma_difference,
            args=(pm, fz, op_cond, allowable_sigma),
            bounds=bounds,
            method=method,
            options={"xatol": 1e-4},
        )

        if not result.success:
            raise ValueError("dy_vault optimisation did not converge.")
        self.dy_vault = result.x
        # print(f"Optimal dy_vault: {self.dy_vault}")
        # print(f"Tresca sigma: {self._tresca_stress(pm, fz, T=T, B=B) / 1e6} MPa")

        return result

    def _sigma_difference(
        self,
        dy_vault: float,
        pm: float,
        fz: float,
        op_cond: OperationalConditions,
        allowable_sigma: float,
    ) -> float:
        """
        Fitness function for the optimisation problem. It calculates the absolute
        difference between the Tresca stress and the allowable stress.

        Parameters
        ----------
        dy_vault:
            The thickness of the vault in the direction perpendicular to the
            applied pressure(m).
        pm:
            The magnetic pressure applied along the radial direction (Pa).
        fz:
            The force applied in the z direction, perpendicular to the case
            cross-section (N).
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material properties.
        allowable_sigma:
            The allowable stress (Pa) for the vault material.

        Returns
        -------
        :
            The absolute difference between the calculated Tresca stress and the
            allowable stress (Pa).

        Notes
        -----
        This function modifies the case's vault thickness
        using the value provided in jacket_thickness.
        """
        self.dy_vault = dy_vault
        sigma = self._tresca_stress(pm, fz, op_cond)
        # bluemira_print(f"sigma: {sigma}, allowable_sigma: {allowable_sigma},
        # diff: {sigma - allowable_sigma}")
        return abs(sigma - allowable_sigma)

    def optimise_jacket_and_vault(
        self,
        pm: float,
        fz: float,
        op_cond: OperationalConditions,
        allowable_sigma: float,
        bounds_cond_jacket: np.ndarray | None = None,
        bounds_dy_vault: np.ndarray | None = None,
        layout: str = "auto",
        wp_reduction_factor: float = 0.8,
        min_gap_x: float = 0.05,
        n_layers_reduction: int = 4,
        max_niter: int = 10,
        eps: float = 1e-8,
        n_conds: int | None = None,
    ):
        """
        Jointly optimise the conductor jacket and case vault thickness
        under electromagnetic loading constraints.

        This method performs an iterative optimisation of:
        - The cross-sectional area of the conductor jacket.
        - The vault radial thickness of the TF coil casing.

        The optimisation loop continues until the relative change in
        jacket area and vault thickness drops below the specified
        convergence threshold `eps`, or `max_niter` is reached.

        Parameters
        ----------
        pm:
            Radial magnetic pressure on the conductor [Pa].
        fz:
            Axial electromagnetic force on the winding pack [N].
        op_cond:
            Operational conditions including temperature, magnetic field, and strain
            at which to calculate the material properties.
        allowable_sigma:
            Maximum allowable stress for structural material [Pa].
        bounds_cond_jacket:
            Min/max bounds for conductor jacket area optimisation [m²].
        bounds_dy_vault:
            Min/max bounds for the case vault thickness optimisation [m].
        layout:
            Cable layout strategy; "auto" or predefined layout name.
        wp_reduction_factor:
            Reduction factor applied to WP footprint during conductor rearrangement.
        min_gap_x:
            Minimum spacing between adjacent conductors [m].
        n_layers_reduction:
            Number of conductor layers to remove when reducing WP height.
        max_niter:
            Maximum number of optimisation iterations.
        eps:
            Convergence threshold for the combined optimisation loop.
        n_conds:
            Target total number of conductors in the winding pack. If None, the self
            number of conductors is used.

        Notes
        -----
        The function modifies the internal state of `conductor` and `self.dy_vault`.
        """
        debug_msg = ["Method optimise_jacket_and_vault"]

        # Initialize convergence array
        self._convergence_array = []

        if n_conds is None:
            n_conds = self.n_conductors

        conductor = self.WPs[0].conductor

        self._check_WPs(self.WPs)

        i = 0
        err_conductor_area_jacket = 10000 * eps
        err_dy_vault = 10000 * eps
        tot_err = err_dy_vault + err_conductor_area_jacket

        self._convergence_array.append([
            i,
            conductor.dy_jacket,
            self.dy_vault,
            err_conductor_area_jacket,
            err_dy_vault,
            self.dy_wp_tot,
            self.geometry.variables.Ri.value - self.geometry.variables.Rk.value,
        ])

        damping_factor = 0.3

        while i < max_niter and tot_err > eps:
            i += 1
            debug_msg.append(f"Internal optimazion - iteration {i}")

            # Store current values
            cond_dx_jacket0 = conductor.dx_jacket
            case_dy_vault0 = self.dy_vault

            debug_msg.append(
                f"before optimisation: conductor jacket area = {conductor.area_jacket}"
            )
            cond_area_jacket0 = conductor.area_jacket
            t_z_cable_jacket = (
                fz
                * self.area_wps_jacket
                / (self.area_case_jacket + self.area_wps_jacket)
                / self.n_conductors
            )
            conductor.optimise_jacket_conductor(
                pm, t_z_cable_jacket, op_cond, allowable_sigma, bounds_cond_jacket
            )
            debug_msg.extend([
                f"t_z_cable_jacket: {t_z_cable_jacket}",
                f"after optimisation: conductor jacket area = {conductor.area_jacket}",
            ])

            conductor.dx_jacket = (
                1 - damping_factor
            ) * cond_dx_jacket0 + damping_factor * conductor.dx_jacket

            err_conductor_area_jacket = (
                abs(conductor.area_jacket - cond_area_jacket0) / cond_area_jacket0
            )

            self.rearrange_conductors_in_wp(
                n_conds,
                wp_reduction_factor,
                min_gap_x,
                n_layers_reduction,
                layout=layout,
            )

            debug_msg.append(f"before optimisation: case dy_vault = {self.dy_vault}")
            self.optimise_vault_radial_thickness(
                pm=pm,
                fz=fz,
                op_cond=op_cond,
                allowable_sigma=allowable_sigma,
                bounds=bounds_dy_vault,
            )

            self.dy_vault = (
                1 - damping_factor
            ) * case_dy_vault0 + damping_factor * self.dy_vault

            delta_case_dy_vault = abs(self.dy_vault - case_dy_vault0)
            err_dy_vault = delta_case_dy_vault / self.dy_vault
            tot_err = err_dy_vault + err_conductor_area_jacket

            debug_msg.append(
                f"after optimisation: case dy_vault = {self.dy_vault}\n"
                f"err_dy_jacket = {err_conductor_area_jacket}\n "
                f"err_dy_vault = {err_dy_vault}\n "
                f"tot_err = {tot_err}"
            )

            # Store iteration results in convergence array
            self._convergence_array.append([
                i,
                conductor.dy_jacket,
                self.dy_vault,
                err_conductor_area_jacket,
                err_dy_vault,
                self.dy_wp_tot,
                self.geometry.variables.Ri.value - self.geometry.variables.Rk.value,
            ])

        # final check
        if i < max_niter:
            bluemira_print(
                f"Optimisation of jacket and vault reached after "
                f"{i} iterations. Total error: {tot_err} < {eps}."
            )

            ax = self.plot(show=False, homogenised=False)
            ax.set_title("Case design after optimisation")
            plt.show()

        else:
            bluemira_warn(
                f"Maximum number of optimisation iterations {max_niter} "
                f"reached. A total of {tot_err} > {eps} has been obtained."
            )

    def plot_convergence(self):
        """
        Plot the evolution of thicknesses and error values over optimisation iterations.

        Raises
        ------
        RuntimeError
            If no convergence data available
        """
        if not hasattr(self, "_convergence_array") or not self._convergence_array:
            raise RuntimeError("No convergence data available. Run optimisation first.")

        convergence_data = np.array(self._convergence_array)

        iterations = convergence_data[:, 0]
        dy_jacket = convergence_data[:, 1]
        dy_vault = convergence_data[:, 2]
        err_dy_jacket = convergence_data[:, 3]
        err_dy_vault = convergence_data[:, 4]
        dy_wp_tot = convergence_data[:, 5]
        Ri_minus_Rk = convergence_data[:, 6]  # noqa: N806

        _, axs = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

        # Top subplot: Thicknesses
        axs[0].plot(iterations, dy_jacket, marker="o", label="dy_jacket [m]")
        axs[0].plot(iterations, dy_vault, marker="s", label="dy_vault [m]")
        axs[0].plot(iterations, dy_wp_tot, marker="^", label="dy_wp_tot [m]")
        axs[0].plot(iterations, Ri_minus_Rk, marker="v", label="Ri - Rk [m]")
        axs[0].set_ylabel("Thickness [m]")
        axs[0].set_title("Evolution of Jacket, Vault, and WP Thicknesses")
        axs[0].legend()
        axs[0].grid(visible=True)

        # Bottom subplot: Errors
        axs[1].plot(iterations, err_dy_jacket, marker="o", label="err_dy_jacket")
        axs[1].plot(iterations, err_dy_vault, marker="s", label="err_dy_vault")
        axs[1].set_ylabel("Relative Error")
        axs[1].set_xlabel("Iteration")
        axs[1].set_title("Evolution of Errors during Optimisation")
        axs[1].set_yscale("log")  # Log scale for better visibility if needed
        axs[1].legend()
        axs[1].grid(visible=True)

        plt.tight_layout()
        plt.show()


def create_case_tf_from_dict(
    case_dict: dict,
    name: str | None = None,
) -> CaseTF:
    """
    Factory function to create a CaseTF (or subclass) from a serialized dictionary.

    Parameters
    ----------
    case_dict:
        Serialized case dictionary, must include 'name_in_registry' field.
    name:
        Name to assign to the created case. If None, uses the name in the dictionary.

    Returns
    -------
    :
        A fully instantiated CaseTF (or subclass) object.

    Raises
    ------
    ValueError
        If no class is registered with the given name_in_registry.
    """
    name_in_registry = case_dict.get("name_in_registry")
    if name_in_registry is None:
        raise ValueError("CaseTF dictionary must include 'name_in_registry' field.")

    case_cls = CASETF_REGISTRY.get(name_in_registry)
    if case_cls is None:
        available = list(CASETF_REGISTRY.keys())
        raise ValueError(
            f"No registered CaseTF class with name_in_registry '{name_in_registry}'. "
            f"Available: {available}"
        )

    return case_cls.from_dict(
        name=name,
        case_dict=case_dict,
    )
