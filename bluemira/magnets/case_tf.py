# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Toroidal Field (TF) Coil 2D Case Class.

This class models and optimizes the cross-sectional layout of the inboard leg of a TF
coil.
It is designed to define and adjust the distribution of structural materials and
winding pack arrangement to achieve optimal performance and mechanical robustness.

Note:
- Focused on the two-dimensional analysis of the inboard leg.
"""

import math
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from bluemira.base.look_and_feel import (
    bluemira_debug,
    bluemira_error,
    bluemira_print,
    bluemira_warn,
)
from bluemira.magnets.registry import RegistrableMeta
from bluemira.magnets.utils import parall_k, serie_k
from bluemira.magnets.winding_pack import WindingPack, create_wp_from_dict
from bluemira.materials.cache import get_cached_material
from bluemira.materials.material import Material

# ------------------------------------------------------------------------------
# Global Registries
# ------------------------------------------------------------------------------
CASETF_REGISTRY = {}


# ------------------------------------------------------------------------------
# TFcoil cross section Geometry Base and Implementations
# ------------------------------------------------------------------------------
class CaseGeometry(ABC):
    """
    Abstract base class for TF case geometry profiles.

    Provides access to radial dimensions and toroidal width calculations
    as well as geometric plotting and area calculation interfaces.
    """

    def __init__(self, Ri: float, Rk: float, theta_TF: float):  # noqa: N803
        """
        Initialize the geometry base.

        Parameters
        ----------
        Ri : float
            External radius of the TF coil case [m].
        Rk : float
            Internal radius of the TF coil case [m].
        theta_TF : float
            Toroidal angular span of the TF coil [degrees].
        """
        self._Ri = None
        self.Ri = Ri

        self._Rk = None
        self.Rk = Rk

        self.theta_TF = theta_TF

    @property
    def Ri(self) -> float:  # noqa: N802
        """
        External (outermost) radius of the TF case at the top [m].

        Returns
        -------
        float
            Outer radius measured from the machine center to the case outer wall [m].
        """
        return self._Ri

    @Ri.setter
    def Ri(self, value: float):  # noqa: N802
        """
        Set the external (outermost) radius of the TF case.

        Parameters
        ----------
        value : float
            Outer radius [m]. Must be a strictly positive number.

        Raises
        ------
        ValueError
            If the provided radius is not positive.
        """
        if value <= 0:
            raise ValueError("Ri must be positive.")
        self._Ri = value

    @property
    def Rk(self) -> float:  # noqa: N802
        """
        Internal (innermost) radius of the TF case at the top [m].

        Returns
        -------
        float
            Inner radius measured from the machine center to the case outer wall [m].
        """
        return self._Rk

    @Rk.setter
    def Rk(self, value: float):  # noqa: N802
        """
        Set the internal (innermost) radius of the TF case.

        Parameters
        ----------
        value : float
            Outer radius [m]. Must be a strictly positive number.

        Raises
        ------
        ValueError
            If the provided radius is not positive.
        """
        if value < 0:
            raise ValueError("Rk must be positive.")
        self._Rk = value

    @property
    def theta_TF(self) -> float:
        """
        Toroidal angular span of the TF coil [degrees].

        Returns
        -------
        float
            Toroidal angular span [°].
        """
        return self._theta_TF

    @theta_TF.setter
    def theta_TF(self, value: float):
        """
        Set the toroidal angular span and update the internal radian representation.

        Parameters
        ----------
        value : float
            New toroidal angular span [degrees].

        Raises
        ------
        ValueError
            If the provided value is not within (0, 360] degrees.
        """
        if not (0.0 < value <= 360.0):  # noqa: PLR2004
            raise ValueError("theta_TF must be in the range (0, 360] degrees.")
        self._theta_TF = value
        self._rad_theta_TF = np.radians(value)

    @property
    def rad_theta_TF(self):
        """
        Toroidal angular span of the TF coil [radians].

        Returns
        -------
        float
            Toroidal aperture converted to radians.
        """
        return self._rad_theta_TF

    def dx_at_radius(self, radius: float) -> float:
        """
        Compute the toroidal width at a given radial position.

        Parameters
        ----------
        radius : float
            Radial position at which to compute the toroidal width [m].

        Returns
        -------
        float
            Toroidal width [m] at the given radius.
        """
        return 2 * radius * np.tan(self.rad_theta_TF / 2)

    @property
    @abstractmethod
    def area(self) -> float:
        """
        Compute the cross-sectional area of the TF case.

        Returns
        -------
        float
            Cross-sectional area [m²] enclosed by the case geometry.

        Notes
        -----
        Must be implemented by each specific geometry class.
        """

    @abstractmethod
    def plot(self, ax=None, *, show: bool = False) -> plt.Axes:
        """
        Plot the cross-sectional geometry of the TF case.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis on which to draw the geometry. If None, a new figure and axis are
            created.
        show : bool, optional
            If True, the plot is displayed immediately using plt.show().
            Default is False.

        Returns
        -------
        matplotlib.axes.Axes
            The axis object containing the plot.

        Notes
        -----
        Must be implemented by each specific geometry class.
        """


class TrapezoidalGeometry(CaseGeometry):
    """
    Geometry of a Toroidal Field (TF) coil case with trapezoidal cross-section.

    The coil cross-section has a trapezoidal shape: wider at the outer radius (Ri)
    and narrower at the inner radius (Rk), reflecting typical TF coil designs
    for magnetic and mechanical optimization.
    """

    @property
    def area(self) -> float:
        """
        Compute the cross-sectional area of the trapezoidal TF case.

        The area is calculated as the average of the toroidal widths at Ri and Rk,
        multiplied by the radial height (Ri - Rk).

        Returns
        -------
        float
            Cross-sectional area [m²].
        """
        return (
            0.5
            * (self.dx_at_radius(self.Ri) + self.dx_at_radius(self.Rk))
            * (self.Ri - self.Rk)
        )

    def build_polygon(self) -> np.ndarray:
        """
        Construct the (x, r) coordinates of the trapezoidal cross-section polygon.

        Returns
        -------
        np.ndarray
            Array of shape (4, 2) representing the corners of the trapezoid.
            Coordinates are ordered counterclockwise starting from the top-left corner:
            [(-dx_outer/2, Ri), (dx_outer/2, Ri), (dx_inner/2, Rk), (-dx_inner/2, Rk)].
        """
        dx_outer = self.dx_at_radius(self.Ri)
        dx_inner = self.dx_at_radius(self.Rk)

        return np.array([
            [-dx_outer / 2, self.Ri],
            [dx_outer / 2, self.Ri],
            [dx_inner / 2, self.Rk],
            [-dx_inner / 2, self.Rk],
        ])

    def plot(self, ax=None, *, show=False) -> plt.Axes:
        """
        Plot the trapezoidal cross-sectional shape of the TF case.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis object on which to draw the geometry. If None, a new figure and axis
            are created.
        show : bool, optional
            If True, the plot is immediately displayed using plt.show(). Default is
            False.

        Returns
        -------
        matplotlib.axes.Axes
            Axis object containing the plotted geometry.
        """
        if ax is None:
            _, ax = plt.subplots()
        poly = self.build_polygon()
        poly = np.vstack([poly, poly[0]])  # Close the polygon
        ax.plot(poly[:, 0], poly[:, 1], "k-", linewidth=2)
        ax.set_aspect("equal")
        if show:
            plt.show()
        return ax


class WedgedGeometry(CaseGeometry):
    """
    TF coil case shaped as a sector of an annulus (wedge with arcs).

    The geometry consists of two circular arcs (inner and outer radii)
    connected by radial lines, forming a wedge-like shape.
    """

    def area(self) -> float:
        """
        Compute the cross-sectional area of the wedge geometry.

        Returns
        -------
        float
            Cross-sectional area [m²] defined by the wedge between outer radius Ri
            and inner radius Rk over the toroidal angle theta_TF.
        """
        return 0.5 * self.rad_theta_TF * (self.Ri**2 - self.Rk**2)

    def build_polygon(self, n_points: int = 50) -> np.ndarray:
        """
        Build the polygon representing the wedge shape.

        The polygon is created by discretizing the outer and inner arcs
        into a series of points connected sequentially.

        Parameters
        ----------
        n_points : int, optional
            Number of points to discretize each arc. Default is 50.

        Returns
        -------
        np.ndarray
            Array of (x, y) coordinates [m] describing the wedge polygon.
        """
        theta1 = -self.rad_theta_TF / 2
        theta2 = -theta1

        angles_outer = np.linspace(theta1, theta2, n_points)
        angles_inner = np.linspace(theta2, theta1, n_points)

        arc_outer = np.column_stack((
            self.Ri * np.sin(angles_outer),
            self.Ri * np.cos(angles_outer),
        ))
        arc_inner = np.column_stack((
            self.Rk * np.sin(angles_inner),
            self.Rk * np.cos(angles_inner),
        ))

        return np.vstack((arc_outer, arc_inner))

    def plot(self, ax=None, *, show=False):
        """
        Plot the wedge-shaped TF coil case cross-section.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis on which to draw the geometry. If None, a new figure and axis are
            created.
        show : bool, optional
            If True, immediately display the plot with plt.show(). Default is False.

        Returns
        -------
        matplotlib.axes.Axes
            The axis object containing the plot.
        """
        if ax is None:
            _, ax = plt.subplots()
        poly = self.build_polygon()
        poly = np.vstack([poly, poly[0]])  # Close the polygon
        ax.plot(poly[:, 0], poly[:, 1], "k-", linewidth=2)
        ax.set_aspect("equal")
        if show:
            plt.show()
        return ax


# ------------------------------------------------------------------------------
# CaseTF Class
# ------------------------------------------------------------------------------
class BaseCaseTF(CaseGeometry, ABC, metaclass=RegistrableMeta):
    """
    Abstract Base Class for Toroidal Field Coil Case configurations.

    Defines the universal properties common to all TF case geometries.
    """

    _registry_ = CASETF_REGISTRY
    _name_in_registry_ = None

    def __init__(
        self,
        Ri: float,  # noqa: N803
        dy_ps: float,
        dy_vault: float,
        theta_TF: float,
        mat_case: Material,
        WPs: list[WindingPack],  # noqa: N803
        name: str = "BaseCaseTF",
    ):
        """
        Initialize a BaseCaseTF instance.

        Parameters
        ----------
        Ri : float
            External radius at the top of the TF coil case [m].
        dy_ps : float
            Radial thickness of the poloidal support region [m].
        dy_vault : float
            Radial thickness of the vault support region [m].
        theta_TF : float
            Toroidal angular aperture of the coil [degrees].
        mat_case : Material
            Structural material assigned to the TF coil case.
        WPs : list[WindingPack]
            List of winding pack objects embedded inside the TF case.
        name : str, optional
            String identifier for the TF coil case instance (default is "BaseCaseTF").
        """
        self._name = None
        self.name = name

        self._dy_ps = None
        self.dy_ps = dy_ps

        self._WPs = None
        self.WPs = WPs

        self._mat_case = None
        self.mat_case = mat_case

        self._Ri = None
        self.Ri = Ri

        self._theta_TF = None
        self.theta_TF = theta_TF

        # super().__init__(Ri=Ri, Rk=0, theta_TF=theta_TF)

        self._dy_vault = None
        self.dy_vault = dy_vault

    def set_wp(self, winding_packs: list[WindingPack]):
        self.WPs = winding_packs

    @property
    def name(self) -> str:
        """
        Name identifier of the TF case.

        Returns
        -------
        str
            Human-readable label for the coil case instance.
        """
        return self._name

    @name.setter
    def name(self, value: str):
        """
        Set the name of the TF case.

        Parameters
        ----------
        value : str
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
    def dy_ps(self) -> float:
        """
        Radial thickness of the poloidal support (PS) region [m].

        Returns
        -------
        float
            Thickness of the upper structural cap between the TF case wall and the
            first winding pack [m].
        """
        return self._dy_ps

    @dy_ps.setter
    def dy_ps(self, value: float):
        """
        Set the thickness of the poloidal support region.

        Parameters
        ----------
        value : float
            Poloidal support thickness [m].

        Raises
        ------
        ValueError
            If value is not positive.
        """
        if value <= 0:
            raise ValueError("dy_ps must be positive.")
        self._dy_ps = value

    @property
    def dy_vault(self) -> float:
        """
        Radial thickness of the vault support region [m].

        Returns
        -------
        float
            Thickness of the lower structural region supporting the winding packs [m].
        """
        return self._dy_vault

    @dy_vault.setter
    def dy_vault(self, value: float):
        """
        Set the thickness of the vault support region.

        Parameters
        ----------
        value : float
            Vault thickness [m].

        Raises
        ------
        ValueError
            If value is not positive.
        """
        if value <= 0:
            raise ValueError("dy_vault must be positive.")
        self._dy_vault = value

        self.Rk = self.R_wp_k[-1] - self._dy_vault

    @property
    @abstractmethod
    def dx_vault(self):
        """
        Average toroidal length of the vault.

        Returns
        -------
        float
            Average length of the vault in the toroidal direction [m].
        """

    @property
    def mat_case(self) -> Material:
        """
        Structural material assigned to the TF case.

        Returns
        -------
        Material
            Material object providing mechanical and thermal properties.
        """
        return self._mat_case

    @mat_case.setter
    def mat_case(self, value: Material):
        """
        Set the structural material assigned to the TF case.

        Parameters
        ----------
        value : Material
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
        list of WindingPack
            Winding pack instances composing the internal coil layout.
        """
        return self._WPs

    @WPs.setter
    def WPs(self, value: list[WindingPack]):  # noqa: N802
        """
        Set the winding pack objects list.

        Parameters
        ----------
        value : list[WindingPack]
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
        if hasattr(self, "dy_vault"):
            self.dy_vault = self.dy_vault

    @property
    def dx_i(self):
        """Toroidal length of the coil case at its maximum radial position [m]"""
        return 2 * self.Ri * np.tan(self._rad_theta_TF / 2)

    @property
    def dx_ps(self):
        """Average toroidal length of the ps plate [m]"""
        return (self.Ri + (self.Ri - self.dy_ps)) * np.tan(self._rad_theta_TF / 2)

    @property
    def n_conductors(self):
        """Total number of conductors in the winding pack."""
        return sum(w.n_conductors for w in self.WPs)

    @property
    def dy_wp_i(self) -> np.ndarray:
        """
        Computes the radial thickness of each winding pack.

        Returns
        -------
        np.ndarray
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
        float
            Total radial thickness [m] summed over all winding packs.
        """
        return sum(self.dy_wp_i)

    @property
    def R_wp_i(self) -> np.ndarray:  # noqa: N802
        """
        Compute the radial positions for the outer edge (start) of each winding pack.

        Returns
        -------
        np.ndarray
            Array of radial positions [m] corresponding to the outer edge of each WP.
        """
        dy_wp_cumsum = np.cumsum(np.concatenate(([0.0], self.dy_wp_i)))
        result_initial = self.Ri - self.dy_ps
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
        np.ndarray
            Array of radial positions [m] corresponding to the outer edge of
            each winding pack.
        """
        return self.R_wp_i - self.dy_wp_i

    @property
    def Rk(self) -> float:  # noqa: N802
        """
        Internal (innermost) radius of the TF case at the top [m].

        Returns
        -------
        float
            Inner radius measured from the machine center to the case outer wall [m].
        """
        return self._Rk

    @Rk.setter
    def Rk(self, value: float):  # noqa: N802
        """
        Set the internal (innermost) radius of the TF case.

        Parameters
        ----------
        value : float
            Outer radius [m]. Must be a strictly positive number.

        Raises
        ------
        ValueError
            If the provided radius is not positive.
        """
        if value < 0:
            raise ValueError("Rk must be positive.")
        self._Rk = value

        self._dy_vault = self.R_wp_k[-1] - self._Rk

    def plot(self, ax=None, *, show: bool = False, homogenized: bool = False):
        """
        Schematic plot of the TF case cross-section including winding packs.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis on which to draw the figure. If `None`, a new figure and axis will be
            created.
        show : bool, optional
            If `True`, displays the plot immediately using `plt.show()`.
            Default is `False`.
        homogenized : bool, optional
            If `True`, plots winding packs as homogenized blocks.
            If `False`, plots individual conductors inside WPs.
            Default is `False`.

        Returns
        -------
        matplotlib.axes.Axes
            The axis object containing the rendered plot.
        """
        if ax is None:
            _, ax = plt.subplots()
            ax.set_aspect("equal", adjustable="box")

        # --------------------------------------
        # Plot external case boundary (delegate)
        # --------------------------------------
        super().plot(ax=ax, show=False)

        # --------------------------------------
        # Plot winding packs
        # --------------------------------------
        for i, wp in enumerate(self.WPs):
            xc_wp = 0.0
            yc_wp = self.R_wp_i[i] - wp.dy / 2
            ax = wp.plot(xc=xc_wp, yc=yc_wp, ax=ax, homogenized=homogenized)

        # --------------------------------------
        # Finalize plot
        # --------------------------------------
        ax.set_xlabel("Toroidal direction [m]")
        ax.set_ylabel("Radial direction [m]")
        ax.set_title(f"TF Case Cross Section: {self.name}")

        if show:
            plt.show()

        return ax

    @property
    def area_case_jacket(self):
        """
        Area of the case jacket (excluding winding pack regions).

        Returns
        -------
        float
            Case jacket area [m²], computed as total area minus total WP area.
        """
        return self.area - self.area_wps

    @property
    def area_wps(self):
        """
        Total area occupied by all winding packs.

        Returns
        -------
        float
            Combined area of the winding packs [m²].
        """
        return np.sum([w.area for w in self.WPs])

    @property
    def area_wps_jacket(self):
        """
        Total jacket area of all winding packs.

        Returns
        -------
        float
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
        float
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
        n_conductors : int
            Total number of conductors to distribute.
        wp_reduction_factor : float
            Fractional reduction of available toroidal space for WPs.
        min_gap_x : float
            Minimum gap between the WP and the case boundary in toroidal direction [m].
        n_layers_reduction : int
            Number of layers to remove after each WP.
        layout : str, optional
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
        n_conductors : int
            Number of conductors to allocate.
        dx_WP : float
            Available toroidal width for the winding pack [m].
        dx_cond : float
            Toroidal width of a single conductor [m].
        dy_cond : float
            Radial height of a single conductor [m].
        layout : str
            Layout type:
            - "auto"    : no constraints
            - "layer"   : enforce even number of turns (ny % 2 == 0)
            - "pancake" : enforce even number of layers (nx % 2 == 0)

        Returns
        -------
        tuple[int, int]
            n_layers_max (nx), n_turns_max (ny)

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
    def optimize_vault_radial_thickness(
        self,
        pm: float,
        fz: float,
        T: float,  # noqa: N803
        B: float,
        allowable_sigma: float,
        bounds: np.ndarray = None,
    ):
        """
        Abstract method to optimize the radial thickness of the vault support region.

        Parameters
        ----------
        pm : float
            Radial magnetic pressure [Pa].
        fz : float
            Axial electromagnetic force [N].
        T : float
            Operating temperature [K].
        B : float
            Magnetic field strength [T].
        allowable_sigma : float
            Allowable maximum stress [Pa].
        bounds : np.ndarray, optional
            Optimization bounds for vault thickness [m].
        """

    def to_dict(self) -> dict:
        """
        Serialize the BaseCaseTF instance into a dictionary.

        Returns
        -------
        dict
            Serialized data representing the TF case, including geometry and material
            information.
        """
        return {
            "name_in_registry": getattr(
                self, "_name_in_registry_", self.__class__.__name__
            ),
            "name": self.name,
            "Ri": self.Ri,
            "dy_ps": self.dy_ps,
            "dy_vault": self.dy_vault,
            "theta_TF": self.theta_TF,
            "mat_case": self.mat_case.name,  # Assume Material has 'name' attribute
            "WPs": [wp.to_dict() for wp in self.WPs],
            # Assume each WindingPack implements to_dict()
        }

    @classmethod
    def from_dict(cls, case_dict: dict, name: str | None = None) -> "BaseCaseTF":
        """
        Deserialize a BaseCaseTF instance from a dictionary.

        Parameters
        ----------
        case_dict : dict
            Dictionary containing serialized TF case data.
        name : str, optional
            Optional name override for the new instance.

        Returns
        -------
        BaseCaseTF
            Reconstructed TF case instance.

        Raises
        ------
        ValueError
            If the 'name_in_registry' field does not match this class.
        """
        name_in_registry = case_dict.get("name_in_registry")
        expected_name_in_registry = getattr(cls, "_name_in_registry_", cls.__name__)

        if name_in_registry != expected_name_in_registry:
            raise ValueError(
                f"Cannot create {cls.__name__} from dictionary with name_in_registry "
                f"'{name_in_registry}'. Expected '{expected_name_in_registry}'."
            )

        mat_case = get_cached_material(case_dict["mat_case"])
        WPs = [create_wp_from_dict(wp_dict) for wp_dict in case_dict["WPs"]]  # noqa:N806

        return cls(
            Ri=case_dict["Ri"],
            dy_ps=case_dict["dy_ps"],
            dy_vault=case_dict["dy_vault"],
            theta_TF=case_dict["theta_TF"],
            mat_case=mat_case,
            WPs=WPs,
            name=name or case_dict.get("name"),
        )

    def __str__(self) -> str:
        """
        Generate a human-readable summary of the TF case.

        Returns
        -------
        str
            Multiline string summarizing key properties of the TF case.
        """
        return (
            f"CaseTF '{self.name}'\n"
            f"  - Ri: {self.Ri:.3f} m\n"
            f"  - Rk: {self.Rk:.3f} m\n"
            f"  - dy_ps: {self.dy_ps:.3f} m\n"
            f"  - dy_vault: {self.dy_vault:.3f} m\n"
            f"  - theta_TF: {self.theta_TF:.2f}°\n"
            f"  - Material: {self.mat_case.name}\n"
            f"  - Winding Packs: {len(self.WPs)} packs\n"
        )


class TrapezoidalCaseTF(BaseCaseTF, TrapezoidalGeometry):
    """
    Toroidal Field Coil Case with Trapezoidal Geometry.
    Note: this class considers a set of Winding Pack with the same conductor (instance).
    """

    _registry_ = CASETF_REGISTRY
    _name_in_registry_ = "TrapezoidalCaseTF"

    def __init__(
        self,
        Ri: float,  # noqa: N803
        dy_ps: float,
        dy_vault: float,
        theta_TF: float,
        mat_case: Material,
        WPs: list[WindingPack],  # noqa: N803
        name: str = "TrapezoidalCaseTF",
    ):
        self._check_WPs(WPs)

        super().__init__(
            Ri=Ri,
            dy_ps=dy_ps,
            dy_vault=dy_vault,
            theta_TF=theta_TF,
            mat_case=mat_case,
            WPs=WPs,
            name=name,
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
        WPs : list of WindingPack
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
        float
            Average length of the vault in the toroidal direction [m].
        """
        return (self.R_wp_k[-1] + self.Rk) * np.tan(self.rad_theta_TF / 2)

    def Kx_ps(self, **kwargs):  # noqa: N802
        """
        Compute the equivalent radial stiffness of the poloidal support (PS) region.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments passed to the material's Young's modulus
            function.

        Returns
        -------
        float
            Equivalent radial stiffness of the poloidal support [Pa].
        """
        return self.mat_case.E(**kwargs) * self.dy_ps / self.dx_ps

    def Kx_lat(self, **kwargs):  # noqa: N802
        """
        Compute the equivalent radial stiffness of the lateral case sections.

        These are the mechanical links between each winding pack and the outer case.
        Each lateral segment is approximated as a rectangular element.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments passed to the material's Young's modulus function.

        Returns
        -------
        np.ndarray
            Array of radial stiffness values for each lateral segment [Pa].
        """
        dx_lat = np.array([
            (self.R_wp_i[i] + self.R_wp_k[i]) / 2 * np.tan(self.rad_theta_TF / 2)
            - w.dx / 2
            for i, w in enumerate(self.WPs)
        ])
        dy_lat = np.array([w.dy for w in self.WPs])
        return self.mat_case.E(**kwargs) * dy_lat / dx_lat

    def Kx_vault(self, **kwargs):  # noqa: N802
        """
        Compute the equivalent radial stiffness of the vault region.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments passed to the material's Young's modulus function.

        Returns
        -------
        float
            Equivalent radial stiffness of the vault [Pa].
        """
        return self.mat_case.E(**kwargs) * self.dy_vault / self.dx_vault

    def Kx(self, **kwargs):  # noqa: N802
        """
        Compute the total equivalent radial stiffness of the entire case structure.

        Combines:
        - Two lateral case elements (left and right) per winding pack in series
            with the WP
        - All such combined units in parallel with the PS and vault regions

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments passed to subcomponent stiffness evaluations.

        Returns
        -------
        float
            Total equivalent radial stiffness of the TF case [Pa].
        """
        temp = [
            serie_k([
                self.Kx_lat(**kwargs)[i],
                w.Kx(**kwargs),
                self.Kx_lat(**kwargs)[i],
            ])
            for i, w in enumerate(self.WPs)
        ]
        return parall_k([self.Kx_ps(**kwargs), self.Kx_vault(**kwargs), *temp])

    def Ky_ps(self, **kwargs):  # noqa: N802
        """
        Compute the equivalent toroidal stiffness of the poloidal support (PS) region.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments passed to the material's Young's modulus function.

        Returns
        -------
        float
            Equivalent toroidal stiffness of the PS region [Pa].
        """
        return self.mat_case.E(**kwargs) * self.dx_ps / self.dy_ps

    def Ky_lat(self, **kwargs):  # noqa: N802
        """
        Compute the equivalent toroidal stiffness of lateral case sections
        per winding pack.

        Each lateral piece is treated as a rectangular beam in the toroidal direction.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments passed to the material's Young's modulus function.

        Returns
        -------
        np.ndarray
            Array of toroidal stiffness values for each lateral segment [Pa].
        """
        dx_lat = np.array([
            (self.R_wp_i[i] + self.R_wp_k[i]) / 2 * np.tan(self._rad_theta_TF / 2)
            - w.dx / 2
            for i, w in enumerate(self.WPs)
        ])
        dy_lat = np.array([w.dy for w in self.WPs])
        return self.mat_case.E(**kwargs) * dx_lat / dy_lat

    def Ky_vault(self, **kwargs):  # noqa: N802
        """
        Compute the equivalent toroidal stiffness of the vault region.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments passed to the material's Young's modulus function.

        Returns
        -------
        float
            Equivalent toroidal stiffness of the vault [Pa].
        """
        return self.mat_case.E(**kwargs) * self.dx_vault / self.dy_vault

    def Ky(self, **kwargs):  # noqa: N802
        """
        Compute the total equivalent toroidal stiffness of the entire case structure.

        Combines:
        - Each winding pack and its adjacent lateral case sections in parallel
        - These parallel combinations are arranged in series with the PS and
            vault regions

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments passed to subcomponent stiffness evaluations.

        Returns
        -------
        float
            Total equivalent toroidal stiffness of the TF case [Pa].
        """
        temp = [
            parall_k([
                self.Ky_lat(**kwargs)[i],
                w.Ky(**kwargs),
                self.Ky_lat(**kwargs)[i],
            ])
            for i, w in enumerate(self.WPs)
        ]
        return serie_k([self.Ky_ps(**kwargs), self.Ky_vault(**kwargs), *temp])

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
        n_conductors : int
            Total number of conductors to be allocated.
        wp_reduction_factor : float
            Fractional reduction of the total available toroidal space for WPs.
        min_gap_x : float
            Minimum allowable toroidal gap between WP and boundary [m].
        n_layers_reduction : int
            Number of horizontal layers to reduce after each WP.
        layout : str, optional
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
        debug_msg.extend(f"dx_WP = {dx_WP}")
        debug_msg.extend(f"self.dx_i = {self.dx_i}")
        debug_msg.extend(f"wp_reduction_factor = {wp_reduction_factor}")
        debug_msg.extend(f"min_gap_x = {min_gap_x}")
        debug_msg.extend(f"n_layers_reduction = {n_layers_reduction}")
        debug_msg.extend(f"layout = {layout}")
        debug_msg.extend(f"n_conductors = {n_conductors}")

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
                    n_layers_max = math.floor(dx_WP / conductor.dx / 2.0) * 2
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

                gap_0 = R_wp_i * np.tan(self.rad_theta_TF / 2) - dx_WP / 2
                gap_1 = min_gap_x

                max_dy = (gap_0 - gap_1) / np.tan(self.rad_theta_TF / 2)
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

    def _tresca_stress(self, pm: float, fz: float, **kwargs):
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
        pm : float
            Radial magnetic pressure acting on the case [Pa].
        fz : float
            Vertical force acting on the inner leg of the case [N].
        **kwargs :
            Additional keyword arguments forwarded to the stiffness calculations (e.g.,
            temperature, material model switches).

        Returns
        -------
        float
            Estimated maximum stress [Pa] acting on the case nose (hoop + vertical
            contribution).
        """
        # The maximum principal stress acting on the case nose is the compressive
        # hoop stress generated in the equivalent shell from the magnetic pressure. From
        # the Shell theory, for an isotropic continuous shell with a thickness ratio:
        beta = self.Rk / (self.Rk + self.dy_vault)
        # the maximum hoop stress, corrected to account for the presence of the WP, is
        # placed at the innermost radius of the case as:
        sigma_theta = (
            2.0 / (1 - beta**2) * pm * self.Kx_vault(**kwargs) / self.Kx(**kwargs)
        )

        # In addition to the radial centripetal force, the second in-plane component
        # to be accounted is the vertical force acting on the TFC inner-leg.
        # t_z = 0.5*np.log(self.Ri / Re) * MU_0_4PI * (360. / self.theta_TF) * I ** 2

        # As conservative approximation, the vertical force is considered to act only
        # on jackets and vault
        sigma_z = fz / (self.area_case_jacket + self.area_wps_jacket)
        return sigma_theta + sigma_z

    def optimize_vault_radial_thickness(
        self,
        pm: float,
        fz: float,
        T: float,  # noqa: N803
        B: float,
        allowable_sigma: float,
        bounds: np.array = None,
    ):
        """
        Optimize the vault radial thickness of the case

        Parameters
        ----------
        pm :
            The magnetic pressure applied along the radial direction (Pa).
        f_z :
            The force applied in the z direction, perpendicular to the case
            cross-section (N).
        T :
            The operating temperature (K).
        B :
            The operating magnetic field (T).
        allowable_sigma :
            The allowable stress (Pa) for the jacket material.
        bounds :
            Optional bounds for the jacket thickness optimization (default is None).

        Returns
        -------
            The result of the optimization process containing information about the
            optimal vault thickness.

        Raises
        ------
        ValueError
            If the optimization process did not converge.
        """
        method = None
        if bounds is not None:
            method = "bounded"

        result = minimize_scalar(
            fun=self._sigma_difference,
            args=(pm, fz, T, B, allowable_sigma),
            bounds=bounds,
            method=method,
            options={"xatol": 1e-4},
        )

        if not result.success:
            raise ValueError("dy_vault optimization did not converge.")
        self.dy_vault = result.x
        # print(f"Optimal dy_vault: {self.dy_vault}")
        # print(f"Tresca sigma: {self._tresca_stress(pm, fz, T=T, B=B) / 1e6} MPa")

        return result

    def _sigma_difference(
        self,
        dy_vault: float,
        pm: float,
        fz: float,
        temperature: float,
        B: float,
        allowable_sigma: float,
    ):
        """
        Fitness function for the optimization problem. It calculates the absolute
        difference between the Tresca stress and the allowable stress.

        Parameters
        ----------
        dy_vault :
            The thickness of the vault in the direction perpendicular to the
            applied pressure(m).
        pm :
            The magnetic pressure applied along the radial direction (Pa).
        fz :
            The force applied in the z direction, perpendicular to the case
            cross-section (N).
        temperature :
            The temperature (K) at which the conductor operates.
        B :
            The magnetic field (T) at which the conductor operates.
        allowable_sigma :
            The allowable stress (Pa) for the vault material.

        Returns
        -------
            The absolute difference between the calculated Tresca stress and the
            allowable stress (Pa).

        Notes
        -----
            This function modifies the case's vault thickness
            using the value provided in jacket_thickness.
        """
        self.dy_vault = dy_vault
        sigma = self._tresca_stress(pm, fz, temperature=temperature, B=B)
        # bluemira_print(f"sigma: {sigma}, allowable_sigma: {allowable_sigma},
        # diff: {sigma - allowable_sigma}")
        return abs(sigma - allowable_sigma)

    def optimize_jacket_and_vault(
        self,
        pm: float,
        fz: float,
        temperature: float,
        B: float,
        allowable_sigma: float,
        bounds_cond_jacket: np.array = None,
        bounds_dy_vault: np.array = None,
        layout: str = "auto",
        wp_reduction_factor: float = 0.8,
        min_gap_x: float = 0.05,
        n_layers_reduction: int = 4,
        max_niter: int = 10,
        eps: float = 1e-8,
        n_conds: int | None = None,
    ):
        """
        Jointly optimize the conductor jacket and case vault thickness
        under electromagnetic loading constraints.

        This method performs an iterative optimization of:
        - The cross-sectional area of the conductor jacket.
        - The vault radial thickness of the TF coil casing.

        The optimization loop continues until the relative change in
        jacket area and vault thickness drops below the specified
        convergence threshold `eps`, or `max_niter` is reached.

        Parameters
        ----------
        pm : float
            Radial magnetic pressure on the conductor [Pa].
        fz : float
            Axial electromagnetic force on the winding pack [N].
        temperature : float
            Operating temperature [K].
        B : float
            Operating magnetic field [T].
        allowable_sigma : float
            Maximum allowable stress for structural material [Pa].
        bounds_cond_jacket : np.ndarray, optional
            Min/max bounds for conductor jacket area optimization [m²].
        bounds_dy_vault : np.ndarray, optional
            Min/max bounds for the case vault thickness optimization [m].
        layout : str, optional
            Cable layout strategy; "auto" or predefined layout name.
        wp_reduction_factor : float, optional
            Reduction factor applied to WP footprint during conductor rearrangement.
        min_gap_x : float, optional
            Minimum spacing between adjacent conductors [m].
        n_layers_reduction : int, optional
            Number of conductor layers to remove when reducing WP height.
        max_niter : int, optional
            Maximum number of optimization iterations.
        eps : float, optional
            Convergence threshold for the combined optimization loop.
        n_conds : int, optional
            Target total number of conductors in the winding pack. If None, the self
            number of conductors is used.

        Notes
        -----
            The function modifies the internal state of `conductor` and `self.dy_vault`.
        """
        debug_msg = ["Method optimize_jacket_and_vault"]

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
            self.Ri - self.Rk,
        ])

        damping_factor = 0.3

        while i < max_niter and tot_err > eps:
            i += 1
            debug_msg.append(f"Internal optimazion - iteration {i}")

            # Store current values
            cond_dx_jacket0 = conductor.dx_jacket
            case_dy_vault0 = self.dy_vault

            debug_msg.append(
                f"before optimization: conductor jacket area = {conductor.area_jacket}"
            )
            cond_area_jacket0 = conductor.area_jacket
            t_z_cable_jacket = (
                fz
                * self.area_wps_jacket
                / (self.area_case_jacket + self.area_wps_jacket)
                / self.n_conductors
            )
            conductor.optimize_jacket_conductor(
                pm, t_z_cable_jacket, temperature, B, allowable_sigma, bounds_cond_jacket
            )
            debug_msg.extend(f"t_z_cable_jacket: {t_z_cable_jacket}")
            debug_msg.extend(
                f"after optimization: conductor jacket area = {conductor.area_jacket}"
            )

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

            debug_msg.extend(f"before optimization: case dy_vault = {self.dy_vault}")
            self.optimize_vault_radial_thickness(
                pm=pm,
                fz=fz,
                T=temperature,
                B=B,
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
                f"after optimization: case dy_vault = {self.dy_vault}\n"
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
                self.Ri - self.Rk,
            ])

        # final check
        if i < max_niter:
            bluemira_print(
                f"Optimization of jacket and vault reached after "
                f"{i} iterations. Total error: {tot_err} < {eps}."
            )

            ax = self.plot(show=False, homogenized=False)
            ax.set_title("Case design after optimization")
            plt.show()

        else:
            bluemira_warn(
                f"Maximum number of optimization iterations {max_niter} "
                f"reached. A total of {tot_err} > {eps} has been obtained."
            )

    def plot_convergence(self):
        """
        Plot the evolution of thicknesses and error values over optimization iterations.

        Raises
        ------
        RuntimeError
            If no convergence data available
        """
        if not hasattr(self, "_convergence_array") or not self._convergence_array:
            raise RuntimeError("No convergence data available. Run optimization first.")

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
        axs[1].set_title("Evolution of Errors during Optimization")
        axs[1].set_yscale("log")  # Log scale for better visibility if needed
        axs[1].legend()
        axs[1].grid(visible=True)

        plt.tight_layout()
        plt.show()
