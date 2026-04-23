# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Definition of a simple panelling optimisation problem for neutronics.
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.look_and_feel import bluemira_print, bluemira_warn
from bluemira.codes.error import FreeCADError
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import revolve_shape, split_open_wire_at_coords

if TYPE_CHECKING:
    from bluemira.geometry.wire import BluemiraWire


class SimplePanellingOptProblem:
    """
    Optimisation problem to minimise the change in
    panel areas (or volume)  panelling a wire,
    assuming the wire is simple.
    """

    def __init__(
        self,
        wire: BluemiraWire,
        scale: float,
        opt_config: dict,
        *,
        by_volume: bool = False,
    ):
        self.wire = wire
        self.scale = scale
        self.by_volume = by_volume
        self.max_fractional_change = opt_config.get("max_fractional_change", 0.05)
        self.plot_convergence = opt_config.get("plot_convergence", False)
        self.min_dscr, self.max_dscr, self.step = opt_config.get(
            "discretisation_range", (10, 100, 10)
        )
        self.x0 = np.array([self.min_dscr], dtype=float)

    def objective(self, x: np.ndarray) -> float:
        """
        Objective function to minimise total deviated area or volume.

        Parameters
        ----------
        x : np.ndarray
            Array with a single value indicating the number of discretisation points.

        Returns
        -------
        float
            Total 'deviated' panel areas or volumes.
        """
        if self.by_volume:
            return self.volume_objective(x)
        return self.area_objective(x)

    def area_objective(self, x: np.ndarray) -> float:
        """
        Objective function to minimise total deviated area.

        Parameters
        ----------
        x : np.ndarray
            Array with a single value indicating the number of discretisation points.

        Returns
        -------
        float
            Total 'deviated' panel areas / Total Area
        """
        n_panels = int(np.clip(round(x[0]), self.min_dscr, self.max_dscr))

        panel_points = self.wire.discretise(ndiscr=n_panels, byedges=True).T
        panel_wires = split_open_wire_at_coords(
            self.wire, Coordinates(panel_points[1:-1])
        )

        deviated_panel_areas = []
        for wire in panel_wires:
            closed_wire = deepcopy(wire)
            closed_wire.close()
            deviated_panel_areas.append(BluemiraFace(closed_wire).area)

        return sum(deviated_panel_areas) / self.scale

    def volume_objective(self, x: np.ndarray) -> float:
        """
        Objective function to minimise total deviated volume.

        (slightly accurate, but slower than the area_objective)

        Parameters
        ----------
        x : np.ndarray
            Array with a single value indicating the number of discretisation points.

        Returns
        -------
        float
            Total 'deviated' panel volumes / Total Vol
        """
        n_panels = int(np.clip(round(x[0]), self.min_dscr, self.max_dscr))

        panel_points = self.wire.discretise(ndiscr=n_panels, byedges=True).T
        panel_wires = split_open_wire_at_coords(
            self.wire, Coordinates(panel_points[1:-1])
        )

        deviated_panel_volumes = []
        for wire in panel_wires:
            closed_wire = deepcopy(wire)
            closed_wire.close()
            revolved_solid = revolve_shape(
                BluemiraFace(closed_wire), [0, 0, 0], [0, 0, 1], 360
            )
            deviated_panel_volumes.append(revolved_solid.volume)

        return sum(deviated_panel_volumes) / self.scale

    def grid_search_n_panels(self) -> int:
        """
        Grid search to find the number of panels balancing
        objective reduction and complexity
        using the elbow method with early stopping based on
        relative improvement and max deviated fraction.

        Returns
        -------
        int
            Best number of panels.
        """
        history_n = []
        history_obj = []
        # Sampling instead of full grid search
        # Assuming a smooth change in the objective

        n_samples = max(30, round((self.max_dscr - self.min_dscr) / 3))
        sample_points = np.unique(
            np.linspace(self.min_dscr, self.max_dscr, n_samples, dtype=int)
        )
        for n_panels in sample_points:
            try:
                obj_val = self.objective(np.array([n_panels], dtype=float))
            except FreeCADError:
                # Just skip
                continue

            history_n.append(n_panels)
            history_obj.append(obj_val)

        best_n_panels = self._find_elbow(history_n, history_obj)

        if self.plot_convergence:
            self.plot_history(history_n, history_obj, best_n_panels)

        return best_n_panels

    def _find_elbow(self, history_n: list[int], history_obj: list[float]) -> int:
        """
        Find elbow point from history data using elbow technique

        Parameters
        ----------
        history_n : list[int]
            Number of panels tested.
        history_obj : list[float]
            Corresponding objective values.

        Returns
        -------
        int
            Selected number of panels by elbow and max deviated fraction criteria.
        """
        points = np.column_stack((history_n, history_obj))
        start, end = points[0], points[-1]
        line_vec = end - start
        line_vec_norm = line_vec / np.linalg.norm(line_vec)
        vec_from_start = points - start
        dist_to_line = np.abs(
            vec_from_start[:, 0] * line_vec_norm[1]
            - vec_from_start[:, 1] * line_vec_norm[0]
        )
        elbow_idx = np.argmax(dist_to_line)
        best_n_panels = history_n[elbow_idx]
        best_obj = history_obj[elbow_idx]

        if best_obj < self.max_fractional_change:
            bluemira_print(f"Best number of panels by elbow method: {best_n_panels}")
            return best_n_panels

        # Find all objectives <= max_fractional_change
        below_tol_indices = [
            i for i, obj in enumerate(history_obj) if obj <= self.max_fractional_change
        ]
        if below_tol_indices:
            chosen_n = history_n[
                below_tol_indices[0]
            ]  # largest n_panels with obj <= max_fractional_change
            bluemira_warn(
                f"Elbow objective {best_obj:.6f} above max deviated fraction. "
                f"Returning panel count {chosen_n} just below threshold."
            )
            return chosen_n

        # No objectives below max_fractional_change; pick panel with minimum objective
        chosen_n = history_n[int(np.argmin(history_obj))]
        bluemira_warn(
            f"Elbow objective {best_obj:.6f} above max deviated fraction "
            f"and no objectives below threshold. "
            f"Returning panel count {chosen_n} with minimum objective."
        )
        return chosen_n

    def plot_history(
        self, history_n: list[int], history_obj: list[float], best_n_panels: int
    ) -> None:
        """
        Plot the convergence history and mark the elbow point.

        Parameters
        ----------
        history_n : list[int]
        history_obj : list[float]
        best_n_panels : int
        """
        plt.figure(figsize=(8, 5))

        # Plot sampled points
        plt.plot(
            history_n, [100 * v for v in history_obj], "o", label="Sampled Objectives"
        )
        # Plot horizontal tolerance line
        plt.axhline(
            self.max_fractional_change * 100,
            color="red",
            linestyle="--",
            label=f"Tolerance ({self.max_fractional_change * 100}%)",
        )

        # Mark Best Panels
        plt.axvline(
            best_n_panels,
            color="blue",
            linestyle="--",
            label=f"Selected n_panel ={best_n_panels}",
        )

        plt.xlabel("Number of Panels (n_panels)")
        ylabel = "Change in volume (%)" if self.by_volume else "Change in area (%)"
        plt.ylabel(ylabel)

        plt.legend()
        plt.tight_layout()
        plt.show()
