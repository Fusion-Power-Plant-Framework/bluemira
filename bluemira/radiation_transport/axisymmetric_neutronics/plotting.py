# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Plotting Routines"""

from pathlib import Path

import matplotlib.pyplot as plt

from bluemira.base.components import Component
from bluemira.base.look_and_feel import bluemira_print
from bluemira.display.plotter import plot_2d


def plot_desplining_comparisons(
    orig_comps: list[Component],
    desplined_comps: list[Component],
    output_path: Path | None = None,
):
    """
    Create comparison plots in the xz plane before and after
    desplining components.

    Raises
    ------
    ValueError
        if original and desplined components do not have the
        same length
    """
    output_path.mkdir(parents=True, exist_ok=True)

    if len(orig_comps) != len(desplined_comps):
        raise ValueError("original and desplined components must have the same length.")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 6),
        sharex=True,
        sharey=True,
    )

    for orig_comp, desplined_comp in zip(
        orig_comps,
        desplined_comps,
        strict=True,
    ):
        orig_comp.plot_2d(ax=axes[0], show=False)
        for wire in orig_comp.leaves[0].shape.boundary:
            plot_2d(wire.vertexes.T, ax=axes[0], show=False)

        desplined_comp.plot_2d(ax=axes[1], show=False)
        for wire in desplined_comp.leaves[0].shape.boundary:
            plot_2d(wire.vertexes.T, ax=axes[1], show=False)

    axes[0].set_title("Original")
    axes[1].set_title("Desplined")

    fig.tight_layout()

    plot_path = output_path / "desplining_comparisons.png"
    fig.savefig(plot_path, bbox_inches="tight")

    bluemira_print(f"Plotted desplining comparisons as {plot_path}")
