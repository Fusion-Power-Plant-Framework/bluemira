# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Plotting for balance of plant
"""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.sankey import Sankey
from scipy.optimize import minimize

from bluemira.base.constants import raw_uc
from bluemira.display.palettes import BLUEMIRA_PALETTE

if TYPE_CHECKING:
    from collections.abc import Iterable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


class SuperSankey(Sankey):
    """
    A sub-class of the Sankey diagram class from matplotlib, which is capable
    of connecting two blocks, instead of just one. This is done using a cute
    sledgehammer approach, using optimisation. Basically, the Sankey object
    is quite complex, and it makes it very hard to calculate the exact lengths
    required to connect two sub-diagrams.
    """

    def add(  # noqa: D102
        self,
        patchlabel: str = "",
        flows: Iterable[float] | None = None,
        orientations: Iterable[float] | None = None,
        labels: str | None | list[str | None] = "",
        trunklength: float = 1.0,
        pathlengths: float | list[float] = 0.25,
        prior: int | None = None,
        future: int | None = None,
        connect: tuple[int, int] | list[tuple[int, int]] = (0, 0),
        rotation: float = 0,
        **kwargs,
    ):
        __doc__ = super().__doc__  # noqa: A001, F841
        # Here we first check if the "add" method has received arguments that
        # the Sankey class can't handle.
        if future is None:
            # There is only one connection, Sankey knows how to do this
            super().add(
                patchlabel,
                flows,
                orientations,
                labels,
                trunklength,
                pathlengths,
                prior,
                connect,
                rotation,
                **kwargs,
            )
        else:
            # There are two connections, use new method
            self._double_connect(
                patchlabel,
                flows,
                orientations,
                labels,
                trunklength,
                pathlengths,
                prior,
                future,
                connect,
                rotation,
                **kwargs,
            )

    def _double_connect(
        self,
        patchlabel: str,
        flows: Iterable[float] | None,
        orientations: Iterable[float] | None,
        labels: str | None | list[str | None],
        trunklength: float,
        pathlengths: list[float],
        prior: int | None,
        future: int | None,
        connect: list[tuple[int, int]],
        rotation: float,
        **kwargs,
    ):
        """
        Handles two connections in a Sankey diagram.

        Parameters
        ----------
        future:
            The index of the diagram to connect to
        connect:
            The list of (int, int) connections.
            - connect[0] is a (prior, this) tuple indexing the flow of the
            prior diagram and the flow of this diagram to connect.
            - connect[1] is a (future, this) tuple indexing of the flow of the
            future diagram and the flow of this diagram to connect.

        See Also
        --------
        Sankey.add for a full description of the various args and kwargs

        """
        # Get the optimum deltas
        dx, dy = self._opt_connect(
            flows, orientations, prior, future, connect, trunklength=trunklength
        )
        # Replace
        pathlengths[0] = dx
        pathlengths[-1] = dy
        self.add(
            patchlabel=patchlabel,
            labels=labels,
            flows=flows,
            orientations=orientations,
            prior=prior,
            connect=connect[0],
            trunklength=trunklength,
            pathlengths=pathlengths,
            rotation=rotation,
            facecolor=kwargs.get("facecolor", None),
        )

    def _opt_connect(
        self,
        flows: Iterable[float] | None,
        orient: Iterable[float] | None,
        prior: int | None,
        future: int | None,
        connect: list[tuple[int, int]],
        trunklength: float,
    ) -> tuple[float, float]:
        """
        Optimises the second connection between Sankey diagrams.

        Returns
        -------
        dx:
            The x pathlength to use to match the tips
        dy:
            The y pathlength to use to match the tips

        Notes
        -----
        This is because Sankey is very complicated, and makes it hard to work
        out the positions of things prior to adding them to the diagrams.
        Because we are bizarrely using a plotting function as a minimisation
        objective, we need to make sure we clean the plot on every call.
        """
        future_index, this_f_index = connect[1]
        labels = [None] * len(flows)
        pathlengths = [0.0] * len(flows)

        # Make a local copy of the Sankey.extent attribute to override any
        # modifications during optimisation
        extent = deepcopy(self.extent)

        def minimise_dxdy(x_opt):
            """
            Minimisation function for the spatial difference between the target
            tip and the actual tip.

            Parameters
            ----------
            x_opt: array_like
                The vector of d_x, d_y delta-vectors to match tip positions

            Returns
            -------
            delta: float
                The sum of the absolute differences
            """
            tip2 = self.diagrams[future].tips[future_index]
            pathlengths[0] = x_opt[0]
            pathlengths[-1] = x_opt[1]
            self.add(
                trunklength=trunklength,
                pathlengths=pathlengths,
                flows=flows,
                prior=prior,
                connect=connect[0],
                orientations=orient,
                labels=labels,
                facecolor="#00000000",
            )
            new_tip = self.diagrams[-1].tips[this_f_index].copy()
            # Clean sankey plot
            self.diagrams.pop()
            self.ax.patches[-1].remove()
            return np.sum(np.abs(tip2 - new_tip))

        x0 = np.zeros(2)
        result = minimize(minimise_dxdy, x0, method="SLSQP")
        self.extent = extent  # Finish clean-up
        return result.x


BALANCE_PLOT_DEFAULTS = {
    # Matplotlib figure
    "facecolor": "k",
    "figsize": (14, 10),
    # Sankey scalings
    "scale": 0.001,
    "gap": 0.25,
    "radius": 0,
    "shoulder": 0,
    "head_angle": 150,
    "trunk_length": 0.7,
    "standard_length": 0.6,
    "medium_length": 1.0,
    # Text font, colour and size
    "unit": "MW",
    "format": "%.0f",
    "font_weight": "bold",
    "font_color": "white",
    "font_size": 14,
    "flow_font_size": 11,
}


class BalanceOfPlantPlotter:
    """
    The plotting object for the BalanceOfPlant system. Builds a relatively
    complicated Sankey diagram, connecting the various flows of energy in the
    reactor.
    """

    plot_options = deepcopy(BALANCE_PLOT_DEFAULTS)

    def __init__(self, **kwargs):
        self.plot_options = {**self.plot_options, **kwargs}

    def _scale_flows(self, flow_dict: dict[str, list[float]]) -> dict[str, list[float]]:
        plot_unit = self.plot_options.get("unit", "MW")
        flow_unit = "W"

        for k, v in flow_dict.items():
            flow_dict[k] = [raw_uc(vi, flow_unit, plot_unit) for vi in v]
        return flow_dict

    def plot(self, flow_dict: dict[str, list[float]], title: str = "") -> Axes:
        """
        Plots the BalanceOfPlant system, based on the inputs and flows.

        Parameters
        ----------
        inputs: dict
            The inputs to BalanceOfPlant (used here to format the title)
        op_mode: str
            The operation mode of the reactor
        flow_dict: dict
            The dictionary of flows for each of the Sankey diagrams.
        """
        flow_dict = self._scale_flows(flow_dict)
        # Build the base figure object
        fig = plt.figure(
            figsize=self.plot_options["figsize"],
            facecolor=self.plot_options["facecolor"],
        )
        ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
        plt.axis("off")

        sankey = self._build_diagram(
            flow_dict,
            SuperSankey(
                ax=ax,
                scale=self.plot_options["scale"],
                format=self.plot_options["format"],
                unit=self.plot_options["unit"],
                gap=self.plot_options["gap"],
                radius=self.plot_options["radius"],
                shoulder=self.plot_options["shoulder"],
                head_angle=self.plot_options["head_angle"],
            ),
        )
        self._polish(fig, sankey)
        fig.suptitle(
            title, color=self.plot_options["font_color"], fontsize=24, weight="bold"
        )
        return ax

    def _build_diagram(
        self, flow_dict: dict[str, list[float]], sankey: SuperSankey
    ) -> SuperSankey:
        """
        Builds the Sankey diagram. This is much more verbose than looping over
        some structs, but that's how it used to be and it was hard to modify.
        This is easier to read and modify.
        """
        trunk_length = self.plot_options["trunk_length"]
        l_s = self.plot_options["standard_length"]
        l_m = self.plot_options["medium_length"]

        # 0: Plasma
        sankey.add(
            patchlabel="Plasma",
            labels=["Fusion Power", None, "Neutrons", "Alphas + Aux"],
            flows=flow_dict["Plasma"],
            orientations=[0, -1, 0, -1],
            prior=None,
            connect=None,
            trunklength=trunk_length,
            pathlengths=[l_m, l_s / 1.5, l_s, l_s],
            facecolor=BLUEMIRA_PALETTE["blue"].as_hex(),
        )
        # 1: H&CD (first block)
        sankey.add(
            patchlabel="H&CD",
            labels=["", "H&CD power", "Losses"],
            flows=flow_dict["H&CD"],
            orientations=[-1, 1, -1],
            prior=0,
            connect=(1, 1),
            trunklength=trunk_length,
            pathlengths=[l_s, l_s / 1.5, l_s],
            facecolor=BLUEMIRA_PALETTE["pink"].as_hex(),
        )
        # 2: Neutrons
        sankey.add(
            patchlabel="Neutrons",
            labels=[None, "Energy Multiplication", "Blanket n", "Divertor n", "Aux n"],
            flows=flow_dict["Neutrons"],
            orientations=[0, 1, 0, -1, -1],
            prior=0,
            connect=(2, 0),
            trunklength=trunk_length,
            pathlengths=[l_s, l_s, l_s, 3 * l_m, l_m],
            facecolor=BLUEMIRA_PALETTE["orange"].as_hex(),
        )
        # 3: Radiation and separatrix
        sankey.add(
            patchlabel="Radiation and\nseparatrix",
            labels=[None, "", "Divertor rad and\n charged p"],
            flows=flow_dict["Radiation and \nseparatrix"],
            orientations=[1, 0, -1],
            prior=0,
            connect=(3, 0),
            trunklength=trunk_length,
            pathlengths=[l_s, l_s, l_s],
            facecolor=BLUEMIRA_PALETTE["red"].as_hex(),
        )
        # 4: Blanket
        sankey.add(
            patchlabel="Blanket",
            labels=[None, "", "", "Decay heat", ""],
            flows=flow_dict["Blanket"],
            orientations=[0, -1, -1, 1, 0],
            prior=2,
            connect=(2, 0),
            trunklength=trunk_length,
            pathlengths=[l_s, l_s, l_s, l_s, l_s],
            facecolor=BLUEMIRA_PALETTE["yellow"].as_hex(),
        )
        # 5: Divertor
        sankey.add(
            patchlabel="Divertor",
            labels=[None, None, ""],
            flows=flow_dict["Divertor"],
            orientations=[1, 0, 0],
            prior=2,
            connect=(3, 0),
            trunklength=trunk_length,
            pathlengths=[l_m, l_s, l_s],
            facecolor=BLUEMIRA_PALETTE["cyan"].as_hex(),
        )
        # 6: First wall
        sankey.add(
            patchlabel="First wall",
            labels=[None, "Auxiliary \n FW", None],
            flows=flow_dict["First wall"],
            orientations=[0, -1, 1],
            prior=3,
            future=4,
            connect=[(1, 0), (1, 2)],
            trunklength=trunk_length,
            pathlengths=[0, l_s, 0],
            facecolor=BLUEMIRA_PALETTE["grey"].as_hex(),
        )
        # 7: BoP
        sankey.add(
            patchlabel="BoP",
            labels=[None, None, "Losses", None],
            flows=flow_dict["BoP"],
            orientations=[0, -1, -1, 0],
            prior=4,
            connect=(4, 0),
            trunklength=trunk_length,
            pathlengths=[l_s, l_m, l_m, 0],
            facecolor=BLUEMIRA_PALETTE["purple"].as_hex(),
        )
        # 8: Electricity
        # Check if we have net electric power
        labels = [
            "$P_{el}$",
            "T plant",
            "P_oth...",
            "Cryoplant",
            "Magnets",
            None,
            None,
            "BB Pumping \n electrical \n power",
            "",
        ]
        orientations = [0, -1, -1, -1, -1, -1, -1, -1, 0]

        if flow_dict["Electricity"][-1] > 0:
            # Conversely, this means "net electric loss"
            labels[-1] = "Grid"
            orientations[-1] = 1

        sankey.add(
            patchlabel="Electricity",
            labels=labels,
            flows=flow_dict["Electricity"],
            orientations=orientations,
            prior=7,
            connect=(3, 0),
            trunklength=trunk_length,
            pathlengths=[
                l_m,
                2 * l_m,
                3 * l_m,
                4 * l_m,
                5 * l_m,
                7 * l_m,
                5 * l_m,
                3 * l_m,
                l_s,
            ],
            facecolor=BLUEMIRA_PALETTE["green"].as_hex(),
        )
        # 9: H&CD return leg
        sankey.add(
            patchlabel="",
            labels=[None, "H&CD Power"],
            flows=flow_dict["_H&CD loop"],
            orientations=[-1, 0],
            prior=8,
            connect=(5, 0),
            trunklength=trunk_length,
            pathlengths=[l_s / 2, 7 * l_m],
            facecolor=BLUEMIRA_PALETTE["green"].as_hex(),
        )
        # 10: Divertor (second block)
        sankey.add(
            patchlabel="",
            labels=[None, None],
            flows=flow_dict["_Divertor 2"],
            orientations=[1, 0],
            prior=3,
            future=5,
            connect=[(2, 0), (1, 1)],
            trunklength=trunk_length,
            pathlengths=[0, 0],
            facecolor=BLUEMIRA_PALETTE["cyan"].as_hex(),
        )
        # 11: H&CD return leg (second half)
        sankey.add(
            patchlabel="",
            labels=[None, None],
            flows=flow_dict["_H&CD loop 2"],
            orientations=[-1, 0],
            prior=9,
            future=1,
            connect=[(1, 0), (0, 1)],
            trunklength=trunk_length,
            pathlengths=[0, 0],
            facecolor=BLUEMIRA_PALETTE["green"].as_hex(),
        )
        # 12: Divertor back into BoP
        sankey.add(
            patchlabel="",
            labels=[None, "", ""],
            flows=flow_dict["_DIV to BOP"],
            orientations=[0, -1, 1],
            prior=5,
            future=7,
            connect=[(2, 0), (1, 2)],
            trunklength=trunk_length,
            pathlengths=[0, l_s / 2, 0],
            facecolor=BLUEMIRA_PALETTE["cyan"].as_hex(),
        )
        # 13: BB electrical pumping loss turn leg
        sankey.add(
            patchlabel="",
            labels=[None, "Losses", "BB coolant \n pumping"],
            flows=flow_dict["_BB coolant loop turn"],
            orientations=[0, 0, -1],
            prior=8,
            connect=(7, 0),
            trunklength=trunk_length,
            pathlengths=[l_s, l_s, l_m * 3],
            facecolor=BLUEMIRA_PALETTE["green"].as_hex(),
        )
        # 14: BB electrical pumping return leg into blanket
        sankey.add(
            patchlabel="",
            labels=[None, None],
            flows=flow_dict["_BB coolant loop blanket"],
            orientations=[0, -1],
            prior=13,
            future=4,
            connect=[(2, 0), (2, 1)],
            trunklength=trunk_length,
            pathlengths=[0, 0],
            facecolor=BLUEMIRA_PALETTE["green"].as_hex(),
        )
        # 15: Divertor electrical pumping loss turn leg
        sankey.add(
            patchlabel="",
            labels=[None, "Losses", "Div coolant \n pumping"],
            flows=flow_dict["_DIV coolant loop turn"],
            orientations=[0, 0, -1],
            prior=8,
            connect=(6, 0),
            trunklength=trunk_length,
            pathlengths=[l_s, l_s / 2, l_m],
            facecolor=BLUEMIRA_PALETTE["green"].as_hex(),
        )
        # 16: Divertor electrical pumping return into divertor
        sankey.add(
            patchlabel="",
            labels=[None, None],
            flows=flow_dict["_DIV coolant loop divertor"],
            orientations=[0, -1],
            prior=15,
            future=12,
            connect=[(2, 0), (1, 1)],
            trunklength=trunk_length,
            pathlengths=[0, 0],
            facecolor=BLUEMIRA_PALETTE["green"].as_hex(),
        )
        return sankey

    def _polish(self, fig: Figure, sankey: SuperSankey):
        """
        Finish up and polish figure, and format text
        """
        diagrams = sankey.finish()
        for diagram in diagrams:
            diagram.text.set_fontweight(self.plot_options["font_weight"])
            diagram.text.set_fontsize(self.plot_options["font_size"])
            diagram.text.set_color(self.plot_options["font_color"])
            for text in diagram.texts:
                text.set_fontsize(self.plot_options["flow_font_size"])
                text.set_color(self.plot_options["font_color"])

        fig.tight_layout()
