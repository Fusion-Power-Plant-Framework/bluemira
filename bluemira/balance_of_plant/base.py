# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
Base classes
"""

import matplotlib.pyplot as plt

from bluemira.balance_of_plant.error import BalanceOfPlantError
from bluemira.balance_of_plant.tools import SuperSankey


class PowerFlow:
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.name}, {self.value}"


class PowerConnection:
    def __init__(self, block_1, block_2, flow_name):
        self.block_1 = block_1
        self.block_2 = block_2
        self.flow_name = flow_name
        # Plotting utility... :(
        self.indices = (len(block_1.flow_names) - 1, len(block_2.flow_names) - 1)

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.block_1.name} --{self.flow_name}--> {self.block_2.name}"


class PowerBlock:
    """
    Class for a single entity of power balance.
    """

    def __init__(self, name):
        self.name = name
        self.in_flows = {}
        self.out_flows = {}
        self.flow_names = []
        self.flow_values = []

    def _add_flow(self, power_flow, positive=True):
        direction = 1 if positive else -1
        self.flow_names.append(power_flow.name)
        self.flow_values.append(direction * power_flow.value)

    def add_in_flow(self, power_flow):
        self.in_flows[power_flow.name] = power_flow
        self._add_flow(power_flow)

    def add_out_flow(self, power_flow):
        self.out_flows[power_flow.name] = power_flow
        self._add_flow(power_flow, positive=False)

    @property
    def total_in(self):
        return sum(flow.value for flow in self.in_flows.values())

    @property
    def total_out(self):
        return sum(flow.value for flow in self.out_flows.values())

    @property
    def remainder(self):
        return self.total_in - self.total_out

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.name}"


class PowerBalance:
    """
    Class for assembling a power balance between various PowerBlocks.
    """

    def __init__(self, blocks=None):
        self.blocks = {}
        self.connections = []
        if blocks is not None:
            for block in blocks:
                self.add_power_block(block)

    def add_power_block(self, block):
        if block.name in self.blocks:
            raise BalanceOfPlantError(
                f"There is already a PowerBlock: {block.name} in PowerBalance."
            )

        self.blocks[block.name] = block

    def connect_blocks(self, block_1, block_2, flow_name, flow_value):
        for name in [block_1.name, block_2.name]:
            if name not in self.blocks:
                raise BalanceOfPlantError(f"No PowerBlock name: {name} in PowerBalance.")

        self.blocks[block_1.name].add_out_flow(PowerFlow(flow_name, flow_value))
        self.blocks[block_2.name].add_in_flow(PowerFlow(flow_name, flow_value))
        self.connections.append(
            PowerConnection(
                self.blocks[block_1.name], self.blocks[block_2.name], flow_name
            )
        )


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
    "trunklength": 0.7,
    "standardlength": 0.6,
    "mediumlength": 1.0,
    # Text font, colour and size
    "unit": "MW",
    "format": "%.0f",
    "fontweight": "bold",
    "fontcolor": "white",
    "fontsize": 14,
    "flowfontsize": 11,
}


class _Indexer:
    def __init__(self):
        self.map = {}

    def add(self, name, index):
        self.map[name] = index

    def get_index(self, name):
        return self.map[name]

    def get_name(self, index):
        return list(self.map.values()).index(index)

    def __len__(self):
        return len(self.map)


class PowerBalancePlotter:
    def __init__(self, **figure_options):
        self.fig_opts = {**BALANCE_PLOT_DEFAULTS, **figure_options}

        self.fig = None
        self.sankey = None
        self._indexer = _Indexer()

    def _prepare_plot(self):
        self.fig = plt.figure(
            figsize=self.fig_opts.pop("figsize"),
            facecolor=self.fig_opts.pop("facecolor"),
        )
        ax = self.fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
        plt.axis("off")

        # NOTE: We don't **self.fig_opts here.. let's control the kwargs carefully.
        self.sankey = SuperSankey(
            ax=ax,
            scale=self.fig_opts["scale"],
            format=self.fig_opts["format"],
            unit=self.fig_opts["unit"],
            gap=self.fig_opts["gap"],
            radius=self.fig_opts["radius"],
            shoulder=self.fig_opts["shoulder"],
            head_angle=self.fig_opts["head_angle"],
        )

    def _make_orientations(self, block):
        orientations = []
        pos_count = 0
        neg_count = 0
        for i, value in enumerate(block.flow_values):
            if value >= 0:
                if pos_count == 0:
                    orientations.append(0)
                else:
                    orientations.append(-1)
                pos_count += 1
            if value < 0:
                neg_count += 1
                if neg_count == len(block.out_flows):
                    orientations.append(0)
                else:
                    orientations.append(-1)
        return orientations

    def _get_connections(self, block):
        """
        Get the first connection
        """
        connections = []
        for connection in self.power_balance.connections:
            if block.name == connection.block_2.name:
                connections.append(connection)
        if not connections:
            raise BalanceOfPlantError(f"PowerBlock {block.name} has no connections.")
        return connections

    def _make_prior(self, block):
        if len(self._indexer) == 1:
            return None
        connections = self._get_connections(block)
        # Get the first connection
        connected_to = connections[0].block_1.name
        prior = self._indexer.get_index(connected_to)
        return prior

    def _make_connect(self, block):
        if len(self._indexer) == 1:
            return None

        connections = self._get_connections(block)

        return connections[0].indices

    def _make_future(self, block):
        return None

    def _make_pathlengths(self, block):
        return self.fig_opts["standardlength"] * len(block.flow_names)

    def _make_sankey_kwargs(self, block, plot_options):
        orientations = plot_options.get("orientations", self._make_orientations(block))
        prior = plot_options.get("prior", self._make_prior(block))
        future = plot_options.get("future", self._make_future(block))
        connect = plot_options.get("connect", self._make_connect(block))
        pathlengths = plot_options.get("pathlengths", self._make_pathlengths(block))
        kwargs = {
            "patchlabel": block.name,
            "labels": block.flow_names,
            "flows": block.flow_values,
            "orientations": orientations,
            "prior": prior,
            "connect": connect,
            "future": future,
            "trunklength": plot_options.get("trunklength", self.fig_opts["trunklength"]),
            "pathlengths": pathlengths,
            "facecolor": plot_options.get("facecolor", None),
        }

        return kwargs

    def _build_diagram(self, block_plot_options):
        for i, (name, block) in enumerate(self.power_balance.blocks.items()):
            self._indexer.add(name, i)
            plot_options = block_plot_options.get(name, {})
            sankey_kwargs = self._make_sankey_kwargs(block, plot_options)
            self.sankey.add(**sankey_kwargs)

    def _finalise_plot(self):
        """
        Finish up and polish figure, and format text
        """
        diagrams = self.sankey.finish()
        for diagram in diagrams:
            diagram.text.set_fontweight(self.fig_opts["fontweight"])
            diagram.text.set_fontsize(self.fig_opts["fontsize"])
            diagram.text.set_color(self.fig_opts["fontcolor"])
            for text in diagram.texts:
                text.set_fontsize(self.fig_opts["flowfontsize"])
                text.set_color(self.fig_opts["fontcolor"])

        self.fig.tight_layout()

    def plot(self, power_balance, block_plot_options=None):
        self.power_balance = power_balance
        self._prepare_plot()
        self._build_diagram(block_plot_options)
        self._finalise_plot()


if __name__ == "__main__":

    p_fusion = 2000  # MW
    p_neutrons = 0.8 * p_fusion
    p_alpha = 0.2 * p_fusion
    p_hcd = 50
    p_nrgm = 0.35 * p_neutrons
    p_neutronic = p_neutrons + p_nrgm
    f_aux_neutrons = 0.05
    f_div_neutrons = 0.07
    f_blk_neutrons = 1 - f_aux_neutrons - f_div_neutrons
    p_aux_neutrons = f_aux_neutrons * p_neutronic
    p_div_neutrons = f_div_neutrons * p_neutronic
    p_blk_neutrons = f_blk_neutrons * p_neutronic
    eta_bop = 0.33
    p_el_cryo = 30
    p_el_mag = 40
    p_el_tfv = 15
    p_el_building = 40

    balance = PowerBalance()

    plasma = PowerBlock("Plasma")
    plasma.add_in_flow(PowerFlow("Fusion power", p_fusion))
    plasma.add_in_flow(PowerFlow("H&CD", p_hcd))
    balance.add_power_block(plasma)

    rad_sep = PowerBlock("Radiation & Separatrix")
    balance.add_power_block(rad_sep)
    balance.connect_blocks(plasma, rad_sep, "Alphas + Aux", p_hcd + p_alpha)

    neutrons = PowerBlock("Neutrons")
    balance.add_power_block(neutrons)
    balance.connect_blocks(plasma, neutrons, "14.1 MeV neutrons", p_neutrons)
    neutrons.add_in_flow(PowerFlow("Energy multiplication", p_nrgm))
    neutrons.add_out_flow(PowerFlow("Aux n", p_aux_neutrons))
    neutrons.add_out_flow(PowerFlow("Divertor n", p_div_neutrons))

    blanket = PowerBlock("Blanket")
    balance.add_power_block(blanket)
    balance.connect_blocks(neutrons, blanket, "Blanket n", p_blk_neutrons)

    bop = PowerBlock("BoP")
    bop.add_out_flow(PowerFlow("Losses", (1 - eta_bop) * p_blk_neutrons))
    balance.add_power_block(bop)
    balance.connect_blocks(blanket, bop, "", p_blk_neutrons)

    electricity = PowerBlock("Electricity")
    balance.add_power_block(electricity)
    balance.connect_blocks(bop, electricity, "$P_{el}$", eta_bop * p_blk_neutrons)
    electricity.add_out_flow(PowerFlow("Magnets", p_el_mag))
    electricity.add_out_flow(PowerFlow("Tritium plant", p_el_tfv))
    electricity.add_out_flow(PowerFlow("Cryoplant", p_el_cryo))
    electricity.add_out_flow(PowerFlow("Grid", electricity.remainder))

    from bluemira.base.constants import BLUEMIRA_PAL_MAP

    # balance.blocks["Environment"].add_out_flow(PowerFlow("People", -2000))

    plotter = PowerBalancePlotter()
    plotter.plot(
        balance,
        {
            "Plasma": {"facecolor": BLUEMIRA_PAL_MAP["blue"]},
            "Neutrons": {"facecolor": BLUEMIRA_PAL_MAP["orange"]},
            "Radiation & Separatrix": {"facecolor": BLUEMIRA_PAL_MAP["red"]},
            "Blanket": {"facecolor": BLUEMIRA_PAL_MAP["yellow"]},
            "BoP": {"facecolor": BLUEMIRA_PAL_MAP["purple"]},
            "Electricity": {"facecolor": BLUEMIRA_PAL_MAP["green"]},
        },
    )
