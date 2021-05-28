# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
Extremely crude cost calculator (proof of principle)
"""
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
from BLUEPRINT.costs.constants import RELATIVE_COST_FRACTIONS, AUXILIARY_COST_FRACTIONS


class VolumeCalculator:
    """
    Calculates volumes of different reactor components, in preparation of cost
    calculations.

    Parameters
    ----------
    reactor: Reactor
        The reactor for which to calculate the volumes
    """

    def __init__(self, reactor):
        reactor.build_CAD()
        self.reactor = reactor
        self.n_TF = reactor.params.n_TF
        self.volumes = {}

    def calculate(self):
        """
        Calculate the volume of all the major tokamak components.
        """
        # IVC volumes
        self.get_bb_volumes()
        self.get_div_volumes()
        # Coil volumes
        self.get_tf_volumes()
        self.get_pf_volumes()
        self.get_cs_volumes()
        self.get_coil_struct_volumes()
        # Structures volumes
        self.get_cr_volume()
        self.get_ts_volume()
        self.get_vv_volume()
        self.get_rs_volume()

        return self.volumes

    def get_bb_volumes(self):
        """
        Calculate the BB volumes.
        """
        # sum the total volume of each segment type
        props = self.reactor.CAD.parts["Breeding blanket"].get_properties()
        v_libs = 0
        v_ribs = 0
        v_lobs = 0
        v_cobs = 0
        v_robs = 0
        for k, v in props.items():
            if k.startswith("LIBS_"):
                v_libs += v["Volume"]
            elif k.startswith("RIBS_"):
                v_ribs += v["Volume"]
            elif k.startswith("LOBS_"):
                v_lobs += v["Volume"]
            elif k.startswith("COBS_"):
                v_cobs += v["Volume"]
            elif k.startswith("ROBS_"):
                v_robs += v["Volume"]

        self.volumes["BB"] = self.n_TF * (v_libs + v_ribs + v_lobs + v_cobs + v_robs)

    def get_div_volumes(self):
        """
        Calculate the divertor volumes.
        """
        props = self.reactor.CAD.parts["Divertor"].get_properties()
        self.volumes["DIV"] = self.n_TF * sum([p["Volume"] for p in props.values()])

    def get_tf_volumes(self):
        """
        Calculate the TF coil volumes.
        """
        props = self.reactor.CAD.parts["Toroidal field coils"].get_properties()
        self.volumes["TF case"] = self.n_TF * props["case"]["Volume"]
        self.volumes["TF wp"] = self.n_TF * props["wp"]["Volume"]

    def get_pf_volumes(self):
        """
        Calculate the PF coil volumes.
        """
        props = self.reactor.CAD.parts["Poloidal field coils"].get_properties()
        self.volumes["PF"] = self.n_TF * sum([p["Volume"] for p in props.values()])

    def get_cs_volumes(self):
        """
        Calculate the central solenoid volume.
        """
        props = self.reactor.CAD.parts["Central solenoid"].get_properties()
        self.volumes["CS"] = self.n_TF * sum([p["Volume"] for p in props.values()])

    def get_coil_struct_volumes(self):
        """
        Calculate the coil structure volumes.
        """
        props = self.reactor.CAD.parts["Coil structures"].get_properties()
        self.volumes["coil structures"] = self.n_TF * sum(
            [p["Volume"] for p in props.values()]
        )

    def get_cr_volume(self):
        """
        Calculate the cryostat vacuum vessel volume.
        """
        props = self.reactor.CAD.parts["Cryostat"].get_properties()
        self.volumes["CR"] = self.n_TF * props["Cryostat vacuum vessel"]["Volume"]

    def get_ts_volume(self):
        """
        Calculate the thermal shield volume.
        """
        props = self.reactor.CAD.parts["Thermal shield"].get_properties()
        self.volumes["TS"] = self.n_TF * props["thermal_shield"]["Volume"]

    def get_vv_volume(self):
        """
        Calculate the vacuum vessel volume.
        """
        props = self.reactor.CAD.parts["Reactor vacuum vessel"].get_properties()
        v_vv = props["vessel"]["Volume"]
        v_ports = props["ports"]["Volume"]
        self.volumes["VV"] = self.n_TF * (v_vv + v_ports)

    def get_rs_volume(self):
        """
        Calculate the radiation shield volume.
        """
        props = self.reactor.CAD.parts["Radiation shield"].get_properties()
        self.volumes["RS"] = self.n_TF * sum([p["Volume"] for p in props.values()])


class CostCalculator:
    """
    Object responsible for the calculation of the total reactor cost.

    This is an extremely crude "middle-up" cost estimate, based on absolutely
    nothing beyond volume and gut feel

    Parameters
    ----------
    reactor: Reactor
        The reactor for which to calculate the cost
    """

    def __init__(self, reactor):
        volume_calc = VolumeCalculator(reactor)
        self.reactor = reactor
        volumes = volume_calc.calculate()
        self.df = DataFrame(
            {
                "Volume": volumes,
                "Cost factor": RELATIVE_COST_FRACTIONS,
                "Auxiliary multiplier": AUXILIARY_COST_FRACTIONS,
            }
        )

    def calculate(self):
        """
        Calculate the component costs.
        """
        self.df["Cost"] = (
            self.df["Volume"] * self.df["Cost factor"] * self.df["Auxiliary multiplier"]
        )
        return sum(self.df["Cost"])

    def plot(self):
        """
        Plot the cost breakdown in a pie chart.
        """

        def pct_format(string):
            return f"{string:.1f}"

        colors = sns.color_palette("Blues_r", len(self.df))
        f, ax = plt.subplots()

        if "Cost" not in self.df.columns:
            self.calculate()

        ax.pie(
            self.df["Cost"],
            startangle=90,
            autopct=pct_format,
            labels=list(self.df.index),
            colors=colors,
        )
        ax.set_title("Percentage cost breakdown")


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
