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
Heating and current drive system
"""
from typing import Type

import numpy as np

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter import ParameterFrame
from BLUEPRINT.systems.baseclass import ReactorSystem
from BLUEPRINT.systems.plotting import ReactorSystemPlotter


class HCDSystem(ReactorSystem):
    """
    Heating and current drive reactor system.
    """

    config: Type[ParameterFrame]
    inputs: dict

    # fmt: off
    default_params = [
        ['g_cd_nb', 'NB current drive efficiency', 0.4, 'MA/MW.m', 'Check units!', 'Input'],
        ['eta_nb', 'NB electrical efficiency', 0.3, 'dimensionless', 'Check units!', 'Input'],
        ['p_nb', 'NB launcher power', 1, 'MA', 'Maximum launcher current drive in a port', 'Input'],
        ['g_cd_ec', 'EC current drive efficiency', 0.15, 'MA/MW.m', 'Check units!', 'Input'],
        ['eta_ec', 'EC electrical efficiency', 0.35, 'dimensionless', 'Check units!', 'Input'],
        ['p_ec', 'EC launcher power', 10, 'MW', 'Maximum launcher power per sector', 'Input'],
        ['f_cd_aux', 'Auxiliary current drive fraction', 0.1, 'dimensionless', None, 'Input'],
        ['f_bs', 'UNKNOWN_2', 0.1, 'dimensionless', None, 'Input'],
        ['op_mode', 'UNKNOWN_3', "str", 'dimensionless', None, 'Input'],
    ]
    # fmt: on

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs
        self._plotter = HCDSystemPlotter()

        self._init_params(self.config)

        self.f_bs = self.params.f_bs
        if self.params.op_mode == "Pulsed":
            self.pulsed = True
        elif self.params.op_mode == "Steady-state":
            self.pulsed = False
        self.NB = NeutralBeam(self.config, self.inputs, 0.5, 0.3, 1)
        self.EC = ElectronCyclotron(self.config, self.inputs, 0.15, 0.35, 10)
        self.P_LH = self.inputs["P_LH"]
        self.requirements["P_LH"] = self.P_LH
        self.allocate("P_LH", f_NBI=0.2, f_ECD=0.8)

    def build(self):
        """
        Build the neutral beams and electorn cyclotrons. IRCH is not cool.
        """
        self.NB.build()
        self.EC.build()

    def set_requirement(self, req, value):
        """
        Set a requirement for the HCDSystem.
        """
        self.requirements[req] = value

    def allocate(self, req, f_NBI, f_ECD=0, f_ICRH=0):  # noqa :N803
        """
        Allocate a requirement to the NB and EC fractionally.
        """
        if f_ICRH != 0:
            f_ICRH = 0  # noqa
            bluemira_warn("Not on my watch.")
        if f_ECD == 0:
            f_ECD = 1 - f_NBI  # noqa
        if f_NBI + f_ECD != 1:
            raise ValueError("{0}+{1}!=1, dipshit.".format(f_NBI, f_ECD))
        self.NB.set_requirement(req, f_NBI * self.requirements[req])
        self.EC.set_requirement(req, f_ECD * self.requirements[req])

    def set_current(self, config):
        """
        Set the HCD driven current.
        """
        f_cd_aux = 1 - self.f_cd_ohm - self.params.f_bs
        self.I_hcd = config["I_p"] * f_cd_aux

    def pulsed_ops(self):
        """
        Check for current consistency.
        """
        # self.f_cd_ohm = 0.5  # Hook this up to CS and flux swing one day
        if self.params.f_bs + self.f_cd_ohm > 1:
            raise ValueError("Current drive fraction greater than 1.")

    def get_heating_power(self):
        """
        Get the HCD heating power.
        """
        return self.NB.params.P_h_ss + self.EC.params.P_h_ss

    def get_electric_power(self):
        """
        Get the HCD required electical power.
        """
        return self.NB.params.P_el + self.EC.params.P_el

    def get_nb_fraction(self):
        """
        Get the fraction of NB current drive.
        """
        return self.NB.requirements["I_cd"] / self.requirements["I_cd"]


class GenericHCD(ReactorSystem):
    """
    Generic HCD sub-system base class.
    """

    config: Type[ParameterFrame]
    inputs: dict

    default_params = [
        ["R_0", "Major radius", 9, "m", None, "Input"],
        ["n_TF", "Number of TF coils", 16, "dimensionless", None, "Input"],
        ["I", "Current drive", 0.1, "dimensionless", None, "Input"],
    ]

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs

        self.params = ParameterFrame(self.default_params.to_records())
        self._init_params(self.config)

        self.R_0 = self.params.R_0
        self.n_TF = self.params.n_TF

    def set_requirement(self, req, value):
        """
        Set a requirement for the GenericHCD object.
        """
        self.requirements[req] = value

    def build(self):
        """
        Build the GenericHCD object.
        """
        if "I_cd" in self.requirements:
            self.add_parameter(
                "I", "Current drive", self.requirements["I_cd"], "MA", None, "BLUEPRINT"
            )
            self.add_parameter(
                "P_h_ss",
                "Steady-state plasma heating power",
                self.inputs["n_20"] * self.R_0 * self.params.I / self.g_cd,
                "MW",
                None,
                "BLUEPRINT",
            )
        elif "P_hcd_ss" in self.requirements:
            self.add_parameter(
                "P_h_ss",
                "Steady-state plasma heating power",
                self.requirements["P_hcd_ss"],
                "MW",
                None,
                "BLUEPRINT",
            )
            self.add_parameter(
                "I",
                "Current drive",
                self.requirements["P_hcd_ss"]
                * self.g_cd
                / (self.inputs["n_20"] * self.R_0),
                "MA",
                None,
                "BLUEPRINT",
            )
        self.add_parameter(
            "P_el",
            "Electric power input",
            self.params.P_h_ss / self.eta_cd,
            "MW",
            None,
            "BLUEPRINT",
        )

    def get_n(self):
        """
        Get the number of GenericHCD objects. Current divided by launcher unit power.
        """
        self.n = int(np.ceil(self.params.I / self.plauncher))
        return self.n

    def add_penetration(self, plug):
        """
        Takes a plug object from BlanketCoverage
        """
        self.geom["Equatorial plug"] = plug["loop"]
        plug_geom = self.build_plug(plug)
        self.geom["feed 3D CAD"] = {"NB equatorial plug": plug_geom}

    def build_plug(self, plug):
        """
        Build a HCD plug.
        """
        alpha = 360 / self.n_TF * plug["tor_width_f"]
        nbplug = plug["loop"].copy()
        nbplug.rotate(-alpha / 2, p1=[0, 0, 0], p2=[0, 0, 1])
        plug_geom = {
            "profile1": nbplug,
            "path": {"angle": alpha, "rotation axis": [(0, 0, 0), (0, 0, 1)]},
        }
        return plug_geom

    @property
    def xz_plot_loop_names(self):
        """
        The x-z loop names to plot.
        """
        return ["Equatorial plug"]

    @property
    def xy_plot_loop_names(self):
        """
        The x-y loop names to plot.
        """
        raise NotImplementedError

    def plot_xy(self, ax=None, **kwargs):
        """
        Plot the GenericHCD object.
        """
        raise NotImplementedError


class NeutralBeam(GenericHCD):
    """
    Neutral Beam system.
    """

    config: Type[ParameterFrame]
    inputs: dict
    # g_cd = FloatBetween(low=0.0, high=1.0)
    # eta_cd = FloatBetween(low=0.0, high=1.0)
    # plauncher = FloatOrInt()  # MA per NBI launcher in sector. Total guess
    default_params = [
        ["R_0", "Major radius", 9, "m", None, "Input"],
        ["n_TF", "Number of TF coils", 16, "dimensionless", None, "Input"],
        ["I", "Current drive", 0.1, "dimensionless", None, "Input"],
        [
            "P_h_ss",
            "Steady-state plasma heating power",
            0.1,
            "dimensionless",
            None,
            "Input",
        ],
        ["P_el", "Electric power input", 0.1, "dimensionless", None, "Input"],
    ]

    def __init__(self, config, inputs, g_cd, eta_cd, plauncher):
        super().__init__(config, inputs)
        self.g_cd = g_cd
        self.eta_cd = eta_cd
        self.plauncher = plauncher


class ElectronCyclotron(GenericHCD):
    """
    Electron Cyclotron system.
    """

    config: Type[ParameterFrame]
    inputs: dict
    # g_cd = FloatBetween(low=0.0, high=1.0)
    # eta_cd = FloatBetween(low=0.0, high=1.0)
    # plauncher = FloatOrInt()  # MW per EC launcher in sector. Total guess
    default_params = [
        ["R_0", "Major radius", 9, "m", None, "Input"],
        ["n_TF", "Number of TF coils", 16, "dimensionless", None, "Input"],
        ["I", "Current drive", 0.1, "dimensionless", None, "Input"],
        [
            "P_h_ss",
            "Steady-state plasma heating power",
            0.1,
            "dimensionless",
            None,
            "Input",
        ],
        ["P_el", "Electric power input", 0.1, "dimensionless", None, "Input"],
    ]

    def __init__(self, config, inputs, g_cd, eta_cd, plauncher):
        super().__init__(config, inputs)
        self.g_cd = g_cd
        self.eta_cd = eta_cd
        self.plauncher = plauncher

    def get_n(self):
        """
        Get the number of ECH launchers. Heating power divided by unit launcher power.
        """
        self.n = int(np.ceil(self.requirements["P_LH"] / self.plauncher))
        self.geom["n"] = self.n
        return self.n


class HCDSystemPlotter(ReactorSystemPlotter):
    """
    The plotter for a Heating and Current Drive System.
    """

    def __init__(self):
        super().__init__()
        self._palette_key = "HCD"

    def plot_xy(self, plot_objects, ax=None, **kwargs):
        """
        Plot the HCDsystem in x-y.
        """
        raise NotImplementedError
