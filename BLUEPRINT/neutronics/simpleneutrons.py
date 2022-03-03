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
Very, very simple TBR and power deposition routines
"""
import json
import os
from typing import Type

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import get_bluemira_path, try_get_bluemira_path
from bluemira.base.look_and_feel import bluemira_print
from bluemira.base.parameter import ParameterFrame
from bluemira.display.auto_config import plot_defaults
from bluemira.geometry._deprecated_tools import innocent_smoothie
from bluemira.geometry.constants import VERY_BIG
from BLUEPRINT.geometry.boolean import boolean_2d_common
from BLUEPRINT.geometry.geomtools import (
    get_angle_between_points,
    join_intersect,
    loop_volume,
)
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.systems.baseclass import ReactorSystem

TBR_DATA_ROOT = "_TBR_data.json"


class TBRData:
    """
    TBR Data class.
    """

    def __init__(self, blanket_type, datadir=None):
        self.angle = None
        self.percTBR = None
        self.dees_angle = None
        self.dees_percTBR = None
        self.decum_TBR = None
        self.ix = None
        self.iy = None
        self.datadir = datadir
        if self.datadir is None:
            self.datadir = get_bluemira_path("neutronics", subfolder="data/BLUEPRINT")
        self.filename = blanket_type + TBR_DATA_ROOT
        self.get_raw_TBR_data()
        self.deescalate_data()
        self.decumulate_data()
        self.interpret_raw(self.dees_angle, self.decum_TBR)

    def get_raw_TBR_data(self):
        """
        The raw data obtained from Pavel Pereslavtsev (KIT) on the 04/04/2017.
        Best IDM reference: 2ME43P
        Data valid for HCPB blankets. Open question on 1.25 or 1.27 ideal TBR.
        0° outboard midplane, anti-clockwise.
        """
        fpath = os.sep.join([self.datadir, self.filename])

        with open(fpath, "r") as file:
            data = json.load(file)

        self.angle = data["poloidal_angle"]
        self.percTBR = data["percentage_TBR"]

    def deescalate_data(self):
        """
        Deescalate raw data.
        """
        self.dees_angle, self.dees_percTBR = [], []
        for i, (a, t) in enumerate(zip(self.angle, self.percTBR)):
            if i == 0:
                self.dees_angle.append(a)
                self.dees_percTBR.append(0)
            elif i % 2 != 0:
                self.dees_angle.append(a)
                self.dees_percTBR.append(t)
        return

    def decumulate_data(self):
        """
        Decumulate raw data.
        """
        ddata = []
        for i in range(len(self.dees_percTBR)):
            if i == len(self.dees_percTBR) - 1:
                ddata.append(
                    (self.dees_percTBR[i] - self.dees_percTBR[i - 1])
                    / (self.dees_angle[i] - self.dees_angle[i - 1])
                )
            else:
                ddata.append(
                    (self.dees_percTBR[i + 1] - self.dees_percTBR[i])
                    / (self.dees_angle[i + 1] - self.dees_angle[i])
                )
            self.decum_TBR = ddata
        return

    def interpret_raw(self, a, TBR):
        """
        Interpret raw data.
        """
        self.ix, self.iy = innocent_smoothie(a, TBR, s=0.006)
        return

    def plot(self, ax=None):
        """
        Plot the TBRData.
        """
        if ax is None:
            f, ax = plt.subplots()
        # Raw data plotting
        ax.plot(self.angle, self.percTBR, label="Cumulative TBR %")
        ax.plot(
            self.dees_angle, self.dees_percTBR, label="Cumulative TBR % (linearised)"
        )
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Poloidal angle [°]")
        ax.set_ylabel("Cumulative % of TBR")
        # Interpreted data
        # ax2 = ax.twinx()
        # next(ax2._get_lines.prop_cycler)
        # ax2.plot(self.dees_angle, self.decum_TBR, label='TBR % per °')
        # ax2.set_xlim([0, 360])
        # ax2.plot(self.ix, self.iy, label='Smoothed data')
        # ax2.legend(loc='lower right')
        return ax


class NeutronicsRulesOfThumb:
    """
    Deliberately heretical interpretation of the neutronics Old Testament.

    Assumes "classical" shielding values. Not to be used in anger.
    """

    def __init__(self):
        # Neutron flux in TF coil insulation [MC: based on PP email 20/02/17 -
        # difficult to read figure] 2:actual 1.4:make it look good
        # NOTE: insulation probably not the problem
        self.tf_ins_nflux = 1.4e13  # [n/m^2/s]
        # Blanket outboard midplane EUROFER dmg rate [PP: 2M7HN3 fig. 20]
        self.blk_dmg = 10.2  # [dpa/FPY]
        # Divertor region CuCrZr dmg rate [dpa]
        # https://iopscience.iop.org/article/10.1088/1741-4326/57/9/092002/pdf]
        self.div_dmg = 3  # [dpa/MW.annum/m^2]
        # VV peak SS316LN-IG dmg rate [MC: PP 2M7HN3 fig. 18]
        self.vv_dmg = 0.3  # [dpa/MW.annum/m^2]


class BlanketCoverage(ReactorSystem):
    """
    Simplified neutron accountancy for TBR and volumetric heat deposition

    Takes in Plasma and FW objects describing the x, z coordinates of the
    separatrix and first wall, ordered about the major radius anticlockwise
    from the outboard midplane.
    Calculates the volume fraction penalty for high grade heat power
    conversion and the TBR using crude neutronics.
    Needs a first wall shape to work out volume fraction penalty
    Atr the moment no neutrons go to the vessel!!

    :math:`\\Lambda = \\Lambda_{max}-\\sum_{i=1}^{n} \\frac{\\alpha_{\\phi_{i}}}{2\\pi}
    \\frac{t_{i}}{T}\\int_{\\alpha_{i_{1}}}^{\\alpha_{i_{2}}} \\lambda(\\theta) d\\theta`

    :math:`\\theta`: poloidal angle
    :math:`\\Lambda_{max}`: total potential TBR
    :math:`\\alpha_{\\phi_{i}}`: toroidal extension of the non-breeding region $i$ (in radians) \n
    :math:`t_{i}`: thickness of the non-breeding region $i$
    :math:`T`: maximum thickness of the in-vessel components
    :math:`\\alpha_{i_{1}}`: starting poloidal angle of the non-breeding region $i$
    :math:`\\alpha_{i_{2}}`: final poloidal angle of the non-breeding region $i$
    :math:`\\lambda(\\theta)`: potential TBR as a function of poloidal angle
    """  # noqa :W505

    config: Type[ParameterFrame]
    inputs: dict

    # fmt: off
    default_params = [
        ["blanket_type", "Blanket type", "HCPB", "dimensionless", None, "Input"],
        ['plasma_type', 'Type of plasma', 'SN', 'dimensionless', None, 'Input'],
        ["vvpfrac", "Fraction of neutrons deposited in VV", 0.04, "dimensionless",
            "simpleneutrons needs a correction for VV n absorbtion", None],
        ["R_0", "Major radius", 9, "m", None, "Input"],
        ["n_TF", "Number of TF coils", 16, "dimensionless", None, "Input"],
    ]
    # fmt: on

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs

        self._init_params(config)

        self.nb_regions = []
        self.lgh_regions = []
        self.aux_n_regions = []
        self.plug_loops = []
        self.max_TBR = self.inputs["max_TBR"]
        datadir = inputs.get(
            "datadir", try_get_bluemira_path("neutronics", subfolder="data/BLUEPRINT")
        )
        self.data = TBRData(self.params.blanket_type, datadir=datadir)

        # Calculated constructors
        self.div_n_frac = 0
        self.aux_n_frac = 0
        self.f_HGH = 0
        self.TBR = 0
        self.V = 0

    def calculate(self):
        """
        Carries out the calculations for the simplified volumetric neutronics
        estimations
        """
        self.volume()
        self.vol_penalty()
        self.calcTBR()
        self.correct_for_VV()

    def volume(self):
        """
        Volume of ideal in-vessel component coverage
        """
        area = self.inputs["ideal_shell"].area
        radius = self.inputs["ideal_shell"].centroid[0]
        self.V = 2 * np.pi * area * radius

    def add_plug(
        self, poloidal_angles, tor_width_f=1, pol_depth_f=1, tor_cont=False, typ=""
    ):
        """
        Add a non-breeding "plug" to the problem and subtract its volume from
        breeding and high-grade heat removal.

        Parameters
        ----------
        poloidal_angles: List(2)
            The start and end poloidal angles of the plug region
        tor_width_f: float
            The fraction of toroidal width that the plug occupies
        pol_depth_f: float
            The fraction of poloidal depth in the ideal shell that the plug
            occupies
        tor_cont: bool
            Whether or not the plug is toroidally continuous (e.g. a divertor)
            This equates to tor_width_f = 1
        typ: str
            The type of plug. typ = "Aux" will count the neutrons in the
            auxiliary areas

        Returns
        -------
        plug: dict
            The plug dictionary, containing the X-Z geometry and the 3-D
            information (tor_width_f, pol_depth_f, tor_cont)
        """
        shell = self.inputs["ideal_shell"]
        angles = np.deg2rad(poloidal_angles)

        x = [
            self.params.R_0,
            self.params.R_0 + np.cos(angles[0]) * VERY_BIG,
            self.params.R_0 + np.cos(angles[1]) * VERY_BIG,
        ]
        z = [
            0,
            np.sin(angles[0]) * VERY_BIG,
            np.sin(angles[1]) * VERY_BIG,
        ]
        triangle = Loop(x=x, y=0, z=z)
        triangle.close()

        loop = boolean_2d_common(shell, triangle)[0]

        plug = {
            "loop": loop,
            "tor_width_f": tor_width_f,
            "pol_depth_f": pol_depth_f,
            "tor_cont": tor_cont,
        }
        weight = pol_depth_f * tor_width_f / self.params.n_TF
        if tor_cont:
            weight = pol_depth_f
            if tor_width_f != 1:
                bluemira_print(
                    "tor_width is being ignored because you have " "specified tor_cont."
                )
        volume = loop_volume(plug["loop"]["x"], plug["loop"]["z"]) * weight
        self.nb_regions.append([poloidal_angles, weight])

        self.lgh_regions.append(volume)
        if typ == "Aux":
            self.aux_n_regions.append(volume)

        self.plug_loops.append(plug["loop"])
        return plug

    def add_divertor(self, div_geom):
        """
        Adds divertor region(s) to the BlanketCoverage and treats these regions
        as non-breeding and as low-grade-heat

        Parameters
        ----------
        div_geom: Union[Loop, MultiLoop]
            The divertor geometry to transform into plug(s)
        """
        if self.params.plasma_type == "DN":
            # Handle double null
            loops = [
                div_geom["lower"]["divertor_gap"],
                div_geom["upper"]["divertor_gap"],
            ]

        else:  # Handle single null
            loops = [div_geom["divertor_gap"]]

        volume = 0  # Divertor (poloidal angle cut(s)) volume tracker
        for cut in loops:

            inner = self.inputs["ideal_shell"].inner
            outer = self.inputs["ideal_shell"].outer
            inner_args = join_intersect(inner, cut, get_arg=True)
            outer_args = join_intersect(outer, cut, get_arg=True)

            # Get maximum angular span of the divertor
            angles = []
            for i in [0, 1]:
                p1 = inner.d2.T[inner_args[i]]
                p2 = outer.d2.T[outer_args[i]]
                angle1 = self.get_mp_angle(p1, self.params.R_0)
                angle2 = self.get_mp_angle(p2, self.params.R_0)
                angles.append([angle1, angle2])

            max_angles = [max(np.abs(angles[0])), min(np.abs(angles[1]))]

            self.add_plug(max_angles, tor_cont=True)

            volume += self.lgh_regions[-1]

        self.div_n_frac = volume / self.V

    @staticmethod
    def get_mp_angle(p, R_0):
        """
        Anticlockwise from outboard midplane
        """
        if p[1] > 0:
            a = get_angle_between_points([R_0 + 1, 0], [R_0, 0], p)
        elif p[1] <= 0:
            a = 360 - get_angle_between_points([R_0 + 1, 0], [R_0, 0], p)
        return a

    def vol_penalty(self):
        """
        Calculates the volume fraction of high grade heat in the
        BlanketCoverage
        """
        v_hgh = self.V - sum(self.lgh_regions)
        self.f_HGH = v_hgh / self.V
        self.aux_n_frac = (sum(self.aux_n_regions)) / self.V

    def correct_for_VV(self):
        """
        Corrections the fraction of neutrons for leakage to the vacuum vessel
        """
        # NOTE: (self.f_HGH+self.div_n_frac+self.aux_n_frac) = 1
        a = 1 - self.params.vvpfrac
        self.f_HGH *= a
        self.div_n_frac *= a
        self.aux_n_frac *= a

    def calcTBR(self):
        """
        Calculates the TBR, based on the available parameterised poloidal angle
        TBR data
        """
        p_tbr = [i * self.max_TBR / 100 for i in self.data.iy]
        angle = np.array(self.data.ix)
        perc = np.trapz(p_tbr, angle)
        for entry in self.nb_regions:
            (end, start), weight = entry

            argin = np.abs(angle - start).argmin()
            argout = np.abs(angle - end).argmin()
            perc -= np.trapz(p_tbr[argin:argout], angle[argin:argout]) * weight
        self.TBR = float(perc)

    def plot(self, title=False):
        """
        Plot the BlanketCoverage result.
        """
        plot_defaults()
        f, ax = plt.subplots(1, 2)
        ax1 = self.data.plot(ax[0])
        for i, region in enumerate(self.nb_regions):
            if region[1] == 1:
                label = "Toroidally continuous \n non-breeding region"
            else:
                label = "Toroidally periodic \n non-breeding region"

            ax1.axvspan(
                region[0][0],
                region[0][1],
                color="r",
                alpha=max(region[1], 0.2),
                label=label,
            )

        if title:
            ax1.set_title(
                "Achievable TBR = {0:.2f} out of a potential "
                "{1:.2f}".format(self.TBR, self.max_TBR)
            )
        ax1.set_xlim([0, 360])

        ax1.legend(bbox_to_anchor=(0.05, 1))

        # Now plot the geometry a bit
        self.inputs["ideal_shell"].plot(ax[1])
        for loop in self.plug_loops:
            loop.plot(ax[1], facecolor="r")
