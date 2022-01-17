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
Thermal shield CAD routines
"""
from BLUEPRINT.base.palettes import BLUE
from BLUEPRINT.cad.cadtools import (
    boolean_cut,
    boolean_fuse,
    extrude,
    make_face,
    make_mixed_face,
    make_mixed_shell,
    make_shell,
    make_vector,
    revolve,
    rotate_shape,
)
from BLUEPRINT.cad.component import ComponentCAD
from BLUEPRINT.cad.mixins import OnionCAD
from BLUEPRINT.geometry.boolean import simplify_loop
from BLUEPRINT.geometry.shell import Shell


class ThermalShieldCAD(OnionCAD, ComponentCAD):
    """
    Thermal shield CAD constructor class

    Parameters
    ----------
    thermal_shield: Systems::ThermalShield object
        The thermal shield object for which to build the CAD
    """

    # TODO: is multiple inheritance required here, is OnionCAD a mixin?
    def __init__(self, thermal_shield, **kwargs):
        super().__init__(
            "Thermal shield",
            thermal_shield.geom,
            thermal_shield.params.n_TF,
            palette=[BLUE["TS"]],
            **kwargs
        )

    def build(self, **kwargs):
        """
        Build the CAD for the ThermalShield
        """
        thermal_shield, n_TF = self.args

        # First get the VVTS shape and simplify it
        vvts_2d = thermal_shield["2D profile"]

        inner = simplify_loop(vvts_2d.inner)
        outer = simplify_loop(vvts_2d.outer)
        vvts_2d = Shell(inner, outer)

        vvts = make_mixed_shell(vvts_2d)

        # rotate it so that we can get a clean sector
        vvts = rotate_shape(vvts, None, -180 / n_TF)
        # Finally make the VVTS sector, with no penetrations
        vvts = revolve(vvts, None, 360 / n_TF)

        # Cutting shape for the internal parts of the ports
        cut = revolve(make_mixed_face(vvts_2d.inner), None, 360)

        up = thermal_shield["Upper port"]
        zref = up.inner["z"][0]
        up = up.translate([0, 0, -zref], update=False)  # Bring to midplane
        up = make_shell(up)
        up_port = extrude(up, length=zref + 5, axis="z")
        upi = thermal_shield["Upper port"].inner.translate([0, 0, -zref], update=False)
        cut_up = extrude(make_face(upi), length=zref + 5, axis="z")

        ep = make_shell(thermal_shield["Equatorial port"])
        eq_port = extrude(ep, length=-10, axis="x")
        cut_eq = extrude(
            make_face(thermal_shield["Equatorial port"].inner), length=-10, axis="x"
        )

        lpvec = make_vector(thermal_shield["LP path"][0], thermal_shield["LP path"][1])
        lpi = make_face(thermal_shield["Lower port"].inner)
        lpi = extrude(lpi, vec=-lpvec * 1.2)
        lower_port = make_shell(thermal_shield["Lower port"])
        lower_port = extrude(lower_port, vec=-lpvec)

        # Cut away holes in the VVTS
        vvts = boolean_cut(vvts, cut_up)
        vvts = boolean_cut(vvts, cut_eq)
        vvts = boolean_cut(vvts, lpi)

        # lpo = boolean_cut(lower_port, cut)

        # Duct
        di = thermal_shield["Lower duct"].inner
        # do = thermal_shield["Lower duct"].outer
        # lpd = make_shell(thermal_shield["Lower duct"])
        # lpd = extrude(lpd, length=-6, axis="x")
        # dp = Loop(*do.xyz)
        # dp = dp.translate([-6, 0, 0], update=False)
        # pad = extrude(make_face(dp), length=-0.05, axis="x")
        # duct = boolean_fuse(pad, lpd)

        duct_cut = extrude(make_face(di), length=-6, axis="x")

        # Cryostat thermal shield
        ctscut = thermal_shield["CTS cut out"].rotate(
            -180 / n_TF, p1=[0, 0, 0], p2=[0, 0, 1], update=False
        )
        ctscut = make_face(ctscut)
        ctscut = revolve(ctscut, None, 360 / n_TF)
        cts = thermal_shield["Cryostat TS"].rotate(
            -180 / n_TF, p1=[0, 0, 0], p2=[0, 0, 1], update=False
        )
        cts = make_face(cts)
        cts = revolve(cts, None, 360 / n_TF)

        # Cut away holes in the CTS
        cts = boolean_cut(cts, duct_cut)
        cts = boolean_cut(cts, cut_eq)
        cts = boolean_cut(cts, cut_up)

        # Carrying out boolean cuts on smaller shapes is cheaper than
        # doing it once... usually
        # Trim the ports internally
        up_port = boolean_cut(up_port, cut)
        eq_port = boolean_cut(eq_port, cut)
        lower_port = boolean_cut(lower_port, cut)

        # Trim the ports externally
        up_port = boolean_cut(up_port, ctscut)
        eq_port = boolean_cut(eq_port, ctscut)
        lower_port = boolean_cut(lower_port, ctscut)

        # build the final CAD object
        final = vvts
        # this is a bit buggy... OCC issues
        # for component in [lower_port, eq_port, up_port, cts]:
        #     final = boolean_fuse(final, component)

        self.add_shape(final, name="thermal_shield")
        self.add_shape(lower_port, name="lower")
        self.add_shape(eq_port, name="eq_port")
        self.add_shape(up_port, name="up_port")
        self.add_shape(cts, name="cts")

    def build_neutronics(self, **kwargs):
        """
        Build the neutronics CAD for the thermal shield
        """
        self.build(**kwargs)

    def for_neutronics(self):
        """
        An old function used for fusing the entire geometry for use in neutronics.
        """
        ts = self.component["shapes"][0]
        for part in self.component["shapes"][1:]:
            ts = boolean_fuse(ts, part)
        self.component = {
            "shapes": [],
            "names": [],
            "sub_names": [],
            "colors": [],
        }  # Phoenix design pattern
        self.add_shape(ts, name="thermal_shield")


class SegmentedThermalShieldCAD(OnionCAD, ComponentCAD):
    """
    Thermal shield CAD constructor class

    Parameters
    ----------
    thermal_shield: Systems::ThermalShield object
        The thermal shield object for which to build the CAD
    """

    # TODO: is multiple inheritance required here, is OnionCAD a mixin?
    def __init__(self, thermal_shield, **kwargs):
        super().__init__(
            "SegmetThermal shield",
            thermal_shield.geom,
            thermal_shield.params.n_TF,
            palette=[BLUE["TS"]],
            **kwargs
        )

    def build(self, **kwargs):
        """
        Build the CAD for the SegmentedThermalShield
        """
        thermal_shield, n_TF = self.args

        if "Cryostat TS" in thermal_shield:
            loop_list = [
                ["inboard_thermal_shield", thermal_shield["Inboard profile"]],
                ["outboard_thermal_shield", thermal_shield["Outboard profile"]],
                ["cryostat_thermal_sheild", thermal_shield["Cryostat TS"]],
            ]
        else:
            loop_list = [
                ["inboard_thermal_shield", thermal_shield["Inboard profile"]],
                ["outboard_thermal_shield", thermal_shield["Outboard profile"]],
            ]

        for name, profile in loop_list:
            # First get the VVTS shape and simplify it
            # Making sure the inner-outter loops are Ok
            profile_vv_2d = simplify_loop(profile)

            # Making the CAD faces
            profile_vv = make_face(profile_vv_2d)

            # rotate it so that we can get a clean sector
            profile_vv = rotate_shape(profile_vv, None, -180 / n_TF)

            # Finally make the VVTS sector, with no penetrations
            profile_vv = revolve(profile_vv, None, 360 / n_TF)

            # Add the TS shapes
            self.add_shape(profile_vv, name=name)
