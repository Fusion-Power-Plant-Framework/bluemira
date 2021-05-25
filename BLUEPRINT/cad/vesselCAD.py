# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.

"""
Vacuum vessel CAD routines
"""
import seaborn as sns

from BLUEPRINT.cad.component import ComponentCAD
from BLUEPRINT.cad.mixins import OnionCAD
from BLUEPRINT.cad.cadtools import (  # noqa (F401)  Used in some variations commented out
    revolve,
    boolean_cut,
    boolean_fuse,
    make_face,
    make_mixed_shell,
    make_shell,
    extrude,
    make_compound,
    make_vector,
)
from BLUEPRINT.base.palettes import BLUE


class VesselCAD(OnionCAD, ComponentCAD):
    """
    Vacuum vessel CAD constructor class

    Parameters
    ----------
    vessel: Systems::VacuumVessel object
        The vacuum vessel object for which to build the CAD
    """

    def __init__(self, vessel, **kwargs):
        if hasattr(vessel, "color"):
            palette = [vessel.color]
        else:
            palette = sns.color_palette([BLUE["VV"]])
        ComponentCAD.__init__(
            self,
            "Reactor vacuum vessel",
            vessel.geom,
            vessel.params.n_TF,
            palette=palette,
            n_colors=4,
            **kwargs
        )

    def build_(self, **kwargs):
        """
        Build the CAD for the vacuum vessel.
        """
        vessel, n_TF = self.args
        vv, _ = self.ring(vessel, n_TF)
        self.add_shape(vv, name="vessel")

    def build_neutronics(self, **kwargs):
        """
        Build the neutronics CAD for the vacuum vessel.
        """
        v_vessel, n_TF = self.args
        vv, _ = self.ring(v_vessel, n_TF, full=True)  # Full revolve
        # vv = fix_shape(vv)
        self.add_shape(vv, name="vessel")

    # =============================================================================
    #         self.build(**kwargs)
    #         vv = self.component['shapes'][0]
    #         ports = self.component['shapes'][1]
    #         sectors = self.part_pattern(vv, n_TF)
    #         ports = self.part_pattern(ports, n_TF)
    #         for i, (s, p) in enumerate(zip(sectors, ports)):
    #             self.add_shape(s, name=f'vessel_sector_{i}')
    #             self.add_shape(p, name=f'ports_sector_{i}')
    #         self.remove_shape('Reactor vacuum vessel_vessel')
    #         self.remove_shape('Reactor vacuum vessel_ports')
    # =============================================================================

    def build(self, **kwargs):  # define component shapes
        """
        Deprecated, but kept for future flexibility
        """
        v_vessel, n_TF = self.args
        v = v_vessel["2D profile"].rotate(
            -180 / n_TF, p1=[0, 0, 0], p2=[0, 0, 1], update=False
        )
        vo, vi = v_vessel["2D profile"].outer.copy(), v_vessel["2D profile"].inner.copy()
        vo.rotate(-180 / n_TF, p1=[0, 0, 0], p2=[0, 0, 1])
        vi.rotate(-180 / n_TF, p1=[0, 0, 0], p2=[0, 0, 1])
        cut = revolve(make_face(vo, spline=True), None, 360)
        # Took a long time to find a good solution here, and sadly didn't find
        # one. The only way seemed was this. Let's hope it doesn't come up
        # again... The offset cuts off potential construction failures.
        # cuti = revolve(make_face(vi.offset(0.001), spline=True), None, 360)
        # Volvera..

        vv = make_shell(v, spline=False)

        vv = revolve(vv, None, 360 / n_TF)
        # vv = boolean_cut(cut, cuti)
        up = v_vessel["Upper port"]
        zref = up.inner["z"][0]
        up = up.translate([0, 0, -zref], update=False)  # Bring to midplane
        up = make_shell(up)
        upo = extrude(up, length=zref, axis="z")
        upi = v_vessel["Upper port"].inner
        # upi = upi.translate([0, 0, -zref], update=False)
        upoo = boolean_cut(upo, cut)
        cut_up = extrude(make_face(upi), length=-zref, axis="z")

        vv = boolean_cut(vv, cut_up)
        # vv = boolean_cut(vv, cuti)

        ep = make_shell(v_vessel["Equatorial port"])
        eqpo = extrude(ep, length=-10, axis="x")
        eqpoo = boolean_cut(eqpo, cut)
        cut_eq = extrude(
            make_face(v_vessel["Equatorial port"].inner), length=-10, axis="x"
        )
        vv = boolean_cut(vv, cut_eq)

        # Messy, slow, painful, verbose. Sad!
        # I will build a great wall - and nobody builds walls better than me
        lpvec = make_vector(v_vessel["LP path"][0], v_vessel["LP path"][1])
        lpi = make_face(v_vessel["Lower port"].inner)
        lp = make_shell(v_vessel["Lower port"])
        lpi = extrude(lpi, vec=-lpvec)
        lp = extrude(lp, vec=-lpvec)
        lpo = boolean_cut(lp, cut)
        vv = boolean_cut(vv, lpi)
        # Duct
        # di = v_vessel["Lower duct"].inner
        # do = v_vessel["Lower duct"].outer
        # dl = v_vessel["LP path"]["x"][-2] - v_vessel["LP path"]["x"][-1]
        # lpd = make_shell(v_vessel["Lower duct"])
        # lpd = extrude(lpd, length=dl, axis="x")
        # dp = do.translate([dl, 0, 0], update=False)

        # pad = extrude(make_face(dp), length=-0.2, axis="x")
        # duct = boolean_fuse(pad, lpd)
        # LPO = boolean_fuse(LPO, duct)
        lpo = boolean_cut(lpo, lpi)
        # duct_cut = extrude(make_face(di), length=dl, axis="x")
        # LPO = boolean_cut(LPO, duct_cut)
        # vv = boolean_fuse(vv, upoo)
        # vv = boolean_fuse(vv, eqpoo)
        # vv = boolean_fuse(vv, LPO)

        self.add_shape(vv, name="vessel")
        ports = make_compound([upoo, eqpoo, lpo])
        self.add_shape(ports, name="ports")

        # self.add_shape(upoo, name='upper_port')
        # self.add_shape(eqpoo, name='eq_port')
        # self.add_shape(LPO, name='lower_port')


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
