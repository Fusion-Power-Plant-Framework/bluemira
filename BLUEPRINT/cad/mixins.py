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
CAD DRY mixin clases
"""
import numpy as np

from BLUEPRINT.cad.cadtools import (
    boolean_cut,
    boolean_fuse,
    extrude,
    make_compound,
    make_face,
    make_shell,
    make_vector,
    revolve,
    sew_shapes,
)
from BLUEPRINT.geometry.loop import Loop


class PlugChopper:
    """
    DRY mixin class for plug handling in RadiationShield and CryostatVV
    """

    def plugs_for_neutronics(self, plugs=None):
        """
        Add the plugs to the shapes for neutronics models.
        """
        if plugs is None:
            for i, p in enumerate(self._plugs):
                self.add_shape(p, name=f"Plugs_{i}")
        else:
            for i, p in enumerate(plugs):
                self.add_shape(p, name=f"Plug_{i}")

    @staticmethod
    def _make_cutters(loop):
        """
        Makes the cutting revolutions for the plug overshoots (details)
        """
        topargs = np.where((loop.x == 0) & (loop.z > 0))[0]
        botargs = np.where((loop.x == 0) & (loop.z < 0))[0]
        cutin = Loop(x=loop.x[: topargs[0] + 1], z=loop.z[: topargs[0] + 1])
        cutin.close()
        cutout = Loop(
            x=loop.x[topargs[1] : botargs[1] + 1][::-1],
            z=loop.z[topargs[1] : botargs[1] + 1][::-1],
        )
        out = cutout.offset(5)
        x = np.concatenate((cutout.x, out.x[::-1]))
        z = np.concatenate((cutout.z, out.z[::-1]))
        cutout = Loop(x=x, z=z)
        cutin = revolve(make_face(cutin), None, angle=360)
        cutout = revolve(make_face(cutout), None, angle=360)
        return cutin, cutout


class OnionCAD:
    """
    DRY class.
    Provides the ability to build an onion ring for VV/TS objects
    """

    def ring(self, geom, n_TF, full=False, ts=False):
        """
        Default ring building,
        """
        onion_ring = geom
        v = onion_ring["2D profile"].rotate(
            -180 / n_TF, p1=[0, 0, 0], p2=[0, 0, 1], update=False
        )
        vo = onion_ring["2D profile"].outer
        cut = revolve(make_face(vo, spline=True), None, 360)
        # vi = onion_ring["2D profile"].inner
        # Took a long time to find a good solution here, and sadly didn't find
        # one. The only way seemed was this. Let's hope it doesn't come up
        # again... The offset cuts off potential construction failures.
        # cuti = outils.revolve(outils.make_face(vli.offset(0.001), spline=True),
        #                      None, 360)
        # Volvera..
        # =====================================================================

        onion = make_shell(v, spline=True)
        if full:
            angle = 360
        else:
            angle = 360 / n_TF
        onion = revolve(onion, None, angle)
        upvec = make_vector([0, 0, onion_ring["Upper port"].inner.z[0]], [0, 0, 0])
        eqvec = make_vector(
            [onion_ring["Equatorial port"].inner.x[0], 0, 0],
            [onion_ring["LP path"].x[0], 0, 0],
        )
        lpvec = make_vector(onion_ring["LP path"][1], onion_ring["LP path"][0])
        ports = {}
        for p, vec in zip(
            ["Upper port", "Equatorial port", "Lower port"], [upvec, eqvec, lpvec]
        ):
            # po, pi = onion_ring[p].outer, onion_ring[p].inner
            s = make_shell(onion_ring[p])
            port = extrude(s, vec=vec)
            port = boolean_cut(port, cut)
            # po, pi = [make_face(f) for f in [po, pi]]
            # po, pi = [extrude(p, vec=vec) for p in [po, pi]]
            # if full:  # Full revolve
            #     pi = make_compound(self.part_pattern(pi, n_TF))
            # vv = boolean_cut(vv, pi)
            ports[p] = port
            # if full:
            #     port = make_compound(self.part_pattern(port, n_TF))
            # vv = sew_shapes(vv, port)
        return onion, ports

    def neutronics_ring(self, geom, n_TF):
        """
        Coring the apple
        """
        onion_ring = geom
        vi = onion_ring["2D profile"].inner
        v = make_shell(onion_ring["2D profile"], spline=False)
        v = revolve(v, None, 360)
        vpcut = make_face(onion_ring["2D profile"].outer.offset(-0.05), spline=False)
        upvec = make_vector([0, 0, onion_ring["Upper port"].inner.z[0]], [0, 0, 0])
        eqvec = make_vector(
            [onion_ring["Equatorial port"].inner.x[0], 0, 0],
            [onion_ring["LP path"].x[0], 0, 0],
        )
        lpvec = make_vector(onion_ring["LP path"][1], onion_ring["LP path"][0])

        for p, vec in zip(
            ["Upper port", "Equatorial port", "Lower port"], [upvec, eqvec, lpvec]
        ):
            po = make_face(onion_ring[p].outer)
            po = extrude(po, vec=vec)
            po = boolean_cut(po, vpcut)
            port = make_compound(self.part_pattern(po, n_TF))

            v = boolean_fuse(v, port)

        for p, vec in zip(
            ["Upper port", "Equatorial port", "Lower port"], [upvec, eqvec, lpvec]
        ):
            po = make_face(onion_ring[p].inner)
            po = extrude(po, vec=vec)
            port = make_compound(self.part_pattern(po, n_TF))
            v = sew_shapes([v, port])
        v = boolean_cut(v, vi)
        vv = v
        return vv, None
