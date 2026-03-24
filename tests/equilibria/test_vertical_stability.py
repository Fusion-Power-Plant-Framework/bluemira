# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import bluemira.equilibria.vertical_stability as vs
from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.coils import (
    Coil,
    CoilSet,
    make_mutual_inductance_matrix,
)
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.vertical_stability import (
    _get_coil_points_along_wire,
    make_coils_along_wire,
)
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.tools import (
    make_circle,
    make_circle_arc_3P,
    make_ellipse,
    make_polygon,
)
from bluemira.geometry.wire import BluemiraWire


class TestDiscretisation:
    @classmethod
    def setup_class(cls):
        cls.circle = make_circle(2, center=(4, 0, 0), axis=(0, 1, 0))
        cls.ellipse = make_ellipse(
            center=(4, 0, 0),
            major_radius=4,
            minor_radius=2,
            major_axis=(0, 0, 1),
            minor_axis=(1, 0, 0),
        )
        points = Coordinates({
            "x": [
                1.0,
                1.0,
                2 - 0.5 * np.sqrt(2),
                2.0,
                6.0,
                2.0,
                2 - 0.5 * np.sqrt(2),
                1.0,
            ],
            "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "z": [
                -4.0,
                4.0,
                4 + 0.5 * np.sqrt(2),
                5.0,
                0.0,
                -5.0,
                -4 - 0.5 * np.sqrt(2),
                -4.0,
            ],
        }).T

        wires = [
            make_polygon(points[:2, :]),
            make_circle_arc_3P(points[1, :], points[2, :], points[3, :]),
            make_circle_arc_3P(points[3, :], points[4, :], points[5, :]),
            make_circle_arc_3P(points[5, :], points[6, :], points[7, :]),
        ]
        cls.dshape = BluemiraWire(wires)

    def test_comparison_plot(self):
        simple_kwargs = {"edgecolor": "red", "facecolor": "white"}
        detailed_kwargs = {"edgecolor": "blue", "facecolor": "white"}
        coils_circle_simple = make_coils_along_wire(self.circle, 0.06)
        coils_circle = make_coils_along_wire(self.circle, 0.06, simple=False)
        coils_ellipse_simple = make_coils_along_wire(self.ellipse, 0.06)
        coils_ellipse = make_coils_along_wire(self.ellipse, 0.06, simple=False)
        coils_dshape_simple = make_coils_along_wire(self.dshape, 0.06)
        coils_dshape = make_coils_along_wire(self.dshape, 0.06, simple=False)
        f = plt.figure()
        ax1 = f.add_subplot(1, 3, 1)
        coils_circle_simple.plot(ax=ax1, **simple_kwargs)
        coils_circle.plot(ax=ax1, **detailed_kwargs)
        ax2 = f.add_subplot(1, 3, 2)
        coils_ellipse_simple.plot(ax=ax2, **simple_kwargs)
        coils_ellipse.plot(ax=ax2, **detailed_kwargs)
        ax3 = f.add_subplot(1, 3, 3)
        coils_dshape_simple.plot(ax=ax3, **simple_kwargs)
        coils_dshape.plot(ax=ax3, **detailed_kwargs)

    def test_coil_points(self):
        coils_circle = make_polygon(_get_coil_points_along_wire(self.circle, 0.06))
        np.testing.assert_allclose(coils_circle.length, self.circle.length, rtol=1e-2)
        coils_ellipse = make_polygon(_get_coil_points_along_wire(self.ellipse, 0.06))
        np.testing.assert_allclose(coils_ellipse.length, self.ellipse.length, rtol=1e-2)
        coils_dshape = make_polygon(_get_coil_points_along_wire(self.dshape, 0.06))
        np.testing.assert_allclose(coils_dshape.length, self.dshape.length, rtol=1e-2)


class TestFilaments:
    @classmethod
    def setup_class(cls):
        coil1 = Coil(
            2.0,
            1.0,
            0.5,
            0.5,
        )
        coil2 = Coil(
            2.0,
            -1.0,
            0.5,
            0.5,
        )
        cls.coilset1 = CoilSet(coil1, coil2)
        coil3 = Coil(2.0, 1.0, 0.5, 0.5, n_turns=4, discretisation=0.5)
        coil4 = Coil(2.0, -1.0, 0.5, 0.5, n_turns=4, discretisation=0.5)
        cls.coilset2 = CoilSet(coil3, coil4)

    def test_filaments(self):
        ind_mat1 = make_mutual_inductance_matrix(self.coilset1, square_coil=True)
        ind_mat2 = make_mutual_inductance_matrix(self.coilset2, square_coil=True)
        np.testing.assert_allclose(ind_mat1, ind_mat2, rtol=1e-8)


class TestRZIp:
    @classmethod
    def setup_class(cls):
        path = get_bluemira_path("equilibria/test_data", subfolder="tests")
        cls.dn = Equilibrium.from_eqdsk(
            Path(path, "DN-DEMO_eqref_withCoilNames.json"),
            from_cocos=3,
            qpsi_positive=False,
        )
        cls.dn.coilset.control = True
        cls.ellipse = make_ellipse(
            center=(9, 0, 0),
            major_radius=6.0,
            minor_radius=3.5,
            major_axis=(0, 0, 1),
            minor_axis=(1, 0, 0),
        )

    def test_run(self):
        print(self.dn.coilset)
        passive_group = make_coils_along_wire(self.ellipse, 0.5)
        print(passive_group)
        self.dn.coilset.add_coil(*passive_group._coils)
        self.dn._remap_greens()
        vs.calculate_rzip_stability_criterion(self.dn)
