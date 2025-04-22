# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Tests for EU-DEMO Maintenance
"""

import numpy as np
import pytest

from bluemira.base.error import BuilderError
from bluemira.base.parameter_frame import Parameter
from bluemira.display.displayer import show_cad
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import (
    boolean_fuse,
    distance_to,
    extrude_shape,
    make_circle,
    make_polygon,
)
from bluemira.geometry.wire import BluemiraWire
from eudemo.maintenance.duct_connection import (
    VVUpperPortDuctBuilder,
    VVUpperPortDuctBuilderParams,
)
from eudemo.maintenance.upper_port import UpperPortKOZDesigner


class TestUpperPortDesigner:
    """Test Upper Port"""

    def test_dummy_blanket_port_design(self):
        """Test Upper Port Optimiser"""
        params = {
            "c_rm": {"value": 0.02, "unit": "m"},
            "R_0": {"value": 9, "unit": "m"},
            "bb_min_angle": {"value": 70, "unit": "degrees"},
            "tk_bb_ib": {"value": 0.8, "unit": "m"},
            "tk_bb_ob": {"value": 1.1, "unit": "m"},
            "tk_vv_double_wall": {"value": 0.04, "unit": "m"},
            "g_vv_ts": {"value": 0.05, "unit": "m"},
            "tk_ts": {"value": 0.06, "unit": "m"},
            "g_ts_tf": {"value": 0.07, "unit": "m"},
            "pf_s_g": {"value": 0.08, "unit": "m"},
            "pf_s_tk_plate": {"value": 0.09, "unit": "m"},
        }
        bb = make_polygon(
            {
                "x": [5, 6, 6, 11, 11, 12, 12, 5],
                "z": [-5, -5, 5, 5, -5, -5, 6, 6],
            },
            closed=True,
        )
        bb = BluemiraFace(bb)
        designer = UpperPortKOZDesigner(params, {}, bb, upper_port_extrema=10)

        up_zone, r_cut, cut_angle = designer.run()

        bbox = up_zone.bounding_box
        assert bbox.x_min == pytest.approx(6.37)
        assert bbox.x_max == pytest.approx(12.41)
        assert bbox.y_min == pytest.approx(bbox.y_max) == pytest.approx(0)
        assert bbox.z_min == pytest.approx(0)
        assert bbox.z_max == pytest.approx(10)
        assert r_cut == pytest.approx(8.52)
        assert cut_angle == pytest.approx(0)


class TestDuctConnection:
    port_kozs = (
        BluemiraFace(
            make_polygon({"x": [5, 10, 10, 5], "z": [4, 4, 10, 10]}, closed=True)
        ),
        BluemiraFace(
            make_polygon({"x": [0.1, 10, 10, 0.1], "z": [4, 4, 15, 15]}, closed=True)
        ),
    )

    def setup_method(self):
        self.params = VVUpperPortDuctBuilderParams(
            Parameter("n_TF", 12, ""),
            Parameter("tf_wp_depth", 0.0, "m"),
            Parameter("g_ts_tf", 0.00, "m"),
            Parameter("tk_ts", 0.00, "m"),
            Parameter("g_vv_ts", 0.00, "m"),
            Parameter("g_cr_ts", 0.00, "m"),
            Parameter("tk_vv_double_wall", 0.1, "m"),
            Parameter("tk_vv_single_wall", 0.05, "m"),
        )
        self.port_koz = self.port_kozs[0]

    def _wires(self, arc: BluemiraWire):
        return [
            make_polygon({
                "x": [0, point[0][0]],
                "y": [0, point[1][0]],
                "z": point[2],
            })
            for point in (arc.start_point(), arc.end_point())
        ]

    def _make_sectors(self, port_koz, angle):
        o_c = make_circle(port_koz.bounding_box.x_max, end_angle=angle)
        i_c = make_circle(port_koz.bounding_box.x_min, end_angle=angle)
        o_wires = self._wires(o_c)
        o_sector = BluemiraWire((o_wires[0], o_c, o_wires[1]))
        i_wires = self._wires(i_c)
        i_sector = BluemiraWire((i_wires[0], i_c, i_wires[1]))
        return o_sector, i_sector, o_wires, o_c

    @pytest.mark.parametrize("port_koz", port_kozs)
    @pytest.mark.parametrize("n_TF", [5, 9, 12, 16])
    @pytest.mark.parametrize("port_wall", np.linspace(0.1, 0.2, num=3))
    @pytest.mark.parametrize("y_offset", np.linspace(0, 1, num=5))
    def test_extrusion_shape(self, port_koz, n_TF, y_offset, port_wall):
        angle = 2 * np.rad2deg(np.pi / n_TF)
        self.params.n_TF.value = n_TF
        self.params.tk_vv_single_wall.value = port_wall
        self.params.tk_vv_double_wall.value = port_wall * 2
        self.params.tf_wp_depth.value = y_offset
        builder = VVUpperPortDuctBuilder(self.params, {}, port_koz, port_koz)
        port = builder.build()
        xy = port.get_component("xy").get_component_properties("shape")
        diff = xy.wires[0].length - xy.wires[1].length

        # is the bigger wire inside the smaller wire
        assert diff > 0

        for no, (e1, e2) in enumerate(
            zip(
                sorted(
                    xy.wires[0].edges,
                    key=lambda x: (*tuple(-x.start_point().xy.flatten()), -x.length),
                ),
                sorted(
                    xy.wires[1].edges,
                    key=lambda x: (*tuple(-x.start_point().xy.flatten()), -x.length),
                ),
                strict=False,
            )
        ):
            # Are the edges of the internal and external wires parallel
            v1 = (e1.end_point().xy - e1.start_point().xy).flatten()
            v2 = (e2.end_point().xy - e2.start_point().xy).flatten()
            assert np.isclose(np.cross(v1, v2), 0)

            # Are they the right distance apart
            if no in {0, 3}:
                assert np.isclose(distance_to(e1, e2)[0], port_wall * 2)
            else:
                assert np.isclose(distance_to(e1, e2)[0], port_wall)

        o_sector, i_sector, o_wires, o_c = self._make_sectors(port_koz, angle)

        # Are they the right distance from the y offset
        for o_w in o_wires:
            assert np.isclose(
                min(distance_to(o_w, edge)[0] for edge in xy.wires[0].edges), y_offset
            )
        # Are the outer edges in the same place
        assert np.isclose(
            min(distance_to(o_c, edge)[0] for edge in xy.wires[0].edges), 0
        )

        # Is the extruded shape sensible
        xyz = port.get_component("xyz").get_component_properties("shape")
        cylinder = extrude_shape(
            BluemiraFace((o_sector, i_sector)),
            (0, 0, port_koz.bounding_box.z_max),
        )
        show_cad([cylinder, xyz])
        finalshape = boolean_fuse([cylinder, xyz])
        assert np.allclose(finalshape.volume, cylinder.volume)

    def test_ValueError_on_zero_wal_thickness(self):
        self.params.tk_vv_single_wall.value = 0

        with pytest.raises(ValueError):  # noqa: PT011
            VVUpperPortDuctBuilder(self.params, {}, self.port_koz, self.port_koz)

    @pytest.mark.parametrize("end", [1, 5])
    def test_BuilderError_on_too_small_port(self, end):
        # raises an error in two different places
        self.params.tk_vv_single_wall.value = 0.5
        self.params.tk_vv_double_wall.value = end
        self.params.tf_wp_depth.value = 2.0
        builder = VVUpperPortDuctBuilder(self.params, {}, self.port_koz, self.port_koz)

        with pytest.raises(BuilderError):
            builder.build()
