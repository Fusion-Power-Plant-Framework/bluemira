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

import os
import numpy as np
import pytest

from bluemira.base.file import get_bluemira_path
from bluemira.geometry._deprecated_loop import Loop
from bluemira.equilibria import Equilibrium
from bluemira.equilibria.error import FluxSurfaceError
from bluemira.equilibria.flux_surfaces import (
    ClosedFluxSurface,
    OpenFluxSurface,
    PartialOpenFluxSurface,
    FieldLineTracer,
)

TEST_PATH = get_bluemira_path("bluemira/equilibria/test_data", subfolder="tests")


class TestOpenFluxSurfaceStuff:
    @classmethod
    def setup_class(cls):
        eq_name = "eqref_OOB.json"
        filename = os.sep.join([TEST_PATH, eq_name])
        cls.eq = Equilibrium.from_eqdsk(filename)

    def test_bad_geometry(self):
        closed_loop = Loop(x=[0, 4, 5, 8, 0], z=[1, 2, 3, 4, 1])
        with pytest.raises(FluxSurfaceError):
            _ = OpenFluxSurface(closed_loop)
        with pytest.raises(FluxSurfaceError):
            _ = PartialOpenFluxSurface(closed_loop)

    def test_connection_length(self):
        """
        Use both a flux surface and field line tracing approach to calculate connection
        length and check they are the same or similar.
        """
        x_start, z_start = 12, 0
        loop = self.eq.get_flux_surface_through_point(x_start, z_start)
        fs = OpenFluxSurface(loop)
        lfs, hfs = fs.split(self.eq.get_OX_points()[0][0])
        l_lfs = lfs.connection_length(self.eq)
        l_hfs = hfs.connection_length(self.eq)

        # test discretisation sensitivity
        lfs_loop = lfs.loop.copy()
        lfs_loop.interpolate(3 * len(lfs_loop))
        lfs_interp = PartialOpenFluxSurface(lfs_loop)
        l_lfs_interp = lfs_interp.connection_length(self.eq)
        assert np.isclose(l_lfs, l_lfs_interp, rtol=5e-3)

        hfs_loop = hfs.loop.copy()
        hfs_loop.interpolate(3 * len(hfs_loop))
        hfs_interp = PartialOpenFluxSurface(hfs_loop)
        l_hfs_interp = hfs_interp.connection_length(self.eq)
        assert np.isclose(l_hfs, l_hfs_interp, rtol=5e-3)

        # compare with field line tracer
        flt = FieldLineTracer(self.eq)
        l_flt_lfs = flt.trace_field_line(x_start, z_start, n_turns_max=20, forward=True)
        l_flt_hfs = flt.trace_field_line(
            x_start, z_start, n_turns_max=20, forward=False
        ).connection_length
        print(len(l_flt_lfs.loop))
        assert np.isclose(l_flt_lfs.connection_length, l_lfs, rtol=2e-2)
        assert np.isclose(l_flt_hfs, l_hfs, rtol=2e-2)


class TestClosedFluxSurface:
    def test_bad_geometry(self):
        open_loop = Loop(x=[0, 4, 5, 8], z=[1, 2, 3, 4])
        with pytest.raises(FluxSurfaceError):
            _ = ClosedFluxSurface(open_loop)


class TestFieldLine:
    @classmethod
    def setup_class(cls):
        eq_name = "eqref_OOB.json"
        filename = os.sep.join([TEST_PATH, eq_name])
        cls.eq = Equilibrium.from_eqdsk(filename)

    def test_connection_length(self):
        flt = FieldLineTracer(self.eq)
        field_line = flt.trace_field_line(13, 0, n_points=1000)
        assert np.isclose(
            field_line.connection_length, field_line.loop.length, rtol=5e-2
        )


if __name__ == "__main__":
    pytest.main([__file__])
