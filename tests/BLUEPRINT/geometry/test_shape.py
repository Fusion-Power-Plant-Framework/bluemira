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

from bluemira.base.file import get_bluemira_path, make_bluemira_path
from bluemira.geometry.error import GeometryError
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.shape import Shape, fit_shape_to_loop


class TestShape:
    def test_init(self):
        for fam in ["S", "A", "P", "D"]:
            for obj in ["L", "V"]:
                Shape(
                    "tester",
                    family=fam,
                    objective=obj,
                    npoints=200,
                    symmetric=False,
                    read_write=False,
                    directory=None,
                )
        with pytest.raises(GeometryError):
            Shape(
                "tester",
                family="J",
                objective="L",
                npoints=200,
                symmetric=False,
                read_write=False,
                directory=None,
            )
        with pytest.raises(GeometryError):
            Shape(
                "tester",
                family="D",
                objective="Y",
                npoints=200,
                symmetric=False,
                read_write=False,
                directory=None,
            )

    def test_write(self):
        write_directory = make_bluemira_path(
            "BLUEPRINT/geometry/test_generated_data", subfolder="tests"
        )
        shp = Shape(
            "tester_S",
            family="S",
            objective="L",
            npoints=200,
            n_TF=18,
            symmetric=False,
            read_write=True,
            write_directory=write_directory,
        )
        shp.write()
        assert os.path.isfile(shp.write_filename)
        os.remove(shp.write_filename)

    def test_readwrite(self):
        read_directory = get_bluemira_path(
            "BLUEPRINT/geometry/test_data", subfolder="tests"
        )
        write_directory = make_bluemira_path(
            "BLUEPRINT/geometry/test_generated_data", subfolder="tests"
        )
        shp = Shape(
            "tester_S_rw",
            family="S",
            objective="V",
            npoints=200,
            n_TF=18,
            symmetric=False,
            read_write=True,
            read_directory=read_directory,
            write_directory=write_directory,
        )
        shp.optimise()
        result = shp.parameterisation.draw()
        shp.write()

        try:
            shp = Shape(
                "tester_S_rw",
                family="S",
                objective="V",
                npoints=200,
                n_TF=18,
                symmetric=False,
                read_write=True,
                read_directory=read_directory,
                write_directory=write_directory,
            )
            result2 = shp.parameterisation.draw()
            for k, v in result.items():
                for kk, vv in result2.items():
                    if k == kk:
                        if isinstance(v, np.ndarray):
                            assert v.all() == vv.all()
                        else:
                            assert v == vv
        finally:
            os.remove(shp.write_filename)


class TestShapeFitting:
    def test_s_shape(self):
        shp = Shape("test", "S")

        shp.adjust_xo("x1", value=5)

        reference = Loop(**shp.parameterisation.draw())

        fitter = fit_shape_to_loop("S", reference)

        assert np.isclose(fitter.parameterisation.xo["x1"]["value"], 5, rtol=1e-3)

    def test_d_shape(self):
        shp = Shape("test", "D")

        shp.adjust_xo("x1", value=5)

        reference = Loop(**shp.parameterisation.draw())

        fitter = fit_shape_to_loop("D", reference)

        assert np.isclose(fitter.parameterisation.xo["x1"]["value"], 5, rtol=1e-3)

    def test_p_shape(self):
        shp = Shape("test", "P")

        shp.adjust_xo("x1", value=5)
        shp.adjust_xo("ri", value=0.0)

        reference = Loop(**shp.parameterisation.draw())

        fitter = fit_shape_to_loop("P", reference)

        assert np.isclose(fitter.parameterisation.xo["x1"]["value"], 5, rtol=1e-3)
