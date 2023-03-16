# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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

import tempfile
from dataclasses import dataclass
from unittest import mock

import numpy as np
import pytest
from dolfin import Mesh

from bluemira.base.components import PhysicalComponent
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.equilibria.fem_fixed_boundary.equilibrium import (
    PlasmaFixedBoundaryParams,
    _update_delta_kappa,
    create_mesh,
    create_plasma_xz_cross_section,
    solve_transport_fixed_boundary,
)
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_circle


def test_update_delta_kappa_it_err():
    pfb = PlasmaFixedBoundaryParams(1, 3, 5, 7, 9, 11)
    iter_err = _update_delta_kappa(pfb, 10, 8, 6, 4, 2)
    assert pfb.delta_u == 12.0
    assert pfb.kappa_u == 6.0
    assert iter_err == 0.5


@dataclass
class TransportSolverParams(ParameterFrame):
    """Transport Solver ParameterFrame"""

    V_p: Parameter[float]
    kappa_95: Parameter[float]
    delta_95: Parameter[float]
    delta: Parameter[float]
    kappa: Parameter[float]


@dataclass
class Params:
    kappa_u: float
    kappa_l: float
    delta_u: float
    delta_l: float

    def tabulate(self):
        return ""


class DummyCircle:
    def create_shape(self):
        return make_circle(center=(1, 1, 1), radius=1)


def test_plasma_xz_cross_section():
    pp = Params(1, 1, 1, 1)
    transport_params = TransportSolverParams.from_dict(
        {
            "V_p": {"value": -2500, "unit": "m^3"},
            "kappa_95": {"value": 1, "unit": ""},
            "delta_95": {"value": 1, "unit": ""},
            "delta": {"value": 0.5, "unit": ""},
            "kappa": {"value": 0.5, "unit": ""},
        }
    )
    lcfs_options = {"face": {"lcar": 1}, "lcfs": {"lcar": 2}}
    plasma = create_plasma_xz_cross_section(
        DummyCircle(), transport_params, pp, 1.652, 0.333, lcfs_options, "source"
    )

    assert isinstance(plasma.shape, BluemiraFace)
    assert plasma.shape.mesh_options.lcar == 1
    assert plasma.shape.boundary[0].mesh_options.lcar == 2
    assert isinstance(plasma, PhysicalComponent)

    assert transport_params.V_p.value == pytest.approx(19.7392)

    for tr in transport_params:
        assert len(tr.history()) == 2
        assert tr.source == "source"


def test_create_mesh():
    tmp_dir = tempfile.mkdtemp()
    plasma = PhysicalComponent("Plasma", shape=BluemiraFace(JohnerLCFS().create_shape()))
    lcfs = plasma.shape
    # Set mesh options
    lcfs.boundary[0].mesh_options = {"lcar": 0.15, "physical_group": "lcfs"}
    lcfs.mesh_options = {"lcar": 0.15, "physical_group": "plasma_face"}
    mesh = create_mesh(plasma, tmp_dir, "file", "file.msh")

    assert isinstance(mesh, Mesh)


@dataclass
class TranspOutParams(ParameterFrame):
    """Transport Solver ParameterFrame"""

    I_p: Parameter[float]
    B_0: Parameter[float]
    R_0: Parameter[float]


class DummyTransportSolver:
    name = "DUMMY"

    def __init__(self):
        self.i = 0

    def execute(self, mode):
        return TranspOutParams.from_dict(
            {
                "I_p": {"value": 5, "unit": "A"},
                "B_0": {"value": 5, "unit": "T"},
                "R_0": {"value": 9, "unit": "m"},
            }
        )

    def get_profile(self, prof):
        self.i += 1
        return np.linspace(0, self.i, num=10)
