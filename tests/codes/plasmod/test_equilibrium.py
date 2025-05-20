# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import tempfile
from dataclasses import dataclass

import numpy as np
import pytest
from dolfinx.mesh import Mesh

from bluemira.base.components import PhysicalComponent
from bluemira.base.constants import EPS
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.codes.plasmod.equilibrium_2d_coupling import (
    PlasmaFixedBoundaryParams,
    TransportSolverParams,
    _update_delta_kappa,
    create_plasma_xz_cross_section,
    solve_transport_fixed_boundary,
)
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemGradShafranovFixedBoundary,
)
from bluemira.equilibria.fem_fixed_boundary.utilities import create_mesh
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_circle


def test_update_delta_kappa_it_err():
    pfb = PlasmaFixedBoundaryParams(1, 3, 5, 7, 9, 11)
    iter_err = _update_delta_kappa(pfb, 10, 8, 6, 4, 2)
    assert pfb.delta_u == pytest.approx(12.0, rel=0, abs=EPS)
    assert pfb.kappa_u == pytest.approx(6.0, rel=0, abs=EPS)
    assert iter_err == pytest.approx(0.5, rel=0, abs=EPS)


@dataclass
class DummyTransportSolverParams(ParameterFrame):
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
    transport_params = DummyTransportSolverParams.from_dict({
        "V_p": {"value": -2500, "unit": "m^3"},
        "kappa_95": {"value": 1, "unit": ""},
        "delta_95": {"value": 1, "unit": ""},
        "delta": {"value": 0.5, "unit": ""},
        "kappa": {"value": 0.5, "unit": ""},
    })
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
    (mesh, _ct, _f), _labels = create_mesh(plasma, tmp_dir, "file.msh")

    assert isinstance(mesh, Mesh)


@dataclass
class TranspOutParams(ParameterFrame):
    """Transport Solver ParameterFrame"""

    I_p: Parameter[float]
    B_0: Parameter[float]
    R_0: Parameter[float]


class DummyTransportSolver:
    name = "DUMMY"
    params = TransportSolverParams.from_dict({
        "I_p": {"value": 15e6, "unit": "A"},
        "B_0": {"value": 5, "unit": "T"},
        "R_0": {"value": 9, "unit": "m"},
        "A": {"value": 3.1, "unit": "m"},
        "V_p": {"value": 2000, "unit": "m^3"},
        "v_burn": {"value": 0.02, "unit": "V"},
        "kappa_95": {"value": 1.5, "unit": ""},
        "delta_95": {"value": 0.4, "unit": ""},
        "delta": {"value": 0.33, "unit": ""},
        "kappa": {"value": 1.6, "unit": ""},
        "q_95": {"value": 3.25, "unit": ""},
        "f_ni": {"value": 0.1, "unit": ""},
    })

    def __init__(self):
        self.i = 0
        self.n = 50
        self.x = np.linspace(0, 1, self.n)

    def execute(self, mode):  # noqa: ARG002
        return self.params

    def get_profile(self, prof):
        self.i += 1
        if prof == "x":
            return self.x
        if prof == "pprime":
            return np.gradient(self.get_profile("psi"))
        if prof == "ffprime":
            return 10 * self.x[::-1]
        if prof == "psi":
            return 5 * self.x
        if prof == "q":
            return 1 + np.linspace(0, np.sqrt(2.25), self.n) ** 2
        if prof == "pressure":
            return 1e6 * self.x[::-1]
        raise ValueError(f"unknown profile {prof}")


class TestSolveTransportFixedBoundary:
    transport_solver = DummyTransportSolver()
    gs_solver = FemGradShafranovFixedBoundary(
        p_order=2,
        maxiter=30,
        iter_err_max=1.0,
        relaxation=0,
    )

    @pytest.mark.parametrize(
        ("maxiter", "message"),
        [
            (1, "did not"),
        ],
    )
    def test_full_run_through(self, maxiter, message, caplog, tmp_path):
        johner_parameterisation = JohnerLCFS({
            "r_0": {"value": 8.9830e00},
            "a": {"value": 8.983 / 3.1},
            "kappa_u": {"value": 1.6},
            "kappa_l": {"value": 1.75},
            "delta_u": {"value": 0.33},
            "delta_l": {"value": 0.45},
        })
        solve_transport_fixed_boundary(
            johner_parameterisation,
            self.transport_solver,
            self.gs_solver,
            1.5,
            0.4,
            iter_err_max=1e-3,
            inner_iter_err_max=1,
            maxiter=maxiter,
            lcar_mesh=0.3,
            refine=True,
            num_levels=1,
            distance=0.1,
            directory=tmp_path,
        )
        assert message in caplog.messages[-1]
