# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later


import numpy as np
import pytest
from eurofusion_materials.library.magnet_branch_mats import (
    NB3SN_MAG,
    SS316_LN_MAG,
)
from matproplib import OperationalConditions
from matproplib.material import MaterialFraction

from bluemira.magnets.cable import RectangularCable, RoundCable, SquareCable
from bluemira.magnets.strand import Strand, SuperconductingStrand
from bluemira.magnets.tfcoil_designer import TFCoilXYDesigner

DummySteel = SS316_LN_MAG
DummySuperconductor = NB3SN_MAG


@pytest.fixture
def sc_strand():
    return SuperconductingStrand(
        name="SC",
        materials=[MaterialFraction(material=DummySuperconductor, fraction=1.0)],
        d_strand=0.001,
        operating_temperature=5.7,
    )


@pytest.fixture
def stab_strand():
    return Strand(
        name="Stab",
        materials=[MaterialFraction(material=DummySteel, fraction=1.0)],
        d_strand=0.001,
        operating_temperature=5.7,
    )


@pytest.fixture
def cable(sc_strand, stab_strand):
    return RectangularCable(
        dx=0.01,
        sc_strand=sc_strand,
        stab_strand=stab_strand,
        n_sc_strand=10,
        n_stab_strand=20,
        d_cooling_channel=0.001,
        void_fraction=0.725,
        cos_theta=0.97,
    )


def test_geometry_and_area(cable):
    assert cable.dx > 0
    assert cable.dy > 0
    assert cable.area > 0
    assert cable.aspect_ratio > 0
    assert cable.area_cooling_channel > 0
    assert cable.area_stab_region > 0
    assert cable.area_sc_region > 0


def test_material_properties(cable):
    temperature = 20  # [K]
    op_cond = OperationalConditions(temperature=temperature)
    assert cable.rho(op_cond) > 0.0
    assert cable.erho(op_cond) > 0.0
    assert cable.Cp(op_cond) > 0.0


def test_str_output(cable):
    summary = str(cable)
    assert "dx" in summary
    assert "sc strand" in summary
    assert "stab strand" in summary


def test_plot(cable):
    ax = cable.plot(show=True)
    assert hasattr(ax, "fill")


def test_temperature_evolution(cable):
    def B_fun(t):  # noqa: ARG001
        return 5

    def I_fun(t):  # noqa: ARG001
        return 1000

    result = cable._temperature_evolution(0, 0.1, 20, B_fun, I_fun)
    assert result.success


def test_optimise_n_stab_ths(sc_strand, stab_strand):
    cable = RectangularCable(
        dx=0.01,
        sc_strand=sc_strand,
        stab_strand=stab_strand,
        n_sc_strand=10,
        n_stab_strand=5,
        d_cooling_channel=0.001,
        void_fraction=0.725,
        cos_theta=0.97,
    )

    def B_fun(t):  # noqa: ARG001
        return 5

    def I_fun(t):  # noqa: ARG001
        return 1000

    result = TFCoilXYDesigner.optimise_cable_n_stab_ths(
        cable,
        t0=0,
        tf=0.1,
        initial_temperature=20,
        target_temperature=80,
        B_fun=B_fun,
        I_fun=I_fun,
        bounds=(1, 100),
    )
    assert result.solution.success


def test_square_and_round_cables(sc_strand, stab_strand):
    square = SquareCable(
        sc_strand=sc_strand,
        stab_strand=stab_strand,
        n_sc_strand=5,
        n_stab_strand=5,
        d_cooling_channel=0.001,
        void_fraction=0.725,
        cos_theta=0.97,
        E=0.1e9,
    )
    round_ = RoundCable(
        sc_strand=sc_strand,
        stab_strand=stab_strand,
        n_sc_strand=5,
        n_stab_strand=5,
        d_cooling_channel=0.001,
        void_fraction=0.725,
        cos_theta=0.97,
        E=0.1e9,
    )
    dummy_op_cond = OperationalConditions(temperature=4.0)
    assert square.dx > 0
    assert square.dy > 0
    assert np.isclose(square.dx, square.dy, rtol=1e-8)
    assert square.E(dummy_op_cond) > 0
    # since dx == dy, Kx == Ky == E
    assert np.isclose(square.Kx(dummy_op_cond), square.E(dummy_op_cond), rtol=1e-8)
    assert np.isclose(square.Ky(dummy_op_cond), square.E(dummy_op_cond), rtol=1e-8)

    assert round_.dx > 0
    assert round_.dy > 0
    assert np.isclose(round_.dx, round_.dy, rtol=1e-8)
    assert round_.E(dummy_op_cond) > 0
    # since dx == dy for round cable, Kx == Ky == E
    assert np.isclose(round_.Kx(dummy_op_cond), round_.E(dummy_op_cond), rtol=1e-8)
    assert np.isclose(round_.Ky(dummy_op_cond), round_.E(dummy_op_cond), rtol=1e-8)


def test_cable_to_dict(sc_strand, stab_strand):
    # Create a RectangularCable for testing
    cable_original = RectangularCable(
        dx=0.01,
        sc_strand=sc_strand,
        stab_strand=stab_strand,
        n_sc_strand=10,
        n_stab_strand=5,
        d_cooling_channel=0.001,
        void_fraction=0.725,
        cos_theta=0.97,
    )

    # Convert to dictionary
    cable_dict = cable_original.to_dict(OperationalConditions(temperature=5))

    assert cable_dict["n_sc_strand"] == 10
