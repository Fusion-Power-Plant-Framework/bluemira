# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later


import matplotlib.pyplot as plt
import numpy as np
import pytest
from eurofusion_materials.library.magnet_branch_mats import (
    NB3SN_MAG,
    SS316_LN_MAG,
)
from matproplib.material import MaterialFraction

from bluemira.magnets.cable import (
    DummyRoundCableLTS,
    DummySquareCableLTS,
    RectangularCable,
)
from bluemira.magnets.strand import Strand, SuperconductingStrand

DummySteel = SS316_LN_MAG
DummySuperconductor = NB3SN_MAG

# -- Pytest Fixtures ----------------------------------------------------------


@pytest.fixture
def sc_strand():
    return SuperconductingStrand(
        name="SC",
        materials=[MaterialFraction(material=DummySuperconductor, fraction=1.0)],
        d_strand=0.001,
    )


@pytest.fixture
def stab_strand():
    return Strand(
        name="Stab",
        materials=[MaterialFraction(material=DummySteel, fraction=1.0)],
        d_strand=0.001,
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
    )


# -- Core Cable Tests ---------------------------------------------------------


def test_geometry_and_area(cable):
    assert cable.dx > 0
    assert cable.dy > 0
    assert cable.area > 0
    assert cable.aspect_ratio > 0
    assert cable.area_cc > 0
    assert cable.area_stab > 0
    assert cable.area_sc > 0


def test_material_properties(cable):
    temperature = 20  # [K]
    assert cable.rho(temperature=temperature) > 0.0
    assert cable.erho(temperature=temperature) > 0.0
    assert cable.Cp(temperature=temperature) > 0.0


def test_str_output(cable):
    summary = str(cable)
    assert "dx" in summary
    assert "sc strand" in summary
    assert "stab strand" in summary


def test_plot(monkeypatch, cable):
    monkeypatch.setattr(plt, "show", lambda: None)
    ax = cable.plot(show=True)
    assert hasattr(ax, "fill")


def test_temperature_evolution(cable):
    def B_fun(t):  # noqa: ARG001
        return 5

    def I_fun(t):  # noqa: ARG001
        return 1000

    result = cable._temperature_evolution(0, 0.1, 20, B_fun, I_fun)
    assert result.success


def test_optimize_n_stab_ths(monkeypatch, sc_strand, stab_strand):
    cable = RectangularCable(
        dx=0.01,
        sc_strand=sc_strand,
        stab_strand=stab_strand,
        n_sc_strand=10,
        n_stab_strand=5,
        d_cooling_channel=0.001,
    )
    monkeypatch.setattr(plt, "show", lambda: None)

    def B_fun(t):  # noqa: ARG001
        return 5

    def I_fun(t):  # noqa: ARG001
        return 1000

    result = cable.optimize_n_stab_ths(
        t0=0,
        tf=0.1,
        initial_temperature=20,
        target_temperature=80,
        B_fun=B_fun,
        I_fun=I_fun,
        bounds=(1, 100),
    )
    assert result.success


def test_invalid_parameters(sc_strand, stab_strand):
    with pytest.raises(ValueError, match="dx must be positive"):
        RectangularCable(
            dx=-0.01,
            sc_strand=sc_strand,
            stab_strand=stab_strand,
            n_sc_strand=10,
            n_stab_strand=5,
            d_cooling_channel=0.001,
        )

    with pytest.raises(ValueError, match="void_fraction must be between 0 and 1"):
        RectangularCable(
            dx=0.01,
            sc_strand=sc_strand,
            stab_strand=stab_strand,
            n_sc_strand=10,
            n_stab_strand=5,
            d_cooling_channel=0.001,
            void_fraction=1.5,
        )


# -- Square & Round Cable Types -----------------------------------------------


def test_square_and_round_cables(sc_strand, stab_strand):
    square = DummySquareCableLTS(
        sc_strand=sc_strand,
        stab_strand=stab_strand,
        n_sc_strand=5,
        n_stab_strand=5,
        d_cooling_channel=0.001,
    )
    round_ = DummyRoundCableLTS(
        sc_strand=sc_strand,
        stab_strand=stab_strand,
        n_sc_strand=5,
        n_stab_strand=5,
        d_cooling_channel=0.001,
    )
    assert square.dx > 0
    assert square.dy > 0
    assert np.isclose(square.dx, square.dy, rtol=1e-8)
    assert square.E() > 0
    # since dx == dy, Kx == Ky == E
    assert np.isclose(square.Kx(), square.E(), rtol=1e-8)
    assert np.isclose(square.Ky(), square.E(), rtol=1e-8)

    assert round_.dx > 0
    assert round_.dy > 0
    assert np.isclose(round_.dx, round_.dy, rtol=1e-8)
    assert round_.E() > 0
    # since dx == dy for round cable, Kx == Ky == E
    assert np.isclose(round_.Kx(), round_.E(), rtol=1e-8)
    assert np.isclose(round_.Ky(), round_.E(), rtol=1e-8)


def test_cable_to_from_dict(sc_strand, stab_strand):
    # Create a RectangularCable for testing
    cable_original = RectangularCable(
        dx=0.01,
        sc_strand=sc_strand,
        stab_strand=stab_strand,
        n_sc_strand=10,
        n_stab_strand=5,
        d_cooling_channel=0.001,
    )

    # Convert to dictionary
    cable_dict = cable_original.to_dict()

    # Reconstruct from dictionary
    cable_reconstructed = RectangularCable.from_dict(cable_dict)

    # Verify key attributes match
    assert cable_original.n_sc_strand == cable_reconstructed.n_sc_strand
    assert cable_original.n_stab_strand == cable_reconstructed.n_stab_strand
    assert cable_original.dx == pytest.approx(cable_reconstructed.dx)
    assert cable_original.d_cooling_channel == pytest.approx(
        cable_reconstructed.d_cooling_channel
    )
    assert cable_original.void_fraction == pytest.approx(
        cable_reconstructed.void_fraction
    )
    assert cable_original.cos_theta == pytest.approx(cable_reconstructed.cos_theta)
    assert cable_original.sc_strand.name == cable_reconstructed.sc_strand.name
    assert cable_original.stab_strand.name == cable_reconstructed.stab_strand.name
