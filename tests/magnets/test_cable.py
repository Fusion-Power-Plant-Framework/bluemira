# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import matplotlib.pyplot as plt
import pytest

from bluemira.magnets.cable import (
    RectangularCable,
    RoundCable,
    SquareCable,
)
from bluemira.magnets.strand import Strand, SuperconductingStrand
from bluemira.materials.material import Superconductor
from bluemira.materials.mixtures import HomogenisedMixture, MixtureFraction

# --------------------------
# Realistic material classes
# --------------------------


class DummyMat(HomogenisedMixture):
    def __init__(self, name="DummyMat"):
        self.name = name

    def erho(self, **kwargs):  # noqa: ARG002
        return 1e-9

    def Cp(self, **kwargs):  # noqa: ARG002
        return 400

    def rho(self, **kwargs):  # noqa: ARG002
        return 8000


class DummySC(Superconductor):
    def __init__(self, name="DummySC"):
        self.name = name

    def erho(self, **kwargs):  # noqa: ARG002
        return 1e-10

    def Cp(self, **kwargs):  # noqa: ARG002
        return 500

    def rho(self, **kwargs):  # noqa: ARG002
        return 8500

    def Jc(self, **kwargs):  # noqa: ARG002
        return 1e8


# -- Pytest Fixtures ----------------------------------------------------------


@pytest.fixture
def sc_strand():
    return SuperconductingStrand(
        name="SC",
        materials=[MixtureFraction(material=DummySC(), fraction=1.0)],
        d_strand=0.001,
    )


@pytest.fixture
def stab_strand():
    return Strand(
        name="Stab",
        materials=[MixtureFraction(material=DummyMat(), fraction=1.0)],
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
    assert cable.rho() > 0.0
    assert cable.erho() > 0.0
    assert cable.Cp() > 0.0


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
    B_fun = lambda t: 5  # noqa: E731, ARG005
    I_fun = lambda t: 1000  # noqa: E731, ARG005, N806
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

    B_fun = lambda t: 5  # noqa: E731, ARG005
    I_fun = lambda t: 1000  # noqa: E731, ARG005, N806

    result = cable.optimize_n_stab_ths(
        t0=0,
        tf=0.1,
        initial_temperature=20,
        target_temperature=80,
        B_fun=B_fun,
        I_fun=I_fun,
        bounds=(1, 10),
        show=True,
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
    square = SquareCable(
        sc_strand=sc_strand,
        stab_strand=stab_strand,
        n_sc_strand=5,
        n_stab_strand=5,
        d_cooling_channel=0.001,
    )
    round_ = RoundCable(
        sc_strand=sc_strand,
        stab_strand=stab_strand,
        n_sc_strand=5,
        n_stab_strand=5,
        d_cooling_channel=0.001,
    )
    assert square.dx > 0
    assert round_.dy == pytest.approx(round_.dx)
    assert square.Kx() > 0
    assert round_.Ky() > 0
