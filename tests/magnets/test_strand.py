# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bluemira.magnets.strand import Strand, SuperconductingStrand
from bluemira.materials.material import Superconductor
from bluemira.materials.mixtures import MixtureFraction


class DummySuperconductor(Superconductor):
    def __init__(self, name="DummySC"):
        self.name = name
        self.Jc0 = 1e8

    def Jc(self, B=None, temperature=None, **kwargs):  # noqa:ARG002
        return self.Jc0  # Constant mock Jc


def test_strand_area():
    mat = MixtureFraction(material=DummySuperconductor(), fraction=1.0)
    strand = Strand("test_strand", materials=[mat], d_strand=0.001)
    expected_area = np.pi * (0.001**2) / 4
    assert np.isclose(strand.area, expected_area)


def test_strand_invalid_diameter():
    mat = MixtureFraction(material=DummySuperconductor(), fraction=1.0)
    with pytest.raises(ValueError, match="positive"):
        Strand("invalid_strand", materials=[mat], d_strand=-0.001)


def test_superconducting_strand_ic_and_jc():
    mat = MixtureFraction(material=DummySuperconductor(), fraction=0.6)
    sc_strand = SuperconductingStrand("sc", materials=[mat], d_strand=0.001)
    expected_area = np.pi * (0.001**2) / 4 * 0.6
    Ic_val = 1e8 * expected_area  # noqa:N806
    assert np.isclose(sc_strand.Ic(B=1.0, temperature=4.2), Ic_val)


def test_superconducting_strand_invalid_materials():
    # Two superconductors — should raise ValueError
    mat1 = MixtureFraction(material=DummySuperconductor("SC1"), fraction=0.5)
    mat2 = MixtureFraction(material=DummySuperconductor("SC2"), fraction=0.5)
    with pytest.raises(ValueError, match="Only one superconductor material"):
        SuperconductingStrand("invalid", materials=[mat1, mat2])

    # No superconductors — should raise ValueError
    class DummyNormal:
        def __init__(self, name="DummyNormal"):
            self.name = name

    mat3 = MixtureFraction(material=DummyNormal(), fraction=1.0)
    with pytest.raises(ValueError, match="No superconductor"):
        SuperconductingStrand("invalid", materials=[mat3])


def test_strand_plot(monkeypatch):
    mat = MixtureFraction(material=DummySuperconductor(), fraction=1.0)
    strand = Strand("plot_test", materials=[mat])

    # Patch plot_2d to skip actual drawing
    monkeypatch.setattr(
        "bluemira.display.plot_2d", lambda *_args, **_kwargs: "mock_plot"
    )
    result = strand.plot(show=False)
    assert result == "mock_plot"


def test_plot_Ic_B(monkeypatch):
    mat = MixtureFraction(material=DummySuperconductor(), fraction=1.0)
    sc_strand = SuperconductingStrand("plot_ic", materials=[mat])

    # Patch plt.show to suppress window
    monkeypatch.setattr(plt, "show", lambda: None)
    B = np.linspace(0, 5, 10)
    ax = sc_strand.plot_Ic_B(B, temperature=4.2, show=True)
    assert ax is not None
    assert hasattr(ax, "plot")
