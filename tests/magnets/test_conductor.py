# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from bluemira.base.file import get_bluemira_path
from bluemira.magnets.cable import RectangularCable
from bluemira.magnets.conductor import Conductor
from bluemira.magnets.strand import Strand, SuperconductingStrand
from bluemira.materials import MaterialCache
from bluemira.materials.mixtures import MixtureFraction

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

directory = get_bluemira_path("magnets", subfolder="tests")
MATERIAL_CACHE = MaterialCache()
MATERIAL_CACHE.load_from_file(Path(directory, "test_materials_mag.json"))


@pytest.fixture
def mat_jacket():
    return MATERIAL_CACHE.get_material("SS316-LN")


@pytest.fixture
def mat_ins():
    return MATERIAL_CACHE.get_material("DummyInsulator")


@pytest.fixture
def sc_strand():
    sc = MATERIAL_CACHE.get_material("Nb3Sn - WST")
    sc.k = lambda **kwargs: 10.0  # noqa: ARG005
    return SuperconductingStrand(
        name="SC",
        materials=[MixtureFraction(material=sc, fraction=1.0)],
        d_strand=0.001,
    )


@pytest.fixture
def stab_strand():
    stab = MATERIAL_CACHE.get_material("SS316-LN")
    stab.k = lambda **kwargs: 15.0  # noqa: ARG005
    return Strand(
        name="Stab",
        materials=[MixtureFraction(material=stab, fraction=1.0)],
        d_strand=0.001,
    )


@pytest.fixture
def rectangular_cable(sc_strand, stab_strand):
    return RectangularCable(
        dx=0.01,
        sc_strand=sc_strand,
        stab_strand=stab_strand,
        n_sc_strand=10,
        n_stab_strand=10,
        d_cooling_channel=0.001,
    )


@pytest.fixture
def conductor(rectangular_cable, mat_jacket, mat_ins):
    return Conductor(
        cable=rectangular_cable,
        mat_jacket=mat_jacket,
        mat_ins=mat_ins,
        dx_jacket=0.002,
        dy_jacket=0.002,
        dx_ins=0.001,
        dy_ins=0.001,
        name="TestConductor",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_geometry_and_area(conductor):
    assert conductor.dx > 0
    assert conductor.dy > 0
    assert conductor.area > 0
    assert conductor.area_jacket > 0
    assert conductor.area_ins > 0


def test_material_properties(conductor):
    temperature = 20  # K
    assert conductor.erho(temperature=temperature) > 0.0
    assert conductor.Cp(temperature=temperature) > 0.0


def test_plot(monkeypatch, conductor):
    monkeypatch.setattr(plt, "show", lambda: None)
    ax = conductor.plot(show=True)
    assert hasattr(ax, "fill")


def test_to_from_dict(conductor):
    config = conductor.to_dict()
    restored = Conductor.from_dict(config)

    assert restored.name == conductor.name
    assert restored.dx_jacket == pytest.approx(conductor.dx_jacket)
    assert restored.dy_jacket == pytest.approx(conductor.dy_jacket)
    assert restored.dx_ins == pytest.approx(conductor.dx_ins)
    assert restored.dy_ins == pytest.approx(conductor.dy_ins)
    assert restored.mat_jacket.name == conductor.mat_jacket.name
    assert restored.mat_ins.name == conductor.mat_ins.name
    assert restored.cable.n_sc_strand == conductor.cable.n_sc_strand
    assert restored.cable.sc_strand.name == conductor.cable.sc_strand.name
