# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later


import matplotlib.pyplot as plt
import pytest
from eurofusion_materials.library.magnet_branch_mats import (
    DUMMY_INSULATOR_MAG,
    NB3SN_MAG,
    SS316_LN_MAG,
)
from matproplib import OperationalConditions
from matproplib.material import MaterialFraction

from bluemira.magnets.cable import RectangularCable
from bluemira.magnets.conductor import Conductor
from bluemira.magnets.strand import Strand, SuperconductingStrand


@pytest.fixture
def mat_jacket():
    return SS316_LN_MAG


@pytest.fixture
def mat_ins():
    return DUMMY_INSULATOR_MAG


@pytest.fixture
def sc_strand():
    sc = NB3SN_MAG
    sc.specific_heat_capacity = lambda *args, **kwargs: 10.0  # noqa: ARG005
    return SuperconductingStrand(
        name="SC",
        materials=[MaterialFraction(material=sc, fraction=1.0)],
        d_strand=0.001,
        operating_temperature=5.7,
    )


@pytest.fixture
def stab_strand():
    stab = SS316_LN_MAG
    stab.thermal_conductivity = lambda *args, **kwargs: 15.0  # noqa: ARG005
    return Strand(
        name="Stab",
        materials=[MaterialFraction(material=stab, fraction=1.0)],
        d_strand=0.001,
        operating_temperature=5.7,
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
        void_fraction=0.725,
        cos_theta=0.97,
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


def test_geometry_and_area(conductor):
    assert conductor.dx > 0
    assert conductor.dy > 0
    assert conductor.area > 0
    assert conductor.area_jacket > 0
    assert conductor.area_ins > 0


def test_material_properties(conductor):
    op_cond = OperationalConditions(temperature=20)
    assert conductor.erho(op_cond) > 0.0
    assert conductor.Cp(op_cond) > 0.0


def test_plot(monkeypatch, conductor):
    monkeypatch.setattr(plt, "show", lambda: None)
    ax = conductor.plot(show=True)
    assert hasattr(ax, "fill")
