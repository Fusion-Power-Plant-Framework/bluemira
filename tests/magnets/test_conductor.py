# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bluemira.magnets.conductor import Conductor, SymmetricConductor

# ---------------------------
# Dummy components for tests
# ---------------------------


class DummyMaterial:
    def __init__(self, name="Dummy"):
        self.name = name

    def erho(self, **kwargs):  # noqa: ARG002
        return 1e-9

    def Cp(self, **kwargs):  # noqa: ARG002
        return 500

    def E(self, **kwargs):  # noqa: ARG002
        return 2e11


class DummyCable:
    def __init__(self, dx=0.01, dy=0.02):
        self.dx = dx
        self.dy = dy
        self.area = dx * dy

    def erho(self, **kwargs):  # noqa: ARG002
        return 2e-9

    def Cp(self, **kwargs):  # noqa: ARG002
        return 400

    def Kx(self, **kwargs):  # noqa: ARG002
        return 1e5

    def Ky(self, **kwargs):  # noqa: ARG002
        return 1e5

    def __str__(self):
        return "DummyCable"

    def plot(self, xc=0, yc=0, *, show=False, ax=None):  # noqa: ARG002
        return ax or plt.gca()


# -----------------------
# Core property testing
# -----------------------


def test_basic_geometry_and_area():
    conductor = Conductor(
        cable=DummyCable(),
        mat_jacket=DummyMaterial(),
        mat_ins=DummyMaterial(),
        dx_jacket=0.001,
        dy_jacket=0.002,
        dx_ins=0.0015,
        dy_ins=0.0025,
    )
    assert np.isclose(conductor.dx, 0.01 + 2 * 0.001 + 2 * 0.0015)
    assert np.isclose(conductor.dy, 0.02 + 2 * 0.002 + 2 * 0.0025)
    assert conductor.area > 0
    assert conductor.area_jacket > 0
    assert conductor.area_ins > 0


def test_thermal_and_electrical_properties():
    conductor = Conductor(
        cable=DummyCable(),
        mat_jacket=DummyMaterial(),
        mat_ins=DummyMaterial(),
        dx_jacket=0.001,
        dy_jacket=0.002,
        dx_ins=0.001,
        dy_ins=0.002,
    )
    assert conductor.erho() > 0
    assert conductor.Cp() > 0


def test_stiffness_properties():
    conductor = Conductor(
        cable=DummyCable(),
        mat_jacket=DummyMaterial(),
        mat_ins=DummyMaterial(),
        dx_jacket=0.001,
        dy_jacket=0.001,
        dx_ins=0.001,
        dy_ins=0.001,
    )
    assert conductor.Kx() > 0
    assert conductor.Ky() > 0


def test_tresca_stress_valid_and_invalid_direction():
    conductor = Conductor(
        cable=DummyCable(),
        mat_jacket=DummyMaterial(),
        mat_ins=DummyMaterial(),
        dx_jacket=0.001,
        dy_jacket=0.001,
        dx_ins=0.001,
        dy_ins=0.001,
    )
    stress = conductor._tresca_sigma_jacket(
        pressure=1e5,
        f_z=1.0,
        temperature=4.2,
        B=0.5,
        direction="x",
    )
    assert stress > 0

    with pytest.raises(ValueError, match="Invalid direction"):
        conductor._tresca_sigma_jacket(
            pressure=1e5,
            f_z=1.0,
            temperature=4.2,
            B=0.5,
            direction="z",
        )


def test_jacket_optimization():
    conductor = Conductor(
        cable=DummyCable(),
        mat_jacket=DummyMaterial(),
        mat_ins=DummyMaterial(),
        dx_jacket=0.001,
        dy_jacket=0.001,
        dx_ins=0.001,
        dy_ins=0.001,
    )
    result = conductor.optimize_jacket_conductor(
        pressure=1e5,
        f_z=1.0,
        temperature=4.2,
        B=0.5,
        allowable_sigma=2e7,
        bounds=(0.0005, 0.005),
        direction="x",
    )
    assert result.success
    assert conductor.dx_jacket == pytest.approx(result.x)


def test_str_method_output():
    conductor = Conductor(
        cable=DummyCable(),
        mat_jacket=DummyMaterial(),
        mat_ins=DummyMaterial(),
        dx_jacket=0.001,
        dy_jacket=0.001,
        dx_ins=0.001,
        dy_ins=0.001,
    )
    s = str(conductor)
    assert "dx_jacket" in s
    assert "DummyCable" in s


def test_plot(monkeypatch):
    conductor = Conductor(
        cable=DummyCable(),
        mat_jacket=DummyMaterial(),
        mat_ins=DummyMaterial(),
        dx_jacket=0.001,
        dy_jacket=0.001,
        dx_ins=0.001,
        dy_ins=0.001,
    )
    monkeypatch.setattr(plt, "show", lambda: None)
    ax = conductor.plot(show=True)
    assert hasattr(ax, "fill")


# -----------------------
# SymmetricConductor
# -----------------------


def test_symmetric_conductor_properties():
    symmetric = SymmetricConductor(
        cable=DummyCable(),
        mat_jacket=DummyMaterial(),
        mat_ins=DummyMaterial(),
        dx_jacket=0.001,
        dx_ins=0.002,
    )
    assert symmetric.dy_jacket == symmetric.dx_jacket
    assert symmetric.dy_ins == symmetric.dx_ins
