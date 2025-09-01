# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.cm import coolwarm
from mpl_toolkits.axes_grid1 import make_axes_locatable

from bluemira.base.file import get_bluemira_path
from bluemira.base.parameter_frame import make_parameter_frame
from bluemira.codes.openmc.params import (
    OpenMCNeutronicsSolverParams,
    PlasmaSourceParameters,
)
from bluemira.codes.openmc.sources import make_tokamak_source

PARAMS_PATH = Path(get_bluemira_path("codes/openmc", subfolder="tests"), "params.json")

with open(PARAMS_PATH) as j:
    params = json.load(j)
openmc_solver_pm_frame = make_parameter_frame(
    params, OpenMCNeutronicsSolverParams, allow_unknown=True
)
pms_params = PlasmaSourceParameters.from_parameterframe(openmc_solver_pm_frame)


def test_major_radius():
    assert pms_params.major_radius == 900.0  # [cm]


def test_plasma_aspect_ratio():
    assert pms_params.aspect_ratio == 3.1  # [dimensionless]


def test_pedestal_location_in_normalized_radius():
    assert pms_params.rho_pedestal == 0.94  # [dimensionless]


def test_minor_radius():
    assert np.isclose(pms_params.minor_radius, 900 / 3.1)


def test_pedestal_radius():
    assert np.isclose(pms_params.pedestal_radius, 900 / 3.1 * 0.94)


def test_last_closed_surface_plasma_elongation():
    assert pms_params.elongation == 1.792  # [dimensionless]


def test_last_closed_surface_plasma_triangularity():
    assert pms_params.triangularity == 0.5  # [dimensionless]


def test_neutronics_reactor_power():
    assert pms_params.reactor_power == 1998.0e6  # [W]


def test_density_profile_alpha_exponent():
    assert pms_params.ion_density_alpha == 1.0  # [dimensionless]


def test_core_plasma_electron_density():
    assert pms_params.ion_density_core == 1.5e20  # [1/m^3]


def test_pedestal_plasma_electron_density():
    assert pms_params.ion_density_ped == 8e19  # [1/m^3]


def test_separatrix_plasma_electron_density():
    assert pms_params.ion_density_sep == 3e19  # [1/m^3]


def test_temperature_profile_alpha_exponent():
    assert pms_params.ion_temperature_alpha == 1.45  # [dimensionless]


def test_temperature_profile_beta_exponent():
    assert pms_params.ion_temperature_beta == 2.0  # [dimensionless]


def test_core_plasma_electron_temperature():
    assert pms_params.ion_temperature_core == 20000.0  # [eV]


def test_pedestal_plasma_electron_temperature():
    assert pms_params.ion_temperature_ped == 5500.0  # [eV]


def test_separatrix_plasma_electron_temperature():
    assert pms_params.ion_temperature_sep == 100.0  # [eV]


def test_shafranov_shift_of_plasma():
    assert pms_params.shaf_shift == 50.0  # [cm]


def test_source_plot():
    tokamak_source = make_tokamak_source(pms_params)
    dd = tokamak_source[:10000]
    tt = tokamak_source[10000:20000]
    dt = tokamak_source[20000:]

    rz, energy, intensity = [], [], []
    for src in dd:
        # position distribution: discrete distribution
        rz.append([src.space.r.x, src.space.z.x])
        intensity.append(src.strength)
    rz = np.array(rz).reshape([len(dd), 2])
    intensity = np.array(intensity)
    fig, ax = plot_one_reaction(*rz.T, intensity)
    ax.set_title("Purported neutron rates from distribution DD reaction")
    plt.show()

    rz, energy, intensity = [], [], []
    for src in tt:
        # position distribution: discrete distribution
        rz.append([src.space.r.x, src.space.z.x])
        intensity.append(src.strength)
    rz = np.array(rz).reshape([len(tt), 2])
    intensity = np.array(intensity)
    fig, ax = plot_one_reaction(*rz.T, intensity)
    ax.set_title("Purported neutron rates from distribution TT reaction")
    plt.show()

    rz, energy, intensity = [], [], []
    for src in dt:
        # position distribution: discrete distribution
        rz.append([src.space.r.x, src.space.z.x])
        intensity.append(src.strength)
    rz = np.array(rz).reshape([len(dt), 2])
    intensity = np.array(intensity)
    fig, ax = plot_one_reaction(*rz.T, intensity)
    ax.set_title("Purported neutron rates from distribution DT reaction")
    plt.show()


@pytest.mark.xfail(reason="The openmc_source is poorly")
def test_unique_energy_distributions():
    """
    Issue: the energy distribution of the DD, TT, and DT are all IDENTICAL.
    This is an issue on the side of openmc_plasma_source
    """
    tokamak_source = make_tokamak_source(pms_params)
    dd_sources = tokamak_source[:10000]
    tt_sources = tokamak_source[10000:20000]
    dt_sources = tokamak_source[20000:30000]
    # These parameters we're expecting them to be different in-general.
    weights = []
    peak_parameters = []
    table_energy, table_prob = [], []

    for dd, tt, dt in zip(dd_sources, tt_sources, dt_sources, strict=False):
        dd_dist = dd.energy
        tt_dist = tt.energy
        dt_dist = dt.energy

        weights.append([dd_dist.probability, tt_dist.probability, dt_dist.probability])

        dd_pk1, dd_pk2 = dd_dist.distribution[:2]
        tt_pk1, tt_pk2 = tt_dist.distribution[:2]
        dt_pk1, dt_pk2 = dt_dist.distribution[:2]
        peak_parameters.append([
            [dd_pk1.mean_value, dd_pk1.std_dev, dd_pk2.mean_value, dd_pk2.std_dev],
            [tt_pk1.mean_value, tt_pk1.std_dev, tt_pk2.mean_value, tt_pk2.std_dev],
            [dt_pk1.mean_value, dt_pk1.std_dev, dt_pk2.mean_value, dt_pk2.std_dev],
        ])

        dd_table = dd_dist.distribution[2]
        tt_table = tt_dist.distribution[2]
        dt_table = dt_dist.distribution[2]
        table_energy.append([dd_table.x, tt_table.x, dt_table.x])
        table_prob.append([dd_table.p, tt_table.p, dt_table.p])
    weights = np.array(weights)
    peak_parameters = np.array(peak_parameters)
    table_energy, table_prob = np.array(table_energy), np.array(table_prob)
    # compare how different the energy distributions are from each other.
    same_weights = np.diff(weights, axis=1) == 0.0
    same_peaks = np.diff(peak_parameters, axis=1) == 0.0
    same_table_energy = np.diff(table_energy, axis=1) == 0.0
    same_table_prob = np.diff(table_prob, axis=1) == 0.0
    assert (not same_weights.all()) or (not same_peaks.all())
    assert (not same_table_energy.all()) or (not same_table_prob.all())


def plot_one_reaction(r, z, intensity):
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    # scaled_intensity = 0.2 * intensity/intensity.max() + 0.8
    scatter = ax.scatter(r, z, marker=".", c=intensity, cmap=coolwarm)
    mpl.colorbar.ColorbarBase(ax=cax, cmap=coolwarm)
    ax.figure.colorbar(scatter, cax=cax, orientation="vertical")
    ax.set_aspect(1.0)
    ax.set_xlabel("r (cm)"), ax.set_ylabel("z (cm)")
    return fig, ax
