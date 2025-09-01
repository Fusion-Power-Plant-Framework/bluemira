# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import json
from pathlib import Path

import numpy as np

from bluemira.base.file import get_bluemira_path
from bluemira.base.parameter_frame import make_parameter_frame
from bluemira.codes.openmc.params import (
    OpenMCNeutronicsSolverParams,
    PlasmaSourceParameters,
)

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
