# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import json
from pathlib import Path

import pytest

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


def test_pedestal_location_in_normalized_radius():
    assert pms_params.rho_pedestal == pytest.approx(0.94)  # [dimensionless]


def test_neutronics_reactor_power():
    assert pms_params.reactor_power == pytest.approx(1998.0e6)  # [W]


def test_density_profile_alpha_exponent():
    assert pms_params.electron_density_alpha == pytest.approx(1.0)  # [dimensionless]


def test_core_plasma_electron_density():
    assert pms_params.electron_density_core == pytest.approx(1.5e20)  # [1/m^3]


def test_pedestal_plasma_electron_density():
    assert pms_params.electron_density_ped == pytest.approx(8e19)  # [1/m^3]


def test_separatrix_plasma_electron_density():
    assert pms_params.electron_density_sep == pytest.approx(3e19)  # [1/m^3]


def test_temperature_profile_alpha_exponent():
    assert pms_params.electron_temperature_alpha == pytest.approx(1.45)  # [dimensionless]


def test_temperature_profile_beta_exponent():
    assert pms_params.electron_temperature_beta == pytest.approx(2.0)  # [dimensionless]


def test_core_plasma_electron_temperature():
    assert pms_params.electron_temperature_core == pytest.approx(20.0)  # [keV]


def test_pedestal_plasma_electron_temperature():
    assert pms_params.electron_temperature_ped == pytest.approx(5.5)  # [keV]


def test_separatrix_plasma_electron_temperature():
    assert pms_params.electron_temperature_sep == pytest.approx(0.1)  # [keV]
