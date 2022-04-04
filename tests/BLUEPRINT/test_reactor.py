# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

# =============================================================================
# Smoke test
# =============================================================================
import pickle  # noqa :S403

import pytest
from matplotlib import pyplot as plt

import tests
from bluemira.base.config import SingleNull
from bluemira.base.file import BM_ROOT
from bluemira.utilities.tools import set_random_seed
from BLUEPRINT.reactor import Reactor

# Chosen by fair dice roll
set_random_seed(7021769)

REACTORNAME = "SMOKE-TEST"

config = {
    "Name": (REACTORNAME, "Input"),
    "P_el_net": (580, "Input"),
    "tau_flattop": (3600, "Input"),
    "plasma_type": ("SN", "Input"),
    "reactor_type": ("Normal", "Input"),
    "CS_material": ("Nb3Sn", "Input"),
    "PF_material": ("NbTi", "Input"),
    "A": (3.1, "Input"),
    "n_CS": (5, "Input"),
    "n_PF": (6, "Input"),
    "f_ni": (0.1, "Input"),
    "fw_psi_n": (1.06, "Input"),
    "tk_ts": (0.05, "Input"),
    "tk_vv_in": (0.3, "Input"),
    "tk_sh_in": (0.3, "Input"),
    "tk_tf_side": (0.1, "Input"),
    "tk_bb_ib": (0.7, "Input"),
    "tk_sol_ib": (0.225, "Input"),
    "LPangle": (-15, "Input"),
}

build_config = {
    "reference_data_root": f"{BM_ROOT}/tests/BLUEPRINT/test_data",
    "generated_data_root": f"{BM_ROOT}/tests/BLUEPRINT/test_generated_data",
    "plot_flag": tests.PLOTTING,
    "process_mode": "mock",  # Tests don't require PROCESS to be installed
    "plasma_mode": "run",
    "tf_mode": "run",
    # TF coil config
    "TF_type": "S",
    "wp_shape": "N",  # This is the winding pack shape choice for the inboard leg
    "TF_objective": "L",
    "conductivity": "SC",
    # FW and VV config
    "VV_parameterisation": "S",
    "FW_parameterisation": "S",
    "BB_segmentation": "radial",
    "lifecycle_mode": "life",
    # Equilibrium modes
    "rm_shuffle": True,
    "force": False,
    "swing": False,
    "HCD_method": "power",
    "GS_type": "JT60SA",
}

build_tweaks = {
    # TF coil optimisation tweakers (n ripple filaments)
    "nr": 1,
    "ny": 1,
    "nrippoints": 10,  # Number of points to check edge ripple on
}


class SmokeTestSingleNullReactor(Reactor):
    config: dict
    build_config: dict
    build_tweaks: dict
    # Input parameter declaration in config.py. Config values will overwrite
    # defaults in Configuration.
    # TODO: transfer functionality to multiple inheritance.. or not?
    default_params = SingleNull().to_records()


@pytest.mark.reactor
def test_runtime(reactor):
    # Check runtime isn't too long
    assert reactor.params["runtime"] < 500


@pytest.mark.reactor
def test_TFV(reactor):
    reactor.build_TFV_system(n=3)
    reactor.life_cycle()


@pytest.mark.reactor
def test_CAD(reactor):
    reactor.build_CAD()
    # reactor.save_as_STEP(filepath)
    # reactor.build_neutronics_model() #ignore this for now.. need Jon Shimwell to update


@pytest.mark.reactor
def test_pickle(reactor):
    original = reactor.params
    serialized = pickle.dumps(reactor)
    loaded = pickle.loads(serialized)  # noqa :S301

    # Loading a reactor replaces the ParameterFrame
    assert loaded.params is not original
    assert loaded.params == original
    if tests.PLOTTING:
        loaded.plot_xz()  # This will probably let you know if any geometry fails
        plt.close("all")
        loaded.EQ.plot_summary()  # smoke test for equilibria pickling
        plt.close("all")
        loaded.build_CAD()  # Another geometry smoke test


@pytest.mark.reactor
def test_pickle_class_unavailable(reactor):
    global SmokeTestSingleNullReactor  # noqa
    path = reactor.save()
    assert path.endswith(f"{REACTORNAME}.pkl")

    OriginalClass = SmokeTestSingleNullReactor  # noqa
    try:
        del SmokeTestSingleNullReactor

        OriginalClass.load(path)
    finally:
        SmokeTestSingleNullReactor = OriginalClass


def test_duplicate_class_caught():
    with pytest.raises(ValueError):

        class SmokeTestSingleNullReactor(Reactor):  # noqa
            default_params = []
