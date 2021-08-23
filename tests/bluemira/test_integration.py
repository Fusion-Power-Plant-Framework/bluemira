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

import numpy as np
import pytest
import tests

from BLUEPRINT.base.baseclass import ReactorSystem
from BLUEPRINT.reactor import Reactor
from BLUEPRINT.systems.config import SingleNull

from examples.bluemira.BLUEPRINT_integration import BluemiraReactor


REACTORNAME = "BLUEPRINT-INTEGRATION-TEST"

config = {
    "Name": REACTORNAME,
    "P_el_net": 580,
    "tau_flattop": 3600,
    "plasma_type": "SN",
    "reactor_type": "Normal",
    "CS_material": "Nb3Sn",
    "PF_material": "NbTi",
    "A": 3.1,
    "n_CS": 5,
    "n_PF": 6,
    "f_ni": 0.1,
    "fw_psi_n": 1.06,
    "tk_ts": 0.05,
    "tk_vv_in": 0.3,
    "tk_sh_in": 0.3,
    "tk_tf_side": 0.1,
    "tk_bb_ib": 0.7,
    "tk_sol_ib": 0.225,
    "LPangle": -15,
}

build_config = {
    "reference_data_root": "!BP_ROOT!/tests/bluemira/test_data",
    "generated_data_root": "!BP_ROOT!/tests/bluemira/test_generated_data",
    "plot_flag": tests.PLOTTING,
    "process_mode": "mock",  # Tests don't require PROCESS to be installed
    "plasma_mode": "run",
    "tf_mode": "run",
    # TF coil config
    "TF_type": "S",
    "TF_objective": "L",
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


class SingleNullBluemiraReactor(BluemiraReactor):
    config: dict
    build_config: dict
    build_tweaks: dict
    # Input parameter declaration in config.py. Config values will overwrite
    # defaults in Configuration.
    # TODO: transfer functionality to multiple inheritance.. or not?
    default_params = SingleNull().to_records()


@pytest.mark.reactor
@pytest.mark.parametrize("bp_system_name,bm_system_name", [("BB", "Blanket")])
def test_xz_systems(
    BLUEPRINT_integration_reactor: BluemiraReactor,
    reactor: Reactor,
    bp_system_name: str,
    bm_system_name: str,
):
    bm_tree = BLUEPRINT_integration_reactor.component_trees["xz"]
    bm_component = bm_tree.get_component(bm_system_name)
    bm_names = [comp.name for comp in bm_component.children]

    bp_system: ReactorSystem = getattr(reactor, bp_system_name)
    missing_systems = []
    for geom_name in bp_system.xz_plot_loop_names:
        if geom_name not in bm_names:
            missing_systems.append(geom_name)

    if len(missing_systems) > 0:
        pytest.fail(f"The components {missing_systems} were not generated.")


@pytest.mark.reactor
@pytest.mark.parametrize("bp_system_name,bm_system_name", [("BB", "Blanket")])
def test_xz_lengths(
    BLUEPRINT_integration_reactor: BluemiraReactor,
    reactor: Reactor,
    bp_system_name: str,
    bm_system_name: str,
):
    bm_tree = BLUEPRINT_integration_reactor.component_trees["xz"]
    bm_component = bm_tree.get_component(bm_system_name)

    bp_system: ReactorSystem = getattr(reactor, bp_system_name)
    bad_length = []
    for geom_name in bp_system.xz_plot_loop_names:
        bm_child = bm_component.get_component(geom_name)

        bp_length = bp_system.geom[geom_name].length
        bm_length = bm_child.shape.length
        if not np.isclose(bp_length, bm_length, atol=0.1, rtol=0.05):
            bad_length.append([geom_name, (bm_length, bp_length)])

    if len(bad_length) > 0:
        pytest.fail(f"The components {bad_length} had bad lengths.")


@pytest.mark.reactor
@pytest.mark.parametrize("bp_system_name,bm_system_name", [("BB", "Blanket")])
def test_xz_areas(
    BLUEPRINT_integration_reactor: BluemiraReactor,
    reactor: Reactor,
    bp_system_name: str,
    bm_system_name: str,
):
    bm_tree = BLUEPRINT_integration_reactor.component_trees["xz"]
    bm_component = bm_tree.get_component(bm_system_name)

    bp_system: ReactorSystem = getattr(reactor, bp_system_name)
    bad_area = []
    for geom_name in bp_system.xz_plot_loop_names:
        bm_child = bm_component.get_component(geom_name)

        bp_area = bp_system.geom[geom_name].area
        bm_area = bm_child.shape.area
        if not np.isclose(bp_area, bm_area, atol=0.1, rtol=0.05):
            bad_area.append([geom_name, (bm_area, bp_area)])

    if len(bad_area) > 0:
        pytest.fail(f"The components {bad_area} had bad areas.")


@pytest.mark.reactor
@pytest.mark.parametrize("bp_system_name,bm_system_name", [("BB", "Blanket")])
def test_xz_centroids(
    BLUEPRINT_integration_reactor: BluemiraReactor,
    reactor: Reactor,
    bp_system_name: str,
    bm_system_name: str,
):
    bm_tree = BLUEPRINT_integration_reactor.component_trees["xz"]
    bm_component = bm_tree.get_component(bm_system_name)
    bm_names = [comp.name for comp in bm_component.children]

    bp_system: ReactorSystem = getattr(reactor, bp_system_name)
    bad_centroid = []
    for geom_name in bp_system.xz_plot_loop_names:
        bm_child = bm_component.get_component(geom_name)

        bp_centroid = bp_system.geom[geom_name].centroid
        bm_centroid = bm_child.shape.center_of_mass[0:3:2]
        if not np.allclose(bp_centroid, bm_centroid, atol=0.1, rtol=0.05):
            bad_centroid.append([geom_name, (bm_centroid, bp_centroid)])

    if len(bad_centroid) > 0:
        pytest.fail(f"The components {bad_centroid} had bad centers of mass.")
