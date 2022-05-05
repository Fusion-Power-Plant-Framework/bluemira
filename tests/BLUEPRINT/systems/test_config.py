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

from bluemira.base.config import Configuration, SingleNull

build_config = {
    "plasma_mode": "read",
    "HCD_method": "power",
    "TF_objective": "L",
    "BB_segmentation": "radial",
    "lifecycle_mode": "life",
    # Equilibrium modes
    "rm_shuffle": False,
    "force": False,
    "swing": False,
}

build_tweaks = {
    # Equilibrium solver tweakers
    "wref": 225,  # Seed flux swing [V.s]
    "rms_limit": 0.2,  # RMS convergence criterion [m]
    "FPFz": 400,  # Vertical force constraints for optimiser [MN]
    # TF coil optimisation tweakers (n ripple filaments)
    "nr": 1,
    "ny": 1,
}


class MockConfig(Configuration):
    pn = [["g_ts_pf", "Clearances to PFs", 0.0009501, "m", None, "Input"]]

    def __init__(self):
        super().__init__()
        self.add_parameters(self.pn)


class TestConfiguration:
    dummy = Configuration()
    c = SingleNull()
    s = MockConfig()

    def test_config(self):
        assert self.c["plasma_type"] == "SN"

    def test_derived(self):
        assert self.s["g_ts_pf"] == 0.0009501

    def test_duplicates(self):
        """
        Check for duplicate variable names
        """
        assert len(set(self.c.keys())) == len(self.c.keys())
