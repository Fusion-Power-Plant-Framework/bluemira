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

"""Test routines for reactor system plotting."""

import filecmp

import pytest
from matplotlib import pyplot as plt

from bluemira.base.file import FileManager
from BLUEPRINT.nova.structure import CoilArchitect
from BLUEPRINT.systems.blanket import BreedingBlanket
from BLUEPRINT.systems.buildings import RadiationShield
from BLUEPRINT.systems.cryostat import Cryostat
from BLUEPRINT.systems.divertor import Divertor
from BLUEPRINT.systems.pfcoils import PoloidalFieldCoils
from BLUEPRINT.systems.plasma import Plasma
from BLUEPRINT.systems.plotting import ReactorPlotter
from BLUEPRINT.systems.tfcoils import ToroidalFieldCoils
from BLUEPRINT.systems.thermalshield import ThermalShield
from BLUEPRINT.systems.vessel import VacuumVessel

REACTORNAME = "SMOKE-TEST"


@pytest.mark.xfail
class TestSystemsPlotting:
    """A class to setup and run systems plotting tests."""

    def setup_method(self):
        """Load systems to be tested from pickles."""
        ReactorPlotter.set_defaults(force=True)
        self.file_manager = FileManager(
            reactor_name=REACTORNAME,
            reference_data_root="tests/BLUEPRINT/test_data",
            generated_data_root="tests/BLUEPRINT/test_generated_data",
        )
        self.file_manager.build_dirs()

        reactor_path = self.file_manager.reference_data_dirs["root"]
        self.ATEC = CoilArchitect.load(f"{reactor_path}/{REACTORNAME}_ATEC.pkl")
        self.BB = BreedingBlanket.load(f"{reactor_path}/{REACTORNAME}_BB.pkl")
        self.CR = Cryostat.load(f"{reactor_path}/{REACTORNAME}_CR.pkl")
        self.DIV = Divertor.load(f"{reactor_path}/{REACTORNAME}_DIV.pkl")
        self.PF = PoloidalFieldCoils.load(f"{reactor_path}/{REACTORNAME}_PF.pkl")
        self.PL = Plasma.load(f"{reactor_path}/{REACTORNAME}_PL.pkl")
        self.RS = RadiationShield.load(f"{reactor_path}/{REACTORNAME}_RS.pkl")
        self.TF = ToroidalFieldCoils.load(f"{reactor_path}/{REACTORNAME}_TF.pkl")
        self.TS = ThermalShield.load(f"{reactor_path}/{REACTORNAME}_TS.pkl")
        self.VV = VacuumVessel.load(f"{reactor_path}/{REACTORNAME}_VV.pkl")

        self.systems_xz = {
            "PL": self.PL,
            "PF": self.PF,
            "TF": self.TF,
            "ATEC": self.ATEC,
            "DIV": self.DIV,
            "BB": self.BB,
            "VV": self.VV,
            "TS": self.TS,
            "CR": self.CR,
            "RS": self.RS,
        }

        self.systems_xy = {
            "PL": self.PL,
            "BB": self.BB,
            "TF": self.TF,
            "VV": self.VV,
            "TS": self.TS,
            "CR": self.CR,
            "RS": self.RS,
        }

    def test_plot_xz(self):
        """
        Test X-Z plot of reactor matches reference image.

        Uses a set of pickled reactor systems for plotting to avoid
        excessive re-baselining when changes are made.
        The saved pickle will need to be updated when significant
        changes are made.
        """
        x = [0, 22]
        z = [-22, 15]
        _, ax = plt.subplots(figsize=[14, 10])

        save_dir = self.file_manager.generated_data_dirs["plots"]
        read_dir = self.file_manager.reference_data_dirs["plots"]

        for _, system in self.systems_xz.items():
            system.plot_xz(ax=ax)

        ax.set_xlim(x)
        ax.set_ylim(z)
        ax.set_aspect("equal", adjustable="box")

        name_old = f"{read_dir}/{REACTORNAME}_XZ_orig.png"
        name_new = f"{save_dir}/{REACTORNAME}_XZ_new.png"

        plt.savefig(name_new)

        assert filecmp.cmp(name_new, name_old, shallow=False)

    def test_plot_xy(self):
        """
        Test X-Y plot of reactor matches reference image.

        Uses a set of pickled reactor systems for plotting to avoid
        excessive re-baselining when changes are made.
        The saved pickle will need to be updated when significant
        changes are made.
        """
        y = [-8, 8]
        x = [1, 20]
        _, ax = plt.subplots(figsize=[14, 10])

        save_dir = self.file_manager.generated_data_dirs["plots"]
        read_dir = self.file_manager.reference_data_dirs["plots"]

        for _, system in self.systems_xy.items():
            system.plot_xy(ax=ax)

        ax.set_xlim(x)
        ax.set_ylim(y)
        ax.set_aspect("equal")

        name_old = f"{read_dir}/{REACTORNAME}_XY_orig.png"
        name_new = f"{save_dir}/{REACTORNAME}_XY_new.png"

        plt.savefig(name_new)

        assert filecmp.cmp(name_new, name_old, shallow=False)

    def test_plot_systems_xy(self):
        """
        Test X-Z plots of reactor systems matches reference images.

        Uses a set of pickled reactor systems for plotting to avoid
        excessive re-baselining when changes are made.
        The saved pickle will need to be updated when significant
        changes are made.
        """
        save_dir = self.file_manager.generated_data_dirs["plots"]
        read_dir = self.file_manager.reference_data_dirs["plots"]

        failed = []
        for name, system in self.systems_xy.items():
            plt.close("all")
            _, axes = plt.subplots(figsize=[14, 10])

            system.plot_xy(
                ax=axes,
                facecolor=["green", "red", "blue"],
                alpha=[0.5, 1, 1],
                edgecolor="k",
                linewidth=2,
            )
            name_old = f"{read_dir}/{REACTORNAME}_{name}_XY_orig.png"
            name_new = f"{save_dir}/{REACTORNAME}_{name}_XY_new.png"

            plt.savefig(name_new)

            if not filecmp.cmp(name_new, name_old, shallow=False):
                failed += [name_new]

        assert failed == []

    def test_plot_systems_xz(self):
        """
        Test X-Z plots of reactor systems matches reference images.

        Uses a set of pickled reactor systems for plotting to avoid
        excessive re-baselining when changes are made.
        The saved pickle will need to be updated when significant
        changes are made.
        """
        save_dir = self.file_manager.generated_data_dirs["plots"]
        read_dir = self.file_manager.reference_data_dirs["plots"]

        failed = []
        for name, system in self.systems_xz.items():
            plt.close("all")
            _, axes = plt.subplots(figsize=[14, 10])

            system.plot_xz(
                ax=axes,
                facecolor=["green", "red", "blue"],
                alpha=[0.5, 1, 1],
                edgecolor="k",
                linewidth=2,
            )

            name_old = f"{read_dir}/{REACTORNAME}_{name}_XZ_orig.png"
            name_new = f"{save_dir}/{REACTORNAME}_{name}_XZ_new.png"

            plt.savefig(name_new)

            if not filecmp.cmp(name_new, name_old, shallow=False):
                failed += [name_new]

        assert failed == []
