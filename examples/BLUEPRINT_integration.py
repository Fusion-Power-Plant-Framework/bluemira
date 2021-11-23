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

"""
Reactor example of merged functionality
"""

from __future__ import annotations

import enum
from typing import Dict, Union
import matplotlib.pyplot as plt
import numpy as np

from BLUEPRINT.systems.baseclass import ReactorSystem
from bluemira.equilibria.coils import Coil
from BLUEPRINT.geometry.geombase import GeomBase
from BLUEPRINT.geometry.shell import Shell, MultiShell
from BLUEPRINT.geometry.loop import Loop, MultiLoop
from BLUEPRINT.reactor import Reactor
from BLUEPRINT.systems.config import SingleNull

from bluemira.base.components import GroupingComponent, PhysicalComponent, ComponentError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry._deprecated_tools import (
    convert_coordinates_to_face,
    convert_coordinates_to_wire,
)


class ConversionMethod(enum.Enum):
    """
    Enumeration of the allowed conversion methods.
    """

    MIXED = "mixed"
    POLYGON = "polygon"
    SPLINE = "spline"


class BluemiraReactor(Reactor):
    """
    Performs the BLUEPRINT build but adds in bluemira Components from the resulting
    BLUEPRINT geometry objects.
    """

    default_params = SingleNull().to_records()

    def __init__(self, config, build_config, build_tweaks):
        super().__init__(config, build_config, build_tweaks)

        self.component_trees = {
            "xy": GroupingComponent(self.params.Name),
            "xz": GroupingComponent(self.params.Name),
            "xyz": GroupingComponent(self.params.Name),
        }

    def _convert_loop(
        self,
        tree: GroupingComponent,
        geom: Loop,
        geom_name: str,
        method: ConversionMethod = ConversionMethod.MIXED,
    ):
        """
        Convert a Loop into a PhysicalComponent with the provided name and add it to the
        tree.
        """
        face = convert_coordinates_to_face(*geom.xyz, method=method.value)
        component = PhysicalComponent(geom_name, face)
        tree.add_child(component)

    def _convert_multiloop(
        self,
        tree: GroupingComponent,
        geom: MultiLoop,
        geom_name: str,
        method: ConversionMethod = ConversionMethod.MIXED,
    ):
        """
        Convert a MultiLoop into a GroupingComponent with the provided name and add it to
        the tree.

        The resulting GroupingComponent is the parent of a set of PhysicalComponents
        representing each Loop.
        """
        component_tree = tree.get_component(geom_name)
        if component_tree is None:
            component_tree = GroupingComponent(geom_name, parent=tree)

        for idx, loop in enumerate(geom.loops):
            self._convert_loop(component_tree, loop, f"{geom_name} {idx}", method)

    def _convert_shell(
        self,
        tree: GroupingComponent,
        geom: Shell,
        geom_name: str,
        method: ConversionMethod = ConversionMethod.MIXED,
    ):
        """
        Convert a Shell into a PhysicalComponent with the provided name and add it to the
        tree.
        """
        inner = convert_coordinates_to_wire(*geom.inner.xyz, method=method.value)
        outer = convert_coordinates_to_wire(*geom.outer.xyz, method=method.value)
        face = BluemiraFace([outer, inner])
        component = PhysicalComponent(geom_name, face)
        tree.add_child(component)

    def _convert_multishell(
        self,
        tree: GroupingComponent,
        geom: MultiShell,
        geom_name: str,
        method: ConversionMethod = ConversionMethod.MIXED,
    ):
        """
        Convert a MultiShell into a GroupingComponent with the provided name and add it
        to the tree.

        The resulting GroupingComponent is the parent of a set of PhysicalComponents
        representing each Shell.
        """
        component_tree = tree.get_component(geom_name)
        if component_tree is None:
            component_tree = GroupingComponent(geom_name, parent=tree)

        for idx, shell in enumerate(geom.shells):
            self._convert_shell(component_tree, shell, f"{geom_name} {idx}", method)

    def _convert_geometry(
        self,
        tree: GroupingComponent,
        geom: GeomBase,
        geom_name: str,
        method: ConversionMethod = ConversionMethod.MIXED,
    ):
        """
        Convert the provided geometry into a Component with the provided name and add it
        to the tree.
        """
        if isinstance(geom, Loop):
            self._convert_loop(tree, geom, geom_name, method)
        elif isinstance(geom, MultiLoop):
            self._convert_multiloop(tree, geom, geom_name, method)
        elif isinstance(geom, Shell):
            self._convert_shell(tree, geom, geom_name, method)
        elif isinstance(geom, MultiShell):
            self._convert_multishell(tree, geom, geom_name, method)
        else:
            raise ComponentError(
                f"Attempt to convert unknown geometry type {type(geom)}."
            )

    def convert_system_xy(
        self,
        system: ReactorSystem,
        system_name: str,
        method: Union[
            ConversionMethod, Dict[str, ConversionMethod]
        ] = ConversionMethod.MIXED,
    ):
        """
        Convert a BLUEPRINT ReactorSystem into a bluemira Component assigned to the tree
        representing the xy build.
        """
        system_comp = GroupingComponent(system_name, parent=self.component_trees["xy"])
        system._generate_xy_plot_loops()
        for geom_name in system.xy_plot_loop_names:
            conversion = method
            if isinstance(conversion, dict):
                conversion = conversion[geom_name]
            self._convert_geometry(
                system_comp, system.geom[geom_name], geom_name.rstrip(" X-Y"), conversion
            )

    def convert_system_xz(
        self,
        system: ReactorSystem,
        system_name: str,
        method: Union[
            ConversionMethod, Dict[str, ConversionMethod]
        ] = ConversionMethod.MIXED,
    ):
        """
        Convert a BLUEPRINT ReactorSystem into a bluemira Component assigned to the tree
        representing the xz build.
        """
        system_comp = GroupingComponent(system_name, parent=self.component_trees["xz"])
        system._generate_xz_plot_loops()
        for geom_name in system.xz_plot_loop_names:
            conversion = method
            if isinstance(conversion, dict):
                conversion = conversion[geom_name]
            self._convert_geometry(
                system_comp, system.geom[geom_name], geom_name, conversion
            )

    def build_IVCs(self):
        """
        Build the in-vessel components (IVCs): i.e. divertor and breeding blanket
        components.

        Converts the Plasma and Divertor systems into Components.
        """
        super().build_IVCs()

        self.convert_system_xy(self.PL, "Plasma", ConversionMethod.SPLINE)
        self.convert_system_xz(self.PL, "Plasma", ConversionMethod.SPLINE)

        self.convert_system_xz(self.DIV, "Divertor", ConversionMethod.MIXED)

    def define_in_vessel_layout(self):
        """
        Define segmentation of the blanket and the divertors.

        Converts the Blanket system into a Component.
        """
        super().define_in_vessel_layout()

        self.convert_system_xy(self.BB, "Blanket", ConversionMethod.MIXED)

        conversion = {}
        geom_name: str
        for geom_name in self.BB.geom:
            if "bss" in geom_name:
                conversion[geom_name] = ConversionMethod.MIXED
            else:
                conversion[geom_name] = ConversionMethod.POLYGON
        self.convert_system_xz(self.BB, "Blanket", conversion)

    def build_TF_coils(
        self, ny=None, nr=None, nrippoints=None, objective=None, shape_type=None
    ):
        """
        Design and optimise the tokamak toroidal field coils.

        Converts the TF Coils system into a Component.

        Parameters
        ----------
        ny: int
            WP discretisation in toroidal direction. Production runs should use
            at least ny=3
        nr: int
            WP discretisation in radial direction. Production runs should use
            at least nr=2
        nrippoints: int
            Number of points along the outer separatrix to check for ripple.
            Lower numbers for speed but careful please
        objective: str from ['L', 'E']
            The optimisation objective:
            - 'L': minimises the length of the winding pack profile. Fast.
            - 'E': minimises the stored energy of the TF coil set. Slow and
            will occasionally cause re-entrant profiles (bad for manufacture)
        shape_type: str from ['S', 'T', 'D', 'P']
            The TF coil shape parameterisation to use:
            - 'S': Spline coil shape (highly parameterised)
            - 'T': triple-arc coil shape
            - 'D': Princeton D coil shape
            - 'P': Picture frame coil shape
        """
        super().build_TF_coils(
            ny=ny,
            nr=nr,
            nrippoints=nrippoints,
            objective=objective,
            shape_type=shape_type,
        )

        self.convert_system_xy(self.TF, "TF Coils", ConversionMethod.MIXED)
        self.convert_system_xz(self.TF, "TF Coils", ConversionMethod.MIXED)

    def build_containments(self):
        """
        Build the cryostat and radiation shield systems.

        Converts the Vacuum Vessel, Thermal Shield, Cryostat, and Radiation Shield
        systems into Components.
        """
        super().build_containments()

        self.convert_system_xy(self.VV, "Vacuum Vessel")
        self.convert_system_xz(self.VV, "Vacuum Vessel")

        conversion = {}
        geom_name: str
        self.TS._generate_xy_plot_loops()
        self.TS._generate_xz_plot_loops()
        for geom_name in self.TS.geom:
            if "port" in geom_name or "Cryostat TS" in geom_name:
                conversion[geom_name] = ConversionMethod.POLYGON
            elif "2D profile" in geom_name:
                conversion[geom_name] = ConversionMethod.MIXED
            else:
                conversion[geom_name] = ConversionMethod.SPLINE

        self.convert_system_xy(self.TS, "Thermal Shield", conversion)
        self.convert_system_xz(self.TS, "Thermal Shield", conversion)

        self.convert_system_xy(self.CR, "Cryostat", ConversionMethod.POLYGON)
        self.convert_system_xz(self.CR, "Cryostat", ConversionMethod.POLYGON)

        self.convert_system_xy(self.RS, "Radiation Shield")
        self.convert_system_xz(self.RS, "Radiation Shield")

    def optimise_coil_cage(self):
        """
        Optimise the TF coil casing.s

        Converts the Coil Architecture system into a Component.
        """
        super().optimise_coil_cage()

        self.convert_system_xz(self.ATEC, "Coil Architecture")

    def build_PF_system(self):
        """
        Design and optimise the reactor poloidal field system.

        Converts the PF Coils system into a Component.
        """
        super().build_PF_system()

        pf_comp = GroupingComponent("PF Coils", parent=self.component_trees["xz"])

        name: str
        coil: Coil
        for name, coil in self.PF.coils.items():
            x = np.append(coil.x_corner, coil.x_corner[0])
            y = np.zeros(len(x))
            z = np.append(coil.z_corner, coil.z_corner[0])
            wire = convert_coordinates_to_wire(x, y, z)
            face = BluemiraFace(wire)
            component = PhysicalComponent(name, face)
            pf_comp.add_child(component)


if __name__ == "__main__":
    REACTORNAME = "EU-DEMO"
    config = {
        "Name": REACTORNAME,
        "P_el_net": 500,
        # TODO: Slightly shorter than 2 hr flat-top..
        "tau_flattop": 6900,
        "plasma_type": "SN",
        "reactor_type": "Normal",
        "blanket_type": "HCPB",
        "CS_material": "Nb3Sn",
        "PF_material": "NbTi",
        "A": 3.1,
        "n_CS": 5,
        "n_PF": 6,
        "n_TF": 18,
        "P_hcd_ss": 50,
        "f_ni": 0.1,
        "fw_psi_n": 1.06,
        "l_i": 0.8,
        # EU-DEMO has no shield component, so set thickness to 0.3 and subtract from VV.
        "tk_sh_in": 0.3,
        "tk_sh_out": 0.3,
        "tk_sh_top": 0.3,
        "tk_sh_bot": 0.3,
        "tk_vv_in": 0.3,
        "tk_vv_out": 0.8,
        "tk_vv_top": 0.3,
        "tk_vv_bot": 0.3,
        "tk_tf_side": 0.1,
        "tk_tf_front_ib": 0.05,
        "tk_bb_ib": 0.755,
        "tk_bb_ob": 1.275,
        "tk_sol_ib": 0.225,
        "tk_sol_ob": 0.225,
        "tk_ts": 0.05,
        "g_cs_tf": 0.05,
        "g_tf_pf": 0.05,
        "g_vv_bb": 0.02,
        "C_Ejima": 0.3,
        "e_nbi": 1000,
        "eta_nb": 0.4,
        "LPangle": -15,
        "bb_e_mult": 1.35,
        "w_g_support": 1.5,
    }

    build_config = {
        "generated_data_root": "!BP_ROOT!/generated_data",
        "plot_flag": False,
        "process_mode": "mock",
        "plasma_mode": "run",
        "tf_mode": "run",
        # TF coil config
        "TF_type": "S",
        "wp_shape": "N",
        "TF_objective": "L",
        "GS_type": "ITER",
        # FW and VV config
        "VV_parameterisation": "S",
        "FW_parameterisation": "S",
        "BB_segmentation": "radial",
        "lifecycle_mode": "life",
        "HCD_method": "power",
    }

    build_tweaks = {
        # TF coil optimisation tweakers (n ripple filaments)
        "nr": 1,
        "ny": 1,
        "nrippoints": 20,  # Number of points to check edge ripple on
    }

    reactor = BluemiraReactor(config, build_config, build_tweaks)
    reactor.build()

    reactor.component_trees["xz"].plot_2d(plane="xz")
    reactor.component_trees["xy"].plot_2d(plane="xy")

    print(reactor.component_trees["xz"].tree())
    print(reactor.component_trees["xy"].tree())

    plt.show()
