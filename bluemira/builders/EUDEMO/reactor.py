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
Perform the EU-DEMO design.
"""

import abc
import numpy as np
from typing import Dict, Optional, Union

from bluemira.base.builder import BuildConfig
from bluemira.base.config import Configuration
from bluemira.base.parameter import ParameterFrame
from bluemira.equilibria.constants import (
    NBTI_J_MAX,
    NBTI_B_MAX,
    NB3SN_J_MAX,
    NB3SN_B_MAX,
)
from bluemira.base.file import FileManager, BM_ROOT
from bluemira.base.look_and_feel import bluemira_print, print_banner
from bluemira.codes import run_systems_code
from bluemira.equilibria import AbInitioEquilibriumProblem
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.geometry._deprecated_loop import Loop
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.tools import offset_wire
from bluemira.utilities.physics_tools import lambda_q


class Reactor(abc.ABC):
    _params: Configuration
    _build_config: BuildConfig
    _file_manager: FileManager

    def __init__(
        self, params: Dict[str, Union[int, float, str]], build_config: BuildConfig
    ):
        self._extract_build_config(build_config)
        self._params = Configuration.from_template(Configuration().keys())
        self._params.update_kw_parameters(params, source="Input")
        self._build_config = build_config  # TODO: Shouldn't be stored on reactor

        self._create_file_manager()

    def _create_file_manager(self):
        """
        Create the FileManager for this Reactor.
        """
        self._file_manager = FileManager(
            reactor_name=self._params.get("Name", "DEFAULT_REACTOR"),
            reference_data_root=self._reference_data_root,
            generated_data_root=self._generated_data_root,
        )
        self._file_manager.build_dirs()

    def _extract_build_config(self, build_config: BuildConfig):
        self._reference_data_root: str = build_config.get(
            "reference_data_root", f"{BM_ROOT}/data"
        )
        self._generated_data_root: str = build_config.get(
            "generated_data_root", f"{BM_ROOT}/data"
        )
        self._plot_flag: bool = build_config.get("plot_flag", False)

    @property
    def build_config(self):
        return self._build_config

    @property
    def params(self):
        return self._params

    @property
    def file_manager(self):
        return self._file_manager

    @abc.abstractmethod
    def build(self):
        print_banner()

    def add_parameters(
        self, params: Dict[str, Union[int, float, str]], source: Optional[str] = None
    ):
        self._params.update_kw_parameters(params, source=source)


class EUDEMO(Reactor):
    def build(self):
        super().build()

        self.run_systems_code()
        self.create_equilibrium()
        # self.build_plasma()
        # self.build_tf_coils()

    def run_systems_code(self):
        """
        Run the systems code module in the requested run mode.
        """
        bluemira_print("Running systems code.")

        PROCESS_output: ParameterFrame = run_systems_code(
            self._params,
            self._build_config,
            self._file_manager.generated_data_dirs["systems_code"],
            self._file_manager.reference_data_dirs["systems_code"],
        )
        self._params.update_kw_parameters(PROCESS_output.to_dict())

    def create_equilibrium(self, qpsi_calcmode=0):
        """
        Creates a reference MHD equilibrium for the Reactor.
        """
        bluemira_print("Generating reference plasma MHD equilibrium.")

        # First make an initial TF coil shape along which to auto-position
        # some starting PF coil locations. We will design the TF later
        rin, rout = self._params["r_tf_in_centre"], self._params["r_tf_out_centre"]

        shape = PrincetonD()
        for key, val in {"x1": rin, "x2": rout, "dz": 0}.items():
            shape.adjust_variable(key, value=val)

        tf_boundary = shape.create_shape()
        if self._params.delta_95 < 0:  # Negative triangularity
            tf_boundary.rotate(
                tf_boundary.center_of_mass, direction=(0, 1, 0), degree=180
            )
        tf_boundary = offset_wire(tf_boundary, -0.5)
        tf_boundary = Loop(*tf_boundary.discretize().T)

        profile = None

        a = AbInitioEquilibriumProblem(
            self._params.R_0.value,
            self._params.B_0.value,
            self._params.A.value,
            self._params.I_p.value * 1e6,  # MA to A
            self._params.beta_p.value / 1.3,  # TODO: beta_N vs beta_p here?
            self._params.l_i.value,
            # TODO: 100/95 problem
            # TODO: This is a parameter patch... switch to strategy pattern
            self._params.kappa_95,
            1.2 * self._params.kappa_95,
            self._params.delta_95,
            1.2 * self._params.delta_95,
            -20,
            5,
            60,
            30,
            self._params.div_L2D_ib,
            self._params.div_L2D_ob,
            self._params.r_cs_in.value + self._params.tk_cs.value / 2,
            self._params.tk_cs.value / 2,
            tf_boundary,
            self._params.n_PF.value,
            self._params.n_CS.value,
            c_ejima=self._params.C_Ejima.value,
            eqtype=self._params.plasma_type.value,
            rtype=self._params.reactor_type.value,
            profile=profile,
        )

        if self._params.PF_material.value == "NbTi":
            j_pf = NBTI_J_MAX
            b_pf = NBTI_B_MAX
        elif self._params.PF_material.value == "Nb3Sn":
            j_pf = NB3SN_J_MAX
            b_pf = NB3SN_B_MAX
        else:
            raise ValueError("Unrecognised material string")

        if self._params.CS_material.value == "NbTi":
            j_cs = NBTI_J_MAX
            b_pf = NBTI_B_MAX
        elif self._params.CS_material.value == "Nb3Sn":
            j_cs = NB3SN_J_MAX
            b_cs = NB3SN_B_MAX

        a.coilset.assign_coil_materials("PF", j_max=j_pf, b_max=b_pf)
        a.coilset.assign_coil_materials("CS", j_max=j_cs, b_max=b_cs)
        a.solve(plot=self._plot_flag)

        directory = self._file_manager.generated_data_dirs["equilibria"]
        a.eq.to_eqdsk(
            self._params["Name"] + "_eqref",
            directory=directory,
            qpsi_calcmode=qpsi_calcmode,
        )
        self.EQ = a
        self.eqref = a.eq.copy()
        self.analyse_equilibrium(self.eqref)

    def analyse_equilibrium(self, eq: Equilibrium):
        """
        Analyse an equilibrium and store important values in the Reactor parameters.
        """
        plasma_dict = eq.analyse_plasma()
        lq = lambda_q(
            self._params.B_0, plasma_dict["q_95"], self._params.P_sep, plasma_dict["R_0"]
        )
        dx_shaf = plasma_dict["dx_shaf"]
        dz_shaf = plasma_dict["dz_shaf"]
        shaf = np.hypot(dx_shaf, dz_shaf)

        # fmt: off
        params = {
            "I_p": plasma_dict["Ip"] / 1e6,
            "q_95": plasma_dict["q_95"],
            "Vp": plasma_dict["V"],
            "beta_p": plasma_dict["beta_p"],
            "li": plasma_dict["li"],
            "li3": plasma_dict["li(3)"],
            "Li": plasma_dict["Li"],
            "Wp": plasma_dict["W"] / 1e6,
            "delta_95": plasma_dict["delta_95"],
            "kappa_95": plasma_dict["kappa_95"],
            "delta": plasma_dict["delta"],
            "kappa": plasma_dict["kappa"],
            "shaf_shift": shaf,
            "lambda_q": lq,
        }
        # fmt: on
        self._params.update_kw_parameters(params, source="equilibria")


if __name__ == "__main__":
    from config import params, build_config

    reactor = EUDEMO(params, build_config)
    reactor.build()
