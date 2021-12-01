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
A builder for Plasma properties and geometry
"""

import numpy as np
from typing import List

from bluemira.base.builder import Builder, BuildConfig
from bluemira.base.config import Configuration
from bluemira.base.file import FileManager
from bluemira.base.look_and_feel import bluemira_print

from bluemira.equilibria.constants import (
    NBTI_J_MAX,
    NBTI_B_MAX,
    NB3SN_J_MAX,
    NB3SN_B_MAX,
)
from bluemira.equilibria import AbInitioEquilibriumProblem
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.geometry._deprecated_loop import Loop
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.tools import offset_wire
from bluemira.utilities.physics_tools import lambda_q


class PlasmaBuilder(Builder):
    """
    A builder for Plasma properties and geometry
    """

    _required_params: List[str] = []
    _params: Configuration
    _file_manager: FileManager

    def reinitialise(self, params, **kwargs) -> None:
        super().reinitialise(params, **kwargs)

    def build(self, **kwargs):
        return super().build(**kwargs)

    def _extract_config(self, build_config: BuildConfig):
        super()._extract_config(build_config)

    def create_equilibrium(self, qpsi_calcmode=0):
        """
        Creates a reference MHD equilibrium for the Reactor.
        """
        bluemira_print("Generating reference plasma MHD equilibrium.")

        # First make an initial TF coil shape along which to auto-position
        # some starting PF coil locations. We will design the TF later
        rin, rout = self._params["r_tf_in_centre"], self._params["r_tf_out_centre"]

        # TODO: Handle other TF coil parameterisations?
        shape = PrincetonD()
        for key, val in {"x1": rin, "x2": rout, "dz": 0}.items():
            shape.adjust_variable(key, value=val)

        tf_boundary = shape.create_shape()
        if self._params.delta_95 < 0:  # Negative triangularity
            tf_boundary.rotate(
                tf_boundary.center_of_mass, direction=(0, 1, 0), degree=180
            )
        tf_boundary = offset_wire(tf_boundary, -0.5)

        # TODO: Avoid converting to (deprecated) Loop
        # TODO: Agree on numpy array dimensionality
        tf_boundary = Loop(*tf_boundary.discretize().T)

        profile = None

        # TODO: Can we make it so that the equilibrium problem being used can be
        # configured?
        a = AbInitioEquilibriumProblem(
            self._params.R_0.value,
            self._params.B_0.value,
            self._params.A.value,
            self._params.I_p.value * 1e6,  # MA to A
            self._params.beta_p.value / 1.3,  # TODO: beta_N vs beta_p here?
            self._params.l_i.value,
            # TODO: 100/95 problem
            # TODO: This is a parameter patch... switch to strategy pattern
            self._params.kappa_95.value,
            1.2 * self._params.kappa_95.value,
            self._params.delta_95.value,
            1.2 * self._params.delta_95.value,
            -20,
            5,
            60,
            30,
            self._params.div_L2D_ib.value,
            self._params.div_L2D_ob.value,
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

        # TODO: Handle these through properties on actual materials.
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
