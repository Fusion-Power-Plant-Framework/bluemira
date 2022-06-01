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

from __future__ import annotations

from typing import List, Optional

import numpy as np

import bluemira.utilities.plot_tools as bm_plot_tools
from bluemira.base.builder import BuildConfig, Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.config import Configuration
from bluemira.base.error import BuilderError
from bluemira.base.look_and_feel import bluemira_print
from bluemira.builders.EUDEMO.equilibria import EUDEMOSingleNullConstraints
from bluemira.builders.EUDEMO.pf_coils import make_coilset
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.opt_problems import UnconstrainedTikhonovCurrentGradientCOP
from bluemira.equilibria.profiles import BetaIpProfile
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.equilibria.solve import DudsonConvergence, PicardIterator
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.tools import make_circle, make_polygon, offset_wire, revolve_shape
from bluemira.geometry.wire import BluemiraWire


class PlasmaComponent(Component):
    """
    A component containing the equilibrium used to build a plasma.
    """

    def __init__(
        self,
        name: str,
        parent: Optional[Component] = None,
        children: Optional[List[Component]] = None,
        equilibrium: Optional[Equilibrium] = None,
    ):
        super().__init__(name, parent=parent, children=children)

        self._equilibrium = equilibrium

    @property
    def equilibrium(self) -> Equilibrium:
        """
        The equilibrium used to build this plasma.
        """
        return self._equilibrium


class PlasmaBuilder(Builder):
    """
    A builder for Plasma properties and geometry.
    """

    _required_params: List[str] = [
        "R_0",
        "B_0",
        "A",
        "I_p",
        "beta_p",
        "P_sep",
        "l_i",
        "kappa_95",
        "kappa",
        "delta_95",
        "delta",
        "q_95",
        "shaf_shift",
        "div_L2D_ib",
        "div_L2D_ob",
        "r_cs_in",
        "r_tf_in_centre",
        "r_tf_out_centre",
        "tk_cs",
        "g_cs_mod",
        "tk_cs_insulation",
        "tk_cs_casing",
        "n_PF",
        "n_CS",
        "PF_jmax",
        "PF_bmax",
        "CS_jmax",
        "CS_bmax",
        "C_Ejima",
        "reactor_type",
        "plasma_type",
    ]

    _params: Configuration
    _boundary: Optional[BluemiraWire] = None
    _equilibrium: Optional[Equilibrium] = None
    _plot_flag: bool
    _eqdsk_path: Optional[str] = None
    _default_runmode: str = "run"

    def _extract_config(self, build_config: BuildConfig):
        super()._extract_config(build_config)

        if self._runmode.name.lower() == "read":
            if build_config.get("eqdsk_path") is None:
                raise BuilderError(
                    "Must supply eqdsk_path in build_config when using 'read' mode."
                )
            self._eqdsk_path = build_config["eqdsk_path"]

        self._plot_flag = build_config.get("plot_flag", False)

    def reinitialise(self, params) -> None:
        """
        Reinitialise the parameters and boundary.

        Parameters
        ----------
        params: dict
            The new parameter values to initialise this builder against.
        """
        super().reinitialise(params)

        self._boundary = None
        self._equilibrium = None

    def run(self):
        """
        Run the plasma equilibrium design problem.
        """
        bluemira_print("Running Plasma equilibrium design problem")
        eq = self._create_equilibrium()
        self._analyse_equilibrium(eq)

    def read(self):
        """
        Read the plasma equilibrium design problem.
        """
        bluemira_print("Reading Plasma equilibrium design problem")
        eq = self._read_equilibrium()
        self._analyse_equilibrium(eq)

    def mock(self):
        """
        Mock the plasma equilibrium design problem using a Johner LCFS.
        """
        bluemira_print(
            "Mocking Plasma equilibrium design problem "
            "- no equilibrium will be produced."
        )
        shape = JohnerLCFS()
        for var, param in {"r_0": "R_0", "a": "A"}.items():
            shape.adjust_variable(var, self._params[param])
        self._boundary = shape.create_shape()
        self._equilibrium = None

    def _ensure_boundary(self):
        """
        Check that the boundary has been set.
        """
        if getattr(self, "_boundary", None) is None:
            raise BuilderError(
                "Boundary not set in Plasma build. "
                "Ensure that one of run, read, or mock has been run before building."
            )

    def _read_equilibrium(self) -> Equilibrium:
        """
        Read a reference MHD equilibrium for the Plasma.
        """
        bluemira_print("Reading reference plasma MHD equilibrium.")

        return Equilibrium.from_eqdsk(self._eqdsk_path)

    def _create_equilibrium(self) -> Equilibrium:
        """
        Creates a reference MHD equilibrium for the Plasma.
        """

        def flatten_shape(x, z):
            """
            Flattens a shape by dragging the lowest and highest point to the minimum
            radius point.
            """
            amin, amax = np.argmin(z), np.argmax(z)
            xmin = np.min(x)
            zmin, zmax = np.min(z), np.max(z)
            xx = np.array(xmin)
            xx = np.append(xx, x[amin:amax])
            xx = np.append(xx, xmin)
            zz = np.array(zmin)
            zz = np.append(zz, z[amin:amax])
            zz = np.append(zz, zmax)
            return xx, zz

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
        x, z = flatten_shape(*tf_boundary.discretize(200, byedges=True).xz)
        tf_boundary = make_polygon({"x": x, "z": z})

        kappa = 1.12 * self._params.kappa_95.value

        coilset = make_coilset(
            tf_boundary,
            R_0=self._params.R_0.value,
            kappa=kappa,
            delta=self._params.delta_95.value,
            r_cs=self._params.r_cs_in.value + self._params.tk_cs.value / 2,
            tk_cs=self._params.tk_cs.value / 2,
            g_cs=self._params.g_cs_mod.value,
            tk_cs_ins=self._params.tk_cs_insulation.value,
            tk_cs_cas=self._params.tk_cs_casing.value,
            n_CS=self._params.n_CS.value,
            n_PF=self._params.n_PF.value,
            CS_jmax=self._params.CS_jmax.value,
            CS_bmax=self._params.CS_bmax.value,
            PF_jmax=self._params.PF_jmax.value,
            PF_bmax=self._params.PF_bmax.value,
        )

        kappa_ul_tweak = 0.05
        kappa_u = (1 - kappa_ul_tweak) * kappa
        kappa_l = (1 + kappa_ul_tweak) * kappa

        eq_targets = EUDEMOSingleNullConstraints(
            R_0=self._params.R_0.value,
            Z_0=0.0,
            A=self._params.A.value,
            kappa_u=kappa_u,
            kappa_l=kappa_l,
            delta_u=self._params.delta_95.value,
            delta_l=self._params.delta_95.value,
            psi_u_neg=0.0,
            psi_u_pos=0.0,
            psi_l_neg=60.0,
            psi_l_pos=30.0,
            div_l_ib=self._params.div_L2D_ib.value,
            div_l_ob=self._params.div_L2D_ob.value,
            psibval=0.0,
            psibtol=1.0e-3,
            lower=True,
            n=100,
        )

        profiles = BetaIpProfile(
            self._params.beta_p.value,
            self._params.I_p.value * 1e6,
            self._params.R_0.value,
            self._params.B_0.value,
        )

        sx, sz = 1.6, 1.7  # grid scales from plasma
        nx, nz = 65, 65
        R_0 = self._params.R_0.value
        A = self._params.A.value
        x_min, x_max = R_0 - sx * (R_0 / A), R_0 + sx * (R_0 / A)
        z_min, z_max = -sz * (kappa * R_0 / A), sz * (kappa * R_0 / A)

        grid = Grid(x_min, x_max, z_min, z_max, nx, nz)

        eq = Equilibrium(
            coilset,
            grid,
            profiles,
        )
        opt_problem = UnconstrainedTikhonovCurrentGradientCOP(
            eq.coilset,
            eq,
            eq_targets,
            gamma=1e-8,
        )
        program = PicardIterator(
            eq,
            opt_problem,
            convergence=DudsonConvergence(),
            relaxation=0.2,
            fixed_coils=True,
            plot=self._plot_flag,
        )
        program()
        self._design_problem = opt_problem
        return eq

    def _analyse_equilibrium(self, eq: Equilibrium):
        """
        Analyse an equilibrium and store important values in the Plasma parameters. Also
        updates the equilibrium and boundary to ensure that they are kept synchronised
        with the parameters.

        Parameters
        ----------
        eq: Equilibrium
            The equilibrium to analyse for use with this builder.
        """
        plasma_dict = eq.analyse_plasma()

        dx_shaf = plasma_dict["dx_shaf"]
        dz_shaf = plasma_dict["dz_shaf"]
        shaf = np.hypot(dx_shaf, dz_shaf)

        params = {
            "I_p": plasma_dict["Ip"] / 1e6,
            "q_95": plasma_dict["q_95"],
            "beta_p": plasma_dict["beta_p"],
            "l_i": plasma_dict["li"],
            "delta_95": plasma_dict["delta_95"],
            "kappa_95": plasma_dict["kappa_95"],
            "delta": plasma_dict["delta"],
            "kappa": plasma_dict["kappa"],
            "shaf_shift": shaf,
        }
        self._params.update_kw_parameters(params, source="equilibria")

        self._equilibrium = eq
        self._boundary = make_polygon(eq.get_LCFS().xyz, "LCFS")

    def build(self) -> PlasmaComponent:
        """
        Build the plasma components.

        Returns
        -------
        plasma_component: Component
            The PlasmaComponent produced by this build.
        """
        super().build()

        component = PlasmaComponent(self._name, equilibrium=self._equilibrium)

        component.add_child(self.build_xz())
        component.add_child(self.build_xy())
        component.add_child(self.build_xyz())

        return component

    def build_xz(self) -> Component:
        """
        Build the xz representation of this plasma.

        Generates the LCFS from the _boundary defined on the builder and includes the
        separatrix from the equilibrium, if the equilibrium is provided.

        Returns
        -------
        component: Component
            The component grouping the results in the xz plane.

        Raises
        ------
        BuilderError
            If the _boundary has not been defined e.g. by running run, read, or mock
            after reinitialisation.
        """
        self._ensure_boundary()

        component = Component("xz")

        if self._equilibrium is not None:
            sep_loop = self._equilibrium.get_separatrix()
            sep_wire = make_polygon(sep_loop.xyz, label="Separatrix")
            sep_component = PhysicalComponent("Separatrix", sep_wire)
            sep_component.plot_options.wire_options["color"] = BLUE_PALETTE["PL"]
            component.add_child(sep_component)

        lcfs_face = BluemiraFace(self._boundary, label="LCFS")
        lcfs_component = PhysicalComponent("LCFS", lcfs_face)
        lcfs_component.plot_options.wire_options["color"] = BLUE_PALETTE["PL"]
        lcfs_component.plot_options.face_options["color"] = BLUE_PALETTE["PL"]
        component.add_child(lcfs_component)

        bm_plot_tools.set_component_view(component, "xz")

        return component

    def build_xy(self) -> Component:
        """
        Build the xy representation of this plasma.

        Generates the LCFS from the _boundary defined on the builder as a ring bound by
        the maximum and minimum radial extent of the boundary.

        Returns
        -------
        component: Component
            The component grouping the results in the xy plane.

        Raises
        ------
        BuilderError
            If the _boundary has not been defined e.g. by running run, read, or mock
            after reinitialisation.
        """
        self._ensure_boundary()

        component = Component("xy")

        inner = make_circle(self._boundary.bounding_box.x_min, axis=[0, 0, 1])
        outer = make_circle(self._boundary.bounding_box.x_max, axis=[0, 0, 1])

        lcfs_face = BluemiraFace([outer, inner], label="LCFS")
        lcfs_component = PhysicalComponent("LCFS", lcfs_face)
        lcfs_component.plot_options.wire_options["color"] = BLUE_PALETTE["PL"]
        lcfs_component.plot_options.face_options["color"] = BLUE_PALETTE["PL"]

        component.add_child(lcfs_component)

        bm_plot_tools.set_component_view(component, "xy")

        return component

    def build_xyz(self, degree: float = 360.0) -> Component:
        """
        Build the 3D representation of this plasma.

        Generates the LCFS from the _boundary defined on the builder by revolving around
        the z axis.

        Parameters
        ----------
        degree: float
            The angle [°] around which to build the components, by default 360.0.

        Returns
        -------
        component: Component
            The component grouping the results in 3D (xyz).

        Raises
        ------
        BuilderError
            If the _boundary has not been defined e.g. by running run, read, or mock
            after reinitialisation.
        """
        self._ensure_boundary()

        shell = revolve_shape(self._boundary, direction=(0, 0, 1), degree=degree)
        component = PhysicalComponent("LCFS", shell)
        component.display_cad_options.color = BLUE_PALETTE["PL"]
        component.display_cad_options.transparency = 0.5

        return Component("xyz").add_child(component)
